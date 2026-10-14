"""Metrics of generated motions (e.g. mdm_validation_test outputs) per directory: distance
of the last frame to the last path node, motion length, jerk, and contact/penetration losses.
Writes one CSV row per directory.

    python -m scripts.motion_tests.compute_losses [--motion_dirs DIR [DIR ...]]
        [--char_file data/assets/humanoid.xml] [--device cuda:0] [--output parc_gen_metrics.csv]

Motion files are MS files named <terrain>_<i>.pkl, with terrain names PATH_TERRAIN_0 ...
PATH_TERRAIN_99, and with path_nodes in their misc_data. Paths may use $DATA_DIR.
"""
import argparse
import csv
import os
import re
import sys
from collections import OrderedDict

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch  # noqa: E402

import parc.anim.kin_char_model as kin_char_model  # noqa: E402
import parc.motion_synthesis.procgen.mdm_path as mdm_path  # noqa: E402
import parc.util.file_io_helper as file_io_helper  # noqa: E402
import parc.util.geom_util as geom_util  # noqa: E402
import parc.util.motion_util as motion_util  # noqa: E402
import parc.util.path_loader as path_loader  # noqa: E402
import parc.util.terrain_util as terrain_util  # noqa: E402

# Load the test motions
#motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_07_model_75000/output/"
#motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_16_model_12500/output/"
####motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_18_model_12500/output/"
#motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_18_model_25000/output/"
#motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_16_no_opt_model_12500/output/"
#motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_21_model_30000/output/"
#motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_21_model_30000_cfg_0/output/"
# motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_21_model_30000_cfg_0_25/output/"
# motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_21_model_30000_cfg_0_5/output/"
# motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_21_model_30000_cfg_0_75/output/"
#motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_21_model_30000_cfg_1/output/"
# motion_dir_path = "$DATA_DIR/tests/val_tests/paths_v_23_model_30000_test_03_cfg_0_55/output/"
#motion_dir_path = "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch2500/output/"
motion_dirs = [
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch2500/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch5000/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch7500/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch10000/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch12500/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch15000/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch17500/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch20000/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch22500/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch25000/output/",
    "$DATA_DIR/tests/val_tests/april_parc/iter1_epoch27500/output/"
]

DEFAULT_CHAR_FILE = "data/assets/humanoid.xml"
NUM_PATH_TERRAINS = 100
MAX_JERK = 11666.3906 #10000.0


def compute_csv_header():
    header = ["exp_name",
              "final_node_dist mean",
              "final_node_dist std",
              "motion_length mean",
              "motion_length std",
              "mean_jerk mean",
              "mean_jerk std",
              "frames_with_jerk_over_X mean",
              "frames_with_jerk_over_X std",
              "contact_loss mean",
              "contact_loss std",
              "pen_loss mean",
              "pen_loss std"]

    for i in range(NUM_PATH_TERRAINS):
        terrain_name = "PATH_TERRAIN_" + str(i)
        header.append(terrain_name + "final node dist mean")
        header.append(terrain_name + "final node dist std")
        header.append(terrain_name + "motion length mean")
        header.append(terrain_name + "motion length std")
        header.append(terrain_name + "mean jerk mean")
        header.append(terrain_name + "mean jerk std")
        header.append(terrain_name + "frames_with_jerk_over_X mean")
        header.append(terrain_name + "frames_with_jerk_over_X std")
        header.append(terrain_name + "contact loss mean")
        header.append(terrain_name + "contact loss std")
        header.append(terrain_name + "pen loss mean")
        header.append(terrain_name + "pen loss std")

    return header


def load_generated_motion(motion_path, device):
    """Returns (MotionFrames [num_frames], terrain, path_nodes, fps) of an MS motion file."""
    file_data = file_io_helper.load_ms_file(motion_path, device=device)
    motion_data = file_data.motion_data
    if motion_data is None or file_data.terrain_data is None:
        raise ValueError("{}: needs motion and terrain data".format(motion_path))
    misc_data = file_data.misc_data if file_data.misc_data is not None else dict()
    if file_io_helper.PATH_NODES_KEY not in misc_data:
        raise ValueError("{}: no path_nodes in misc_data".format(motion_path))

    motion_frames = motion_util.MotionFrames(root_pos=motion_data.root_pos, root_rot=motion_data.root_rot,
                                             joint_rot=motion_data.joint_rot, contacts=motion_data.body_contacts)
    terrain = terrain_util.SubTerrain.from_ms_terrain_data(file_data.terrain_data, device=device)
    return motion_frames, terrain, misc_data[file_io_helper.PATH_NODES_KEY], motion_data.fps


def compute_csv_row(motion_dir_path, char_model_path=DEFAULT_CHAR_FILE, device=None):
    compute_geom_losses = True
    max_jerk = MAX_JERK

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(str(path_loader.resolve_path(char_model_path)))

    body_points = geom_util.get_char_point_samples(char_model)

    motion_dir_path = str(path_loader.resolve_path(motion_dir_path))
    motion_files = sorted(os.listdir(motion_dir_path))

    output_data = [motion_dir_path]


    # TODO: organize losses based on terrain too

    total_losses = OrderedDict()
    for i in range(NUM_PATH_TERRAINS):
        terrain_name = "PATH_TERRAIN_" + str(i)

        total_losses[terrain_name] = {
            "final_node_dist": [],
            "motion_length": [],
            "pen_loss": [],
            "contact_loss": [],
            "mean_jerk": [],
            "frames_with_jerk_over_X": []
        }

    num_motion_files = len(motion_files)
    for i in range(num_motion_files):
        if i % 50 == 0:
            print(i, "/", num_motion_files)
        motion_file = motion_files[i]
        if os.path.splitext(motion_file)[1] != ".pkl":
            continue

        terrain_name = re.sub(r'_\d+$', '', os.path.splitext(motion_file)[0])
        if terrain_name not in total_losses:
            raise ValueError("{}: expected a file named PATH_TERRAIN_<0-{}>_<i>.pkl".format(
                motion_file, NUM_PATH_TERRAINS - 1))

        motion_path = os.path.join(motion_dir_path, motion_file)

        motion_frames, terrain, path_nodes, fps = load_generated_motion(motion_path, device)
        root_pos = motion_frames.root_pos

        final_node_dist = root_pos[-1, 0:2] - path_nodes[-1, 0:2]

        final_node_dist = torch.linalg.norm(final_node_dist).item()

        total_losses[terrain_name]["final_node_dist"].append(final_node_dist)

        motion_length = root_pos.shape[0] / fps
        total_losses[terrain_name]["motion_length"].append(motion_length)

        body_pos, body_rot = char_model.forward_kinematics(root_pos, motion_frames.root_rot, motion_frames.joint_rot)

        dt = 1.0 / fps
        body_vel = (body_pos[1:] - body_pos[:-1]) / dt
        body_acc = (body_vel[1:] - body_vel[:-1]) / dt
        body_jerk = (body_acc[1:] - body_acc[:-1]) / dt
        body_jerk_magnitude = torch.linalg.norm(body_jerk, dim=-1)
        mean_jerk = torch.mean(body_jerk_magnitude).item()
        total_losses[terrain_name]["mean_jerk"].append(mean_jerk)


        frames_with_jerk_over_X = torch.count_nonzero(body_jerk_magnitude > max_jerk).item()
        frames_with_jerk_over_X = frames_with_jerk_over_X / body_jerk_magnitude.shape[0]
        total_losses[terrain_name]["frames_with_jerk_over_X"].append(frames_with_jerk_over_X)

        if compute_geom_losses:
            # compute_motion_loss indexes contacts by body id
            contacts = motion_frames.contacts
            if contacts is None:
                contacts = torch.zeros(size=[root_pos.shape[0], char_model.get_num_contact_bodies()],
                                       dtype=torch.float32, device=device)
            if contacts.shape[-1] != char_model.get_num_bodies():
                contacts = char_model.create_full_contact_tensor(contacts)
            compute_motion_frames = motion_util.MotionFrames(root_pos=root_pos,
                                                             root_rot=motion_frames.root_rot,
                                                             joint_rot=motion_frames.joint_rot,
                                                             contacts=contacts).unsqueeze(0)

            losses = mdm_path.compute_motion_loss(
                motion_frames=compute_motion_frames,
                path_nodes=path_nodes,
                terrain=terrain,
                char_model=char_model,
                body_points=body_points,
                w_contact=1.0,
                w_pen=1.0,
                w_path=1.0,
                verbose=False)

            contact_loss = losses["contact_loss"].item()
            pen_loss = losses["pen_loss"].item()

            total_losses[terrain_name]["contact_loss"].append(contact_loss)
            total_losses[terrain_name]["pen_loss"].append(pen_loss)


    all_terrain_final_node_dists = []
    all_terrain_motion_lengths = []
    all_terrain_mean_jerks = []
    all_terrain_max_jerk_frames = []
    all_terrain_contact_losses = []
    all_terrain_pen_losses = []

    terrain_output_data = []
    for terrain_name in total_losses:
        final_node_dists = total_losses[terrain_name]["final_node_dist"]
        final_node_dists = torch.tensor(final_node_dists, dtype=torch.float64, device="cpu")
        terrain_output_data.append(final_node_dists.mean().item())
        terrain_output_data.append(final_node_dists.std().item())
    #print(final_node_dists.max().item())

        motion_lengths = total_losses[terrain_name]["motion_length"]
        motion_lengths = torch.tensor(motion_lengths, dtype=torch.float64, device="cpu")
        terrain_output_data.append(motion_lengths.mean().item())
        terrain_output_data.append(motion_lengths.std().item())


        mean_jerks = total_losses[terrain_name]["mean_jerk"]
        mean_jerks = torch.tensor(mean_jerks, dtype=torch.float64, device="cpu")
        terrain_output_data.append(mean_jerks.mean().item())
        terrain_output_data.append(mean_jerks.std().item())


        num_frames_max_jerk = total_losses[terrain_name]["frames_with_jerk_over_X"]
        num_frames_max_jerk = torch.tensor(num_frames_max_jerk, dtype=torch.float64, device="cpu")
        terrain_output_data.append(num_frames_max_jerk.mean().item())
        terrain_output_data.append(num_frames_max_jerk.std().item())

        all_terrain_final_node_dists.append(final_node_dists)
        all_terrain_motion_lengths.append(motion_lengths)
        all_terrain_mean_jerks.append(mean_jerks)
        all_terrain_max_jerk_frames.append(num_frames_max_jerk)

        if compute_geom_losses:
            #print(terrain_name)
            motion_contact_losses = total_losses[terrain_name]["contact_loss"]
            motion_contact_losses = torch.tensor(motion_contact_losses, dtype=torch.float64, device="cpu")
            terrain_output_data.append(motion_contact_losses.mean().item())
            terrain_output_data.append(motion_contact_losses.std().item())

            motion_pen_losses = total_losses[terrain_name]["pen_loss"]
            motion_pen_losses = torch.tensor(motion_pen_losses, dtype=torch.float64, device="cpu")
            terrain_output_data.append(motion_pen_losses.mean().item())
            terrain_output_data.append(motion_pen_losses.std().item())

            all_terrain_contact_losses.append(motion_contact_losses)
            all_terrain_pen_losses.append(motion_pen_losses)

    print("")
    print("*****Final Metrics*****")
    all_terrain_final_node_dists = torch.cat(all_terrain_final_node_dists)
    all_terrain_motion_lengths = torch.cat(all_terrain_motion_lengths)
    all_terrain_mean_jerks = torch.cat(all_terrain_mean_jerks)
    all_terrain_max_jerk_frames = torch.cat(all_terrain_max_jerk_frames)

    output_data.append(all_terrain_final_node_dists.mean().item())
    output_data.append(all_terrain_final_node_dists.std().item())
    output_data.append(all_terrain_motion_lengths.mean().item())
    output_data.append(all_terrain_motion_lengths.std().item())
    output_data.append(all_terrain_mean_jerks.mean().item())
    output_data.append(all_terrain_mean_jerks.std().item())
    output_data.append(all_terrain_max_jerk_frames.mean().item())
    output_data.append(all_terrain_max_jerk_frames.std().item())
    if compute_geom_losses:
        all_terrain_contact_losses = torch.cat(all_terrain_contact_losses)
        all_terrain_pen_losses = torch.cat(all_terrain_pen_losses)
        output_data.append(all_terrain_contact_losses.mean().item())
        output_data.append(all_terrain_contact_losses.std().item())
        output_data.append(all_terrain_pen_losses.mean().item())
        output_data.append(all_terrain_pen_losses.std().item())

    output_data.extend(terrain_output_data)
    return output_data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Metrics of generated motions, one CSV row per directory.")
    parser.add_argument("--motion_dirs", nargs="+", default=motion_dirs)
    parser.add_argument("--char_file", default=DEFAULT_CHAR_FILE)
    parser.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
    parser.add_argument("--output", default="parc_gen_metrics.csv")
    args = parser.parse_args(argv)

    data = []
    data.append(compute_csv_header())

    for motion_dir_path in args.motion_dirs:
        data.append(compute_csv_row(motion_dir_path, char_model_path=args.char_file, device=args.device))

    with open(str(path_loader.resolve_path(args.output)), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
