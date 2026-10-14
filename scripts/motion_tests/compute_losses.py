import sys

sys.path.insert(1, sys.path[0] + ("/../.."))
import csv
import os
import re
from collections import OrderedDict

import torch
import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.motion_synthesis.procgen.mdm_path as mdm_path
import parc.util.geom_util as geom_util
import parc.util.motion_edit_lib as medit_lib
import parc.util.motion_util as motion_util

# Load the test motions
#motion_dir_path = "../tests/val_tests/paths_v_07_model_75000/output/"
#motion_dir_path = "../tests/val_tests/paths_v_16_model_12500/output/"
####motion_dir_path = "../tests/val_tests/paths_v_18_model_12500/output/"
#motion_dir_path = "../tests/val_tests/paths_v_18_model_25000/output/"
#motion_dir_path = "../tests/val_tests/paths_v_16_no_opt_model_12500/output/"
#motion_dir_path = "../tests/val_tests/paths_v_21_model_30000/output/"
#motion_dir_path = "../tests/val_tests/paths_v_21_model_30000_cfg_0/output/"
# motion_dir_path = "../tests/val_tests/paths_v_21_model_30000_cfg_0_25/output/"
# motion_dir_path = "../tests/val_tests/paths_v_21_model_30000_cfg_0_5/output/"
# motion_dir_path = "../tests/val_tests/paths_v_21_model_30000_cfg_0_75/output/"
#motion_dir_path = "../tests/val_tests/paths_v_21_model_30000_cfg_1/output/"
# motion_dir_path = "../tests/val_tests/paths_v_23_model_30000_test_03_cfg_0_55/output/"
#motion_dir_path = "../tests/val_tests/april_parc/iter1_epoch2500/output/"
motion_dirs = [
    "../tests/val_tests/april_parc/iter1_epoch2500/output/",
    "../tests/val_tests/april_parc/iter1_epoch5000/output/",
    "../tests/val_tests/april_parc/iter1_epoch7500/output/",
    "../tests/val_tests/april_parc/iter1_epoch10000/output/",
    "../tests/val_tests/april_parc/iter1_epoch12500/output/",
    "../tests/val_tests/april_parc/iter1_epoch15000/output/",
    "../tests/val_tests/april_parc/iter1_epoch17500/output/",
    "../tests/val_tests/april_parc/iter1_epoch20000/output/",
    "../tests/val_tests/april_parc/iter1_epoch22500/output/",
    "../tests/val_tests/april_parc/iter1_epoch25000/output/",
    "../tests/val_tests/april_parc/iter1_epoch27500/output/"
]


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
    
    for i in range(100):
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


def compute_csv_row(motion_dir_path):
    compute_geom_losses = True
    max_jerk = 11666.3906 #10000.0

    char_model_path = "data/assets/humanoid.xml"
    device = "cuda:0"
    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(char_model_path)

    body_points = geom_util.get_char_point_samples(char_model)

    motion_files = os.listdir(motion_dir_path)

    final_node_dists = []
    motion_lengths = []

    # NOTE: maybe try this stat just for completed motions?
    motion_pen_losses = []
    motion_contact_losses = []

    output_data = [motion_dir_path]


    # TODO: organize losses based on terrain too

    total_losses = OrderedDict()
    for i in range(100):
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

        
        terrain_name = result = re.sub(r'_\d+$', '', os.path.splitext(motion_file)[0])
        #print(terrain_name)
        if terrain_name not in total_losses:
            assert False

        motion_path = os.path.join(motion_dir_path, motion_file)

        motion_data = medit_lib.load_motion_file(motion_path, device=device)
        #motion_data.set_device(device=device)

        path_nodes = motion_data._data["path_nodes"]
        motion_frames = motion_data.get_frames()
        terrain = motion_data.get_terrain()
        contact_frames = motion_data.get_contacts()

        if len(motion_frames.shape) == 3 and motion_frames.shape[0] == 1:
            motion_frames = motion_frames.squeeze(0)

        if len(contact_frames.shape) == 3 and contact_frames.shape[0] == 1:
            contact_frames = contact_frames.squeeze(0)

        final_node_dist = motion_frames[-1, 0:2] - path_nodes[-1, 0:2]

        final_node_dist = torch.linalg.norm(final_node_dist).item()

        total_losses[terrain_name]["final_node_dist"].append(final_node_dist)

        motion_length = motion_frames.shape[0] / 30.0
        total_losses[terrain_name]["motion_length"].append(motion_length)


        # TODO: fix
        root_pos, root_rot, joint_rot = char_model.extract_frame_data(motion_frames)
        
        body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        dt = 1.0 / 30.0
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
            compute_motion_frames = motion_util.motion_frames_from_mlib_format(
                motion_frames, char_model=char_model, contacts=contact_frames
            ).unsqueeze(0)

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
        terrain_output_data.append(num_frames_max_jerk.mean())
        terrain_output_data.append(num_frames_max_jerk.std())

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


    

if __name__ == "__main__":

    data = []
    data.append(compute_csv_header())

    for motion_dir_path in motion_dirs:
        data.append(compute_csv_row(motion_dir_path))
    
    with open('parc_gen_metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)