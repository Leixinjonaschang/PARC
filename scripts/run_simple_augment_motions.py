"""
Script used to augment initial dataset with different terrain heights.
Also randomly rotates/squishes/stretches motions for a bit more spatial variation.
Motions have to then be post-processed with kinematic optimization, and then deepmimic for best results.

    python -m scripts.run_simple_augment_motions --config parc/motion_synthesis/motion_opt/config/platform_aug.yaml

Config keys (paths may use $DATA_DIR): motions_yaml_path (a .yaml listing
`motions: [{file, weight}]` of MS files), num_new_motions, device, output_folder_path,
char_model, use_wandb, num_iters, step_size, w_root_pos, w_root_rot, w_joint_rot,
w_smoothness, w_penetration, w_contact, w_sliding, max_motion_len, terrain_aug_type
(HEIGHT_SCALE: min_h_scale, max_h_scale, optional bad_h_range; BOXES_ALONG_PATH: min_len,
max_len, min_angle, max_angle, min_num_boxes, max_num_boxes, min_h, max_h; NONE),
char_point_samples. Optional (defaults leave the motion unchanged): min/max_heading_angle
(degrees, 0), x_scale/y_scale ([1, 1]), terrain_padding_size (0), slice_terrain (False),
sample_weight_by_length (False), w_jerk (0, off), max_jerk (1000).

Writes <output_folder_path>/<name>_aug<k>.pkl (MS format) and logs to <output_folder_path>/log/.
"""
import argparse
import enum
import os
import random
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch  # noqa: E402

import parc.motion_synthesis.motion_opt.motion_optimization as moopt  # noqa: E402
import parc.motion_synthesis.motion_opt.script_util as script_util  # noqa: E402
import parc.util.file_io as file_io  # noqa: E402
import parc.util.file_io_helper as file_io_helper  # noqa: E402
import parc.util.terrain_util as terrain_util  # noqa: E402
import parc.util.torch_util as torch_util  # noqa: E402

DEFAULT_CONFIG = "parc/motion_synthesis/motion_opt/config/platform_aug.yaml"


class TerrainAugmentationType(enum.Enum):
    HEIGHT_SCALE = 0
    BOXES_ALONG_PATH = 1
    NONE = 2

def rand_float(f_min, f_max):
    return random.random() * (f_max - f_min) + f_min


def load_motions(motions_yaml_path, char_model, device):
    motion_filepaths, _ = script_util.fetch_motion_files(motions_yaml_path)
    motions = []
    for filepath in motion_filepaths:
        frames, terrain, _, fps = script_util.load_motion(filepath, char_model, device)
        if terrain is None:
            raise ValueError("{}: no terrain; terrain augmentation needs one".format(filepath))
        motions.append((filepath, frames, terrain, fps))
    return motions


def augment_motion(frames, terrain, fps, cfg, terrain_aug_type, device):
    """Returns (augmented frames, augmented terrain) before optimization; modifies copies."""
    frames = frames.get_copy(device)
    terrain = terrain.torch_copy()
    terrain.set_device(device)

    max_num_frames = int(round(fps * cfg["max_motion_len"]))
    src_num_frames = frames.root_pos.shape[0]
    if src_num_frames > max_num_frames:
        # pick a random starting frame then slice motion and terrain
        start_frame_idx = random.randint(0, src_num_frames - max_num_frames - 1)
        frames = frames.get_slice(slice(start_frame_idx, start_frame_idx + max_num_frames))

    root_pos = frames.root_pos.clone()

    # randomly rotate around the origin
    heading_angle = rand_float(cfg.get("min_heading_angle", 0.0), cfg.get("max_heading_angle", 0.0))
    heading_angle = torch.tensor([heading_angle * torch.pi / 180.0], dtype=torch.float32, device=device)
    root_pos[..., 0:2] = torch_util.rotate_2d_vec(root_pos[..., 0:2], heading_angle)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    delta_heading_quat = torch_util.axis_angle_to_quat(z_axis, heading_angle)
    frames.root_rot = torch_util.quat_multiply(delta_heading_quat, frames.root_rot)

    # also randomly stretch/shrink motions along a random direction
    x_scale = cfg.get("x_scale", [1.0, 1.0])
    y_scale = cfg.get("y_scale", [1.0, 1.0])
    root_pos[..., 0] *= rand_float(x_scale[0], x_scale[1])
    root_pos[..., 1] *= rand_float(y_scale[0], y_scale[1])

    terrain_padding_size = cfg.get("terrain_padding_size", 0)
    if terrain_padding_size > 0:
        terrain.pad(padding_size=terrain_padding_size)
    if cfg.get("slice_terrain", False):
        terrain, root_pos = terrain_util.slice_terrain_around_motion(root_pos, terrain,
                                                                     padding=terrain.dxdy[0].item() * 2)
    frames.root_pos = root_pos

    # randomly change height
    if terrain_aug_type == TerrainAugmentationType.HEIGHT_SCALE:
        bad_h_range = cfg.get("bad_h_range", None)
        height_scale = rand_float(cfg["min_h_scale"], cfg["max_h_scale"])
        while bad_h_range is not None and bad_h_range[0] < height_scale < bad_h_range[1]:
            height_scale = rand_float(cfg["min_h_scale"], cfg["max_h_scale"])
        terrain.hf = height_scale * terrain.hf
    elif terrain_aug_type == TerrainAugmentationType.BOXES_ALONG_PATH:
        num_boxes = random.randint(cfg["min_num_boxes"], cfg["max_num_boxes"])
        frame_inds = torch.randint(0, frames.root_pos.shape[0], size=[num_boxes], device=device, dtype=torch.int64)
        box_centers = frames.root_pos[frame_inds, 0:2] - terrain.min_point
        box_centers = box_centers / terrain.dxdy
        terrain_util.add_boxes_to_hf_at_xy_points(
            box_centers=box_centers,
            hf=terrain.hf,
            min_h=cfg["min_h"],
            max_h=cfg["max_h"],
            min_len=cfg["min_len"],
            max_len=cfg["max_len"],
            min_angle=cfg["min_angle"],
            max_angle=cfg["max_angle"]
        )

    return frames, terrain


def main(argv=None):
    parser = argparse.ArgumentParser(description="Augment MS motions with terrain variations, then optimize them.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="augmentation config yaml")
    args = parser.parse_args(argv)
    print("loading config from", args.config)
    cfg = script_util.load_config(args.config)

    device = cfg["device"]
    num_new_motions = cfg["num_new_motions"]
    output_folder_path = script_util.resolve(cfg["output_folder_path"])
    log_folder_path = os.path.join(output_folder_path, "log")
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(log_folder_path, exist_ok=True)

    terrain_aug_type = TerrainAugmentationType[cfg["terrain_aug_type"]]

    char_model = script_util.load_char_model(cfg["char_model"], device)
    body_points = script_util.get_body_points(char_model, cfg["char_point_samples"])

    try:
        motions = load_motions(cfg["motions_yaml_path"], char_model, device)
    except file_io.MSFileFormatError as exc:
        sys.exit("error: {}".format(exc))

    num_initial_motions = len(motions)
    num_variations_per_motion = [0] * num_initial_motions

    motion_lengths = [frames.root_pos.shape[0] / fps for _, frames, _, fps in motions]
    motion_lengths = torch.tensor(motion_lengths, dtype=torch.float32, device=device)
    if cfg.get("sample_weight_by_length", False):
        sample_weights = motion_lengths
    else:
        sample_weights = torch.ones_like(motion_lengths)
    file_indices = torch.multinomial(sample_weights, num_samples=num_new_motions, replacement=True)

    for new_motion_idx in range(num_new_motions):
        # change selection to be multinomial, based on length of each motion
        file_index = int(file_indices[new_motion_idx].item())
        filepath, src_frames, src_terrain, fps = motions[file_index]
        input_motion_name = os.path.basename(os.path.splitext(filepath)[0])
        print("AUGMENTING MOTION:", input_motion_name)
        print("MOTION", str(new_motion_idx) + "/" + str(num_new_motions))
        num_variations_per_motion[file_index] += 1

        frames, terrain = augment_motion(src_frames, src_terrain, fps, cfg, terrain_aug_type, device)

        output_motion_name = input_motion_name + "_aug" + str(num_variations_per_motion[file_index]).zfill(3)
        log_file = os.path.join(log_folder_path, "log_" + output_motion_name + ".txt")

        opt_frames = moopt.motion_contact_optimization(src_motion_frames=frames,
                                                       body_points=body_points,
                                                       terrain=terrain,
                                                       char_model=char_model,
                                                       num_iters=cfg["num_iters"],
                                                       step_size=cfg["step_size"],
                                                       w_root_pos=cfg["w_root_pos"],
                                                       w_root_rot=cfg["w_root_rot"],
                                                       w_joint_rot=cfg["w_joint_rot"],
                                                       w_smoothness=cfg["w_smoothness"],
                                                       w_penetration=cfg["w_penetration"],
                                                       w_contact=cfg["w_contact"],
                                                       w_sliding=cfg["w_sliding"],
                                                       w_body_constraints=0.0,
                                                       w_jerk=cfg.get("w_jerk", 0.0),
                                                       max_jerk=cfg.get("max_jerk", 1000.0),
                                                       body_constraints=None,
                                                       exp_name=output_motion_name,
                                                       use_wandb=cfg["use_wandb"],
                                                       log_file=log_file)

        output_filepath = os.path.join(output_folder_path, output_motion_name + ".pkl")
        opt_frames.set_device("cpu")
        terrain.set_device("cpu")
        file_io_helper.save_ms_file(output_filepath,
                                    motion_frames=opt_frames,
                                    fps=fps,
                                    loop_mode="CLAMP",
                                    terrain=terrain)
        print("Saved", output_filepath)


if __name__ == "__main__":
    main()
