"""Kinematic optimization of MS motion files (penetration, contact, sliding, smoothness and
jerk losses, optional body constraints).

    python -m scripts.run_optimize_motions --config parc/motion_synthesis/motion_opt/config/motion_opt.yaml

Config keys (paths may use $DATA_DIR): motions_yaml_path (an MS .pkl, or a .yaml listing
`motions: [{file, weight}]`), char_model, device, output_folder_path, num_iters, step_size,
w_root_pos, w_root_rot, w_joint_rot, w_smoothness, w_penetration, w_contact, w_sliding,
w_body_constraints, frame_stride, use_wandb, char_point_samples. Optional: w_jerk (default
0, i.e. off), max_jerk (1000), auto_compute_body_constraints (False: use the constraints
stored in the file, if any).

Writes <output_folder_path>/<name>_opt.pkl (MS format) and logs to <output_folder_path>/log/.
Legacy motion pickles must be converted first (scripts/convert_legacy_motions.py).
"""
import argparse
import math
import os
import sys
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import parc.motion_synthesis.motion_opt.motion_optimization as moopt  # noqa: E402
import parc.motion_synthesis.motion_opt.script_util as script_util  # noqa: E402
import parc.util.file_io as file_io  # noqa: E402
import parc.util.file_io_helper as file_io_helper  # noqa: E402

DEFAULT_CONFIG = "parc/motion_synthesis/motion_opt/config/motion_opt.yaml"


def fetch_motion_files(motion_file):
    return script_util.fetch_motion_files(motion_file)


def stride_body_constraints(body_constraints, frame_stride, device):
    # constraint frame ranges refer to the full-rate motion; map them to the strided frames
    for body_constraint_list in body_constraints:
        for constraint in body_constraint_list:
            constraint.start_frame_idx = int(math.ceil(constraint.start_frame_idx / frame_stride))
            constraint.end_frame_idx = int(math.floor(constraint.end_frame_idx // frame_stride))
            constraint.constraint_point = constraint.constraint_point.to(device=device)
    return body_constraints


def optimize_motion_file(filepath, cfg, char_model, body_points, output_folder_path, log_folder_path):
    device = cfg["device"]
    frame_stride = cfg["frame_stride"]

    input_motion_name = os.path.basename(os.path.splitext(filepath)[0])
    output_motion_name = input_motion_name + "_opt"

    src_frames, terrain, misc_data, fps = script_util.load_motion(filepath, char_model, device)
    if terrain is None:
        raise ValueError("{}: no terrain; motion optimization needs one".format(filepath))

    if cfg.get("auto_compute_body_constraints", False):
        print("Computing approx body constraints...")
        body_constraints = moopt.compute_approx_body_constraints(root_pos=src_frames.root_pos,
                                                                 root_rot=src_frames.root_rot,
                                                                 joint_rot=src_frames.joint_rot,
                                                                 contacts=src_frames.contacts,
                                                                 char_model=char_model,
                                                                 terrain=terrain)
        print("Finished computing approx body constraints.")
    else:
        body_constraints = misc_data.get(file_io_helper.BODY_CONSTRAINTS_KEY, None)

    if body_constraints is not None:
        body_constraints = stride_body_constraints(body_constraints, frame_stride, device)

    src_frames = src_frames.get_slice(slice(None, None, frame_stride))
    if fps % frame_stride != 0:
        print("Note: fps {} is not a multiple of frame_stride {}; saving at {} fps".format(
            fps, frame_stride, fps // frame_stride))

    log_file = os.path.join(log_folder_path, "log_" + output_motion_name + ".txt")

    opt_frames = moopt.motion_contact_optimization(src_motion_frames=src_frames,
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
                                                   w_body_constraints=cfg["w_body_constraints"],
                                                   w_jerk=cfg.get("w_jerk", 0.0),
                                                   max_jerk=cfg.get("max_jerk", 1000.0),
                                                   body_constraints=body_constraints,
                                                   exp_name=output_motion_name,
                                                   use_wandb=cfg["use_wandb"],
                                                   log_file=log_file)

    output_filepath = os.path.join(output_folder_path, output_motion_name + ".pkl")
    opt_frames.set_device("cpu")
    terrain.set_device("cpu")
    file_io_helper.save_ms_file(output_filepath,
                                motion_frames=opt_frames,
                                fps=fps // frame_stride,
                                loop_mode="CLAMP",
                                terrain=terrain,
                                body_constraints=body_constraints)
    print("Saved", output_filepath)
    return output_filepath


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kinematic optimization of MS motion files.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="motion-opt config yaml")
    args = parser.parse_args(argv)
    print("loading config from", args.config)
    cfg = script_util.load_config(args.config)

    motion_filepaths, _ = fetch_motion_files(cfg["motions_yaml_path"])
    num_motions = len(motion_filepaths)

    output_folder_path = script_util.resolve(cfg["output_folder_path"])
    log_folder_path = os.path.join(output_folder_path, "log")
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(log_folder_path, exist_ok=True)

    char_model = script_util.load_char_model(cfg["char_model"], cfg["device"])
    body_points = script_util.get_body_points(char_model, cfg["char_point_samples"])

    start_time = time.time()
    for motion_idx, filepath in enumerate(motion_filepaths):
        print("OPTIMIZING MOTION:", os.path.basename(filepath))
        print("MOTION", str(motion_idx) + "/" + str(num_motions))
        try:
            optimize_motion_file(filepath, cfg, char_model, body_points, output_folder_path, log_folder_path)
        except file_io.MSFileFormatError as exc:
            sys.exit("error: {}".format(exc))
    end_time = time.time()

    print("Total optimization time for", num_motions, "motions:", end_time - start_time, "seconds.")


if __name__ == "__main__":
    main()
