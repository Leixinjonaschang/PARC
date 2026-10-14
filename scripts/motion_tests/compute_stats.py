"""Body velocity, acceleration and jerk statistics of a motion dataset (e.g. to pick the
max_jerk of the motion optimization).

    python -m scripts.motion_tests.compute_stats [--motion_file <MS .pkl or dataset .yaml>]
        [--char_file data/assets/humanoid.xml] [--device cpu] [--output output/body_pos_derivatives.txt]

Paths may use $DATA_DIR. The output file is a pickle of the per-frame vel/acc/jerk arrays.
"""
import argparse
import os
import pickle
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch  # noqa: E402

import parc.anim.kin_char_model as kin_char_model  # noqa: E402
import parc.anim.motion_lib as motion_lib  # noqa: E402
import parc.util.path_loader as path_loader  # noqa: E402

DEFAULT_MOTION_FILE = "$DATA_DIR/Data/parkour_dataset/pre_auto_aug_dataset.yaml"
DEFAULT_CHAR_FILE = "data/assets/humanoid.xml"
DEFAULT_OUTPUT = "output/body_pos_derivatives.txt"

key_body_names = ["left_foot", "right_foot", "left_hand", "right_hand"]


def compute_body_derivatives(mlib: motion_lib.MotionLib):
    """Returns (vel, acc, jerk) of all bodies, concatenated over the motions."""
    all_body_vels = []
    all_body_accs = []
    all_body_jerks = []

    # compute velocity, acceleration, and jerk for every frame
    for motion_id in mlib._motion_ids:
        dt = 1.0 / mlib.get_motion_fps(motion_id)
        body_pos = mlib.get_frames_for_id(motion_id, compute_fk=True).body_pos

        body_vels = (body_pos[1:] - body_pos[:-1]) / dt
        body_accs = (body_vels[1:] - body_vels[:-1]) / dt
        body_jerks = (body_accs[1:] - body_accs[:-1]) / dt

        all_body_vels.append(body_vels)
        all_body_accs.append(body_accs)
        all_body_jerks.append(body_jerks)

    return torch.cat(all_body_vels, dim=0), torch.cat(all_body_accs, dim=0), torch.cat(all_body_jerks, dim=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Body velocity/acceleration/jerk statistics of a motion dataset.")
    parser.add_argument("--motion_file", default=DEFAULT_MOTION_FILE, help="MS motion file or dataset yaml")
    parser.add_argument("--char_file", default=DEFAULT_CHAR_FILE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="pickle of the vel/acc/jerk arrays")
    args = parser.parse_args(argv)

    device = args.device
    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(str(path_loader.resolve_path(args.char_file)))
    mlib = motion_lib.MotionLib.from_file(args.motion_file, char_model, device, contact_info=True)

    key_body_ids = [char_model.get_body_id(name) for name in key_body_names]
    print("Key bodies:", {name: char_model.get_body_name(b) for name, b in zip(key_body_names, key_body_ids)})

    all_body_vels, all_body_accs, all_body_jerks = compute_body_derivatives(mlib)

    vel_mag = all_body_vels.norm(dim=-1)
    acc_mag = all_body_accs.norm(dim=-1)
    jerk_mag = all_body_jerks.norm(dim=-1)

    print("Max speed:", torch.max(vel_mag))
    print("Max acceleration (magnitude):", torch.max(acc_mag))
    print("Max jerk (magnitude):", torch.max(jerk_mag))

    print("Mean speed:", torch.mean(vel_mag))
    print("Mean acceleration (magnitude):", torch.mean(acc_mag))
    print("Mean jerk (magnitude):", torch.mean(jerk_mag))

    print("Std speed:", torch.std(vel_mag))
    print("Std acceleration (magnitude):", torch.std(acc_mag))
    print("Std jerk (magnitude):", torch.std(jerk_mag))

    # save stats to a pkl file
    output = {
        "vel": all_body_vels.cpu().numpy(),
        "acc": all_body_accs.cpu().numpy(),
        "jerk": all_body_jerks.cpu().numpy()
    }

    output_path = str(path_loader.resolve_path(args.output))
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    print("Saved", output_path)

    quantiles = [0.999, 0.99, 0.95, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5]
    quantiles = torch.tensor(quantiles, dtype=torch.float32, device=device)
    jerk_quantiles = torch.quantile(jerk_mag, quantiles)
    acc_quantiles = torch.quantile(acc_mag, quantiles)
    vel_quantiles = torch.quantile(vel_mag, quantiles)

    for i in range(quantiles.shape[0]):
        print("Quantile:", quantiles[i].item() * 100.0)
        print("Jerk:", jerk_quantiles[i].item())
        print("Acc:", acc_quantiles[i].item())
        print("Vel:", vel_quantiles[i].item())
        print("")


if __name__ == "__main__":
    main()
