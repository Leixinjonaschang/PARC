"""Lists the motions of a dataset with the largest root height changes, longest root paths,
most hand contacts, fewest foot contacts and largest root-to-terrain distances.

    python -m scripts.find_interesting_motions [--motion_file <dataset .yaml or MS .pkl>]
        [--char_file data/assets/humanoid.xml] [--device cpu]
"""
import argparse
import os
import sys
from pathlib import Path

import torch



# Ensure the repository root is on the import path so "parc" resolves consistently
# across platforms (editable installs on Windows sometimes fail to place the repo
# ahead of site-packages when invoking this script directly).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.path_loader as path_loader


#from diffusion.mdm_heightfield_contact_motion_sampler import MDMHeightfieldContactMotionSampler

DEFAULT_MOTION_FILE = "output/datasets/rerun_april272025_iter_5_siggraph_terrain.yaml"
DEFAULT_CHAR_FILE = "data/assets/humanoid.xml"

key_body_names = ["left_foot", "right_foot", "left_hand", "right_hand"]


def compute_root_height_delta(root_pos):
    return (root_pos[-1, 2] - root_pos[0, 2]).item()


def compute_root_trajectory_length(root_pos):
    root_deltas = root_pos[1:] - root_pos[:-1]
    return torch.linalg.norm(root_deltas, dim=-1).sum().item()


def count_contact_frames(contacts, body_ids):
    if contacts is None:
        return 0
    body_contacts = contacts[:, body_ids]
    return torch.any(body_contacts > 0.5, dim=-1).sum().item()


def compute_cumulative_root_terrain_distance(root_pos, terrain):
    assert terrain is not None, "Expected terrain data for cumulative root-to-terrain distance computation."
    root_xy = root_pos[:, 0:2]
    terrain_z = terrain.get_hf_val_from_points(root_xy)
    root_terrain_deltas = torch.abs(root_pos[:, 2] - terrain_z)
    return root_terrain_deltas.sum().item()


def format_top_entries(entries, reverse=True):
    return sorted(entries, key=lambda item: item[0], reverse=reverse)[:10]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find motions with extreme root/contact statistics in a dataset.")
    parser.add_argument("--motion_file", default=DEFAULT_MOTION_FILE, help="dataset yaml or MS motion file")
    parser.add_argument("--char_file", default=DEFAULT_CHAR_FILE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    device = args.device
    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(str(path_loader.resolve_path(args.char_file)))
    mlib = motion_lib.MotionLib.from_file(motion_file=args.motion_file, char_model=char_model, device=device, contact_info=True)

    # body ids (generic names resolve through KinCharModel's aliases, e.g. on the G1)
    key_body_ids = torch.tensor([char_model.get_body_id(name) for name in key_body_names],
                                dtype=torch.int64, device=device)

    motion_names = mlib.get_motion_names()
    num_motions = mlib.num_motions()

    pos_height_deltas = []
    neg_height_deltas = []
    trajectory_lengths = []
    hand_contact_counts = []
    foot_contact_counts = []
    foot_contact_ratios = []
    root_terrain_distances = []

    hand_body_ids = key_body_ids[2:].tolist()
    foot_body_ids = key_body_ids[:2].tolist()

    for motion_id in range(num_motions):
        frames = mlib.get_frames_for_id(motion_id, compute_fk=False)
        root_pos = frames.root_pos
        contacts = frames.contacts
        if contacts is not None:
            contacts = char_model.create_full_contact_tensor(contacts) # indexed by body id below
        name = motion_names[motion_id]

        height_delta = compute_root_height_delta(root_pos)
        pos_height_deltas.append((height_delta, name))
        neg_height_deltas.append((height_delta, name))

        trajectory_length = compute_root_trajectory_length(root_pos)
        trajectory_lengths.append((trajectory_length, name))

        hand_contact_frames = count_contact_frames(contacts, hand_body_ids)
        hand_contact_counts.append((hand_contact_frames, name))

        foot_contact_frames = count_contact_frames(contacts, foot_body_ids)
        foot_contact_counts.append((foot_contact_frames, name))
        total_frames = root_pos.shape[0]
        foot_contact_ratio = foot_contact_frames / total_frames if total_frames else 0.0
        foot_contact_ratios.append((foot_contact_ratio, name, foot_contact_frames, total_frames))

        terrain = mlib._terrains[motion_id]
        root_terrain_distance = compute_cumulative_root_terrain_distance(root_pos, terrain)
        root_terrain_distances.append((root_terrain_distance, name))

    print("\nTop 10 positive root height deltas:")
    for value, name in format_top_entries(pos_height_deltas, reverse=True):
        print(f"{name}: {value:.4f}")

    print("\nTop 10 negative root height deltas:")
    for value, name in format_top_entries(neg_height_deltas, reverse=False):
        print(f"{name}: {value:.4f}")

    print("\nTop 10 root trajectory lengths:")
    for value, name in format_top_entries(trajectory_lengths, reverse=True):
        print(f"{name}: {value:.4f}")

    print("\nTop 10 hand contact frame counts:")
    for value, name in format_top_entries(hand_contact_counts, reverse=True):
        print(f"{name}: {value}")

    print("\nTop 10 lowest foot contact frame counts:")
    for value, name in format_top_entries(foot_contact_counts, reverse=False):
        print(f"{name}: {value}")

    print("\nTop 10 lowest foot contact frame ratios:")
    for ratio, name, contact_frames, total_frames in format_top_entries(foot_contact_ratios, reverse=False):
        print(f"{name}: {ratio:.4f} ({contact_frames}/{total_frames} frames)")

    print("\nTop 10 cumulative root-to-terrain distances:")
    for value, name in format_top_entries(root_terrain_distances, reverse=True):
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
