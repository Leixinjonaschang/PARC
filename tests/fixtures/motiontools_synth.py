"""Small synthetic MS motions for the motion-tools tests and smoke runs.

There is no real G1 data on the test machine, so these build a short motion for any
character: the root moves along +x over a terrain with a low box, the joints oscillate a
little, the feet alternate contact and the hands touch the box for a few frames, so that
compute_approx_body_constraints finds constraints on both feet and both hands.

Also usable from the command line (from the repo root), e.g. to make inputs for
scripts/run_optimize_motions.py:

    python tests/fixtures/motiontools_synth.py --char g1 --out /tmp/g1_synth.pkl
"""
import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import parc.anim.kin_char_model as kin_char_model  # noqa: E402
import parc.util.file_io_helper as file_io_helper  # noqa: E402
import parc.util.motion_util as motion_util  # noqa: E402
import parc.util.terrain_util as terrain_util  # noqa: E402

CHAR_FILES = {
    "humanoid": REPO_ROOT / "data" / "assets" / "humanoid.xml",
    "g1": REPO_ROOT / "data" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
}

FOOT_NAMES = ("left_foot", "right_foot")
HAND_NAMES = ("left_hand", "right_hand")
HAND_CONTACT_FRAMES = (12, 18)  # [start, end) frames where both hands touch the box


def load_char(char, device="cpu"):
    char_file = CHAR_FILES.get(char, char)
    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(str(char_file))
    return char_model


def make_motion(char_model, num_frames=30, fps=30, speed=0.6, seed=0):
    """Returns (MotionFrames [num_frames], terrain). Contacts are per contact body."""
    device = char_model._device
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(num_frames, dtype=torch.float32) / fps

    dof_size = char_model.get_dof_size()
    amp = 0.15 * torch.rand(dof_size, generator=gen)
    phase = 2.0 * torch.pi * torch.rand(dof_size, generator=gen)
    joint_dof = amp * torch.sin(2.0 * torch.pi * 1.5 * t.unsqueeze(-1) + phase)
    joint_dof = joint_dof.to(device)
    joint_rot = char_model.dof_to_rot(joint_dof)

    root_rot = torch.zeros(num_frames, 4, device=device)
    root_rot[:, 3] = 1.0

    # put the lowest body origin of the first frame on the ground
    zero = torch.zeros(num_frames, 3, device=device)
    body_pos, _ = char_model.forward_kinematics(zero, root_rot, joint_rot)
    root_z = -torch.min(body_pos[0, :, 2]).item() + 0.02

    root_pos = torch.zeros(num_frames, 3, device=device)
    root_pos[:, 0] = speed * t.to(device)
    root_pos[:, 2] = root_z

    contacts = torch.zeros(num_frames, char_model.get_num_contact_bodies(), device=device)
    half_cycle = max(num_frames // 4, 1)
    for i, name in enumerate(FOOT_NAMES):
        cid = char_model.get_contact_body_id(char_model.resolve_body_name(name))
        for f in range(num_frames):
            if (f // half_cycle) % 2 == i % 2 or f < 3:
                contacts[f, cid] = 1.0
    for name in HAND_NAMES:
        cid = char_model.get_contact_body_id(char_model.resolve_body_name(name))
        contacts[HAND_CONTACT_FRAMES[0]:HAND_CONTACT_FRAMES[1], cid] = 1.0

    terrain = make_terrain(root_pos, device=device)
    frames = motion_util.MotionFrames(root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot,
                                      contacts=contacts)
    return frames, terrain


def make_terrain(root_pos, dx=0.1, padding=1.5, box_height=0.3, device="cpu"):
    min_x = root_pos[:, 0].min().item() - padding
    min_y = root_pos[:, 1].min().item() - padding
    max_x = root_pos[:, 0].max().item() + padding
    max_y = root_pos[:, 1].max().item() + padding
    x_dim = int(round((max_x - min_x) / dx)) + 1
    y_dim = int(round((max_y - min_y) / dx)) + 1
    terrain = terrain_util.SubTerrain(x_dim=x_dim, y_dim=y_dim, dx=dx, dy=dx,
                                      min_x=min_x, min_y=min_y, device=device)
    # a low box ahead of the start, to the side of the path
    i0 = int(round((root_pos[-1, 0].item() - min_x) / dx))
    j0 = int(round((root_pos[0, 1].item() + 0.5 - min_y) / dx))
    terrain.hf[i0 - 3:i0 + 3, j0:j0 + 4] = box_height
    return terrain


def make_path_nodes(root_pos, terrain):
    xy = root_pos[[0, root_pos.shape[0] // 2, -1], 0:2]
    z = terrain.get_hf_val_from_points(xy)
    return torch.cat([xy, z.unsqueeze(-1)], dim=-1)


def save_motion(filepath, char_model, frames, terrain, fps=30, path_nodes=None, body_constraints=None):
    file_io_helper.save_ms_file(str(filepath), motion_frames=frames.get_copy("cpu"), fps=fps,
                                loop_mode="CLAMP", terrain=terrain.torch_copy(),
                                path_nodes=path_nodes, body_constraints=body_constraints)
    return filepath


def write_synthetic_motion(filepath, char="humanoid", num_frames=30, fps=30, seed=0, with_path=True):
    char_model = load_char(char)
    frames, terrain = make_motion(char_model, num_frames=num_frames, fps=fps, seed=seed)
    path_nodes = make_path_nodes(frames.root_pos, terrain) if with_path else None
    return save_motion(filepath, char_model, frames, terrain, fps=fps, path_nodes=path_nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--char", default="humanoid", help="humanoid, g1, or a character xml path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_synthetic_motion(args.out, char=args.char, num_frames=args.num_frames, seed=args.seed)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
