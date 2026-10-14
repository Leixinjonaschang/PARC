"""Synthetic Unitree G1 motions for the pipeline tests and the G1 pipeline smoke runs.

There is no G1 dataset on the test machines, so this writes simple walking-like clips
on box terrains as MS files (file_io_helper.save_ms_file): joint_rot [N, 35, 4]
(29 hinges and 6 fixed joints), 36 contact bodies, 30 fps, CLAMP. The motions are
kinematically plausible (stance foot on the ground, alternating contacts) but not
physically meaningful.

Each class gets its own folder, which is what create_dataset expects:
    <out_dir>/walk/g1_walk_<i>.pkl   flat floor with a box beside the path
    <out_dir>/step/g1_step_<i>.pkl   0.15 m step up across the path

Run from the repo root (in the uv env):
    python tests/fixtures/pipeline_g1_synthetic.py <out_dir> [--num_motions_per_class 3] [--num_frames 60]
"""
import argparse
import math
import pathlib
import sys
from typing import List, Optional, Sequence

import torch

import parc.anim.kin_char_model as kin_char_model
import parc.util.file_io_helper as file_io_helper
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
G1_CHAR_FILE = REPO_ROOT / "data" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
FPS = 30
SOLE_HEIGHT = 0.035  # ankle roll link origin above the sole (foot sphere radius)
CLASSES = ("walk", "step")

# terrain grid: 40 x 24 cells of 0.1 m (within create_dataset's max_terrain_dim of 45)
TERRAIN_DIMS = (40, 24)
TERRAIN_DX = 0.1
TERRAIN_MIN_POINT = (-0.5, -1.2)


def load_g1(device="cpu") -> kin_char_model.KinCharModel:
    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(str(G1_CHAR_FILE))
    return char_model


def _dof_indices(char_model: kin_char_model.KinCharModel):
    ids = dict()
    for j in range(1, char_model.get_num_joints()):
        joint = char_model.get_joint(j)
        if joint.get_dof_dim() == 1:
            ids[joint.name] = joint.dof_idx
    return ids


def make_terrain(kind: str) -> terrain_util.SubTerrain:
    terrain = terrain_util.SubTerrain("terrain", x_dim=TERRAIN_DIMS[0], y_dim=TERRAIN_DIMS[1],
                                      dx=TERRAIN_DX, dy=TERRAIN_DX,
                                      min_x=TERRAIN_MIN_POINT[0], min_y=TERRAIN_MIN_POINT[1], device="cpu")
    if kind == "walk":
        terrain.hf[15:25, 19:23] = 0.4  # x in [1.0, 1.9], y in [0.7, 1.0]: beside the path
    elif kind == "step":
        terrain.hf[21:, :] = 0.15  # x >= 1.6
    else:
        raise ValueError(kind)
    return terrain


def make_walk_frames(char_model: kin_char_model.KinCharModel, terrain: terrain_util.SubTerrain,
                     num_frames: int, speed: float, phase: float, start_xy: Sequence[float],
                     heading: float = 0.0) -> motion_util.MotionFrames:
    dof_ids = _dof_indices(char_model)
    t = torch.arange(num_frames, dtype=torch.float32) / FPS
    s = torch.sin(2.0 * math.pi * 1.6 * t + phase)

    dof = torch.zeros(num_frames, char_model.get_dof_size())
    for side, sign in (("left", 1.0), ("right", -1.0)):
        swing = torch.clamp(sign * s, min=0.0)
        dof[:, dof_ids[side + "_hip_pitch_joint"]] = -0.35 * sign * s
        dof[:, dof_ids[side + "_knee_joint"]] = 0.25 + 0.5 * swing
        dof[:, dof_ids[side + "_ankle_pitch_joint"]] = -0.12 - 0.2 * swing
        dof[:, dof_ids[side + "_shoulder_pitch_joint"]] = 0.3 * sign * s
        dof[:, dof_ids[side + "_shoulder_roll_joint"]] = 0.2 * sign
        dof[:, dof_ids[side + "_elbow_joint"]] = 0.6
    joint_rot = char_model.dof_to_rot(dof)

    # walk along `heading`, with a little yaw wobble and side sway
    root_rot = torch_util.heading_to_quat(heading + 0.08 * torch.sin(math.pi * 1.6 * t + phase))
    root_pos = torch.zeros(num_frames, 3)
    root_pos[:, 0] = start_xy[0] + speed * t * math.cos(heading) - 0.02 * s * math.sin(heading)
    root_pos[:, 1] = start_xy[1] + speed * t * math.sin(heading) + 0.02 * s * math.cos(heading)

    # put the lower foot's sole on the terrain
    body_pos, _ = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    foot_ids = [char_model.get_body_id("left_foot"), char_model.get_body_id("right_foot")]
    toe_ids = [char_model.get_body_id("left_toe_link"), char_model.get_body_id("right_toe_link")]
    foot_pos = body_pos[:, foot_ids, :]  # [n, 2, 3]
    ground_h = terrain.get_hf_val_from_points(foot_pos[..., 0:2])  # [n, 2]
    clearance = foot_pos[..., 2] - SOLE_HEIGHT - ground_h  # with the root at z = 0
    root_pos[:, 2] = -torch.min(clearance, dim=-1)[0]

    clearance = clearance + root_pos[:, 2:3]
    contacts = torch.zeros(num_frames, char_model.get_num_bodies())
    for k in range(2):
        in_contact = (clearance[:, k] < 0.02).to(torch.float32)
        contacts[:, foot_ids[k]] = in_contact
        contacts[:, toe_ids[k]] = in_contact

    return motion_util.MotionFrames(root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot, contacts=contacts)


def write_synthetic_g1_motions(out_dir, num_motions_per_class: int = 3, num_frames: int = 60,
                               classes: Sequence[str] = CLASSES,
                               char_model: Optional[kin_char_model.KinCharModel] = None,
                               seed: int = 0) -> List[pathlib.Path]:
    out_dir = pathlib.Path(out_dir)
    if char_model is None:
        char_model = load_g1()
    gen = torch.Generator().manual_seed(seed)
    paths = []
    for kind in classes:
        class_dir = out_dir / kind
        class_dir.mkdir(parents=True, exist_ok=True)
        terrain = make_terrain(kind)
        for i in range(num_motions_per_class):
            r = torch.rand(4, generator=gen)
            # stays clear of the "walk" box (x in [1.0, 1.9], y >= 0.7) for clips up to ~2 s
            frames = make_walk_frames(char_model, terrain, num_frames,
                                      speed=0.6 + 0.3 * r[0].item(),
                                      phase=2.0 * math.pi * r[1].item(),
                                      start_xy=(0.2, -0.3 + 0.5 * r[2].item()),
                                      heading=0.3 * (r[3].item() - 0.5))
            path = class_dir / "g1_{}_{}.pkl".format(kind, i)
            file_io_helper.save_ms_file(str(path), motion_frames=frames, fps=FPS, loop_mode="CLAMP",
                                        terrain=terrain)
            paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir")
    parser.add_argument("--num_motions_per_class", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=60)
    args = parser.parse_args(argv)
    paths = write_synthetic_g1_motions(args.out_dir, args.num_motions_per_class, args.num_frames)
    print("wrote {} synthetic G1 motions to {}".format(len(paths), args.out_dir))


if __name__ == "__main__":
    sys.exit(main())
