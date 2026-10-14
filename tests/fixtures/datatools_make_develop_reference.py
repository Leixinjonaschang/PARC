"""Generate robot_motion_preprocess parity fixtures with develop's own code.

Not a test: tests/test_datatools_robot_motion_preprocess.py uses its output.  It
imports develop's flat modules (anim.*, util.*, zmotion_editing_tools.*) and runs
develop's tools/motion_file_proc/robot_motion_preprocess.py, so run it with a
develop checkout's Python (torch 1.13 venv), from any directory:

    PYTHONDONTWRITEBYTECODE=1 <develop>/.venv/bin/python <this file> \\
        --develop-root <develop checkout> --out <port repo>/tests/fixtures/datatools_rmp

All motions are synthetic, 40 frames at 30 fps.  Writes, under --out:

  source/walk.pkl      legacy humanoid motion (develop's save_motion_data): frames,
                       15-body contacts, terrain (flat ground, a box, a step)
  source/stand_h.pkl   legacy humanoid motion: stands, lifts the left leg, holds it,
                       puts it down; terrain with a box
  gmr/walk_to_unitree_g1.pkl
                       G1 {frames, fps} as GMR writes it (a NumPy 2 pickle): stands
                       with its soles 2 cm above the ground, then walks; its source
                       motion is source/walk.pkl
  gmr/lift_to_unitree_g1.pkl
                       G1 {frames, fps}: stands with its soles 4 mm above the ground,
                       lifts the left foot ~12 cm and holds it still; no source motion
  gmr_humanoid/stand_h.pkl
                       humanoid {frames, fps}: source/stand_h.pkl raised by 5 mm
  develop_out/g1/*.pkl, develop_out/humanoid/*.pkl
                       output of develop's robot_motion_preprocess.py (legacy format)
                       for gmr/ (G1 model) and gmr_humanoid/ (humanoid model), both
                       with --terrain_dir source and the CLI defaults
"""
import argparse
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--develop-root", required=True, help="develop checkout (flat layout)")
parser.add_argument("--out", required=True, help="output directory")
args = parser.parse_args()

DEV_ROOT = os.path.abspath(args.develop_root)
OUT_DIR = os.path.abspath(args.out)
sys.path.insert(0, DEV_ROOT)

import anim.kin_char_model as kin_char_model
import util.geom_util as geom_util
import util.terrain_util as terrain_util
import util.torch_util as torch_util
import zmotion_editing_tools.motion_edit_lib as medit_lib

HUMANOID_XML = os.path.join(DEV_ROOT, "data/assets/humanoid.xml")
G1_XML = os.path.join(DEV_ROOT, "data/assets/unitree_g1/g1_mocap_29dof.xml")
DEVELOP_TOOL = os.path.join(DEV_ROOT, "tools/motion_file_proc/robot_motion_preprocess.py")

NUM_FRAMES = 40
FPS = 30
T = np.arange(NUM_FRAMES, dtype=np.float64)


def load_char_model(path):
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(path)
    return char_model


def dof_index(char_model, joint_name):
    for j in range(1, char_model.get_num_joints()):
        if char_model.get_joint(j).name == joint_name:
            return char_model.get_joint_dof_idx(j)
    raise KeyError(joint_name)


def ramp(start, end):
    # 0 up to frame `start`, smoothstep to 1 at frame `end`
    x = np.clip((T - start) / float(end - start), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def forward_kinematics(char_model, frames):
    frames = torch.tensor(frames, dtype=torch.float32)
    root_rot = torch_util.exp_map_to_quat(frames[:, 3:6])
    joint_rot = char_model.dof_to_rot(frames[:, 6:])
    return char_model.forward_kinematics(frames[:, 0:3], root_rot, joint_rot)


def g1_sole_z(char_model, frames, body_name):
    # lowest point of the foot's collision spheres, per frame
    body_pos, body_rot = forward_kinematics(char_model, frames)
    b = char_model.get_body_id(body_name)
    zs = []
    for geom in char_model.get_geoms(b):
        if geom._shape_type == kin_char_model.GeomType.SPHERE:
            offset = geom._offset.unsqueeze(0).expand(frames.shape[0], -1)
            center = body_pos[:, b] + torch_util.quat_rotate(body_rot[:, b], offset)
            zs.append(center[:, 2] - geom._dims.reshape(-1)[0])
    return torch.stack(zs, dim=-1).min(dim=-1).values.numpy()


def humanoid_sole_z(char_model, frames, body_name):
    body_pos, body_rot = forward_kinematics(char_model, frames)
    b = char_model.get_body_id(body_name)
    geom = char_model.get_geoms(b)[0]
    points = geom_util.get_box_points_batch(body_pos[:, b], body_rot[:, b], geom._dims, geom._offset)
    return points[..., 2].min(dim=-1).values.numpy()


def save_gmr_pickle(path, frames, fps):
    # GMR runs under NumPy 2, whose pickles reference numpy._core.*.  Pickle with
    # protocol 2 (module names are newline-terminated GLOBAL strings) and rename.
    data = pickle.dumps({"frames": frames, "fps": fps}, protocol=2)
    assert b"numpy.core.multiarray\n" in data
    data = data.replace(b"numpy.core.multiarray\n", b"numpy._core.multiarray\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def run_develop_tool(input_dir, output_dir, char_file):
    cmd = [sys.executable, DEVELOP_TOOL, "--input", input_dir, "--output", output_dir,
           "--robot_model", char_file, "--terrain_dir", os.path.join(OUT_DIR, "source")]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=OUT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    log = "\n".join(line for line in result.stdout.splitlines() if not line.startswith("Loaded mesh"))
    print(log)
    assert result.returncode == 0
    return log


humanoid = load_char_model(HUMANOID_XML)
g1 = load_char_model(G1_XML)
rng = np.random.RandomState(7)

# --- source/walk.pkl: humanoid source motion of the G1 walk -------------------
h_dof = humanoid.get_dof_size()
root_pos = np.stack([0.025 * np.maximum(T - 10.0, 0.0),
                     0.04 * np.sin(2.0 * np.pi * T / 40.0),
                     0.92 + 0.02 * np.cos(2.0 * np.pi * T / 20.0)], axis=-1)
root_rot = np.stack([0.03 * np.sin(2.0 * np.pi * T / 40.0),
                     0.02 * np.cos(2.0 * np.pi * T / 40.0),
                     0.1 + 0.004 * T], axis=-1)
amp = rng.uniform(0.1, 0.5, size=h_dof)
freq = rng.uniform(0.02, 0.06, size=h_dof)
phase = rng.uniform(0.0, 2.0 * np.pi, size=h_dof)
joint_dof = amp * np.sin(2.0 * np.pi * freq * T[:, None] + phase)
walk_src_frames = np.concatenate([root_pos, root_rot, joint_dof], axis=-1).astype(np.float32)

walk_contacts = np.zeros((NUM_FRAMES, humanoid.get_num_joints()), dtype=np.float32)
walk_contacts[list(range(0, 15)) + list(range(25, 40)), humanoid.get_body_id("left_foot")] = 1.0
walk_contacts[list(range(0, 7)) + list(range(12, 31)), humanoid.get_body_id("right_foot")] = 1.0
walk_contacts[20:23, humanoid.get_body_id("right_hand")] = 1.0
# single-frame groups: the trailing one is dropped by develop's constraint grouping
walk_contacts[[33, 39], humanoid.get_body_id("left_hand")] = 1.0
walk_contacts[34:37, humanoid.get_body_id("pelvis")] = 1.0      # no G1 mapping by default
walk_contacts[5:9, humanoid.get_body_id("left_shin")] = 1.0     # no G1 counterpart

walk_terrain = terrain_util.SubTerrain("terrain", x_dim=40, y_dim=26, dx=0.1, dy=0.1,
                                       min_x=-1.03, min_y=-1.27, device="cpu")
walk_terrain.hf[22:25, 10:14] = 0.25   # a box beside the path
walk_terrain.hf[30:, :] = 0.4          # a step ahead

os.makedirs(os.path.join(OUT_DIR, "source"), exist_ok=True)
medit_lib.save_motion_data(os.path.join(OUT_DIR, "source", "walk.pkl"),
                           motion_frames=torch.tensor(walk_src_frames), contact_frames=torch.tensor(walk_contacts),
                           terrain=walk_terrain, fps=FPS, loop_mode="CLAMP")

# --- G1 poses ------------------------------------------------------------------
g1_dof = g1.get_dof_size()
stance = np.zeros(g1_dof)
for side in ["left", "right"]:
    # hip pitch + knee + ankle pitch sum to zero: knees bent, soles level
    stance[dof_index(g1, side + "_hip_pitch_joint")] = -0.25
    stance[dof_index(g1, side + "_knee_joint")] = 0.5
    stance[dof_index(g1, side + "_ankle_pitch_joint")] = -0.25
    stance[dof_index(g1, side + "_elbow_joint")] = 0.6


def g1_root_height(dof, sole_height):
    frame = np.zeros((1, 6 + g1_dof))
    frame[0, 6:] = dof
    sole = min(g1_sole_z(g1, frame, "left_ankle_roll_link")[0], g1_sole_z(g1, frame, "right_ankle_roll_link")[0])
    return sole_height - sole


# --- gmr/walk_to_unitree_g1.pkl: soles 2 cm up for 10 frames, then walks --------
walk_dof = np.tile(stance, (NUM_FRAMES, 1))
gait = ramp(9, 13)
swing = np.sin(2.0 * np.pi * np.maximum(T - 10.0, 0.0) / 20.0) * gait
for side, sign in [("left", 1.0), ("right", -1.0)]:
    walk_dof[:, dof_index(g1, side + "_hip_pitch_joint")] += -0.35 * sign * swing
    walk_dof[:, dof_index(g1, side + "_knee_joint")] += 0.45 * np.maximum(sign * swing, 0.0)
    walk_dof[:, dof_index(g1, side + "_ankle_pitch_joint")] += 0.1 * sign * swing
    walk_dof[:, dof_index(g1, side + "_shoulder_pitch_joint")] += 0.4 * sign * swing
walk_dof[:, dof_index(g1, "waist_yaw_joint")] = 0.15 * swing
walk_dof[:, dof_index(g1, "left_wrist_roll_joint")] = 0.2 * np.sin(2.0 * np.pi * T / 25.0)
walk_root_pos = np.stack([0.025 * np.maximum(T - 10.0, 0.0),
                          0.02 * np.sin(2.0 * np.pi * T / 40.0),
                          g1_root_height(stance, 0.02) + 0.015 * gait * np.cos(4.0 * np.pi * T / 20.0)], axis=-1)
walk_root_rot = np.stack([np.zeros(NUM_FRAMES), np.zeros(NUM_FRAMES), 0.1 + 0.005 * np.maximum(T - 10.0, 0.0)], axis=-1)
walk_frames = np.concatenate([walk_root_pos, walk_root_rot, walk_dof], axis=-1)  # float64, like GMR's qpos
save_gmr_pickle(os.path.join(OUT_DIR, "gmr", "walk_to_unitree_g1.pkl"), walk_frames, FPS)

# --- gmr/lift_to_unitree_g1.pkl: soles 4 mm up, then the left foot lifts and holds ---
lift_dof = np.tile(stance, (NUM_FRAMES, 1))
lift = ramp(10, 19) * (1.0 - ramp(30, 39))
lift_dof[:, dof_index(g1, "left_hip_pitch_joint")] += -0.5 * lift
lift_dof[:, dof_index(g1, "left_knee_joint")] += 0.85 * lift
lift_dof[:, dof_index(g1, "left_ankle_pitch_joint")] += -0.35 * lift
lift_root_pos = np.zeros((NUM_FRAMES, 3))
lift_root_pos[:, 2] = g1_root_height(stance, 0.004)
lift_root_rot = np.zeros((NUM_FRAMES, 3))
lift_root_rot[:, 2] = -0.3
lift_frames = np.concatenate([lift_root_pos, lift_root_rot, lift_dof], axis=-1)
left_sole = g1_sole_z(g1, lift_frames, "left_ankle_roll_link")
assert left_sole[20:30].min() > 0.1, left_sole
save_gmr_pickle(os.path.join(OUT_DIR, "gmr", "lift_to_unitree_g1.pkl"), lift_frames, FPS)

# --- source/stand_h.pkl and gmr_humanoid/stand_h.pkl: humanoid lifts the left leg ---
stand_dof = np.zeros((NUM_FRAMES, h_dof))
lift = ramp(10, 19) * (1.0 - ramp(30, 39))
hip = humanoid.get_joint_dof_idx(humanoid.get_body_id("left_thigh"))
knee = humanoid.get_joint_dof_idx(humanoid.get_body_id("left_shin"))
ankle = humanoid.get_joint_dof_idx(humanoid.get_body_id("left_foot"))
stand_dof[:, hip + 1] = -0.8 * lift
stand_dof[:, knee] = 1.4 * lift
stand_dof[:, ankle + 1] = -0.6 * lift
stand_dof[:, humanoid.get_joint_dof_idx(humanoid.get_body_id("right_lower_arm"))] = 0.3 * np.sin(2.0 * np.pi * T / 40.0)
stand_root_pos = np.zeros((NUM_FRAMES, 3))
stand_root_rot = np.zeros((NUM_FRAMES, 3))
stand_root_rot[:, 2] = 0.2
stand_frames = np.concatenate([stand_root_pos, stand_root_rot, stand_dof], axis=-1)
stand_frames[:, 2] = -humanoid_sole_z(humanoid, stand_frames, "right_foot")[0]
stand_frames = stand_frames.astype(np.float32)

stand_contacts = np.zeros((NUM_FRAMES, humanoid.get_num_joints()), dtype=np.float32)
stand_contacts[:, humanoid.get_body_id("right_foot")] = 1.0
stand_contacts[list(range(0, 11)) + list(range(36, 40)), humanoid.get_body_id("left_foot")] = 1.0
stand_terrain = terrain_util.SubTerrain("terrain", x_dim=21, y_dim=23, dx=0.1, dy=0.1,
                                        min_x=-1.02, min_y=-1.11, device="cpu")
stand_terrain.hf[14:17, 4:8] = 0.3
medit_lib.save_motion_data(os.path.join(OUT_DIR, "source", "stand_h.pkl"),
                           motion_frames=torch.tensor(stand_frames), contact_frames=torch.tensor(stand_contacts),
                           terrain=stand_terrain, fps=FPS, loop_mode="CLAMP")

stand_input = stand_frames.copy()
stand_input[:, 2] += 0.005
save_gmr_pickle(os.path.join(OUT_DIR, "gmr_humanoid", "stand_h.pkl"), stand_input, FPS)

# --- develop's robot_motion_preprocess.py on both ---------------------------------
run_develop_tool(os.path.join(OUT_DIR, "gmr"), os.path.join(OUT_DIR, "develop_out", "g1"), G1_XML)
run_develop_tool(os.path.join(OUT_DIR, "gmr_humanoid"), os.path.join(OUT_DIR, "develop_out", "humanoid"),
                 HUMANOID_XML)

for root, _, files in sorted(os.walk(OUT_DIR)):
    for name in sorted(files):
        path = os.path.join(root, name)
        print(os.path.relpath(path, OUT_DIR), os.path.getsize(path))
