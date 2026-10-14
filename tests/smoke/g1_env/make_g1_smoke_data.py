"""Writes synthetic Unitree G1 motions for the G1 tracker env smoke checks (see README.md).

There is no real G1 data in the repo, so this makes two standing motions on flat ground: the legs
hold the default pose (all DOFs 0, feet flat) and the waist and arms move sinusoidally within
their joint limits. Output, under --out_dir (default output/smoke/g1_env/):
    motions/synthetic/g1_sway.pkl, g1_reach.pkl   MS files: 90 frames at 30 fps, CLAMP
    motions.yaml                                  dataset yaml listing both

Run from the repo root: uv run python tests/smoke/g1_env/make_g1_smoke_data.py
"""
import argparse
import math
import os

import torch
import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.motion_tracker.envs.char_dof_props as char_dof_props
import parc.util.file_io_helper as file_io_helper
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util

G1_FILE = "data/assets/unitree_g1/g1_mocap_29dof.xml"
FPS = 30
NUM_FRAMES = 90
TERRAIN_DX = 0.4  # the env's dm.heightmap.horizontal_scale
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# the G1 feet collide through 4 spheres of radius 0.005 at z=-0.03 in the ankle roll frame
FOOT_SOLE_OFFSET = 0.035

# joint name -> (offset, amplitude, period in s, phase), DOF = offset + amplitude * sin(2 pi t / period + phase)
MOTIONS = {
    "g1_sway": {
        "waist_yaw_joint": (0.0, 0.3, 3.0, 0.0),
        "waist_roll_joint": (0.0, 0.08, 1.5, 0.0),
        "left_shoulder_pitch_joint": (0.0, 0.4, 1.5, 0.0),
        "right_shoulder_pitch_joint": (0.0, 0.4, 1.5, math.pi),
        "left_elbow_joint": (0.6, 0.3, 1.5, 0.0),
        "right_elbow_joint": (0.6, 0.3, 1.5, math.pi),
        "left_wrist_roll_joint": (0.0, 0.3, 1.0, 0.0),
        "right_wrist_roll_joint": (0.0, 0.3, 1.0, 0.0),
    },
    "g1_reach": {
        "waist_pitch_joint": (0.0, 0.05, 3.0, 0.0),
        "left_shoulder_roll_joint": (0.5, 0.4, 3.0, 0.0),
        "right_shoulder_roll_joint": (-0.5, 0.4, 3.0, math.pi),
        "left_shoulder_yaw_joint": (0.0, 0.3, 2.0, 0.0),
        "right_shoulder_yaw_joint": (0.0, 0.3, 2.0, math.pi),
        "left_elbow_joint": (0.8, 0.4, 3.0, 0.0),
        "right_elbow_joint": (0.8, 0.4, 3.0, 0.0),
        "left_wrist_pitch_joint": (0.0, 0.4, 1.5, 0.0),
        "right_wrist_pitch_joint": (0.0, 0.4, 1.5, 0.0),
    },
}


def standing_root_height(char_model):
    # root height that puts the foot soles on the ground (z = 0) in the default pose
    root_pos = torch.zeros(1, 3)
    root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    joint_rot = char_model.dof_to_rot(torch.zeros(1, char_model.get_dof_size()))
    body_pos, _ = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    foot_z = min(body_pos[0, char_model.get_body_id(name), 2].item() for name in FOOT_BODIES)
    return FOOT_SOLE_OFFSET - foot_z


def make_motion(char_model, joint_waves, root_height):
    t = torch.arange(NUM_FRAMES, dtype=torch.float32) / FPS
    dof_names = char_dof_props.get_dof_joint_names(char_model)
    dof = torch.zeros(NUM_FRAMES, char_model.get_dof_size())
    for joint_name, (offset, amplitude, period, phase) in joint_waves.items():
        dof[:, dof_names.index(joint_name)] = offset + amplitude * torch.sin(2.0 * math.pi * t / period + phase)

    lower, upper = char_dof_props.get_kin_dof_limits(char_model)
    assert torch.all(dof >= torch.from_numpy(lower)) and torch.all(dof <= torch.from_numpy(upper))

    root_pos = torch.zeros(NUM_FRAMES, 3)
    root_pos[:, 2] = root_height
    root_rot = torch.zeros(NUM_FRAMES, 4)
    root_rot[:, 3] = 1.0
    contacts = torch.zeros(NUM_FRAMES, char_model.get_num_contact_bodies())
    for name in FOOT_BODIES:
        contacts[:, char_model.get_contact_body_id(name)] = 1.0

    return motion_util.MotionFrames(root_pos=root_pos, root_rot=root_rot,
                                    joint_rot=char_model.dof_to_rot(dof), contacts=contacts)


def make_flat_terrain():
    # 4.4 m x 4.4 m around the origin, where the character stands
    num_cells = 11
    min_xy = -0.5 * (num_cells - 1) * TERRAIN_DX
    return terrain_util.SubTerrain("flat", num_cells, num_cells, TERRAIN_DX, TERRAIN_DX, min_xy, min_xy, device="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_dir", default="output/smoke/g1_env")
    out_dir = parser.parse_args().out_dir

    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(G1_FILE)
    root_height = standing_root_height(char_model)
    print("standing root height: {:.4f} m".format(root_height))

    motion_dir = os.path.join(out_dir, "motions", "synthetic")
    os.makedirs(motion_dir, exist_ok=True)
    dataset = []
    for name, joint_waves in MOTIONS.items():
        motion_file = os.path.join(motion_dir, name + ".pkl")
        file_io_helper.save_ms_file(motion_file, motion_frames=make_motion(char_model, joint_waves, root_height),
                                    fps=FPS, loop_mode="CLAMP", terrain=make_flat_terrain())
        dataset.append({"file": motion_file, "weight": 1.0})
        print("wrote", motion_file)

    dataset_file = os.path.join(out_dir, "motions.yaml")
    with open(dataset_file, "w") as f:
        yaml.safe_dump({"motions": dataset}, f)
    print("wrote", dataset_file)
    return


if __name__ == "__main__":
    main()
