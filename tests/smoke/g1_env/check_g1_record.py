"""GPU check of record mode on the G1 (porting plan A5 e).

Runs record_dm_motions (the code behind run_tracker --mode record and parc_4) on the synthetic motions
from make_g1_smoke_data.py, with an oracle policy whose PD targets are the reference motion's DOFs, so
that motions get recorded without a trained tracker. The oracle does not balance: the G1 tips forward
after about a second, so early termination is off and every motion runs to its end and is saved.
Then loads every recorded file with MotionLib.from_file and compares its first 0.5 s with the reference.

Run from the repo root (see README.md): uv run python tests/smoke/g1_env/check_g1_record.py
"""
import argparse
import glob
import os

import isaacgym  # noqa: F401  (IsaacGym must be imported before torch)

import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.motion_tracker.learning.agent_builder as agent_builder
import parc.util.file_io as file_io
import parc.util.path_loader as path_loader
from parc.anim.motion_lib import MotionLib
from parc.motion_tracker.envs import env_builder
from parc.motion_tracker.learning.dm_motion_recorder import record_dm_motions

ENV_CONFIG = "data/configs/tracker_config/dm_env_g1.yaml"
AGENT_CONFIG = "data/configs/tracker_config/dm_agent_g1.yaml"
G1_FILE = "data/assets/unitree_g1/g1_mocap_29dof.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_dir", default="output/smoke/g1_env")
    args = parser.parse_args()

    dataset_file = os.path.join(args.out_dir, "motions.yaml")
    record_dir = os.path.join(args.out_dir, "record_check", "recorded_motions")
    config = path_loader.load_config(ENV_CONFIG)
    config["env"]["dm"]["motion_file"] = dataset_file
    config["env"]["dm"]["terrain_save_path"] = os.path.join(args.out_dir, "record_check", "terrain.pkl")
    config["env"]["output_motion_dir"] = record_dir
    config["env"]["enable_early_termination"] = False
    os.makedirs(os.path.dirname(record_dir), exist_ok=True)
    config_file = os.path.join(args.out_dir, "record_check", "record_env.yaml")
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(G1_FILE)
    ref_lib = MotionLib.from_file(dataset_file, char_model, "cpu", contact_info=True)
    num_motions = ref_lib.num_motions()

    # one env per motion, as in parc_4
    env = env_builder.build_env(config_file, num_motions, "cuda:0", False)
    agent = agent_builder.build_agent(AGENT_CONFIG, env, "cuda:0")
    agent._decide_action = lambda obs, info: (env._ref_dof_pos.clone(), dict())
    record_dm_motions(agent)

    recorded_files = sorted(glob.glob(os.path.join(record_dir, "**", "*.pkl"), recursive=True))
    assert len(recorded_files) == num_motions, recorded_files
    ref_names = ref_lib.get_motion_names()
    for recorded_file in recorded_files:
        file_data = file_io.load_ms_file(recorded_file)  # an MS file, not a legacy pickle
        obs = file_data.misc_data["obs"]
        assert obs.shape[1] == 2363 and list(file_data.misc_data["obs_shapes"]) == \
            ["char_obs", "tar_obs", "tar_contacts", "char_contacts", "hf"]

        lib = MotionLib.from_file(recorded_file, char_model, "cpu", contact_info=True)
        name = lib.get_motion_names()[0]
        ref_id = ref_names.index(name[:-len("_dm")])
        num_frames = lib._motion_num_frames[0].item()
        assert abs(num_frames - ref_lib._motion_num_frames[ref_id].item()) <= 1, num_frames

        n = 15  # the first 0.5 s, before the oracle tips over
        ref_start = ref_lib._motion_start_idx[ref_id].item()
        dof = lib.joint_rot_to_dof(lib._frame_joint_rot[:n])
        ref_dof = ref_lib.joint_rot_to_dof(ref_lib._frame_joint_rot[ref_start:ref_start + n])
        dof_err = (dof - ref_dof).abs().mean().item()
        root_h_err = (lib._frame_root_pos[:n, 2] - ref_lib._frame_root_pos[ref_start:ref_start + n, 2]).abs().max().item()
        foot_ids = [char_model.get_contact_body_id(b) for b in ["left_ankle_roll_link", "right_ankle_roll_link"]]
        foot_contact = lib._frame_contacts[:n, foot_ids].mean().item()
        print("CHECK {}: MS file, {} frames at {:g} fps, obs {}; first 0.5 s: mean |dof - ref| {:.4f} rad, "
              "max |root height - ref| {:.4f} m, foot contact {:.2f}".format(
                  os.path.relpath(recorded_file), num_frames, lib.get_motion_fps(0), tuple(obs.shape),
                  dof_err, root_h_err, foot_contact))
        assert dof_err < 0.1 and root_h_err < 0.02 and foot_contact > 0.9
    print("CHECK G1 record OK")
    return


if __name__ == "__main__":
    main()
