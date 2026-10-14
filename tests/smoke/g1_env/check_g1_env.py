"""GPU check of the G1 tracker env (porting plan A5 a, b): builds data/configs/tracker_config/dm_env_g1.yaml
on the synthetic motions from make_g1_smoke_data.py and checks the mass, the DOF properties, the
observation layout (porting plan, appendix A) and a short random-action rollout.

Run from the repo root (see README.md): uv run python tests/smoke/g1_env/check_g1_env.py
"""
import argparse
import os
from collections import OrderedDict

import isaacgym  # noqa: F401  (IsaacGym must be imported before torch)

import numpy as np
import torch
import yaml

import parc.util.path_loader as path_loader
from parc.motion_tracker.envs import env_builder

ENV_CONFIG = "data/configs/tracker_config/dm_env_g1.yaml"
G1_MASS = 34.377  # sum of the MJCF masses; 5 bodies have none
OBS_SHAPES = OrderedDict([("char_obs", (266,)), ("tar_obs", (6, 234)), ("tar_contacts", (6, 36)),
                          ("char_contacts", (36,)), ("hf", (441,))])
KEY_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link", "head_link", "left_wrist_yaw_link",
              "right_wrist_yaw_link"]


def write_env_config(out_dir):
    config = path_loader.load_config(ENV_CONFIG)
    config["env"]["dm"]["motion_file"] = os.path.join(out_dir, "motions.yaml")
    config["env"]["dm"]["terrain_save_path"] = os.path.join(out_dir, "check_env_terrain.pkl")
    config_file = os.path.join(out_dir, "check_env.yaml")
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)
    return config, config_file


def first_matching_gain(dof_name, params):
    for key, value in params.items():
        if key.lower() in dof_name.lower():
            return float(value)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_dir", default="output/smoke/g1_env")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    config, config_file = write_env_config(args.out_dir)
    env = env_builder.build_env(config_file, args.num_envs, "cuda:0", False)
    gym, env_ptr, handle = env._gym, env._envs[0], env._get_char_actor_handle()

    # (a) mass and DOF properties
    mass = sum(rb.mass for rb in gym.get_actor_rigid_body_properties(env_ptr, handle))
    print("CHECK total mass {:.4f} kg (expected {:.3f})".format(mass, G1_MASS))
    assert abs(mass - G1_MASS) < 0.01, "a massless body got a default mass"

    dof_names = gym.get_actor_dof_names(env_ptr, handle)
    dof_prop = gym.get_actor_dof_properties(env_ptr, handle)
    kin_lower = env._kin_char_model._lower_dof_limits.cpu().numpy()
    kin_upper = env._kin_char_model._upper_dof_limits.cpu().numpy()
    assert len(dof_names) == 29
    # the env's limits are the kinematic model's; IsaacGym keeps them only where hasLimits
    np.testing.assert_array_equal(env._char_dof_lower, kin_lower)
    np.testing.assert_array_equal(env._char_dof_upper, kin_upper)
    has_limits = dof_prop["hasLimits"]
    np.testing.assert_array_equal(dof_prop["lower"][has_limits], kin_lower[has_limits])
    np.testing.assert_array_equal(dof_prop["upper"][has_limits], kin_upper[has_limits])
    pd_params = config["env"]["pd_params"]
    for i, name in enumerate(dof_names):
        assert dof_prop["stiffness"][i] == first_matching_gain(name, pd_params["stiffness"]), name
        assert dof_prop["damping"][i] == first_matching_gain(name, pd_params["damping"]), name
    print("CHECK 29 DOFs: limits = kinematic model (radians), stiffness/damping = pd_params")
    print("CHECK hasLimits (limits enforced by PhysX): {} of 29 DOFs; effort: {}".format(
        int(np.sum(has_limits)), sorted(set(float(e) for e in dof_prop["effort"]))))

    action_space = env.get_action_space()
    mid, half_range = 0.5 * (kin_upper + kin_lower), 0.5 * (kin_upper - kin_lower)
    np.testing.assert_allclose(action_space.low, mid - 1.4 * half_range, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(action_space.high, mid + 1.4 * half_range, rtol=1e-5, atol=1e-6)
    print("CHECK action dim {}, bounds = joint range mid -/+ 0.7 range".format(action_space.shape[0]))

    body_names = env._kin_char_model.get_body_names()
    assert [body_names[i] for i in env._key_body_ids.tolist()] == KEY_BODIES
    assert env._kin_char_model.get_num_contact_bodies() == 36

    # (b) observation layout
    obs_shapes = env._compute_obs(ret_obs_shapes=True)
    shapes = OrderedDict((key, tuple(info["shape"])) for key, info in obs_shapes.items())
    assert shapes == OBS_SHAPES, shapes
    obs, info = env.reset()
    total = sum(int(np.prod(shape)) for shape in OBS_SHAPES.values())
    assert obs.shape == (args.num_envs, total) == (args.num_envs, 2363), obs.shape
    print("CHECK OBS SHAPES = appendix A, obs dim {}, action dim {}".format(obs.shape[-1], action_space.shape[0]))

    # random actions inside the action bounds
    low = torch.tensor(action_space.low, device="cuda:0")
    high = torch.tensor(action_space.high, device="cuda:0")
    root_h, num_resets = [], 0
    for _ in range(args.steps):
        action = low + (high - low) * torch.rand((args.num_envs, low.shape[0]), device="cuda:0")
        obs, reward, done, info = env.step(action)
        assert torch.all(torch.isfinite(obs)) and torch.all(torch.isfinite(reward)), "NaN or inf in obs or reward"
        root_h.append(env._char_root_pos[:, 2].mean().item())
        done_ids = torch.nonzero(done != 0).flatten()
        num_resets += len(done_ids)
        if (len(done_ids) > 0):
            obs, info = env.reset(done_ids)
    print("CHECK {} random-action steps x {} envs: obs and rewards finite, {} resets, mean root height {:.3f}-{:.3f} m"
          .format(args.steps, args.num_envs, num_resets, min(root_h), max(root_h)))
    print("CHECK G1 env OK")
    return


if __name__ == "__main__":
    main()
