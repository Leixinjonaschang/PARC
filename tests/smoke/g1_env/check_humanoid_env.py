"""GPU regression check for the humanoid: builds the civilization example env and compares its action
bounds and character DOF properties with tests/fixtures/env_humanoid_civilization_props.npz, captured
on develop_migration 89444d8 before Lane A's changes (kinematic joint limits, config PD gains).

Run from the repo root (see README.md): uv run python tests/smoke/g1_env/check_humanoid_env.py
"""
import argparse

import isaacgym  # noqa: F401  (IsaacGym must be imported before torch)

import numpy as np

from parc.motion_tracker.envs import env_builder

ENV_CONFIG = "data/motion_terrains/dm_env_civilization.yaml"
REFERENCE = "tests/fixtures/env_humanoid_civilization_props.npz"
TOLERANCE = 1e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num_envs", type=int, default=4)
    args = parser.parse_args()

    env = env_builder.build_env(ENV_CONFIG, args.num_envs, "cuda:0", False)
    action_space = env.get_action_space()
    dof_prop = env._gym.get_actor_dof_properties(env._envs[0], env._get_char_actor_handle())
    obs, _ = env.reset()
    current = dict(action_low=action_space.low, action_high=action_space.high,
                   bound_low=env._action_bound_low.cpu().numpy(), bound_high=env._action_bound_high.cpu().numpy(),
                   obs_dim=np.array(obs.shape[-1]))
    for field in ["lower", "upper", "hasLimits", "effort", "stiffness", "damping", "driveMode"]:
        current["dof_" + field] = np.array(dof_prop[field])

    with np.load(REFERENCE) as reference:
        for key in reference.files:
            ref, cur = reference[key], current[key]
            assert ref.shape == cur.shape, (key, ref.shape, cur.shape)
            if (ref.dtype.kind == "f"):
                delta = float(np.max(np.abs(ref.astype(np.float64) - cur.astype(np.float64)))) if ref.size else 0.0
                print("CHECK {:14s} max |delta| {:.2e}".format(key, delta))
                assert delta < TOLERANCE, key
            else:
                print("CHECK {:14s} equal".format(key))
                assert np.array_equal(ref, cur), key
    print("CHECK humanoid env unchanged")
    return


if __name__ == "__main__":
    main()
