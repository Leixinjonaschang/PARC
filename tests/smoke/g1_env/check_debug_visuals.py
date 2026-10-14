"""Headless dry run of the tracker env's visualize and debug_visuals code paths (porting plan A2, A3).

IsaacGym's viewer needs a display. This builds the G1 env with visualize=True and debug_visuals=True
through a proxy of the gym that stubs out the viewer calls and records the drawing calls, then checks
what one frame draws: the heightfield samples of every env (one add_lines call, 3 lines per point) and
the colours of the camera env's 36 contact bodies, for the character and the reference character.
It does not show anything: look at the drawing on a machine with a display (see README.md).

Run from the repo root after make_g1_smoke_data.py: uv run python tests/smoke/g1_env/check_debug_visuals.py
"""
import argparse
import os

import isaacgym.gymapi as gymapi  # IsaacGym must be imported before torch

import numpy as np
import torch
import yaml

import parc.motion_tracker.envs.ig_env as ig_env
import parc.util.path_loader as path_loader
from parc.motion_tracker.envs import env_builder

ENV_CONFIG = "data/configs/tracker_config/dm_env_g1.yaml"
NUM_HF_POINTS = 441


class ViewerlessGym:
    """Forwards to the gym, except for the viewer calls, which are stubbed out or recorded."""

    def __init__(self, gym):
        self._gym = gym
        self.lines = []  # num_lines of each add_lines call
        self.colors = []  # (actor, body) of each set_rigid_body_color call
        return

    def __getattr__(self, name):
        return getattr(self._gym, name)

    def create_viewer(self, sim, props):
        return "viewer"

    def add_lines(self, viewer, env, num_lines, vertices, colors):
        assert vertices.dtype == np.float32 and vertices.shape == (2 * num_lines, 3), vertices.shape
        assert colors.dtype == np.float32 and colors.shape == (num_lines, 3), colors.shape
        self.lines.append(num_lines)
        return

    def set_rigid_body_color(self, env, actor, body, mesh_type, color):
        self.colors.append((actor, body))
        return

    def get_viewer_camera_transform(self, viewer, env):
        return gymapi.Transform()

    def query_viewer_has_closed(self, viewer):
        return False

    def query_viewer_action_events(self, viewer):
        return []

    def _no_op(self, *args):
        return

    subscribe_viewer_keyboard_event = viewer_camera_look_at = step_graphics = draw_viewer = _no_op
    poll_viewer_events = clear_lines = _no_op


def build_sim_without_graphics(self, config):
    # create the sim without a graphics device, then hide the viewer behind the proxy
    visualize = self._visualize
    self._visualize = False
    build_sim(self, config)
    self._visualize = visualize
    self._gym = ViewerlessGym(self._gym)
    return


build_sim = ig_env.IGEnv._build_sim
ig_env.IGEnv._build_sim = build_sim_without_graphics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_dir", default="output/smoke/g1_env")
    parser.add_argument("--num_envs", type=int, default=4)
    args = parser.parse_args()

    config = path_loader.load_config(ENV_CONFIG)
    config["env"]["dm"]["motion_file"] = os.path.join(args.out_dir, "motions.yaml")
    config["env"]["dm"]["terrain_save_path"] = os.path.join(args.out_dir, "check_env_terrain.pkl")
    config_file = os.path.join(args.out_dir, "check_debug_visuals.yaml")
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

    env = env_builder.build_env(config_file, args.num_envs, "cuda:0", True, debug_visuals=True)
    gym = env._gym
    assert env._debug_visuals and len(env._ref_char_handles) == args.num_envs
    env.reset()

    action = torch.zeros((args.num_envs, env.get_action_space().shape[0]), device="cuda:0")
    num_steps = 5
    for _ in range(num_steps):
        gym.lines.clear()
        gym.colors.clear()
        env.step(action)

    # one frame
    assert gym.lines == [3 * NUM_HF_POINTS * args.num_envs], gym.lines
    contact_body_ids = env._kin_char_model.get_contact_body_ids().tolist()
    char_colored = [body for actor, body in gym.colors if actor == env._char_handles[0]]
    ref_colored = [body for actor, body in gym.colors if actor == env._ref_char_handles[0]]
    assert char_colored == ref_colored == contact_body_ids, (char_colored, ref_colored)
    print("CHECK visualize + debug_visuals, {} envs, {} steps: per frame 1 add_lines call with {} lines "
          "({} hf points x {} envs), {} contact bodies coloured on the character and on the reference"
          .format(args.num_envs, num_steps, gym.lines[0], NUM_HF_POINTS, args.num_envs, len(char_colored)))
    return


if __name__ == "__main__":
    main()
