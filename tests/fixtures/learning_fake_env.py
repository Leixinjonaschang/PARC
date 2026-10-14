"""CPU stand-in for IGParkourEnv, shared by the learning-stack tests (tests/test_learning_*.py).

FakeEnv implements only the env surface that the tracker agents use, and by default reports the
G1 observation contract of doc/porting_plan.md, Appendix A:

    key            shape      normalized
    char_obs       (266,)     yes
    tar_obs        (6, 234)   yes
    tar_contacts   (6, 36)    no
    char_contacts  (36,)      no
    hf             (441,)     no         -> 2363 in total

Observations and rewards are random; every env finishes its episode after `episode_length` steps.
"""
import collections
import copy
import os

import gym
import numpy as np
import torch
import yaml

import parc.motion_tracker.envs.base_env as base_env

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACKER_CONFIG_DIR = os.path.join(REPO_ROOT, "data", "configs", "tracker_config")

# (key, shape, use_normalizer), in the env's concatenation order.
G1_OBS_SHAPES = (
    ("char_obs", (266,), True),
    ("tar_obs", (6, 234), True),
    ("tar_contacts", (6, 36), False),
    ("char_contacts", (36,), False),
    ("hf", (441,), False),
)
G1_OBS_DIM = 2363
G1_NUM_BODIES = 36
G1_ACTION_SIZE = 29
REWARD_KEYS = ("pose_r", "vel_r", "total_r")


def calc_obs_dim(obs_shapes):
    return int(sum(np.prod(shape) for _, shape, _ in obs_shapes))


class FakeEnv:
    def __init__(self, num_envs=4, obs_shapes=G1_OBS_SHAPES, obs_dim=None, num_bodies=G1_NUM_BODIES,
                 action_size=G1_ACTION_SIZE, episode_length=4, seed=0):
        self._device = "cpu"
        self._num_envs = num_envs
        self._config = {"env": {"env_name": "fake"}}
        self._report_tracking_error = False
        self._obs_shapes = tuple(obs_shapes)
        self._obs_dim = calc_obs_dim(obs_shapes) if obs_dim is None else obs_dim
        self._char_contact_forces = torch.zeros([num_envs, num_bodies, 3])

        action_high = np.ones(action_size, dtype=np.float32)
        self._action_space = gym.spaces.Box(low=-action_high, high=action_high, dtype=np.float32)

        self._episode_length = episode_length
        self._generator = torch.Generator().manual_seed(seed)
        self._timestep = torch.zeros([num_envs], dtype=torch.int)
        self._ep_num = torch.zeros([num_envs], dtype=torch.int64)
        self._mode = base_env.EnvMode.TRAIN
        self._info = dict()
        self.nan_obs = False # set to True to emit NaN observations
        self._obs_buf = self._sample_obs()
        return

    def __deepcopy__(self, memo):
        # The real env holds IsaacGym handles, which cannot be deep-copied. Models must not keep a
        # reference to the env.
        raise TypeError("FakeEnv must not be deep-copied")

    def _compute_obs(self, env_ids=None, ret_obs_shapes=False):
        if ret_obs_shapes:
            obs_shapes = collections.OrderedDict()
            for key, shape, use_normalizer in self._obs_shapes:
                obs_shapes[key] = {"use_normalizer": use_normalizer, "shape": torch.Size(shape)}
            return obs_shapes

        obs = self._obs_buf
        if env_ids is not None:
            obs = obs[env_ids]
        return obs

    def get_obs_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

    def get_action_space(self):
        return self._action_space

    def get_num_envs(self):
        return self._num_envs

    def set_mode(self, mode):
        self._mode = mode
        return

    def get_reward_bounds(self):
        return (-np.inf, np.inf)

    def get_reward_fail(self):
        return 0.0

    def get_reward_succ(self):
        return 0.0

    def has_dm_envs(self):
        return False

    def get_extra_log_info(self):
        return dict()

    def post_test_update(self):
        return

    def _update_reward(self):
        self._info["rewards"] = {key: torch.rand([self._num_envs], generator=self._generator) for key in REWARD_KEYS}
        return

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs)
        if len(env_ids) > 0:
            self._timestep[env_ids] = 0
            self._ep_num[env_ids] += 1
            self._obs_buf[env_ids] = self._sample_obs()[env_ids]
        self._update_info()
        return self._obs_buf, self._info

    def step(self, action):
        assert action.shape == (self._num_envs, self._action_space.shape[0])
        self._timestep += 1
        self._obs_buf = self._sample_obs()
        self._char_contact_forces[:] = torch.rand(self._char_contact_forces.shape, generator=self._generator)

        self._update_reward()
        reward = self._info["rewards"]["total_r"]
        done = torch.full([self._num_envs], base_env.DoneFlags.NULL.value, dtype=torch.int)
        done[self._timestep >= self._episode_length] = base_env.DoneFlags.TIME.value
        self._update_info()
        return self._obs_buf, reward, done, self._info

    def _sample_obs(self):
        obs = torch.randn([self._num_envs, self._obs_dim], generator=self._generator)
        if self.nan_obs:
            obs[:, 0] = float("nan")
        return obs

    def _update_info(self):
        if "rewards" not in self._info:
            self._update_reward()
        self._info["timestep"] = self._timestep.clone()
        self._info["ep_num"] = self._ep_num.clone()
        self._info["compute_time"] = 0.0
        self._info["char_contact_forces"] = self._char_contact_forces.clone()
        return


def load_tracker_config(filename):
    with open(os.path.join(TRACKER_CONFIG_DIR, filename), "r") as f:
        return yaml.safe_load(f)


def make_small_config(base_file="dm_agent_default.yaml", **overrides):
    """An agent config from data/configs/tracker_config with small nets and short iterations, for CPU tests.

    Keyword overrides replace top-level keys; `model` overrides are merged into the model section.
    """
    config = load_tracker_config(base_file)
    config["model"]["actor_net"] = "fc_2layers_1024units"
    config["model"]["critic_net"] = "fc_2layers_1024units"
    config["steps_per_iter"] = 4
    config["batch_size"] = 2
    config["update_epochs"] = 2
    config["iters_per_output"] = 1
    config["iters_per_checkpoint"] = 1
    config["test_episodes"] = 4
    config["use_wandb"] = False

    model_overrides = overrides.pop("model", None)
    config.update(copy.deepcopy(overrides))
    if model_overrides is not None:
        config["model"].update(copy.deepcopy(model_overrides))
    return config
