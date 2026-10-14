"""Regression tests for the upstream tracker learning-stack fixes (porting plan, Lane B tasks B0 and B1).

CPU only: the agents run on tests/fixtures/learning_fake_env.FakeEnv.
"""
import os
import sys

import gym
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))
import learning_fake_env as fake  # noqa: E402

import parc.motion_tracker.learning.dm_ppo_agent as dm_ppo_agent  # noqa: E402
import parc.motion_tracker.learning.normalizer as normalizer  # noqa: E402
import parc.motion_tracker.learning.ppo_agent as ppo_agent  # noqa: E402
import parc.util.wandb_logger as wandb_logger  # noqa: E402


def build_agent(agent_class, env, **config_overrides):
    torch.manual_seed(0)
    config = fake.make_small_config(agent_name=agent_class.NAME, **config_overrides)
    return agent_class(config=config, env=env, device="cpu")


def rollout(agent):
    # Fills the experience buffer with one iteration of data, like the start of BaseAgent._train_iter.
    agent._curr_obs, agent._curr_info = agent.get_env().reset()
    agent._init_iter()
    agent.eval()
    agent.set_mode(ppo_agent.AgentMode.TRAIN)
    agent._rollout_train(agent._steps_per_iter)
    return


@pytest.fixture
def no_wandb(monkeypatch):
    # With use_wandb: false, WandbLogger calls wandb.init(mode="offline"), which writes ./wandb and
    # overrides WANDB_MODE=disabled. Tests don't need wandb at all.
    monkeypatch.setattr(wandb_logger.wandb, "init", lambda *args, **kwargs: None)
    return


def test_plain_ppo_agent_trains(tmp_path, no_wandb):
    # B0: BaseAgent.train_model called _build_logger(logger_type, log_file, config), but
    # _build_logger only takes log_file, so training a plain PPO agent raised a TypeError.
    torch.manual_seed(0)
    env = fake.FakeEnv(num_envs=4)
    config = fake.make_small_config(agent_name=ppo_agent.PPOAgent.NAME)
    agent = ppo_agent.PPOAgent(config=config, env=env, device="cpu")

    out_model_file = str(tmp_path / "model.pt")
    agent.train_model(max_samples=32, out_model_file=out_model_file, int_output_dir="",
                      logger_type="wandb", log_file=str(tmp_path / "log.txt"))

    assert agent._iter == 2
    assert os.path.isfile(out_model_file)
    assert os.path.isfile(str(tmp_path / "log.txt"))


@pytest.mark.parametrize("num_bodies", [15, 36])
def test_dm_agent_contact_force_buffers_match_env_bodies(num_bodies):
    # B1 (34d6f97): the contact-force buffers were hardcoded to 15 bodies (the humanoid); G1 has 36.
    env = fake.FakeEnv(num_envs=4, num_bodies=num_bodies)
    agent = build_agent(dm_ppo_agent.DMPPOAgent, env)

    for name in ["prev_char_contact_forces", "next_char_contact_forces"]:
        assert agent._exp_buffer.get_data(name).shape == (agent._steps_per_iter, 4, num_bodies, 3)

    agent._curr_obs, agent._curr_info = env.reset()
    agent._init_train()
    agent._train_iter()
    recorded = agent._exp_buffer.get_data("next_char_contact_forces")[-1]
    assert torch.equal(recorded, env._char_contact_forces)


def test_normalizer_init_std_is_clamped_to_min_std():
    # B1 (e04ee11): a zero init_std made normalize() divide by zero.
    init_std = torch.tensor([0.0, 1e-6, 2.0])
    norm = normalizer.Normalizer(init_std.shape, device="cpu", init_mean=torch.zeros(3), init_std=init_std,
                                 min_std=1e-4)
    assert torch.equal(norm.get_std(), torch.tensor([1e-4, 1e-4, 2.0]))
    assert torch.isfinite(norm.normalize(torch.ones([2, 3]))).all()


def test_action_normalizer_handles_zero_width_action_range():
    env = fake.FakeEnv(num_envs=4)
    low = -np.ones(fake.G1_ACTION_SIZE, dtype=np.float32)
    high = np.ones(fake.G1_ACTION_SIZE, dtype=np.float32)
    low[0] = high[0] = 0.5 # e.g. a joint whose lower and upper limits coincide
    env._action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    agent = build_agent(dm_ppo_agent.DMPPOAgent, env)

    assert agent._a_norm.get_std()[0].item() == pytest.approx(1e-4)
    actions = torch.zeros([4, fake.G1_ACTION_SIZE])
    assert torch.isfinite(agent._a_norm.normalize(actions)).all()


@pytest.mark.parametrize("agent_class", [ppo_agent.PPOAgent, dm_ppo_agent.DMPPOAgent])
def test_nan_in_training_data_raises(agent_class):
    # B1 (809c625): NaN checks on the training data, gated by ppo_agent.NANCHECK.
    env = fake.FakeEnv(num_envs=4)
    agent = build_agent(agent_class, env)
    env.nan_obs = True
    rollout(agent)

    with pytest.raises(FloatingPointError, match="NaN in obs"):
        agent._build_train_data()


def test_nan_check_can_be_disabled(monkeypatch):
    agent = build_agent(dm_ppo_agent.DMPPOAgent, fake.FakeEnv(num_envs=4))
    nan_tensor = torch.tensor([0.0, float("nan")])
    with pytest.raises(FloatingPointError):
        agent._check_nan(nan_tensor, "x")

    monkeypatch.setattr(ppo_agent, "NANCHECK", False)
    agent._check_nan(nan_tensor, "x")


def test_nan_loss_raises_instead_of_exiting_with_status_zero(tmp_path, monkeypatch):
    # Upstream called exit() on a NaN loss, which ends the process with status 0.
    monkeypatch.chdir(tmp_path)
    agent = build_agent(dm_ppo_agent.DMPPOAgent, fake.FakeEnv(num_envs=4))
    rollout(agent)
    agent._build_train_data()
    agent.train()

    batch = agent._exp_buffer.sample(8)
    batch["tar_val"][:] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN loss"):
        agent._compute_loss(batch)
    assert (tmp_path / "output" / "debug_batch.pkl").is_file()


@pytest.mark.parametrize("agent_class", [ppo_agent.PPOAgent, dm_ppo_agent.DMPPOAgent])
def test_advantages_stay_finite_without_random_actions(agent_class):
    # With no random actions the advantage statistics fall back to all samples instead of being NaN.
    agent = build_agent(agent_class, fake.FakeEnv(num_envs=4), exp_prob_beg=0.0, exp_prob_end=0.0)
    rollout(agent)
    assert not agent._exp_buffer.get_data("rand_action_mask").any()

    info = agent._build_train_data()
    assert torch.isfinite(agent._exp_buffer.get_data("adv")).all()
    assert torch.isfinite(info["adv_std"]) and torch.isfinite(info["adv_mean"])
