"""Regression tests for the upstream tracker learning-stack fixes (porting plan, Lane B tasks B0 and B1).

CPU only: the agents run on tests/fixtures/learning_fake_env.FakeEnv.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))
import learning_fake_env as fake  # noqa: E402

import parc.motion_tracker.learning.ppo_agent as ppo_agent  # noqa: E402
import parc.util.wandb_logger as wandb_logger  # noqa: E402


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
