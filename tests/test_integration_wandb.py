"""P3 integration: which loggers start a wandb run (CPU, wandb.init stubbed).

- The motion optimization with use_wandb false starts no wandb run. It used to start an offline run
  for every optimized motion that had a log file (parc_2_kin_gen, run_optimize_motions, ...).
- Tracker training keeps upstream's behaviour: WandbLogger starts an offline run when use_wandb is
  false. WandbLogger.write_log also works without set_step_key.
"""
import os
import sys

import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import motiontools_synth as synth  # noqa: E402

import parc.motion_synthesis.motion_opt.motion_optimization as moopt  # noqa: E402
import parc.util.geom_util as geom_util  # noqa: E402
import parc.util.wandb_logger as wandb_logger  # noqa: E402


@pytest.fixture
def wandb_calls(monkeypatch):
    calls = {"init": [], "log": [], "finish": 0}

    def finish(*args, **kwargs):
        calls["finish"] += 1

    monkeypatch.setattr(wandb_logger.wandb, "init", lambda *args, **kwargs: calls["init"].append(kwargs))
    monkeypatch.setattr(wandb_logger.wandb, "log", lambda *args, **kwargs: calls["log"].append(args))
    monkeypatch.setattr(wandb_logger.wandb, "finish", finish)
    return calls


@pytest.fixture(scope="module")
def humanoid_motion():
    char_model = synth.load_char("humanoid")
    frames, terrain = synth.make_motion(char_model)
    return char_model, frames, terrain


def optimize(humanoid_motion, use_wandb, log_file):
    char_model, frames, terrain = humanoid_motion
    return moopt.motion_contact_optimization(
        src_motion_frames=frames, body_points=geom_util.get_char_point_samples(char_model),
        terrain=terrain, char_model=char_model, num_iters=2, step_size=1e-2,
        w_root_pos=1.0, w_root_rot=10.0, w_joint_rot=1.0, w_smoothness=10.0, w_penetration=100.0,
        w_contact=100.0, w_sliding=10.0, w_body_constraints=0.0, w_jerk=100.0,
        body_constraints=None, max_jerk=1000.0, exp_name="motion_0", use_wandb=use_wandb,
        log_file=str(log_file))


def test_motion_optimization_without_wandb_starts_no_run(wandb_calls, humanoid_motion, tmp_path):
    log_file = tmp_path / "log" / "log_motion_0.txt"
    optimize(humanoid_motion, use_wandb=False, log_file=log_file)

    char_model, frames, _ = humanoid_motion
    simplified_log_file = tmp_path / "log" / "log_motion_0_simplified.txt"
    moopt.simplified_motion_optimization(src_motion_frames=frames, char_model=char_model, num_iters=2,
                                         step_size=1e-2, w_root_pos=1.0, w_root_rot=10.0, w_joint_rot=1.0,
                                         w_jerk=100.0, use_wandb=False, log_file=str(simplified_log_file))

    assert wandb_calls == {"init": [], "log": [], "finish": 0}
    for path in (log_file, simplified_log_file):
        assert "TOTAL WEIGHTED LOSS" in path.read_text()


def test_motion_optimization_with_wandb_starts_an_online_run(wandb_calls, humanoid_motion, tmp_path):
    optimize(humanoid_motion, use_wandb=True, log_file=tmp_path / "log.txt")
    assert [kwargs.get("mode") for kwargs in wandb_calls["init"]] == [None]
    assert wandb_calls["init"][0]["name"] == "motion_0"
    assert wandb_calls["finish"] == 1


def test_wandb_logger_without_connection_starts_an_offline_run(wandb_calls, monkeypatch, tmp_path):
    # tracker training (BaseAgent._build_logger) with use_wandb: false
    log = wandb_logger.WandbLogger(project_name="parkour_g1", exp_name="exp", connect_to_wandb=False)
    log.configure_output_file(str(tmp_path / "log.txt"))
    assert wandb_calls["init"] == [{"project": "parkour_g1", "name": "exp", "config": None, "mode": "offline"}]

    # an active run, no step key: the row count is the step, and nothing is sent without a connection
    monkeypatch.setattr(wandb_logger.wandb, "run", object())
    log.log("Train_Return", 1.5)
    log.write_log()
    assert wandb_calls["log"] == []
    assert "Train_Return" in (tmp_path / "log.txt").read_text()
