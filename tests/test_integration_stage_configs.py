"""P3 integration: the stage scripts keep the key order of the env configs they copy (CPU).

env.pd_params is matched first key first (char_dof_props.build_pd_gains), so parc_0 (the
p3_tracker/input_dm_env.yaml copy), parc_3 (dm_env.yaml) and parc_4 (record_env.yaml) must not
sort the keys when they re-save an env config. parc_3 and parc_4 run against a stub run_tracker.
"""
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest
import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.motion_tracker.envs.char_dof_props as char_dof_props
import parc.util.path_loader as path_loader
import scripts.parc_0_setup_iter as parc_0

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "data" / "configs"
G1_CHAR_FILE = "data/assets/unitree_g1/g1_mocap_29dof.xml"

# overlapping keys: hip_pitch must come first, or the hip pitch joints get the "hip" gains
PD_PARAMS = {"stiffness": {"hip_pitch": 300, "hip": 100, "knee": 150, "ankle": 40},
             "damping": {"hip_pitch": 6, "hip": 2, "knee": 4, "ankle": 2}}


@pytest.fixture(autouse=True)
def repo_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))


@pytest.fixture
def env_file(tmp_path):
    env_config = path_loader.load_config(CONFIG_DIR / "tracker_config" / "dm_env_g1.yaml")
    env_config["env"]["pd_params"] = PD_PARAMS
    path = tmp_path / "dm_env_overlapping_pd_params.yaml"
    path.write_text(yaml.safe_dump(env_config, sort_keys=False))
    return path


def assert_same_key_order(written_path, source_path):
    written = yaml.safe_load(pathlib.Path(written_path).read_text())
    source = yaml.safe_load(pathlib.Path(source_path).read_text())
    assert list(written) == list(source)
    # parc_4 adds env.output_motion_dir
    assert [key for key in written["env"] if key in source["env"]] == list(source["env"])
    for group in ("stiffness", "damping"):
        assert list(written["env"]["pd_params"][group]) == list(PD_PARAMS[group])

    g1 = kin_char_model.KinCharModel("cpu")
    g1.load_char_file(G1_CHAR_FILE)
    nan = np.full(g1.get_dof_size(), np.nan)
    kp, kd, _ = char_dof_props.build_pd_gains(g1, nan, nan, written["env"]["pd_params"])
    names = char_dof_props.get_dof_joint_names(g1)
    assert kp[names.index("left_hip_pitch_joint")] == 300 and kp[names.index("left_hip_roll_joint")] == 100
    assert kd[names.index("right_hip_pitch_joint")] == 6 and kd[names.index("right_hip_yaw_joint")] == 2


def load_with_fake_run_tracker(monkeypatch, script):
    # parc_3 and parc_4 import scripts.run_tracker, which imports IsaacGym
    calls = []
    fake_run_tracker = types.ModuleType("scripts.run_tracker")
    fake_run_tracker.main = lambda argv: calls.append(argv)
    import scripts
    monkeypatch.setitem(sys.modules, "scripts.run_tracker", fake_run_tracker)
    monkeypatch.setattr(scripts, "run_tracker", fake_run_tracker, raising=False)
    spec = importlib.util.spec_from_file_location(script + "_under_test", REPO_ROOT / "scripts" / (script + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def write_dataset(tmp_path):
    dataset_file = tmp_path / "motions.yaml"
    dataset_file.write_text(yaml.safe_dump({"motions": [{"file": "a.pkl", "weight": 1.0}]}))
    return dataset_file


def test_setup_iter_keeps_env_key_order(tmp_path, env_file):
    tracker_config = path_loader.load_config(CONFIG_DIR / "parc_3_tracker_g1.yaml")
    tracker_config["env_config"] = str(env_file)
    tracker_config_path = tmp_path / "tracker.yaml"
    tracker_config_path.write_text(yaml.safe_dump(tracker_config))
    config = path_loader.load_config(CONFIG_DIR / "parc_0_setup_iter_config_g1.yaml")
    config.update(output_dir=str(tmp_path / "iter_1"),
                  iter_start_dataset_path=str(tmp_path / "iter_1" / "start_motions.yaml"),
                  input_tracker_config_path=str(tracker_config_path), kin_gen_num_batches_of_motions=1)

    parc_0.setup_iter(config)

    written_tracker = yaml.safe_load((tmp_path / "iter_1" / "p3_tracker" / "tracker.yaml").read_text())
    assert written_tracker["agent_config"] == "data/configs/tracker_config/dm_agent_g1.yaml"
    assert written_tracker["env_config"] == str(tmp_path / "iter_1" / "p3_tracker" / "input_dm_env.yaml")
    assert_same_key_order(written_tracker["env_config"], env_file)


def test_tracker_keeps_env_and_agent_key_order(monkeypatch, tmp_path, env_file):
    parc_3, calls = load_with_fake_run_tracker(monkeypatch, "parc_3_tracker")
    agent_file = CONFIG_DIR / "tracker_config" / "dm_agent_g1.yaml"
    output_dir = tmp_path / "p3_tracker"
    config = {"env_config": str(env_file), "agent_config": str(agent_file), "output_dir": str(output_dir),
              "num_envs": 4, "max_samples": 1000, "device": "cpu", "dataset_file": str(write_dataset(tmp_path)),
              "in_model_file": None}

    parc_3.train_tracker(config)

    assert len(calls) == 1
    assert_same_key_order(output_dir / "dm_env.yaml", env_file)
    agent = yaml.safe_load((output_dir / "agent_config.yaml").read_text())
    assert list(agent) == list(yaml.safe_load(agent_file.read_text()))
    assert list(agent["model"]) == list(yaml.safe_load(agent_file.read_text())["model"])


def test_phys_record_keeps_env_key_order(monkeypatch, tmp_path, env_file):
    parc_4, calls = load_with_fake_run_tracker(monkeypatch, "parc_4_phys_record")
    output_dir = tmp_path / "p4_phys_record"
    config = {"device": "cpu", "output_dir": str(output_dir), "model_file": str(tmp_path / "model.pt"),
              "agent_file": str(tmp_path / "agent_config.yaml"), "env_file": str(env_file),
              "dataset_file": str(write_dataset(tmp_path))}

    parc_4.record_motions(config)

    assert len(calls) == 1
    assert_same_key_order(output_dir / "record_env.yaml", env_file)
