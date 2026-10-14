"""The Unitree G1 tracker configs (porting plan, Lane A task A4): data/configs/tracker_config/dm_env_g1.yaml
and data/configs/parc_3_tracker_g1.yaml. CPU only.
"""
import pathlib

import numpy as np
import pytest
import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.motion_tracker.envs.char_dof_props as char_dof_props

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "data" / "configs"
G1_ENV = CONFIG_DIR / "tracker_config" / "dm_env_g1.yaml"
DEFAULT_ENV = CONFIG_DIR / "tracker_config" / "dm_env_default.yaml"

# keys of develop's G1 env config that nothing on main reads, and dead keys of dm_env_default.yaml
DROPPED_KEYS = ["fixed_camera_id", "has_target_xy_obs", "rel_deepmimic_w", "rel_task_w", "target_radius",
                "task1_w", "task2_w", "target_motion_height_buffer", "mgdm", "enable_replan_timer_obs"]


def load_yaml(path):
    with open(str(path)) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def g1_env():
    return load_yaml(G1_ENV)["env"]


@pytest.fixture(scope="module")
def g1():
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(REPO_ROOT / load_yaml(G1_ENV)["env"]["char_file"]))
    return char_model


def test_char_file_is_the_g1(g1_env, g1):
    assert g1_env["char_file"] == "data/assets/unitree_g1/g1_mocap_29dof.xml"
    assert (g1.get_num_bodies(), g1.get_dof_size()) == (36, 29)


def test_list_lengths_match_the_g1(g1_env, g1):
    assert len(g1_env["contact_weights"]) == 36 == g1.get_num_contact_bodies()
    assert len(g1_env["init_pose"]) == 35 == 6 + g1.get_dof_size()
    assert len(g1_env["joint_err_w"]) == 35 == g1.get_num_joints() - 1
    assert len(g1_env["pose_termination_dist"]) == 35 == g1.get_num_bodies() - 1


def test_key_bodies_exist(g1_env, g1):
    assert g1_env["key_bodies"] == ["left_ankle_roll_link", "right_ankle_roll_link", "head_link",
                                    "left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert all(g1.has_body(name) for name in g1_env["key_bodies"])


def test_pd_params_set_every_dof(g1_env, g1):
    nan = np.full(29, np.nan)
    kp, kd, gain_keys = char_dof_props.build_pd_gains(g1, nan, nan, g1_env["pd_params"])
    assert not np.any(np.isnan(kp)) and not np.any(np.isnan(kd))
    assert sorted(set(kp.tolist())) == [40.0, 80.0, 100.0, 150.0]
    assert sorted(set(kd.tolist())) == [1.0, 2.0, 4.0]
    # every key is used, and no joint matches two keys (parc_3 re-saves the config with sorted keys)
    names = char_dof_props.get_dof_joint_names(g1)
    for i, group in enumerate(["stiffness", "damping"]):
        params = g1_env["pd_params"][group]
        assert all(len(char_dof_props.match_pd_params(name, params)) == 1 for name in names)
        assert sorted(set(keys[i] for keys in gain_keys)) == sorted(params)


def test_same_keys_as_the_default_env(g1_env):
    default_config = load_yaml(DEFAULT_ENV)
    default_env = default_config["env"]
    assert set(g1_env) == (set(default_env) - {"mgdm", "enable_replan_timer_obs"}) | {"pd_params"}
    assert set(g1_env["dm"]) == set(default_env["dm"])
    assert {k: v for k, v in load_yaml(G1_ENV).items() if k != "env"} == \
        {k: v for k, v in default_config.items() if k != "env"}
    for key in DROPPED_KEYS:
        assert key not in g1_env and key not in g1_env["dm"]
    assert not any(key.startswith("target_xy") for key in g1_env["dm"])

    assert g1_env["fraction_dm_envs"] == 1.0 and g1_env["debug_visuals"] is False
    for key in [k for k in default_env if k.startswith("ray_")] + ["min_obs_h", "max_obs_h", "tar_obs_steps"]:
        assert g1_env[key] == default_env[key], key


def test_parc_3_tracker_g1_config():
    config = load_yaml(CONFIG_DIR / "parc_3_tracker_g1.yaml")
    assert set(config) == set(load_yaml(CONFIG_DIR / "parc_3_tracker_default.yaml"))
    assert config["env_config"] == "data/configs/tracker_config/dm_env_g1.yaml"
    assert config["agent_config"] == "data/configs/tracker_config/dm_agent_g1.yaml"
    assert (REPO_ROOT / config["env_config"]).is_file() and (REPO_ROOT / config["agent_config"]).is_file()
    assert config["in_model_file"] is None
    assert config["create_dataset_config"] == "data/configs/create_dataset_config_g1.yaml"
