"""Per-DOF joint limits and PD gains for the tracker env (porting plan, Lane A task A1).

CPU only: parc.motion_tracker.envs.char_dof_props does not import IsaacGym.
"""
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import parc.anim.kin_char_model as kin_char_model
import parc.motion_tracker.envs.char_dof_props as char_dof_props

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
G1_FILE = REPO_ROOT / "data" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
HUMANOID_FILE = REPO_ROOT / "data" / "assets" / "humanoid.xml"

# develop's PARC/tracker_config/dm_env_g1.yaml (e7d8c2b)
DEVELOP_G1_PD_PARAMS = {
    "stiffness": {"hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40,
                  "waist": 100, "shoulder": 100, "elbow": 80, "wrist": 40},
    "damping": {"hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2,
                "waist": 2, "shoulder": 2, "elbow": 2, "wrist": 1},
}
FLT_MAX = np.finfo(np.float32).max


def load_char(char_file):
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(char_file))
    return char_model


@pytest.fixture(scope="module")
def g1():
    return load_char(G1_FILE)


@pytest.fixture(scope="module")
def humanoid():
    return load_char(HUMANOID_FILE)


def mjcf_hinge_ranges(char_file):
    # (name, [lo, hi]) of every joint with a range, in document (depth-first) order
    return [(elem.attrib["name"], [float(v) for v in elem.attrib["range"].split()])
            for elem in ET.parse(str(char_file)).getroot().iter("joint") if "range" in elem.attrib]


# ---------------------------------------------------------------- joint limits

def test_g1_dof_limits_are_mjcf_ranges_in_radians(g1):
    ranges = mjcf_hinge_ranges(G1_FILE)
    lower, upper = char_dof_props.get_kin_dof_limits(g1)

    assert lower.dtype == np.float32 and upper.dtype == np.float32
    assert lower.shape == upper.shape == (29,)
    # the G1 MJCF has <compiler angle="radian">: ranges are taken verbatim
    np.testing.assert_allclose(lower, [r[0] for _, r in ranges], rtol=1e-6)
    np.testing.assert_allclose(upper, [r[1] for _, r in ranges], rtol=1e-6)
    np.testing.assert_array_equal(lower, g1._lower_dof_limits.numpy())
    np.testing.assert_array_equal(upper, g1._upper_dof_limits.numpy())

    knee = char_dof_props.get_dof_joint_names(g1).index("left_knee_joint")
    np.testing.assert_allclose([lower[knee], upper[knee]], [-0.087267, 2.8798], rtol=1e-6)


def test_humanoid_dof_limits_expand_spherical_joints(humanoid):
    lower, upper = char_dof_props.get_kin_dof_limits(humanoid)

    assert lower.shape == upper.shape == (28,)
    # abdomen: three hinges (degrees in the MJCF) merged into one spherical joint, DOFs 0-2
    np.testing.assert_allclose(lower[0:3], np.deg2rad([-60.0, -60.0, -70.0]), rtol=1e-6)
    np.testing.assert_allclose(upper[0:3], np.deg2rad([60.0, 90.0, 70.0]), rtol=1e-6)
    np.testing.assert_array_equal(lower, humanoid._lower_dof_limits.numpy())
    np.testing.assert_array_equal(upper, humanoid._upper_dof_limits.numpy())


def test_dof_joint_names_follow_mjcf_order(g1, humanoid):
    # IsaacGym names each DOF after its MJCF joint, depth first; IGCharEnv asserts that every
    # simulated DOF name starts with the kinematic joint name returned here.
    assert char_dof_props.get_dof_joint_names(g1) == [name for name, _ in mjcf_hinge_ranges(G1_FILE)]

    names = char_dof_props.get_dof_joint_names(humanoid)
    assert len(names) == 28
    assert names[0:4] == ["abdomen", "abdomen", "abdomen", "neck"]
    sim_dof_names = [name for name, _ in mjcf_hinge_ranges(HUMANOID_FILE)]
    assert all(sim.startswith(kin) for sim, kin in zip(sim_dof_names, names))


# ---------------------------------------------------------------- pd_params matching

def test_match_pd_params_is_ordered_and_case_insensitive():
    params = {"HIP": 1.0, "hip_pitch": 2.0, "knee": 3.0}
    assert char_dof_props.match_pd_params("left_hip_pitch_joint", params) == ["HIP", "hip_pitch"]
    assert char_dof_props.match_pd_params("LEFT_KNEE_JOINT", params) == ["knee"]
    assert char_dof_props.match_pd_params("waist_yaw_joint", {"hip_yaw": 1.0}) == []


def test_g1_gains_from_develop_pd_params(g1):
    # IsaacGym reads FLT_MAX stiffness and zero damping for the G1 MJCF, which sets neither
    kp, kd, gain_keys = char_dof_props.build_pd_gains(g1, np.full(29, FLT_MAX), np.zeros(29),
                                                      DEVELOP_G1_PD_PARAMS)
    names = char_dof_props.get_dof_joint_names(g1)
    gains = {name: (kp[i], kd[i]) for i, name in enumerate(names)}
    expected = {
        "left_hip_pitch_joint": (100, 2), "right_hip_roll_joint": (100, 2), "left_hip_yaw_joint": (100, 2),
        "right_knee_joint": (150, 4), "left_ankle_pitch_joint": (40, 2), "right_ankle_roll_joint": (40, 2),
        "waist_yaw_joint": (100, 2), "waist_pitch_joint": (100, 2), "left_shoulder_yaw_joint": (100, 2),
        "right_elbow_joint": (80, 2), "left_wrist_pitch_joint": (40, 1), "right_wrist_yaw_joint": (40, 1),
    }
    for name, (exp_kp, exp_kd) in expected.items():
        assert gains[name] == (exp_kp, exp_kd), name

    assert kp.dtype == np.float32 and kd.dtype == np.float32
    # every DOF is set from the config
    assert np.all(kp <= 150) and np.all(kd > 0)
    assert all(kp_key is not None and kd_key is not None for kp_key, kd_key in gain_keys)
    assert gain_keys[names.index("waist_yaw_joint")] == ("waist", "waist")


def test_gains_default_to_asset_values(humanoid):
    default_kp = np.arange(28, dtype=np.float32)
    default_kd = 0.1 * np.arange(28, dtype=np.float32)
    for pd_params in [None, {}, {"stiffness": {}, "damping": None}]:
        kp, kd, gain_keys = char_dof_props.build_pd_gains(humanoid, default_kp, default_kd, pd_params)
        np.testing.assert_array_equal(kp, default_kp)
        np.testing.assert_array_equal(kd, default_kd)
        assert gain_keys == [(None, None)] * 28
    # the inputs are not modified
    np.testing.assert_array_equal(default_kp, np.arange(28, dtype=np.float32))


def test_gains_apply_to_all_dofs_of_a_joint(humanoid):
    names = char_dof_props.get_dof_joint_names(humanoid)
    kp, kd, gain_keys = char_dof_props.build_pd_gains(
        humanoid, np.ones(28), np.ones(28), {"stiffness": {"Shoulder": 9}, "damping": {"left_knee": 3}})

    shoulder = [i for i, name in enumerate(names) if "shoulder" in name]
    assert len(shoulder) == 6  # two spherical joints
    np.testing.assert_array_equal(kp[shoulder], 9.0)
    np.testing.assert_array_equal(np.delete(kp, shoulder), 1.0)
    np.testing.assert_array_equal(kd[names.index("left_knee")], 3.0)
    assert np.sum(kd != 1.0) == 1
    assert gain_keys[shoulder[0]] == ("Shoulder", None)


def test_first_matching_key_wins_and_ambiguity_is_reported(g1, capsys):
    pd_params = {"stiffness": {"hip": 10, "hip_pitch": 20, "unused_key": 5}}
    kp, _, gain_keys = char_dof_props.build_pd_gains(g1, np.zeros(29), np.zeros(29), pd_params)
    names = char_dof_props.get_dof_joint_names(g1)

    assert kp[names.index("left_hip_pitch_joint")] == 10
    assert gain_keys[names.index("left_hip_pitch_joint")] == ("hip", None)
    output = capsys.readouterr().out
    assert "left_hip_pitch_joint matches pd_params.stiffness keys ['hip', 'hip_pitch']" in output
    assert "stiffness.hip_pitch" in output and "stiffness.unused_key" in output


@pytest.mark.parametrize("pd_params", [{"kp": {"knee": 1}}, {"stiffness": {"knee": -1}}])
def test_invalid_pd_params_raise(g1, pd_params):
    with pytest.raises(ValueError):
        char_dof_props.build_pd_gains(g1, np.zeros(29), np.zeros(29), pd_params)
