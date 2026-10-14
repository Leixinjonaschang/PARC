"""Humanoid regression test for the G1 port of KinCharModel, torch_util and geom_util.

tests/fixtures/charmodel_humanoid_snapshot.npz was written by ``compute_snapshot()``
below on develop_migration @956fc75, i.e. *before* the G1 changes to
parc/anim/kin_char_model.py, parc/util/torch_util.py and parc/util/geom_util.py.
The humanoid must reproduce it bit for bit (exact equality, CPU).

Regenerate only when a humanoid change is intended:
    uv run python tests/test_charmodel_humanoid.py --write-snapshot
"""
import argparse
import math
import pathlib

import numpy as np
import pytest
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.util.geom_util as geom_util
import parc.util.torch_util as torch_util

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HUMANOID_FILE = REPO_ROOT / "data" / "assets" / "humanoid.xml"
SNAPSHOT_FILE = REPO_ROOT / "tests" / "fixtures" / "charmodel_humanoid_snapshot.npz"

INPUT_KEYS = ("in_root_pos", "in_root_rot", "in_dof", "in_exp_map")


def load_humanoid():
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(HUMANOID_FILE))
    return char_model


def make_inputs(dof_size):
    gen = torch.Generator().manual_seed(20261014)
    num_poses = 32
    root_pos = torch.rand(num_poses, 3, generator=gen) * 2.0 - 1.0
    root_rot = torch_util.quat_normalize(torch.randn(num_poses, 4, generator=gen))

    # Spherical DOFs are exp maps; |theta| up to ~4.3 rad exercises the angle wrap past pi.
    dof = (torch.rand(num_poses, dof_size, generator=gen) * 2.0 - 1.0) * 2.5
    dof[0] = 0.0  # zero exp maps (identity pose)
    dof[1] *= 1e-6  # exp maps below the 1e-5 threshold of exp_map_to_axis_angle
    dof[2] *= 3.0  # |theta| up to ~13 rad, beyond 4 pi

    exp_map = (torch.rand(256, 3, generator=gen) * 2.0 - 1.0) * (4.0 * math.pi / math.sqrt(3.0))
    exp_map[0] = 0.0
    exp_map[1] = torch.tensor([1e-7, -2e-7, 3e-7])
    exp_map[2] = torch.tensor([0.0, 0.0, math.pi])
    exp_map[3] = torch.tensor([0.0, 2.0 * math.pi, 0.0])
    return {
        "in_root_pos": root_pos.numpy(),
        "in_root_rot": root_rot.numpy(),
        "in_dof": dof.numpy(),
        "in_exp_map": exp_map.numpy(),
    }


def _pad(values, length, fill=np.nan):
    out = np.full([length], fill, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out[:values.shape[0]] = values
    return out


def compute_snapshot(inputs=None):
    char_model = load_humanoid()
    if inputs is None:
        inputs = make_inputs(char_model.get_dof_size())

    out = dict(inputs)
    body_names = char_model.get_body_names()
    out["body_names"] = np.array(body_names)
    out["counts"] = np.array([char_model.get_num_bodies(), char_model.get_num_joints(),
                              char_model.get_dof_size()], dtype=np.int64)
    out["parent_indices"] = char_model._parent_indices.numpy()
    out["local_translation"] = char_model._local_translation.numpy()
    out["local_rotation"] = char_model._local_rotation.numpy()
    out["body_ids_by_name"] = np.array([char_model.get_body_id(n) for n in body_names], dtype=np.int64)
    out["joint_ids_by_name"] = np.array([char_model.get_joint_id(n) for n in body_names], dtype=np.int64)
    out["contact_body_names"] = np.array(char_model._contact_body_names)
    out["contact_body_ids"] = char_model.get_contact_body_ids().numpy()

    joints = char_model._joints
    out["joint_names"] = np.array([j.name for j in joints])
    out["joint_types"] = np.array([j.joint_type.value for j in joints], dtype=np.int64)
    out["joint_dof_idx"] = np.array([j.dof_idx for j in joints], dtype=np.int64)
    out["joint_axes"] = np.stack([_pad([] if j.axis is None else j.axis.numpy(), 3) for j in joints])
    out["joint_limits"] = np.stack([_pad([] if j.limits is None else j.limits.numpy(), 6) for j in joints])
    out["lower_dof_limits"] = char_model._lower_dof_limits.numpy()
    out["upper_dof_limits"] = char_model._upper_dof_limits.numpy()

    out["default_geom_type"] = np.array(char_model._default_geom_type.value, dtype=np.int64)
    geom_rows = [(b, g) for b in range(char_model.get_num_bodies()) for g in char_model.get_geoms(b)]
    out["geom_body_ids"] = np.array([b for b, _ in geom_rows], dtype=np.int64)
    out["geom_names"] = np.array([str(g._name) for _, g in geom_rows])
    out["geom_types"] = np.array([g._shape_type.value for _, g in geom_rows], dtype=np.int64)
    out["geom_offsets"] = np.stack([g._offset.numpy() for _, g in geom_rows])
    out["geom_dims"] = np.stack([_pad(g._dims.numpy(), 3) for _, g in geom_rows])
    out["geom_quats"] = np.stack([_pad([] if g._quat is None else g._quat, 4) for _, g in geom_rows])
    out["geom_radius"] = np.array([np.nan if g._radius is None else g._radius for _, g in geom_rows])

    root_pos = torch.from_numpy(inputs["in_root_pos"])
    root_rot = torch.from_numpy(inputs["in_root_rot"])
    dof = torch.from_numpy(inputs["in_dof"])
    joint_rot = char_model.dof_to_rot(dof)
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    out["joint_rot"] = joint_rot.numpy()
    out["rot_to_dof"] = char_model.rot_to_dof(joint_rot).numpy()
    out["fk_body_pos"] = body_pos.numpy()
    out["fk_body_rot"] = body_rot.numpy()

    exp_map = torch.from_numpy(inputs["in_exp_map"])
    axis, angle = torch_util.exp_map_to_axis_angle(exp_map)
    out["exp_map_axis"] = axis.numpy()
    out["exp_map_angle"] = angle.numpy()
    out["exp_map_quat"] = torch_util.exp_map_to_quat(exp_map).numpy()

    points = geom_util.get_minimal_char_point_samples(char_model)
    out["min_points_counts"] = np.array([p.shape[0] for p in points], dtype=np.int64)
    out["min_points"] = torch.cat(points, dim=0).numpy()
    return out


def _load_snapshot():
    with np.load(SNAPSHOT_FILE, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


SNAPSHOT = _load_snapshot() if SNAPSHOT_FILE.exists() else {}


@pytest.fixture(scope="module")
def actual():
    assert SNAPSHOT, "missing {}".format(SNAPSHOT_FILE)
    return compute_snapshot({k: SNAPSHOT[k] for k in INPUT_KEYS})


def test_snapshot_has_same_keys(actual):
    assert sorted(actual.keys()) == sorted(SNAPSHOT.keys())


@pytest.mark.parametrize("key", sorted(k for k in SNAPSHOT if k not in INPUT_KEYS))
def test_matches_pre_port_snapshot(actual, key):
    assert actual[key].dtype == SNAPSHOT[key].dtype, key
    assert actual[key].shape == SNAPSHOT[key].shape, key
    np.testing.assert_array_equal(actual[key], SNAPSHOT[key], err_msg=key)


def test_minimal_point_samples_per_body(actual):
    counts = actual["min_points_counts"]
    assert counts.shape[0] == actual["counts"][0]
    assert np.all(counts >= 1)
    assert counts.sum() == actual["min_points"].shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-snapshot", action="store_true")
    args = parser.parse_args()
    if args.write_snapshot:
        SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SNAPSHOT_FILE, **compute_snapshot())
        print("wrote", SNAPSHOT_FILE)
