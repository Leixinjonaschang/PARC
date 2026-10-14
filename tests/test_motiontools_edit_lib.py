"""motion_edit_lib: legacy save of body constraints, MotionData.set_device, change_motion_fps,
and the box-feet / sphere-hands checks of the contact helpers. CPU only."""
import os
import pickle
import sys
import warnings

import numpy as np
import pytest
import torch

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import motiontools_synth as synth  # noqa: E402

import parc.motion_synthesis.motion_opt.motion_optimization as moopt  # noqa: E402
import parc.util.motion_edit_lib as medit_lib  # noqa: E402
import parc.util.terrain_util as terrain_util  # noqa: E402


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))
    with warnings.catch_warnings():
        # MotionData, load_motion_file and save_motion_data are deprecated on purpose
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


@pytest.fixture(scope="module")
def humanoid():
    return synth.load_char("humanoid")


def _dof_frames(char_model, num_frames=8):
    frames, terrain = synth.make_motion(char_model, num_frames=num_frames)
    dof = torch.cat([frames.root_pos, torch.zeros(num_frames, 3), char_model.rot_to_dof(frames.joint_rot)], dim=-1)
    return dof, frames.contacts, terrain


def _constraints(num_bodies):
    constraints = [[] for _ in range(num_bodies)]
    constraints[2] = [moopt.BodyConstraint(start_frame_idx=1, end_frame_idx=3,
                                           constraint_point=torch.tensor([0.5, -0.25, 0.0]))]
    constraints[-1] = [moopt.BodyConstraint(start_frame_idx=0, end_frame_idx=0,
                                            constraint_point=torch.tensor([1.0, 2.0, 3.0])),
                       moopt.BodyConstraint(start_frame_idx=4, end_frame_idx=6,
                                            constraint_point=torch.tensor([-1.0, 0.0, 0.25]))]
    return constraints


def _assert_same_constraints(actual, expected):
    assert len(actual) == len(expected)
    for body_actual, body_expected in zip(actual, expected):
        assert len(body_actual) == len(body_expected)
        for a, e in zip(body_actual, body_expected):
            assert isinstance(a, moopt.BodyConstraint)
            assert (a.start_frame_idx, a.end_frame_idx) == (e.start_frame_idx, e.end_frame_idx)
            assert torch.equal(a.constraint_point.cpu(), e.constraint_point.cpu())


def test_save_motion_data_stores_body_constraints_as_numpy(tmp_path, humanoid):
    # upstream called the non-existent moopt.body_constraint_to_numpy_recursive here
    dof, contacts, terrain = _dof_frames(humanoid)
    constraints = _constraints(humanoid.get_num_bodies())
    hf_mask_inds = [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4, 5]])]
    path = tmp_path / "legacy.pkl"

    medit_lib.save_motion_data(str(path), motion_frames=dof, contact_frames=contacts, terrain=terrain,
                               fps=30, loop_mode="CLAMP", hf_mask_inds=hf_mask_inds,
                               **{"opt:body_constraints": constraints})

    with open(path, "rb") as f:
        raw = pickle.load(f)
    stored = raw["opt:body_constraints"]
    assert isinstance(stored[2][0], dict)
    assert isinstance(stored[2][0]["constraint_point"], np.ndarray)
    assert stored[-1][1]["start_frame_idx"] == 4
    assert all(isinstance(x, np.ndarray) for x in raw["hf_mask_inds"])

    motion_data = medit_lib.load_motion_file(str(path))
    _assert_same_constraints(motion_data.get_opt_body_constraints(), constraints)

    # MotionData.save_to_file keeps them too
    resaved = tmp_path / "resaved.pkl"
    motion_data.save_to_file(str(resaved), verbose=False)
    _assert_same_constraints(medit_lib.load_motion_file(str(resaved)).get_opt_body_constraints(), constraints)


def test_motion_data_set_device_moves_tensors(humanoid):
    # upstream's `... or "target_xy"` was always true, so set_device moved nothing.
    # The meta device stands in for a GPU: every tensor must end up there.
    dof, contacts, terrain = _dof_frames(humanoid)
    data = {
        "frames": dof,
        "contacts": contacts,
        "terrain": terrain,
        "fps": 30,
        "loop_mode": "CLAMP",
        "path_nodes": torch.zeros(3, 3),
        "hf_mask_inds": [torch.tensor([[0, 1]]), np.array([[2, 3]])],
        "opt:body_constraints": _constraints(humanoid.get_num_bodies()),
        "min_point_offset": torch.tensor([0.1, 0.2]),
        "target_xy": np.zeros((4, 2), dtype=np.float32),
        "loss": 1.5,
        "obs": np.ones(3, dtype=np.float32),
    }
    motion_data = medit_lib.MotionData(data)
    motion_data.set_device("meta")

    stored = motion_data._data
    for key in ("frames", "contacts", "path_nodes", "min_point_offset"):
        assert stored[key].device.type == "meta", key
    assert stored["terrain"].hf.device.type == "meta"
    assert stored["terrain"].min_point.device.type == "meta"
    assert all(isinstance(x, torch.Tensor) and x.device.type == "meta" for x in stored["hf_mask_inds"])
    for body_constraints in stored["opt:body_constraints"]:
        for constraint in body_constraints:
            assert constraint.constraint_point.device.type == "meta"
    # non-tensor values are left alone
    assert stored["fps"] == 30 and stored["loop_mode"] == "CLAMP" and stored["loss"] == 1.5
    assert isinstance(stored["target_xy"], np.ndarray) and isinstance(stored["obs"], np.ndarray)


def test_change_motion_fps(humanoid):
    # upstream called the old MotionLib(frames, ..., "motion_frames", ...) constructor
    dof, _, _ = _dof_frames(humanoid, num_frames=31)
    new_frames = medit_lib.change_motion_fps(dof, src_fps=30, tar_fps=15, char_model=humanoid)
    # 1 s at 15 fps; the float time accumulation may add a sample at t ~= 1 s
    assert new_frames.shape[0] in (15, 16) and new_frames.shape[1] == dof.shape[1]
    # frames 0 and 2 of the source are samples 0 and 1 (up to the dof -> quat -> dof round trip)
    assert torch.allclose(new_frames[0], dof[0], atol=1e-3)
    assert torch.allclose(new_frames[1], dof[2], atol=1e-3)


def test_contact_helpers_reject_mesh_feet_and_hands():
    g1 = synth.load_char("g1")
    dof, _, terrain = _dof_frames(g1)
    with pytest.raises(ValueError, match="left_ankle_roll_link.*MESH"):
        medit_lib.compute_ground_plane_foot_contacts(dof, g1)
    with pytest.raises(ValueError, match="left_ankle_roll_link.*MESH"):
        medit_lib.compute_hf_foot_contacts_and_correct_pen(dof, terrain, g1)
    with pytest.raises(ValueError, match="left_wrist_yaw_link.*MESH"):
        medit_lib.compute_motion_terrain_hand_contacts(dof, terrain, g1)
    with pytest.raises(ValueError, match="left_ankle_roll_link"):
        medit_lib.correct_foot_ground_pen(dof, g1)


def test_contact_helpers_still_work_on_the_humanoid(humanoid):
    dof, _, terrain = _dof_frames(humanoid)
    num_bodies = humanoid.get_num_bodies()
    foot_ids = [humanoid.get_body_id("left_foot"), humanoid.get_body_id("right_foot")]
    hand_ids = [humanoid.get_body_id("left_hand"), humanoid.get_body_id("right_hand")]

    contacts = medit_lib.compute_ground_plane_foot_contacts(dof, humanoid)
    assert contacts.shape == (dof.shape[0], num_bodies)
    assert contacts[:, foot_ids].sum() > 0  # synthetic motion starts with a foot on the ground

    corrected, contacts = medit_lib.compute_hf_foot_contacts_and_correct_pen(dof, terrain, humanoid)
    assert corrected.shape == dof.shape and contacts.shape == (dof.shape[0], num_bodies)

    hand_contacts = medit_lib.compute_motion_terrain_hand_contacts(dof, terrain, humanoid)
    assert hand_contacts.shape == (dof.shape[0], num_bodies)
    other = [b for b in range(num_bodies) if b not in hand_ids]
    assert hand_contacts[:, other].sum() == 0

    assert medit_lib.correct_foot_ground_pen(dof, humanoid).shape == dof.shape
