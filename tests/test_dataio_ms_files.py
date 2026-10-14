"""MS file IO: body constraint round trip, save_ms_file defaults, format errors,
MotionLib's zero-contact fallback."""
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.file_io as file_io
import parc.util.file_io_helper as file_io_helper
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util

REPO_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_XML = REPO_ROOT / "data" / "assets" / "humanoid.xml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LEGACY_MOTION = FIXTURES / "dataio_legacy_humanoid.pkl"
BUNDLED_MS_FILES = sorted((REPO_ROOT / "data" / "motion_terrains").glob("*.pkl"))


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    # MotionLib.from_file resolves paths through path_loader, which needs a DATA_DIR
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))


def _load_char(char_file=HUMANOID_XML):
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(char_file))
    return char_model


def _motion_frames(char_model, num_frames=6, contacts=True):
    num_joints = char_model.get_num_joints()
    root_pos = torch.zeros(num_frames, 3)
    root_pos[:, 0] = torch.linspace(0.0, 0.5, num_frames)
    root_pos[:, 2] = 0.9
    joint_dof = 0.1 * torch.sin(torch.arange(num_frames * char_model.get_dof_size(), dtype=torch.float32))
    joint_rot = char_model.dof_to_rot(joint_dof.reshape(num_frames, -1))
    contact_vals = None
    if contacts:
        contact_vals = (torch.arange(num_frames * char_model.get_num_contact_bodies()) % 3 == 0)
        contact_vals = contact_vals.to(torch.float32).reshape(num_frames, -1)
    return motion_util.MotionFrames(root_pos=root_pos,
                                    root_rot=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_frames, 1),
                                    joint_rot=joint_rot,
                                    contacts=contact_vals)


def _body_constraints(num_bodies):
    constraints = [[] for _ in range(num_bodies)]
    constraints[3] = [moopt.BodyConstraint(start_frame_idx=0, end_frame_idx=2,
                                           constraint_point=torch.tensor([0.1, -0.2, 0.0]))]
    constraints[num_bodies - 1] = [
        moopt.BodyConstraint(start_frame_idx=1, end_frame_idx=1, constraint_point=torch.tensor([0.5, 0.25, 0.75])),
        moopt.BodyConstraint(start_frame_idx=3, end_frame_idx=5, constraint_point=torch.tensor([-1.0, 2.0, 0.125])),
    ]
    return constraints


def _flatten_constraints(constraints):
    rows = []
    for body_id, body_constraints in enumerate(constraints):
        for c in body_constraints:
            point = c["constraint_point"] if isinstance(c, dict) else c.constraint_point
            start = c["start_frame_idx"] if isinstance(c, dict) else c.start_frame_idx
            end = c["end_frame_idx"] if isinstance(c, dict) else c.end_frame_idx
            if isinstance(point, torch.Tensor):
                point = point.cpu().numpy()
            rows.append((body_id, start, end, tuple(np.asarray(point, dtype=np.float64).tolist())))
    return rows


def test_body_constraint_fields():
    point = torch.tensor([1.0, 2.0, 3.0])
    constraint = moopt.BodyConstraint(start_frame_idx=2, end_frame_idx=5, constraint_point=point)
    fields = asdict(constraint)  # was {} before the fields were annotated
    assert sorted(fields) == ["constraint_point", "end_frame_idx", "start_frame_idx"]
    assert (fields["start_frame_idx"], fields["end_frame_idx"]) == (2, 5)
    assert torch.equal(fields["constraint_point"], point)

    default = moopt.BodyConstraint()  # creators fill the attributes in afterwards
    assert (default.start_frame_idx, default.end_frame_idx, default.constraint_point) == (0, 0, None)

    # identity semantics (no field-wise __eq__ comparing tensors), hashable like develop's class
    twin = moopt.BodyConstraint(start_frame_idx=2, end_frame_idx=5, constraint_point=point.clone())
    assert constraint == constraint and constraint != twin
    assert len({constraint, twin}) == 2


def test_body_constraints_round_trip(tmp_path):
    char_model = _load_char()
    frames = _motion_frames(char_model)
    constraints = _body_constraints(char_model.get_num_bodies())
    terrain = terrain_util.SubTerrain(x_dim=8, y_dim=6, dx=0.1, dy=0.1, min_x=-0.2, min_y=-0.3, device="cpu")
    path = tmp_path / "motion.pkl"

    file_io_helper.save_ms_file(str(path), frames, 30, "CLAMP", terrain, body_constraints=constraints)

    # on disk: portable dicts of numpy values
    raw = file_io.load_ms_file(str(path)).misc_data[file_io_helper.BODY_CONSTRAINTS_KEY]
    assert all(isinstance(c, dict) and isinstance(c["constraint_point"], np.ndarray) for b in raw for c in b)
    assert _flatten_constraints(raw) == _flatten_constraints(constraints)

    # through the helper: BodyConstraint objects with float32 tensors on the device
    loaded = file_io_helper.load_ms_file(str(path), device="cpu").misc_data[file_io_helper.BODY_CONSTRAINTS_KEY]
    assert len(loaded) == char_model.get_num_bodies()
    for body_constraints in loaded:
        for c in body_constraints:
            assert isinstance(c, moopt.BodyConstraint)
            assert c.constraint_point.dtype == torch.float32 and c.constraint_point.device.type == "cpu"
    assert _flatten_constraints(loaded) == _flatten_constraints(constraints)


def test_torchify_body_constraints():
    as_dict = {"start_frame_idx": 1, "end_frame_idx": 2, "constraint_point": np.array([1.0, 2.0, 3.0])}
    as_numpy_obj = moopt.BodyConstraint(3, 4, np.array([4.0, 5.0, 6.0], dtype=np.float32))
    as_tensor_obj = moopt.BodyConstraint(5, 6, torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

    out = file_io_helper.torchify_body_constraints([[as_dict], [], [as_numpy_obj, as_tensor_obj]], "cpu")

    assert [len(b) for b in out] == [1, 0, 2]
    for c, expected in zip([out[0][0], out[2][0], out[2][1]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]):
        assert isinstance(c, moopt.BodyConstraint)
        assert c.constraint_point.dtype == torch.float32
        assert c.constraint_point.tolist() == expected
    assert out[2][0] is as_numpy_obj  # objects are updated in place and returned


def test_torchify_drops_constraints_without_point():
    # files saved while BodyConstraint had no annotated fields hold {} per constraint
    with pytest.warns(UserWarning, match="without a constraint point"):
        out = file_io_helper.torchify_body_constraints([[{}], [{}, {"start_frame_idx": 1, "end_frame_idx": 1,
                                                                   "constraint_point": np.zeros(3)}]], "cpu")
    assert [len(b) for b in out] == [0, 1]


def test_numpify_body_constraints_accepts_dicts():
    constraints = [[{"start_frame_idx": 0, "end_frame_idx": 1, "constraint_point": torch.ones(3)}],
                   [moopt.BodyConstraint(2, 3, torch.zeros(3, requires_grad=True))]]
    out = file_io_helper.numpify_body_constraints(constraints)
    assert all(isinstance(c["constraint_point"], np.ndarray) for b in out for c in b)


def test_save_ms_file_optional_arguments(tmp_path):
    char_model = _load_char()

    # only the required motion arguments: used to crash on the typing-object defaults
    path = tmp_path / "plain.pkl"
    file_io_helper.save_ms_file(str(path), _motion_frames(char_model, contacts=False), 30, "WRAP", None)
    file_data = file_io.load_ms_file(str(path))
    assert file_data.misc_data == {}
    assert file_data.terrain_data is None
    assert file_data.motion_data.body_contacts is None
    assert (file_data.motion_data.fps, file_data.motion_data.loop_mode) == (30, "WRAP")

    # terrain-only file, the way parc_2_kin_gen.py saves with only_gen
    terrain_path = tmp_path / "terrain_only.pkl"
    terrain = terrain_util.SubTerrain(x_dim=4, y_dim=3, dx=0.1, dy=0.1, device="cpu")
    path_nodes = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]])
    file_io_helper.save_ms_file(str(terrain_path), terrain=terrain, path_nodes=path_nodes)
    file_data = file_io_helper.load_ms_file(str(terrain_path))
    assert file_data.motion_data is None
    assert tuple(file_data.terrain_data.hf.shape) == (4, 3)
    assert torch.equal(file_data.misc_data[file_io_helper.PATH_NODES_KEY], path_nodes)

    with pytest.raises(ValueError, match="fps and loop_mode"):
        file_io_helper.save_ms_file(str(path), _motion_frames(char_model))


def test_load_ms_file_rejects_legacy_file():
    for load in (file_io.load_ms_file, file_io_helper.load_ms_file):
        with pytest.raises(file_io.MSFileFormatError, match="legacy motion format") as exc_info:
            load(str(LEGACY_MOTION))
        assert "scripts/convert_legacy_motions.py" in str(exc_info.value)

    # MotionLib used to fail later with an AttributeError on motion_data=None
    with pytest.raises(file_io.MSFileFormatError, match="legacy motion format"):
        motion_lib.MotionLib.from_file(str(LEGACY_MOTION), _load_char(), "cpu", contact_info=True)


def test_load_ms_file_rejects_other_pickles(tmp_path):
    path = tmp_path / "list.pkl"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(file_io.MSFileFormatError, match="not an MS motion file"):
        file_io.load_ms_file(str(path))


def test_load_ms_file_raises_on_broken_section(tmp_path):
    path = tmp_path / "broken.pkl"
    container = {file_io.MOTION_DATA_KEY: b"not a pickle", file_io.TERRAIN_DATA_KEY: None,
                 file_io.MISC_DATA_KEY: pickle.dumps({})}
    path.write_bytes(pickle.dumps(container))
    with pytest.raises(file_io.MSFileFormatError, match="'motion_data' section"):
        file_io.load_ms_file(str(path))


@pytest.mark.parametrize("path", BUNDLED_MS_FILES, ids=lambda p: p.name)
def test_bundled_ms_files_load(path):
    file_data = file_io_helper.load_ms_file(str(path))
    assert file_data.motion_data.root_pos.shape[-1] == 3
    assert file_data.terrain_data is not None


def test_motion_lib_rejects_terrain_only_file(tmp_path):
    path = tmp_path / "terrain_only.pkl"
    file_io_helper.save_ms_file(str(path), terrain=terrain_util.SubTerrain(x_dim=2, y_dim=2, device="cpu"))
    with pytest.raises(ValueError, match="no motion data"):
        motion_lib.MotionLib.from_file(str(path), _load_char(), "cpu", contact_info=True)


def test_motion_lib_from_frames_matches_from_file(tmp_path):
    char_model = _load_char()
    frames = _motion_frames(char_model, num_frames=8)
    path = tmp_path / "motion.pkl"
    file_io_helper.save_ms_file(str(path), frames, 30, "CLAMP", None)

    from_file = motion_lib.MotionLib.from_file(str(path), char_model, "cpu", contact_info=True)
    from_frames = motion_lib.MotionLib.from_frames(frames, char_model, "cpu", motion_lib.LoopMode.CLAMP, 30,
                                                   contact_info=True)

    assert torch.abs(from_file._frame_dof_vel).max() > 0.1
    for name in ["_frame_root_pos", "_frame_root_vel", "_frame_joint_rot", "_frame_dof_vel", "_frame_contacts"]:
        torch.testing.assert_close(getattr(from_frames, name), getattr(from_file, name), atol=1e-4, rtol=1e-4,
                                   msg=name)


def test_motion_lib_zero_contact_fallback_uses_contact_bodies(tmp_path):
    # a character whose contact bodies are a subset of its bodies
    xml = HUMANOID_XML.read_text().replace(
        "</mujoco>",
        '<custom_contact_info><body name="left_foot"/><body name="right_foot"/>'
        '<body name="right_hand"/></custom_contact_info>\n</mujoco>')
    char_file = tmp_path / "humanoid_contact_subset.xml"
    char_file.write_text(xml)
    char_model = _load_char(char_file)
    assert char_model.get_num_contact_bodies() == 3 < char_model.get_num_joints()

    num_frames = 6
    frames = _motion_frames(char_model, num_frames=num_frames, contacts=False)
    path = tmp_path / "no_contacts.pkl"
    file_io_helper.save_ms_file(str(path), frames, 30, "CLAMP", None)

    mlib = motion_lib.MotionLib.from_file(str(path), char_model, "cpu", contact_info=True)
    assert tuple(mlib._frame_contacts.shape) == (num_frames, 3)

    batched = motion_util.MotionFrames(root_pos=frames.root_pos.unsqueeze(0).repeat(2, 1, 1),
                                       root_rot=frames.root_rot.unsqueeze(0).repeat(2, 1, 1),
                                       joint_rot=frames.joint_rot.unsqueeze(0).repeat(2, 1, 1, 1))
    mlib = motion_lib.MotionLib.from_frames(batched, char_model, "cpu", motion_lib.LoopMode.CLAMP, 30,
                                            contact_info=True)
    assert tuple(mlib._frame_contacts.shape) == (2 * num_frames, 3)
