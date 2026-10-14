"""Legacy (develop-era) motion files -> MS files with scripts/convert_legacy_motions.py.

The fixtures were written by develop's own code (tests/fixtures/dataio_make_legacy_fixtures.py,
run in the develop checkout): dataio_legacy_humanoid.pkl is laid out like the
output of develop's robot_motion_preprocess.py and holds develop-era classes
(util.terrain_util.SubTerrain, tools.motion_opt.motion_optimization.BodyConstraint)
and GPU tensors; dataio_legacy_humanoid_ref.npz holds what develop's MotionLib
read from it.
"""
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.file_io as file_io
import parc.util.file_io_helper as file_io_helper
import parc.util.pickle_compat as pickle_compat
import scripts.convert_legacy_motions as convert

REPO_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_XML = REPO_ROOT / "data" / "assets" / "humanoid.xml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LEGACY_MOTION = FIXTURES / "dataio_legacy_humanoid.pkl"
LEGACY_TERRAIN_ONLY = FIXTURES / "dataio_legacy_terrain_only.pkl"
REFERENCE = FIXTURES / "dataio_legacy_humanoid_ref.npz"
MS_SAMPLE = REPO_ROOT / "data" / "motion_terrains" / "sfu.pkl"


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Run the converter CLI with CUDA hidden over a directory of mixed inputs."""
    root = tmp_path_factory.mktemp("convert")
    in_dir, out_dir = root / "in", root / "out"
    (in_dir / "terrains").mkdir(parents=True)
    shutil.copy(str(LEGACY_MOTION), str(in_dir / LEGACY_MOTION.name))
    shutil.copy(str(LEGACY_TERRAIN_ONLY), str(in_dir / "terrains" / LEGACY_TERRAIN_ONLY.name))
    shutil.copy(str(MS_SAMPLE), str(in_dir / MS_SAMPLE.name))
    (in_dir / "not_a_motion.pkl").write_bytes(pickle.dumps([1, 2, 3]))

    # the legacy fixture holds tensors pickled on cuda:0; they must load without CUDA
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PARC_DATA_DIR=str(root), WANDB_MODE="disabled")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.convert_legacy_motions", "--inputs", str(in_dir),
         "--outputs", str(out_dir), "--char-file", str(HUMANOID_XML)],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    return {"root": root, "out": out_dir, "log": result.stdout}


@pytest.fixture
def char_model():
    model = kin_char_model.KinCharModel("cpu")
    model.load_char_file(str(HUMANOID_XML))
    return model


@pytest.fixture
def reference():
    return np.load(str(REFERENCE))


def test_fixture_is_a_develop_era_gpu_pickle():
    data = LEGACY_MOTION.read_bytes()
    assert b"util.terrain_util" in data and b"tools.motion_opt.motion_optimization" in data
    assert b"parc." not in data
    assert b"cuda" in data  # tensor storages pickled on the GPU


def test_cli_summary(converted):
    log, out = converted["log"], converted["out"]
    assert "Total converted files: 2, copied MS files: 1, skipped non-motion files: 1, failed: 0" in log
    assert (out / LEGACY_MOTION.name).is_file()
    assert (out / "terrains" / LEGACY_TERRAIN_ONLY.name).is_file()
    assert (out / MS_SAMPLE.name).read_bytes() == MS_SAMPLE.read_bytes()
    assert not (out / "not_a_motion.pkl").exists()


def test_converted_motion_matches_develop(converted, char_model, reference, monkeypatch):
    monkeypatch.setenv("PARC_DATA_DIR", str(converted["root"]))
    mlib = motion_lib.MotionLib.from_file(str(converted["out"] / LEGACY_MOTION.name), char_model, "cpu",
                                          contact_info=True)

    np.testing.assert_array_equal(mlib._motion_num_frames.numpy(), reference["num_frames"])
    np.testing.assert_array_equal(mlib._motion_fps.numpy(), reference["fps"])
    np.testing.assert_array_equal(mlib._motion_loop_modes.numpy(), reference["loop_modes"])

    assert np.abs(mlib._frame_root_pos.numpy() - reference["root_pos"]).max() < 1e-6
    root_rot, ref_root_rot = mlib._frame_root_rot.numpy(), reference["root_rot"]
    root_rot_err = np.minimum(np.abs(root_rot - ref_root_rot).max(-1), np.abs(root_rot + ref_root_rot).max(-1))
    assert root_rot_err.max() < 1e-6  # q and -q are the same rotation

    dof = mlib.joint_rot_to_dof(mlib._frame_joint_rot).numpy()
    assert np.abs(dof - reference["dof"]).max() < 1e-4  # develop's joint_rot_to_dof
    assert np.abs(dof - reference["frames"][:, 6:]).max() < 1e-4  # the DOFs stored in the legacy file

    np.testing.assert_array_equal(mlib._frame_contacts.numpy(), reference["contacts"])

    terrain = mlib._terrains[0]
    np.testing.assert_array_equal(terrain.hf.numpy(), reference["hf"])
    np.testing.assert_array_equal(terrain.hf_maxmin.numpy(), reference["hf_maxmin"])
    np.testing.assert_array_equal(terrain.min_point.numpy(), reference["min_point"])
    np.testing.assert_array_equal(terrain.dxdy.numpy(), reference["dxdy"])

    hf_mask_inds = mlib._hf_mask_inds[0]
    np.testing.assert_array_equal([x.shape[0] for x in hf_mask_inds], reference["hf_mask_inds_lengths"])
    np.testing.assert_array_equal(torch.cat(hf_mask_inds).numpy(), reference["hf_mask_inds"])


def test_converted_body_constraints_survive(converted, reference):
    path = str(converted["out"] / LEGACY_MOTION.name)

    raw = file_io.load_ms_file(path).misc_data[file_io_helper.BODY_CONSTRAINTS_KEY]
    assert all(isinstance(c, dict) and isinstance(c["constraint_point"], np.ndarray) for b in raw for c in b)

    constraints = file_io_helper.load_ms_file(path).misc_data[file_io_helper.BODY_CONSTRAINTS_KEY]
    rows = []
    for body_id, body_constraints in enumerate(constraints):
        for c in body_constraints:
            assert isinstance(c, moopt.BodyConstraint)
            rows.append([body_id, c.start_frame_idx, c.end_frame_idx] + c.constraint_point.double().tolist())
    assert len(rows) > 0
    np.testing.assert_array_equal(np.array(rows), reference["body_constraints"])


def test_converted_terrain_only_file(converted):
    legacy = pickle_compat.load(LEGACY_TERRAIN_ONLY)
    file_data = file_io_helper.load_ms_file(str(converted["out"] / "terrains" / LEGACY_TERRAIN_ONLY.name))

    assert file_data.motion_data is None
    np.testing.assert_array_equal(file_data.terrain_data.hf.numpy(), legacy["terrain"].hf.numpy())
    np.testing.assert_array_equal(file_data.terrain_data.hf_maxmin.numpy(), legacy["terrain"].hf_maxmin.numpy())
    np.testing.assert_array_equal(file_data.terrain_data.min_point, legacy["terrain"].min_point.numpy())
    assert np.float32(file_data.terrain_data.dx) == legacy["terrain"].dxdy[0].item()
    assert torch.equal(file_data.misc_data[file_io_helper.PATH_NODES_KEY], legacy["path_nodes"])


def test_defaults_and_old_numpy_terrain(char_model):
    raw = pickle_compat.load(LEGACY_MOTION)
    del raw["fps"], raw["loop_mode"]
    # a terrain saved as numpy before hf_maxmin existed
    old_terrain = raw["terrain"].numpy_copy()
    del old_terrain.hf_maxmin
    raw["terrain"] = old_terrain

    ms_file = convert.legacy_to_ms_file(raw, char_model)

    assert ms_file.motion_data.fps == 30 and isinstance(ms_file.motion_data.fps, int)
    assert ms_file.motion_data.loop_mode == "CLAMP"
    np.testing.assert_array_equal(ms_file.terrain_data.hf, old_terrain.hf)
    assert (ms_file.terrain_data.hf_maxmin[..., 0] == 1.0).all()
    assert (ms_file.terrain_data.hf_maxmin[..., 1] == -1.0).all()


def test_frames_with_extra_motion_dim(char_model):
    raw = pickle_compat.load(LEGACY_MOTION)
    expected = convert.legacy_to_ms_file(pickle_compat.load(LEGACY_MOTION), char_model).motion_data
    raw["frames"] = raw["frames"][None]
    raw["contacts"] = torch.from_numpy(raw["contacts"][None])
    motion_data = convert.legacy_to_ms_file(raw, char_model).motion_data
    np.testing.assert_array_equal(motion_data.joint_rot, expected.joint_rot)
    np.testing.assert_array_equal(motion_data.body_contacts, expected.body_contacts)


def test_wrong_char_file_is_rejected(char_model):
    raw = pickle_compat.load(LEGACY_MOTION)
    raw["frames"] = raw["frames"][:, :-1]
    with pytest.raises(ValueError, match="wrong --char-file"):
        convert.legacy_to_ms_file(raw, char_model)


def test_non_motion_dict_is_rejected(char_model):
    with pytest.raises(convert.NotAMotionFileError):
        convert.legacy_to_ms_file({"fps": 30}, char_model)


@pytest.mark.parametrize("value, expected", [
    (None, 30), (30, 30), (30.0, 30), (np.int64(60), 60), (np.array(30.0), 30), (torch.tensor(50.0), 50),
    (29.97, 29.97),
])
def test_normalize_fps(value, expected):
    fps = convert.normalize_fps(value)
    assert fps == expected and type(fps) is type(expected)


@pytest.mark.parametrize("value, expected", [
    (None, "CLAMP"), ("WRAP", "WRAP"), (motion_lib.LoopMode.WRAP, "WRAP"), (0, "CLAMP"),
])
def test_normalize_loop_mode(value, expected):
    assert convert.normalize_loop_mode(value) == expected


@pytest.mark.parametrize("bad_fps", [0, -30])
def test_normalize_fps_rejects_invalid(bad_fps):
    with pytest.raises(ValueError):
        convert.normalize_fps(bad_fps)


def test_normalize_loop_mode_rejects_invalid():
    with pytest.raises(ValueError):
        convert.normalize_loop_mode("BOUNCE")
