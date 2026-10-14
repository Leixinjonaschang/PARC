"""scripts/tools/pkl_extract_frames_fps.py and extract_motion.sh: PARC motion files -> {frames, fps}."""
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
import parc.util.file_io_helper as file_io_helper
import parc.util.motion_util as motion_util
import parc.util.pickle_compat as pickle_compat
import scripts.convert_legacy_motions as convert
from scripts.tools import pkl_extract_frames_fps as extract

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "scripts" / "tools"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
G1_XML = REPO_ROOT / "data" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
HUMANOID_XML = REPO_ROOT / "data" / "assets" / "humanoid.xml"
# develop-format humanoid motion with develop-era classes and tensors pickled on cuda:0
LEGACY_HUMANOID = FIXTURES / "dataio_legacy_humanoid.pkl"
LEGACY_TERRAIN_ONLY = FIXTURES / "dataio_legacy_terrain_only.pkl"
GMR_WALK = FIXTURES / "datatools_rmp" / "gmr" / "walk_to_unitree_g1.pkl"


def _load_char(path):
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(path))
    return char_model


def _read(path):
    with open(str(path), "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def legacy_frames():
    return np.asarray(pickle_compat.load(LEGACY_HUMANOID)["frames"], dtype=np.float32)


def test_ms_round_trip_humanoid(tmp_path, legacy_frames):
    # legacy frames -> MS file (convert_legacy_motions) -> frames again
    ms_file = tmp_path / "humanoid_ms.pkl"
    assert convert.convert_motion_file(LEGACY_HUMANOID, ms_file, char_model=_load_char(HUMANOID_XML)) == convert.CONVERTED
    out = tmp_path / "frames.pkl"
    assert extract.main(["--input", str(ms_file), "--output", str(out), "--char-file", str(HUMANOID_XML)]) == 0

    data = _read(out)
    assert sorted(data.keys()) == ["fps", "frames"]
    assert data["fps"] == 30
    assert data["frames"].dtype == np.float32 and data["frames"].shape == legacy_frames.shape
    np.testing.assert_array_equal(data["frames"][:, 0:3], legacy_frames[:, 0:3])
    np.testing.assert_allclose(data["frames"][:, 3:], legacy_frames[:, 3:], rtol=0, atol=1e-5)


def test_ms_round_trip_g1(tmp_path):
    g1 = _load_char(G1_XML)
    frames = torch.as_tensor(np.asarray(pickle_compat.load(GMR_WALK)["frames"], dtype=np.float32))
    ms_file = tmp_path / "g1_ms.pkl"
    file_io_helper.save_ms_file(str(ms_file), motion_frames=motion_util.motion_frames_from_mlib_format(frames, g1),
                                fps=30, loop_mode="CLAMP")
    out = tmp_path / "frames.pkl"
    assert extract.main(["--input", str(ms_file), "--output", str(out), "--char-file", str(G1_XML)]) == 0
    np.testing.assert_allclose(_read(out)["frames"], frames.numpy(), rtol=0, atol=1e-5)


def test_ms_file_with_wrong_char_file_fails(tmp_path):
    ms_file = tmp_path / "humanoid_ms.pkl"
    convert.convert_motion_file(LEGACY_HUMANOID, ms_file, char_model=_load_char(HUMANOID_XML))
    out = tmp_path / "frames.pkl"
    assert extract.main(["--input", str(ms_file), "--output", str(out), "--char-file", str(G1_XML)]) == 1
    assert not out.exists()


def test_legacy_file_without_parc_or_cuda(tmp_path, legacy_frames):
    # the legacy path must not need PARC: run it from outside the repository and check
    # that no parc module was imported; tensors pickled on cuda:0 load with CUDA hidden
    out = tmp_path / "frames.pkl"
    code = ("import runpy, sys; sys.argv = [sys.argv[1], '--input', sys.argv[2], '--output', sys.argv[3]]; "
            "status = 0\n"
            "try:\n    runpy.run_path(sys.argv[0], run_name='__main__')\n"
            "except SystemExit as exc:\n    status = exc.code\n"
            "print('STATUS', status, 'PARC_IMPORTED', any(m == 'parc' or m.startswith('parc.') for m in sys.modules))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code, str(TOOLS / "pkl_extract_frames_fps.py"),
                             str(LEGACY_HUMANOID), str(out)], cwd=str(tmp_path), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert "STATUS 0 PARC_IMPORTED False" in result.stdout, result.stdout
    data = _read(out)
    np.testing.assert_array_equal(data["frames"], legacy_frames)
    assert data["fps"] == 30


def test_numpy2_pickle(tmp_path):
    # GMR-style input written under NumPy 2 (numpy._core module paths)
    src = tmp_path / "gmr.pkl"
    shutil.copy(str(GMR_WALK), str(src))
    assert b"numpy._core" in src.read_bytes()
    out = tmp_path / "frames.pkl"
    assert extract.main(["--input", str(src), "--output", str(out)]) == 0
    np.testing.assert_array_equal(_read(out)["frames"], pickle_compat.load(GMR_WALK)["frames"])


def test_directory_default_names(tmp_path):
    in_dir = tmp_path / "in"
    (in_dir / "run").mkdir(parents=True)
    shutil.copy(str(LEGACY_HUMANOID), str(in_dir / "run" / "a.pkl"))
    shutil.copy(str(LEGACY_TERRAIN_ONLY), str(in_dir / "terrain.pkl"))  # no frames: skipped
    convert.convert_motion_file(LEGACY_HUMANOID, in_dir / "b.pkl", char_model=_load_char(HUMANOID_XML))

    assert extract.main(["--input", str(in_dir)]) == 0
    assert sorted(p.name for p in in_dir.rglob("*_cleaned.pkl")) == ["a_cleaned.pkl", "b_cleaned.pkl"]
    np.testing.assert_allclose(_read(in_dir / "b_cleaned.pkl")["frames"], _read(in_dir / "run" / "a_cleaned.pkl")["frames"],
                               rtol=0, atol=1e-5)
    # a second run skips the *_cleaned files it wrote
    assert extract.main(["--input", str(in_dir)]) == 0
    assert len(list(in_dir.rglob("*_cleaned*.pkl"))) == 2


def test_extract_motion_sh(tmp_path, legacy_frames):
    in_dir = tmp_path / "in"
    (in_dir / "run").mkdir(parents=True)
    shutil.copy(str(LEGACY_HUMANOID), str(in_dir / "run" / "a.pkl"))
    convert.convert_motion_file(LEGACY_HUMANOID, in_dir / "b.pkl", char_model=_load_char(HUMANOID_XML))
    out_dir = tmp_path / "out"

    # run from another directory: the tool is found next to the script (develop's path was wrong)
    env = dict(os.environ, PYTHON=sys.executable, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(["bash", str(TOOLS / "extract_motion.sh"), str(in_dir), str(out_dir),
                             "--char-file", str(HUMANOID_XML)], cwd=str(tmp_path), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    np.testing.assert_array_equal(_read(out_dir / "run" / "a.pkl")["frames"], legacy_frames)
    np.testing.assert_allclose(_read(out_dir / "b.pkl")["frames"], legacy_frames, rtol=0, atol=1e-5)

    result = subprocess.run(["bash", str(TOOLS / "extract_motion.sh"), str(tmp_path / "missing")],
                            cwd=str(tmp_path), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    assert result.returncode == 1 and "Input directory does not exist" in result.stdout
