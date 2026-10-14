"""scripts/tools/view_terrain.py: terrain loading from MS files and legacy pickles (no GUI)."""
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import parc.util.file_io_helper as file_io_helper
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util
from scripts.tools import obj_to_heightfield, view_terrain

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _sub_terrain():
    terrain = terrain_util.SubTerrain(x_dim=7, y_dim=5, dx=0.2, dy=0.2, min_x=1.5, min_y=-0.6, device="cpu")
    terrain.hf[2:4, 1:3] = 0.75
    return terrain


@pytest.mark.parametrize("make_file, expected", [
    # in-repo MS motion file
    (lambda tmp: REPO_ROOT / "data" / "motion_terrains" / "sfu.pkl", ((32, 32), 0.0, 0.0, 0.4)),
    # develop-era terrain-only file ({"terrain", "path_nodes"}, tensors pickled on cuda:0, loaded on the CPU)
    (lambda tmp: FIXTURES / "dataio_legacy_terrain_only.pkl", ((6, 5), -0.4, -0.5, 0.2)),
    # develop's robot_motion_preprocess output (legacy motion dict with a terrain)
    (lambda tmp: FIXTURES / "datatools_rmp" / "develop_out" / "g1" / "walk_to_unitree_g1.pkl",
     ((40, 26), -1.03, -1.27, 0.1)),
], ids=["ms", "legacy_terrain_only", "legacy_motion"])
def test_extract_terrain_data(make_file, expected, tmp_path):
    hf, min_x, min_y, dx = view_terrain.extract_terrain_data(str(make_file(tmp_path)))
    shape, exp_min_x, exp_min_y, exp_dx = expected
    assert isinstance(hf, np.ndarray) and hf.shape == shape
    assert (min_x, min_y, dx) == pytest.approx((exp_min_x, exp_min_y, exp_dx), abs=1e-6)


def test_ms_terrain_only_file(tmp_path):
    path = tmp_path / "terrain_ms.pkl"
    file_io_helper.save_ms_file(str(path), terrain=_sub_terrain())
    hf, min_x, min_y, dx = view_terrain.extract_terrain_data(str(path))
    np.testing.assert_array_equal(hf, _sub_terrain().hf.numpy())
    assert (min_x, min_y, dx) == pytest.approx((1.5, -0.6, 0.2), abs=1e-6)


def test_ms_file_without_terrain(tmp_path):
    motion_frames = motion_util.MotionFrames(root_pos=torch.zeros(3, 3), root_rot=torch.zeros(3, 4),
                                             joint_rot=torch.zeros(3, 2, 4))
    path = tmp_path / "motion_only.pkl"
    file_io_helper.save_ms_file(str(path), motion_frames=motion_frames, fps=30, loop_mode="CLAMP")
    assert view_terrain.extract_terrain_data(str(path)) == (None, None, None, None)
    assert view_terrain.main([str(path)]) == 1


def test_obj_to_heightfield_output(tmp_path):
    obj_path = tmp_path / "step.obj"
    obj_path.write_text("v 0 0 0\nv 2 0 0\nv 2 1 0\nv 0 1 0\nv 1 0 0.3\nv 2 0 0.3\nv 2 1 0.3\nv 1 1 0.3\n"
                        "f 1 2 3\nf 1 3 4\nf 5 6 7\nf 5 7 8\n")
    out = tmp_path / "step.pkl"
    assert obj_to_heightfield.main([str(obj_path), str(out), "--resolution", "0.25"]) == 0
    hf, min_x, min_y, dx = view_terrain.extract_terrain_data(str(out))
    assert hf.shape == (9, 5) and (min_x, min_y, dx) == pytest.approx((0.0, 0.0, 0.25))
    np.testing.assert_allclose(hf[5:, :], 0.3, atol=1e-5)  # x >= 1.25 is on the step
    np.testing.assert_allclose(hf[:4, :], 0.0, atol=1e-5)


def test_plain_dict(tmp_path):
    path = tmp_path / "plain.pkl"
    with open(str(path), "wb") as f:
        pickle.dump({"hf": np.ones((3, 4), dtype=np.float32), "dxdy": np.array([0.2, 0.2]),
                     "min_point": np.array([1.0, 2.0])}, f)
    hf, min_x, min_y, dx = view_terrain.extract_terrain_data(str(path))
    assert hf.shape == (3, 4) and (min_x, min_y, dx) == pytest.approx((1.0, 2.0, 0.2))


def test_missing_file(tmp_path):
    assert view_terrain.main([str(tmp_path / "missing.pkl")]) == 1


def test_2d_view_headless(tmp_path):
    path = tmp_path / "terrain_ms.pkl"
    file_io_helper.save_ms_file(str(path), terrain=_sub_terrain())
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "tools" / "view_terrain.py"), str(path),
                             "--vis_type", "2d"], cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    assert "Terrain: hf (7, 5), dx 0.2, min point (1.500, -0.600), heights [0.000, 0.750]" in result.stdout
