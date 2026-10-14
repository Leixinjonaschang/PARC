"""scripts/tools/pkl_preview.py and pkl_preview_example.py on MS and legacy files."""
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.file_io as file_io
import parc.util.file_io_helper as file_io_helper
import parc.util.motion_util as motion_util
import parc.util.pickle_compat as pickle_compat
import parc.util.terrain_util as terrain_util
from scripts.tools import pkl_preview, pkl_preview_example

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "scripts" / "tools"
G1_XML = REPO_ROOT / "data" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "datatools_rmp"
GMR_WALK = FIXTURES / "gmr" / "walk_to_unitree_g1.pkl"
# develop's robot_motion_preprocess output: pickles util.terrain_util.SubTerrain and
# tools.motion_opt.motion_optimization.BodyConstraint
LEGACY_G1 = FIXTURES / "develop_out" / "g1" / "walk_to_unitree_g1.pkl"


@pytest.fixture(scope="module")
def g1():
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(G1_XML))
    return char_model


@pytest.fixture(scope="module")
def g1_frames():
    return torch.as_tensor(np.asarray(pickle_compat.load(GMR_WALK)["frames"], dtype=np.float32))


@pytest.fixture(scope="module")
def g1_ms_file(tmp_path_factory, g1, g1_frames):
    """A synthetic G1 MS file: 40 frames, contacts, a terrain, hf_mask_inds and a body constraint."""
    num_frames = g1_frames.shape[0]
    contacts = torch.zeros(size=(num_frames, g1.get_num_bodies()))
    contacts[:10, g1.get_body_id("left_foot")] = 1.0
    motion_frames = motion_util.motion_frames_from_mlib_format(g1_frames, g1, contacts)
    terrain = terrain_util.SubTerrain(x_dim=12, y_dim=8, dx=0.1, dy=0.1, min_x=-0.5, min_y=-0.4, device="cpu")
    hf_mask_inds = [torch.tensor([[3, 4], [3, 5]], dtype=torch.int64) for _ in range(num_frames)]
    body_constraints = [[] for _ in range(g1.get_num_bodies())]
    constraint = moopt.BodyConstraint(start_frame_idx=0, end_frame_idx=9, constraint_point=torch.tensor([0.0, 0.1, 0.0]))
    body_constraints[g1.get_body_id("left_foot")] = [constraint]
    path = tmp_path_factory.mktemp("preview") / "g1_motion.pkl"
    file_io_helper.save_ms_file(str(path), motion_frames=motion_frames, fps=30, loop_mode="CLAMP", terrain=terrain,
                                hf_mask_inds=hf_mask_inds, body_constraints=body_constraints)
    return path


def _run(tool, *args):
    return subprocess.run([sys.executable, str(TOOLS / tool)] + [str(a) for a in args], cwd=str(REPO_ROOT),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def _row(output, key):
    rows = [line for line in output.splitlines() if line.startswith(key + " ")]
    assert len(rows) == 1, output
    return rows[0]


def test_preview_ms_file(g1_ms_file):
    result = _run("pkl_preview.py", g1_ms_file, "--char-file", G1_XML)
    assert result.returncode == 0, result.stdout
    out = result.stdout
    assert "Format: MS motion file" in out
    assert "Shape: (40, 35, 4)" in _row(out, "joint_rot")
    assert "Shape: (40, 4)" in _row(out, "root_rot")
    assert "Shape: (40, 36)" in _row(out, "body_contacts")
    assert "Shape: (40, 29)" in _row(out, "joint_dof (rot_to_dof)")
    assert "Value: 30" in _row(out, "fps")
    assert "Shape: (12, 8)" in _row(out, "hf")
    assert "Length: 40" in _row(out, "hf_mask_inds")
    assert "1 items in total" in _row(out, "opt:body_constraints")


def test_preview_legacy_file_needs_pickle_compat():
    # develop-era module paths: a plain unpickler cannot load the file
    with pytest.raises(ModuleNotFoundError):
        with open(str(LEGACY_G1), "rb") as f:
            pickle.load(f)

    result = _run("pkl_preview.py", LEGACY_G1)
    assert result.returncode == 0, result.stdout
    out = result.stdout
    assert "Format: legacy pickle" in out
    assert "Shape: (40, 35)" in _row(out, "frames")
    assert "Shape: (40, 36)" in _row(out, "contacts")
    terrain_row = _row(out, "terrain")
    assert "SubTerrain" in terrain_row and "hf (40, 26)" in terrain_row and "dx 0.1" in terrain_row
    assert "Length: 36, Item Type: list, 6 items in total" in _row(out, "opt:body_constraints")


def test_preview_missing_file(tmp_path):
    result = _run("pkl_preview.py", tmp_path / "missing.pkl")
    assert result.returncode == 1
    assert "File not found" in result.stdout


def test_frame_view_ms_file(g1_ms_file, g1, g1_frames, capsys):
    assert pkl_preview_example.main([str(g1_ms_file), "5", "--char-file", str(G1_XML)]) == 0
    out = capsys.readouterr().out
    assert "frame 5 of 40" in out
    assert "Joint Rotations: 35 quaternions" in out
    assert "Joint DOFs (29 dims, rot_to_dof)" in out
    assert "Contact States (36 bodies)" in out
    assert "Occupied Grid Indices (Count: 2)" in out
    left_foot = g1.get_body_id("left_foot")
    assert "Constraints active at this frame (body id, start, end): [({}, 0, 9)]".format(left_foot) in out

    # the DOFs shown are the DOFs the motion was made from
    ms_data = file_io.load_ms_file(str(g1_ms_file))
    dofs = pkl_preview.joint_dofs(ms_data.motion_data.joint_rot[5:6], g1)
    np.testing.assert_allclose(dofs[0], g1_frames[5, 6:].numpy(), rtol=0, atol=1e-5)


def test_frame_view_without_char_file(g1_ms_file, capsys):
    assert pkl_preview_example.main([str(g1_ms_file), "-1"]) == 0
    out = capsys.readouterr().out
    assert "frame 39 of 40" in out
    assert "pass --char-file" in out


def test_frame_view_legacy_file(capsys):
    legacy = pickle_compat.load(LEGACY_G1)
    assert pkl_preview_example.main([str(LEGACY_G1), "-1"]) == 0
    out = capsys.readouterr().out
    assert "Root Position (x, y, z): {}".format(pkl_preview_example.format_array(legacy["frames"][-1][0:3])) in out
    assert "Joint DOFs (29 dims)" in out
    assert "Contact States (36 bodies)" in out


def test_frame_view_out_of_bounds(g1_ms_file, capsys):
    assert pkl_preview_example.main([str(g1_ms_file), "40"]) == 1
    assert "out of bounds" in capsys.readouterr().out


def test_no_global_numpy_core_alias():
    # develop's tools aliased numpy._core to numpy.core in sys.modules (doc/porting_plan.md DB6)
    code = ("import sys; import scripts.tools.pkl_preview, scripts.tools.pkl_preview_example; "
            "import numpy; print(sys.modules.get('numpy._core') is numpy.core)")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    assert result.stdout.strip().splitlines()[-1] == "False"
