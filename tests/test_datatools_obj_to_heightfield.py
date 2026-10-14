"""scripts/tools/obj_to_heightfield.py: OBJ terrain -> legacy terrain dict with a SubTerrain."""
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

import parc.util.pickle_compat as pickle_compat
from scripts.tools import obj_to_heightfield

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "scripts" / "tools" / "obj_to_heightfield.py"

GROUND_MIN = np.array([-1.0, -0.5])
GROUND_MAX = np.array([3.0, 2.5])
BOX_MIN = np.array([1.0, 0.8])
BOX_MAX = np.array([1.6, 1.2])
BOX_HEIGHT = 0.5
DX = 0.1
BORDER = 0.05  # grid nodes fall 5 cm from every box edge


def _write_obj(path):
    """Ground plane and a closed 0.6 x 0.4 x 0.5 m box, as two objects with two materials."""
    ground_v = np.array([[GROUND_MIN[0], GROUND_MIN[1], 0.0], [GROUND_MAX[0], GROUND_MIN[1], 0.0],
                         [GROUND_MAX[0], GROUND_MAX[1], 0.0], [GROUND_MIN[0], GROUND_MAX[1], 0.0]])
    ground_f = np.array([[0, 1, 2], [0, 2, 3]])
    box = trimesh.creation.box(extents=[BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], BOX_HEIGHT])
    box.apply_translation([(BOX_MIN[0] + BOX_MAX[0]) / 2, (BOX_MIN[1] + BOX_MAX[1]) / 2, BOX_HEIGHT / 2])

    lines, offset = [], 1
    for name, verts, faces in [("ground", ground_v, ground_f), ("box", box.vertices, box.faces)]:
        lines.append("o {}".format(name))
        lines += ["v {:.6f} {:.6f} {:.6f}".format(*v) for v in verts]
        lines.append("usemtl {}_material".format(name))
        lines += ["f {} {} {}".format(*(f + offset)) for f in faces]
        offset += len(verts)
    path.write_text("\n".join(lines) + "\n")
    # several materials: trimesh loads this as a Scene of two meshes
    assert isinstance(trimesh.load(str(path)), trimesh.Scene)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("obj")
    obj_path = root / "box.obj"
    _write_obj(obj_path)
    out = root / "terrains" / "box.pkl"
    result = subprocess.run(
        [sys.executable, str(TOOL), str(obj_path), str(out), "--resolution", str(DX), "--border-offset", str(BORDER),
         "--path-nodes", "0,0", "1.3,1.0"],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    return out


def test_pickles_parc_subterrain(converted):
    # parc_2_kin_gen (FILE mode) and mdm_validation_test use plain pickle.load
    with open(str(converted), "rb") as f:
        data = pickle.load(f)
    assert sorted(data.keys()) == ["path_nodes", "terrain"]
    terrain = data["terrain"]
    assert type(terrain).__module__ == "parc.util.terrain_util" and type(terrain).__name__ == "SubTerrain"
    assert isinstance(terrain.hf, np.ndarray)  # saved with numpy fields, as develop did

    compat = pickle_compat.load(converted)
    np.testing.assert_array_equal(compat["terrain"].hf, terrain.hf)


def test_grid_spacing_is_dx(converted):
    terrain = pickle_compat.load(converted)["terrain"]
    terrain.to_torch("cpu")  # as parc_2_kin_gen and mdm_validation_test do
    np.testing.assert_allclose(terrain.min_point.numpy(), GROUND_MIN + BORDER, rtol=0, atol=1e-6)
    np.testing.assert_allclose(terrain.dxdy.numpy(), [DX, DX], rtol=0, atol=1e-7)
    assert tuple(terrain.hf.shape) == (40, 30)

    # every node holds the height at min_point + ij * dx (develop sampled with another spacing)
    ij = torch.stack(torch.meshgrid(torch.arange(40), torch.arange(30), indexing="ij"), dim=-1)
    xy = terrain.get_point(ij).numpy()
    inside = np.all((xy > BOX_MIN) & (xy < BOX_MAX), axis=-1)
    np.testing.assert_allclose(terrain.hf.numpy(), np.where(inside, BOX_HEIGHT, 0.0), rtol=0, atol=1e-5)


def test_box_height_and_edges(converted):
    terrain = pickle_compat.load(converted)["terrain"]
    terrain.to_torch("cpu")
    center = torch.tensor((BOX_MIN + BOX_MAX) / 2, dtype=torch.float32)
    assert terrain.get_hf_val_from_points(center).item() == pytest.approx(BOX_HEIGHT, abs=1e-5)

    ij = torch.nonzero(terrain.hf > BOX_HEIGHT / 2)
    box_xy = terrain.get_point(ij).numpy()
    np.testing.assert_allclose(box_xy.min(axis=0), BOX_MIN, rtol=0, atol=DX)
    np.testing.assert_allclose(box_xy.max(axis=0), BOX_MAX, rtol=0, atol=DX)

    np.testing.assert_allclose(terrain.hf_maxmin[..., 0].numpy(), BOX_HEIGHT, rtol=0, atol=1e-5)
    np.testing.assert_allclose(terrain.hf_maxmin[..., 1].numpy(), 0.0, rtol=0, atol=1e-5)


def test_path_nodes(converted):
    path_nodes = pickle_compat.load(converted)["path_nodes"]
    assert isinstance(path_nodes, torch.Tensor) and path_nodes.dtype == torch.float32
    np.testing.assert_allclose(path_nodes.numpy(), [[0.0, 0.0, 0.0], [1.3, 1.0, BOX_HEIGHT]], rtol=0, atol=1e-5)


def test_no_path_nodes_by_default(tmp_path):
    obj_path = tmp_path / "box.obj"
    _write_obj(obj_path)
    out = tmp_path / "box.pkl"
    assert obj_to_heightfield.main([str(obj_path), str(out)]) == 0
    data = pickle_compat.load(out)
    assert list(data.keys()) == ["terrain"]
    # without a border offset the grid starts at the mesh's minimum corner
    np.testing.assert_allclose(data["terrain"].min_point, GROUND_MIN, rtol=0, atol=1e-6)
    assert data["terrain"].hf.shape == (41, 31)


def test_border_offset_too_large(tmp_path):
    obj_path = tmp_path / "box.obj"
    _write_obj(obj_path)
    assert obj_to_heightfield.main([str(obj_path), str(tmp_path / "out.pkl"), "--border-offset", "2.0"]) == 1
    assert not (tmp_path / "out.pkl").exists()
