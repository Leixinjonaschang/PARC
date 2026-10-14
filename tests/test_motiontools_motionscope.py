"""MotionScope helpers that do not need polyscope to be initialised (global_header calls
ps.init() at import, which needs a display or a headless EGL backend).

Covered: character trimeshes for the G1 (mesh and cylinder geoms, also from a GPU model,
bacd697) and the character identifiers of motionscope_config.yaml.
"""
import os
import sys

import pytest
import torch

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import motiontools_synth as synth  # noqa: E402

import parc.anim.kin_char_model as kin_char_model  # noqa: E402
import parc.motionscope.char_paths as char_paths  # noqa: E402
import parc.motionscope.polyscope_util as ps_util  # noqa: E402


def _check_trimeshes(char_model):
    trimeshes = ps_util.create_char_trimeshes(char_model)
    assert len(trimeshes) == char_model.get_num_bodies()
    for body_id, body_meshes in enumerate(trimeshes):
        # bodies without geoms get a small placeholder sphere
        assert len(body_meshes) == max(len(char_model.get_geoms(body_id)), 1)
    return trimeshes


def test_g1_char_trimeshes():
    g1 = synth.load_char("g1")
    trimeshes = _check_trimeshes(g1)
    geom_types = {g._shape_type for b in range(g1.get_num_bodies()) for g in g1.get_geoms(b)}
    assert kin_char_model.GeomType.MESH in geom_types and kin_char_model.GeomType.CYLINDER in geom_types
    # a mesh geom is placed at its offset: the torso's third geom has a non-zero one
    torso_id = g1.get_body_id("torso_link")
    offset = g1.get_geoms(torso_id)[2]._offset.numpy()
    assert abs(offset).sum() > 0
    mesh = g1._meshes[g1.get_geoms(torso_id)[2]._mesh_name]
    assert abs(trimeshes[torso_id][2].vertices.mean(0) - (mesh.vertices.mean(0) + offset)).max() < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU character model")
def test_char_trimeshes_from_a_gpu_char_model():
    # bacd697: cylinder and mesh offsets were written into numpy arrays as CUDA tensors
    _check_trimeshes(synth.load_char("g1", device="cuda:0"))
    _check_trimeshes(synth.load_char("humanoid", device="cuda:0"))


@pytest.mark.parametrize("identifier,expected", [
    ("humanoid", "data/assets/humanoid.xml"),
    ("g1", "data/assets/unitree_g1/g1_mocap_29dof.xml"),
    ("G1", "data/assets/unitree_g1/g1_mocap_29dof.xml"),
    ("data/assets/custom.xml", "data/assets/custom.xml"),
])
def test_resolve_char_path(identifier, expected):
    assert char_paths.resolve_char_path(identifier) == expected


def test_known_char_paths_exist():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in char_paths.CHAR_PATHS.values():
        assert os.path.isfile(os.path.join(repo_root, path)), path
