"""scripts/robot_motion_preprocess.py: parity with develop's tool, and its fixes.

The fixtures in tests/fixtures/datatools_rmp were written by develop's own code and
develop's tools/motion_file_proc/robot_motion_preprocess.py (develop @8385a15, CLI
defaults), see tests/fixtures/datatools_make_develop_reference.py:

  source/walk.pkl      legacy humanoid source motion: terrain and 15-body contacts
  gmr/walk_to_unitree_g1.pkl   G1 {frames, fps} (NumPy 2 pickle), soles 2 cm above the ground
  gmr/lift_to_unitree_g1.pkl   G1, soles 4 mm up, lifts the left foot and holds it; no source
  source/stand_h.pkl, gmr_humanoid/stand_h.pkl   humanoid source and input (5 mm up)
  develop_out/g1/*.pkl, develop_out/humanoid/stand_h.pkl   develop's output (legacy format)

Develop's output is converted with scripts/convert_legacy_motions.py and compared
with the port's MS output.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.create_dataset as create_dataset
import parc.util.file_io as file_io
import parc.util.file_io_helper as file_io_helper
import parc.util.pickle_compat as pickle_compat
import parc.util.terrain_util as terrain_util
import scripts.convert_legacy_motions as convert
import scripts.robot_motion_preprocess as rmp

REPO_ROOT = Path(__file__).resolve().parents[1]
G1_XML = REPO_ROOT / "data" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
HUMANOID_XML = REPO_ROOT / "data" / "assets" / "humanoid.xml"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "datatools_rmp"
SOURCE_DIR = FIXTURES / "source"
GMR_DIR = FIXTURES / "gmr"
WALK = "walk_to_unitree_g1.pkl"
LIFT = "lift_to_unitree_g1.pkl"

NUM_FRAMES = 40
G1_SOLE_OFFSET = 0.035  # 4 spheres of radius 0.005 at z = -0.03 under left/right_ankle_roll_link
DEVELOP_SOLE_OFFSET = 0.05


def _load_char(path):
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(path))
    return char_model


@pytest.fixture(scope="module")
def g1():
    return _load_char(G1_XML)


@pytest.fixture(scope="module")
def humanoid():
    return _load_char(HUMANOID_XML)


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("rmp")


def _develop_ms(name, char_model, tmp_root):
    """develop's legacy output, converted to an MS file with scripts/convert_legacy_motions.py."""
    dst = tmp_root / "develop_ms" / name
    assert convert.convert_motion_file(FIXTURES / "develop_out" / name, dst, char_model=char_model) == convert.CONVERTED
    return file_io.load_ms_file(str(dst))


def _run_port(*argv):
    code = rmp.main([str(a) for a in argv])
    assert code == 0


@pytest.fixture(scope="module")
def cli_run(tmp_root):
    """The G1 directory (walk with a source, lift without) through the CLI, with develop's sole offset."""
    out_dir = tmp_root / "g1_develop_offset"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PARC_DATA_DIR=str(tmp_root), WANDB_MODE="disabled")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.robot_motion_preprocess", "--input", str(GMR_DIR), "--output", str(out_dir),
         "--char_file", str(G1_XML), "--source_dir", str(SOURCE_DIR), "--sole_offset", str(DEVELOP_SOLE_OFFSET)],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    return {"out": out_dir, "log": result.stdout}


@pytest.fixture(scope="module")
def g1_default(tmp_root):
    out_dir = tmp_root / "g1_default"
    _run_port("--input", GMR_DIR, "--output", out_dir, "--source_dir", SOURCE_DIR)
    return out_dir


def _foot_sole_heights(ms_data, char_model, foot):
    """Lowest collision point of a foot minus the terrain height below it, per frame."""
    motion_data = ms_data.motion_data
    terrain = terrain_util.SubTerrain.from_ms_terrain_data(ms_data.terrain_data, device="cpu")
    root_pos = torch.from_numpy(motion_data.root_pos)
    root_rot = torch.from_numpy(motion_data.root_rot)
    joint_rot = torch.from_numpy(motion_data.joint_rot)
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    body_id = char_model.get_body_id(foot)
    sole_z = rmp.lowest_point_z(rmp.collision_geoms(char_model, body_id), body_pos[:, body_id], body_rot[:, body_id])
    return (sole_z - terrain.get_hf_val_from_points(body_pos[:, body_id, 0:2])).numpy()


def _constraint_rows(ms_data):
    rows = []
    for body_id, constraints in enumerate(ms_data.misc_data[file_io_helper.BODY_CONSTRAINTS_KEY]):
        for c in constraints:
            rows.append((body_id, int(c["start_frame_idx"]), int(c["end_frame_idx"]),
                         np.asarray(c["constraint_point"], dtype=np.float64)))
    return rows


def assert_same_motion(port, ref, height_atol=1e-5):
    pm, rm = port.motion_data, ref.motion_data
    assert (pm.fps, pm.loop_mode) == (rm.fps, rm.loop_mode)
    np.testing.assert_array_equal(pm.body_contacts, rm.body_contacts)
    np.testing.assert_allclose(pm.root_pos, rm.root_pos, rtol=0, atol=height_atol)
    np.testing.assert_allclose(pm.root_rot, rm.root_rot, rtol=0, atol=1e-6)
    np.testing.assert_allclose(pm.joint_rot, rm.joint_rot, rtol=0, atol=1e-6)

    pt, rt = port.terrain_data, ref.terrain_data
    np.testing.assert_array_equal(pt.hf, rt.hf)
    np.testing.assert_array_equal(pt.hf_maxmin, rt.hf_maxmin)
    np.testing.assert_allclose(pt.min_point, rt.min_point, rtol=0, atol=1e-6)
    assert float(pt.dx) == pytest.approx(float(rt.dx), abs=1e-7)

    p_inds = port.misc_data[file_io_helper.HF_MASK_INDS_KEY]
    r_inds = ref.misc_data[file_io_helper.HF_MASK_INDS_KEY]
    assert len(p_inds) == len(r_inds) == pm.root_pos.shape[0]
    for p, r in zip(p_inds, r_inds):
        np.testing.assert_array_equal(p, r)

    p_rows, r_rows = _constraint_rows(port), _constraint_rows(ref)
    assert [row[:3] for row in p_rows] == [row[:3] for row in r_rows]
    for p, r in zip(p_rows, r_rows):
        np.testing.assert_allclose(p[3], r[3], rtol=0, atol=1e-5)


def test_cli_help():
    result = subprocess.run([sys.executable, "-m", "scripts.robot_motion_preprocess", "--help"], cwd=str(REPO_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert result.returncode == 0, result.stdout
    for option in ["--input", "--output", "--char_file", "--source_dir", "--source_char_file", "--contact_bodies",
                   "--geometric_contacts", "--sole_offset", "--height_fix_frames"]:
        assert option in result.stdout


def test_gmr_fixture_is_numpy2_pickle():
    # GMR runs under NumPy 2; the port must read such pickles through pickle_compat
    assert b"numpy._core.multiarray" in (GMR_DIR / WALK).read_bytes()


def test_parity_g1_with_source_contacts(cli_run, g1, tmp_root):
    port = file_io.load_ms_file(str(cli_run["out"] / WALK))
    ref = _develop_ms("g1/" + WALK, g1, tmp_root)
    assert_same_motion(port, ref)

    # contacts were copied by name: feet and hands of the humanoid source, nothing else
    contacts = port.motion_data.body_contacts
    assert contacts.shape == (NUM_FRAMES, 36)
    mapped = sorted(np.nonzero(contacts.sum(axis=0))[0].tolist())
    assert mapped == sorted(g1.get_body_id(n) for n in ["left_foot", "right_foot", "left_hand", "right_hand"])
    # develop's grouping drops a trailing single-frame contact run (left hand at frame 39)
    rows = [row[:3] for row in _constraint_rows(port)]
    assert (g1.get_body_id("left_hand"), 33, 33) in rows
    assert all(row[1] != 39 for row in rows if row[0] == g1.get_body_id("left_hand"))


def test_parity_humanoid_geometric_contacts(humanoid, tmp_root):
    # develop ignores source contacts for the humanoid and detects foot contacts from the
    # foot boxes; the humanoid's derived sole offset is develop's 0.05
    out = tmp_root / "humanoid" / "stand_h.pkl"
    _run_port("--input", FIXTURES / "gmr_humanoid" / "stand_h.pkl", "--output", out, "--char_file", HUMANOID_XML,
              "--source_dir", SOURCE_DIR, "--geometric_contacts")
    port = file_io.load_ms_file(str(out))
    ref = _develop_ms("humanoid/stand_h.pkl", humanoid, tmp_root)
    assert_same_motion(port, ref)
    left = port.motion_data.body_contacts[:, humanoid.get_body_id("left_foot")]
    assert left[:5].all() and not left[15:30].any()  # the lifted foot is not in contact


def test_derived_sole_offsets(g1, humanoid):
    g1_feet = [g1.get_body_id(n) for n in rmp.FOOT_NAMES]
    humanoid_feet = [humanoid.get_body_id(n) for n in rmp.FOOT_NAMES]
    assert rmp.derive_sole_offset(g1, g1_feet, "cpu") == pytest.approx(G1_SOLE_OFFSET, abs=1e-6)
    assert rmp.derive_sole_offset(humanoid, humanoid_feet, "cpu") == pytest.approx(DEVELOP_SOLE_OFFSET, abs=1e-6)


def test_default_sole_offset_puts_feet_on_terrain(g1_default, cli_run, g1):
    # the height fix uses the feet in contact during the first 10 frames (both soles level)
    for out_dir, expected in [(g1_default, 0.0), (cli_run["out"], DEVELOP_SOLE_OFFSET - G1_SOLE_OFFSET)]:
        ms_data = file_io.load_ms_file(str(out_dir / WALK))
        for foot in rmp.FOOT_NAMES:
            heights = _foot_sole_heights(ms_data, g1, foot)[:10]
            in_contact = ms_data.motion_data.body_contacts[:10, g1.get_body_id(foot)] > 0.5
            assert in_contact.any()
            np.testing.assert_allclose(heights[in_contact], expected, rtol=0, atol=1e-3)


def test_g1_geometric_contacts_use_collision_spheres(cli_run, g1, tmp_root):
    # no source motion: foot contacts come from geometry, on a flat terrain
    port = file_io.load_ms_file(str(cli_run["out"] / LIFT))
    ref = _develop_ms("g1/" + LIFT, g1, tmp_root)
    left, right = g1.get_body_id("left_foot"), g1.get_body_id("right_foot")

    heights = _foot_sole_heights(port, g1, "left_foot")
    assert heights[20:30].min() > 0.1  # the left foot is held 12 cm up, without moving
    # develop judged "near the ground" with the G1's foot mesh as a 1 m box: still feet were always in contact
    assert ref.motion_data.body_contacts[20:30, left].all()
    assert not port.motion_data.body_contacts[18:31, left].any()
    assert port.motion_data.body_contacts[:10, left].all()
    assert port.motion_data.body_contacts[:, right].all()

    # develop wrote no terrain and no hf_mask_inds here; the port writes a flat terrain under the motion
    assert ref.terrain_data is None
    assert port.terrain_data is not None and not port.terrain_data.hf.any()
    assert len(port.misc_data[file_io_helper.HF_MASK_INDS_KEY]) == NUM_FRAMES


def test_ms_source_gives_same_result_as_legacy_source(cli_run, humanoid, tmp_root):
    ms_source = tmp_root / "ms_source" / "walk.pkl"
    assert convert.convert_motion_file(SOURCE_DIR / "walk.pkl", ms_source, char_model=humanoid) == convert.CONVERTED
    assert file_io.is_ms_container(pickle_compat.load(ms_source))
    out = tmp_root / "from_ms_source" / WALK
    _run_port("--input", GMR_DIR / WALK, "--output", out, "--source", ms_source,
              "--sole_offset", DEVELOP_SOLE_OFFSET)
    assert_same_motion(file_io.load_ms_file(str(out)), file_io.load_ms_file(str(cli_run["out"] / WALK)),
                       height_atol=0.0)


def test_contact_bodies_all(cli_run, g1, humanoid, tmp_root):
    out = tmp_root / "all_bodies" / WALK
    _run_port("--input", GMR_DIR / WALK, "--output", out, "--source_dir", SOURCE_DIR,
              "--sole_offset", DEVELOP_SOLE_OFFSET, "--contact_bodies", "all")
    contacts = file_io.load_ms_file(str(out)).motion_data.body_contacts
    default = file_io.load_ms_file(str(cli_run["out"] / WALK)).motion_data.body_contacts
    src = pickle_compat.load(SOURCE_DIR / "walk.pkl")["contacts"]

    pelvis = g1.get_body_id("pelvis")
    np.testing.assert_array_equal(contacts[:, pelvis], src[:, humanoid.get_body_id("pelvis")])
    assert contacts[:, pelvis].sum() == 3
    others = [b for b in range(g1.get_num_bodies()) if b != pelvis]
    np.testing.assert_array_equal(contacts[:, others], default[:, others])  # e.g. left_shin has no G1 body


def test_output_loads_in_motion_lib_and_create_dataset(cli_run, g1, tmp_path, monkeypatch):
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))
    motion_dir = tmp_path / "motions" / "walking"
    motion_dir.mkdir(parents=True)
    shutil.copy(str(cli_run["out"] / WALK), str(motion_dir / WALK))

    mlib = motion_lib.MotionLib.from_file(str(motion_dir / WALK), g1, "cpu", contact_info=True)
    assert mlib.num_motions() == 1
    assert tuple(mlib._frame_contacts.shape) == (NUM_FRAMES, 36)
    assert len(mlib._hf_mask_inds[0]) == NUM_FRAMES

    # develop's G1 dataset config: terrains up to 45 x 45 cells, at least 16 frames
    dataset_yaml = tmp_path / "dataset.yaml"
    create_dataset.create_dataset_yaml(
        folder_paths=[tmp_path / "motions"], save_path=dataset_yaml, char_filepath=str(G1_XML),
        compute_preprocessing_data=True, override_old_hf_mask_inds=True, cut_some_classes_in_half=False,
        motion_classes_to_cut_in_half=[], max_terrain_dim_x=45, max_terrain_dim_y=45, min_num_frames=16)
    assert WALK in dataset_yaml.read_text()
    mlib = motion_lib.MotionLib.from_file(str(dataset_yaml), g1, "cpu", contact_info=True)
    assert mlib.num_motions() == 1 and mlib._terrains[0].hf.shape == (40, 26)


def test_wrong_char_file_fails(tmp_path):
    out = tmp_path / "out.pkl"
    code = rmp.main(["--input", str(GMR_DIR / WALK), "--output", str(out), "--char_file", str(HUMANOID_XML)])
    assert code == 1
    assert not out.exists()


def test_find_source_file(tmp_path):
    source_dir = tmp_path / "source"
    (source_dir / "sub").mkdir(parents=True)
    (source_dir / "sub" / "walk.pkl").write_bytes(b"")
    (source_dir / "jump.pkl").write_bytes(b"")
    input_root = tmp_path / "in"
    suffix = rmp.DEFAULT_NAME_SUFFIX

    found = rmp.find_source_file(input_root / "sub" / "walk_to_unitree_g1.pkl", input_root, source_dir, suffix)
    assert found == source_dir / "sub" / "walk.pkl"
    # falls back to the top of source_dir
    found = rmp.find_source_file(input_root / "sub" / "jump_to_unitree_g1.pkl", input_root, source_dir, suffix)
    assert found == source_dir / "jump.pkl"
    assert rmp.find_source_file(input_root / "jump_to_unitree_g1.pkl", None, source_dir, "") is None
    assert rmp.find_source_file(input_root / "jump.pkl", None, source_dir, suffix) == source_dir / "jump.pkl"


def test_flat_terrain_covers_motion_away_from_origin(g1):
    # starts at (5, 3), goes 1.2 m forward, then back past its start
    phase = torch.linspace(0.0, 1.5 * np.pi, 30)
    frames = torch.zeros(size=(30, 6 + g1.get_dof_size()))
    frames[:, 0] = 5.0 + 1.2 * torch.sin(phase)
    frames[:, 1] = 3.0 - 0.7 * torch.sin(0.5 * phase)
    frames[:, 2] = 0.8
    terrain = rmp.create_flat_terrain(frames, g1, "cpu")
    min_point = terrain.min_point.numpy()
    max_point = terrain.get_max_point().numpy()
    root_xy = frames[:, 0:2].numpy()
    margin = rmp.FLAT_TERRAIN_PADDING - rmp.FLAT_TERRAIN_DX
    assert (min_point <= root_xy.min(axis=0) - margin).all()
    assert (max_point >= root_xy.max(axis=0) + margin).all()
    assert not terrain.hf.any()
