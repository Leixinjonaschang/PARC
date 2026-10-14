"""scripts/motion_tests: mdm_validation_test's config handling (terrains as a string or a
list, MS terrain files with path nodes), and compute_stats / compute_losses on small MS
motions. CPU only; generating with a real MDM is not covered here."""
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import motiontools_synth as synth  # noqa: E402

import parc.util.file_io as file_io  # noqa: E402
import scripts.motion_tests.compute_losses as compute_losses  # noqa: E402
import scripts.motion_tests.compute_stats as compute_stats  # noqa: E402
import scripts.motion_tests.mdm_validation_test as mdm_validation_test  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))


@pytest.mark.parametrize("terrains", ["$DATA_DIR/a.pkl", ["$DATA_DIR/a.pkl", "rel/b.pkl"]])
def test_terrain_list_accepts_a_string_or_a_list(tmp_path, terrains):
    # 39bc984: `terrains:` may be a single path
    terrains_yaml = tmp_path / "terrains.yaml"
    terrains_yaml.write_text(yaml.safe_dump({"terrains": terrains}))
    files = mdm_validation_test.load_terrain_file_list("$DATA_DIR/terrains.yaml")
    expected = [str(tmp_path / "a.pkl")] + (["rel/b.pkl"] if isinstance(terrains, list) else [])
    assert files == expected


def test_validation_config_resolves_data_dir(tmp_path):
    config = mdm_validation_test.load_config(str(REPO_ROOT / "scripts" / "motion_tests" / "mdm_validation_test.yaml"))
    for key in ("input_terrains", "mdm_model", "output_folder"):
        assert config[key].startswith(str(tmp_path) + "/tests/"), key
    assert config["mdm_model"].endswith(".ckpt")


def test_load_terrain_and_path_nodes(tmp_path):
    motion_file = synth.write_synthetic_motion(tmp_path / "terrain.pkl", char="humanoid")
    terrain, path_nodes = mdm_validation_test.load_terrain_and_path_nodes(str(motion_file))
    assert path_nodes.shape == (3, 3)
    assert terrain.hf.shape == (37, 31)

    no_path = synth.write_synthetic_motion(tmp_path / "no_path.pkl", char="humanoid", with_path=False)
    with pytest.raises(ValueError, match="no path_nodes"):
        mdm_validation_test.load_terrain_and_path_nodes(str(no_path))

    legacy_terrain = REPO_ROOT / "tests" / "fixtures" / "dataio_legacy_terrain_only.pkl"
    with pytest.raises(file_io.MSFileFormatError, match="convert_legacy_motions"):
        mdm_validation_test.load_terrain_and_path_nodes(str(legacy_terrain))


@pytest.mark.parametrize("char", ["humanoid", "g1"])
def test_compute_stats(tmp_path, char):
    motion_file = synth.write_synthetic_motion(tmp_path / "walk.pkl", char=char, num_frames=20)
    output = tmp_path / "stats" / "derivatives.pkl"
    compute_stats.main(["--motion_file", str(motion_file), "--char_file", str(synth.CHAR_FILES[char]),
                        "--output", str(output)])
    with open(output, "rb") as f:
        stats = pickle.load(f)
    num_bodies = synth.load_char(char).get_num_bodies()
    assert stats["vel"].shape == (19, num_bodies, 3)
    assert stats["acc"].shape == (18, num_bodies, 3)
    assert stats["jerk"].shape == (17, num_bodies, 3)
    assert np.isfinite(stats["jerk"]).all()


@pytest.mark.parametrize("char", ["humanoid", "g1"])
def test_compute_losses(tmp_path, char):
    motion_dir = tmp_path / "gen"
    motion_dir.mkdir()
    for seed in range(2):
        synth.write_synthetic_motion(motion_dir / "PATH_TERRAIN_7_{}.pkl".format(seed), char=char, seed=seed)
    output = tmp_path / "metrics.csv"
    compute_losses.main(["--motion_dirs", "$DATA_DIR/gen", "--char_file", str(synth.CHAR_FILES[char]),
                         "--device", "cpu", "--output", str(output)])

    rows = list(csv.reader(open(output)))
    assert len(rows) == 2
    header, row = rows
    assert len(header) == len(row) == 13 + 12 * compute_losses.NUM_PATH_TERRAINS
    summary = dict(zip(header[1:13], (float(x) for x in row[1:13])))
    assert summary["motion_length mean"] == pytest.approx(1.0)  # 30 frames at 30 fps
    assert summary["final_node_dist mean"] == pytest.approx(0.0, abs=1e-5)  # synthetic path ends at the motion
    assert all(np.isfinite(v) for v in summary.values())


def test_compute_losses_rejects_unexpected_file_names(tmp_path):
    motion_dir = tmp_path / "gen"
    motion_dir.mkdir()
    synth.write_synthetic_motion(motion_dir / "spiral_0.pkl", char="humanoid")
    with pytest.raises(ValueError, match="PATH_TERRAIN_"):
        compute_losses.compute_csv_row(str(motion_dir), device="cpu")
