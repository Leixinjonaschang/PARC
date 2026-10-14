"""Lane C1: dataset yaml creation (CPU).

- create_dataset skips log/ folders, warns about and skips classes without usable
  motions (D10, develop 342741a), keeps every weight > 0 and balances the classes;
- scripts/create_dataset_yaml.py is a --config CLI that does nothing on import.

The motions are synthetic G1 clips (tests/fixtures/pipeline_g1_synthetic.py).
"""
import importlib
import os
import pathlib
import shutil
import subprocess
import sys
import warnings

import pytest
import yaml

import parc.util.create_dataset as create_dataset
import parc.util.file_io as file_io

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = str(REPO_ROOT / "tests" / "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import pipeline_g1_synthetic as synthetic  # noqa: E402

G1_DATASET_CONFIG = REPO_ROOT / "data" / "configs" / "create_dataset_config_g1.yaml"
NUM_FRAMES = 40


@pytest.fixture(autouse=True)
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))


@pytest.fixture(scope="module")
def motion_templates(tmp_path_factory):
    # written once, copied per test: create_dataset re-saves the files it reads
    root = tmp_path_factory.mktemp("g1_templates")
    char_model = synthetic.load_g1()
    synthetic.write_synthetic_g1_motions(root / "regular", num_motions_per_class=2, num_frames=NUM_FRAMES,
                                         char_model=char_model)
    synthetic.write_synthetic_g1_motions(root / "short", num_motions_per_class=1, num_frames=15,
                                         classes=("walk",), char_model=char_model)
    return root


def dataset_config(save_path, folder_paths, **overrides):
    with open(G1_DATASET_CONFIG) as f:
        config = yaml.safe_load(f)
    config.update(save_path=str(save_path), folder_paths=[str(p) for p in folder_paths],
                  char_filepath=str(synthetic.G1_CHAR_FILE), cut_some_classes_in_half=False)
    config.update(overrides)
    return config


def build_layout(templates, root):
    """initial/{walk,step,empty,log}/ and recorded/walk/ with
    walk: 2 + 1 clips (same class name in both folders), step: 2 clips,
    empty: one 15-frame clip only (dropped by min_num_frames: 16), log: a text log."""
    shutil.copytree(templates / "regular", root / "initial")
    (root / "initial" / "empty").mkdir()
    shutil.copy(templates / "short" / "walk" / "g1_walk_0.pkl", root / "initial" / "empty" / "too_short.pkl")
    (root / "initial" / "log").mkdir()
    (root / "initial" / "log" / "log_boxes_0_0.txt").write_text("motion optimization log\n")
    (root / "recorded" / "walk").mkdir(parents=True)
    shutil.copy(templates / "regular" / "walk" / "g1_walk_0.pkl", root / "recorded" / "walk" / "g1_walk_rec.pkl")
    return [root / "initial", root / "recorded"]


def test_dataset_skips_log_and_empty_classes(motion_templates, tmp_path):
    folder_paths = build_layout(motion_templates, tmp_path)
    save_path = tmp_path / "out" / "motions.yaml"  # out/ does not exist yet

    with pytest.warns(UserWarning, match="motion class 'empty' has no usable motions"):
        create_dataset.create_dataset_yaml_from_config(dataset_config(save_path, folder_paths))

    motions = yaml.safe_load(save_path.read_text())["motions"]
    files = [pathlib.Path(m["file"]) for m in motions]
    assert len(files) == len(set(files)) == 5  # walk 3, step 2; no duplicates
    assert all(f.suffix == ".pkl" for f in files)
    assert not any(f.parent.name in ("empty", "log") for f in files)

    weights = {f: m["weight"] for f, m in zip(files, motions)}
    assert all(w > 0.0 for w in weights.values())
    # classes are balanced: equal total weight per class (walk has 3 clips, step 2)
    walk = sum(w for f, w in weights.items() if f.parent.name == "walk")
    step = sum(w for f, w in weights.items() if f.parent.name == "step")
    assert walk == pytest.approx(step)

    # compute_preprocessing_data wrote the hf mask data into the motion files
    for f in files:
        data = file_io.load_ms_file(str(f))
        assert len(data.misc_data["hf_mask_inds"]) == NUM_FRAMES


def test_dataset_without_usable_motions_raises(motion_templates, tmp_path):
    (tmp_path / "data" / "walk").mkdir(parents=True)
    shutil.copy(motion_templates / "short" / "walk" / "g1_walk_0.pkl", tmp_path / "data" / "walk")
    config = dataset_config(tmp_path / "motions.yaml", [tmp_path / "data"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="no usable motions"):
            create_dataset.create_dataset_yaml_from_config(config)
    assert not (tmp_path / "motions.yaml").exists()


def test_create_dataset_yaml_cli(motion_templates, tmp_path):
    shutil.copytree(motion_templates / "regular", tmp_path / "data")
    save_path = tmp_path / "motions.yaml"
    config_path = tmp_path / "create_dataset.yaml"
    config = dataset_config(save_path, [tmp_path / "data"], compute_preprocessing_data=False)
    config_path.write_text(yaml.safe_dump(config))

    # importing the script does nothing
    module = importlib.import_module("scripts.create_dataset_yaml")
    assert not save_path.exists()
    assert module.main(["--config", str(tmp_path / "missing.yaml")]) == 1

    env = dict(os.environ, PARC_DATA_DIR=str(tmp_path), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-m", "scripts.create_dataset_yaml", "--config", str(config_path)],
                            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(yaml.safe_load(save_path.read_text())["motions"]) == 4

    result = subprocess.run([sys.executable, "-m", "scripts.create_dataset_yaml", "--bogus"],
                            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
