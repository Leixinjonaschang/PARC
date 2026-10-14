"""scripts/run_optimize_motions.py and scripts/run_simple_augment_motions.py at tiny scale on
synthetic MS motions (humanoid and G1), plus the motion-opt configs' $DATA_DIR paths. CPU only."""
import os
import sys
import warnings
from pathlib import Path

import pytest
import torch
import yaml

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import motiontools_synth as synth  # noqa: E402

import parc.anim.motion_lib as motion_lib  # noqa: E402
import parc.motion_synthesis.motion_opt.script_util as script_util  # noqa: E402
import parc.util.file_io as file_io  # noqa: E402
import parc.util.file_io_helper as file_io_helper  # noqa: E402
import parc.util.wandb_logger as wandb_logger  # noqa: E402
import scripts.run_optimize_motions as run_optimize_motions  # noqa: E402
import scripts.run_simple_augment_motions as run_simple_augment_motions  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "parc" / "motion_synthesis" / "motion_opt" / "config"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))
    # the optimizer's logger starts an (offline) wandb run whenever it writes a log file
    monkeypatch.setattr(wandb_logger.wandb, "init", lambda *args, **kwargs: None)


def _write_config(path, base_config, **overrides):
    cfg = yaml.safe_load(open(CONFIG_DIR / base_config))
    cfg.update(device="cpu", use_wandb=False, **overrides)
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


@pytest.mark.parametrize("char", ["humanoid", "g1"])
def test_run_optimize_motions(tmp_path, char):
    motion_file = synth.write_synthetic_motion(tmp_path / "{}_walk.pkl".format(char), char=char)
    motions_yaml = tmp_path / "motions.yaml"
    motions_yaml.write_text(yaml.safe_dump({"motions": [{"file": "$DATA_DIR/{}".format(motion_file.name), "weight": 1.0}]}))
    config = _write_config(tmp_path / "opt.yaml", "motion_opt.yaml",
                           motions_yaml_path="$DATA_DIR/motions.yaml",
                           output_folder_path="$DATA_DIR/out/",
                           char_model=str(synth.CHAR_FILES[char]),
                           num_iters=3, frame_stride=2, auto_compute_body_constraints=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_optimize_motions.main(["--config", config])
    d16 = [w for w in caught if "D16" in str(w.message)]
    assert (len(d16) > 0) == (char == "g1")

    out_file = tmp_path / "out" / "{}_walk_opt.pkl".format(char)
    assert (tmp_path / "out" / "log" / "log_{}_walk_opt.txt".format(char)).is_file()
    file_data = file_io_helper.load_ms_file(str(out_file))
    assert file_data.motion_data.root_pos.shape[0] == 15  # 30 frames, frame_stride 2
    assert file_data.motion_data.fps == 15
    assert file_data.terrain_data is not None
    constraints = file_data.misc_data[file_io_helper.BODY_CONSTRAINTS_KEY]
    # hand contact frames [12, 17] map to [6, 8] after the stride
    left_hand = synth.load_char(char).get_body_id("left_hand")
    assert [(c.start_frame_idx, c.end_frame_idx) for c in constraints[left_hand]] == [(6, 8)]

    char_model = synth.load_char(char)
    mlib = motion_lib.MotionLib.from_file(str(out_file), char_model, "cpu", contact_info=True)
    assert mlib.num_motions() == 1


@pytest.mark.parametrize("char,base_config,overrides", [
    ("humanoid", "mid_climbing_aug.yaml", {}),
    ("humanoid", "flat_running_aug.yaml", {}),  # BOXES_ALONG_PATH; has none of the optional keys
    ("g1", "vaulting_aug.yaml", {"slice_terrain": True, "terrain_padding_size": 3}),
])
def test_run_simple_augment_motions(tmp_path, char, base_config, overrides):
    motion_file = synth.write_synthetic_motion(tmp_path / "{}_walk.pkl".format(char), char=char)
    motions_yaml = tmp_path / "motions.yaml"
    motions_yaml.write_text(yaml.safe_dump({"motions": [{"file": "$DATA_DIR/{}".format(motion_file.name), "weight": 1.0}]}))
    config = _write_config(tmp_path / "aug.yaml", base_config,
                           motions_yaml_path="$DATA_DIR/motions.yaml",
                           output_folder_path="$DATA_DIR/aug/",
                           char_model=str(synth.CHAR_FILES[char]),
                           num_iters=2, num_new_motions=2, **overrides)

    run_simple_augment_motions.main(["--config", config])

    src = file_io_helper.load_ms_file(str(motion_file))
    for k in (1, 2):
        out = file_io_helper.load_ms_file(str(tmp_path / "aug" / "{}_walk_aug{:03d}.pkl".format(char, k)))
        assert out.motion_data.root_pos.shape == src.motion_data.root_pos.shape
        assert out.motion_data.joint_rot.shape == src.motion_data.joint_rot.shape
        assert torch.isfinite(out.motion_data.root_pos).all()
        if overrides.get("slice_terrain"):
            assert out.terrain_data.hf.shape[0] < src.terrain_data.hf.shape[0]
            # slicing moves the start to the origin; the optimizer then moves it a little
            assert torch.allclose(out.motion_data.root_pos[0, 0:2], torch.zeros(2), atol=0.05)


def test_load_motion_errors(tmp_path):
    humanoid = synth.load_char("humanoid")
    legacy = REPO_ROOT / "tests" / "fixtures" / "dataio_legacy_humanoid.pkl"
    with pytest.raises(file_io.MSFileFormatError, match="convert_legacy_motions"):
        script_util.load_motion(str(legacy), humanoid, "cpu")

    g1_file = synth.write_synthetic_motion(tmp_path / "g1.pkl", char="g1")
    with pytest.raises(ValueError, match="35 joint rotations.*14 non-root joints"):
        script_util.load_motion(str(g1_file), humanoid, "cpu")


@pytest.mark.parametrize("config_file", sorted(p.name for p in CONFIG_DIR.glob("*.yaml")))
def test_motion_opt_configs_resolve_data_dir(tmp_path, config_file):
    raw = yaml.safe_load(open(CONFIG_DIR / config_file))
    assert not raw["motions_yaml_path"].startswith(".."), "use $DATA_DIR instead of ../"
    cfg = script_util.load_config(str(CONFIG_DIR / config_file))
    if raw["motions_yaml_path"].startswith("$DATA_DIR"):
        assert cfg["motions_yaml_path"].startswith(str(tmp_path) + "/Data/")
