"""Lane C1: stages 1 and 2 end to end on CPU with a tiny G1 MDM.

Trains a 1-layer MDM for 2 epochs on synthetic G1 motions through parc_1's train_mdm
(G1 config, so the hf-collision and target losses are on), then runs parc_2's
mdm_procgen with the motion optimization, in only_gen mode, and in FILE mode on the
only_gen terrains. Every motion it writes must load in MotionLib with G1 shapes.
wandb is stubbed out. Takes about half a minute.
"""
import pathlib
import random
import sys

import numpy as np
import pytest
import torch
import wandb
import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.file_io_helper as file_io_helper
import parc.util.path_loader as path_loader
import scripts.parc_1_train_gen as parc_1
import scripts.parc_2_kin_gen as parc_2

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "data" / "configs"
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))
import pipeline_g1_synthetic as synthetic  # noqa: E402

NUM_JOINTS = 35  # non-root joints: 29 hinges + 6 fixed
NUM_BODIES = 36


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _FakeRun:
    def finish(self):
        pass


def stub_wandb(mp):
    mp.setattr(wandb, "init", lambda *args, **kwargs: _FakeRun())
    mp.setattr(wandb, "log", lambda *args, **kwargs: None)
    mp.setattr(wandb, "login", lambda *args, **kwargs: None)


@pytest.fixture(scope="module")
def pipeline_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("g1_pipeline")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PARC_DATA_DIR", str(root))
        mp.chdir(REPO_ROOT)
        stub_wandb(mp)
        yield root


@pytest.fixture(scope="module")
def trained_mdm_dir(pipeline_env):
    root = pipeline_env
    seed_everything(0)
    synthetic.write_synthetic_g1_motions(root / "motions", num_motions_per_class=2)

    dataset_config = path_loader.load_config(CONFIG_DIR / "create_dataset_config_g1.yaml")
    dataset_config.update(save_path=str(root / "motions.yaml"), folder_paths=[str(root / "motions")])
    (root / "create_dataset.yaml").write_text(yaml.safe_dump(dataset_config))

    config = path_loader.load_config(CONFIG_DIR / "parc_1_train_gen_g1.yaml")
    p1_dir = root / "p1_train_gen"
    config.update(
        device="cpu", use_wandb=False, epochs=2, epochs_per_checkpoint=1, iters_per_epoch=1,
        batch_size=4, test_size=4,
        # tiny network; features and losses as in the G1 config
        diffusion_timesteps=21, d_model=64, num_heads=4, d_hid=64, num_layers=1,
        in_mlp_layers=[64], out_mlp_layers=[64],
        motion_lib_file=str(root / "motions.yaml"),
        create_dataset_config=str(root / "create_dataset.yaml"),
        sampler_save_filepath=str(p1_dir / "sampler.pkl"),
        sampler_stats_filepath=str(p1_dir / "sampler_stats.txt"),
        output_dir=str(p1_dir))
    parc_1.train_mdm(config)
    return p1_dir


def kin_gen_config(root, p1_dir, name, **overrides):
    config = path_loader.load_config(CONFIG_DIR / "parc_2_kin_gen_g1.yaml")
    config.update(device="cpu", mdm_model_path=str(p1_dir / "checkpoints"),
                  output_dir=str(root / name / "raw"), procgen_mode="BOXES", save_name=name,
                  motion_id_offset=0, num_new_motions=1,
                  # an untrained MDM does not reach the real loss thresholds
                  max_contact_loss=1e9, max_pen_loss=1e9, max_total_loss=1e9)
    config["opt"].update(device="cpu", output_dir=str(root / name / "opt"), use_wandb=False, num_iters=3)
    config["mdm_path"].update(mdm_batch_size=2, top_k=1, max_motion_length=1.0)
    config.update(overrides)
    return config


@pytest.fixture(scope="module")
def g1_char_model():
    return synthetic.load_g1()


def check_g1_motion_file(path, char_model):
    data = file_io_helper.load_ms_file(str(path))
    motion = data.motion_data
    num_frames = motion.root_pos.shape[0]
    assert num_frames >= 2
    assert tuple(motion.joint_rot.shape) == (num_frames, NUM_JOINTS, 4)
    assert tuple(motion.body_contacts.shape) == (num_frames, NUM_BODIES)
    assert (motion.fps, motion.loop_mode) == (30, "CLAMP")
    assert data.terrain_data is not None
    mlib = motion_lib.MotionLib.from_file(str(path), char_model, "cpu", contact_info=True)
    assert mlib.num_motions() == 1
    return data


def test_train_gen_writes_checkpoints_and_stats(trained_mdm_dir):
    assert (trained_mdm_dir / "checkpoints" / "model_1.ckpt").is_file()
    assert (trained_mdm_dir / "final_model.ckpt").is_file()
    assert (trained_mdm_dir / "sampler_stats.txt").is_file()
    assert (trained_mdm_dir / "sampler.pkl").is_file()
    checkpoint = torch.load(str(trained_mdm_dir / "final_model.ckpt"), map_location="cpu")
    assert "ema_model" in checkpoint
    assert checkpoint["cfg"]["use_hf_collision_loss"] and checkpoint["cfg"]["use_target_loss"]
    assert checkpoint["cfg"]["char_file"].endswith("unitree_g1/g1_mocap_29dof.xml")


def test_kin_gen_with_optimization_writes_g1_motions(pipeline_env, trained_mdm_dir, g1_char_model):
    root = pipeline_env
    seed_everything(1)
    parc_2.mdm_procgen(kin_gen_config(root, trained_mdm_dir, "boxes"))

    raw = check_g1_motion_file(root / "boxes" / "raw" / "boxes_0_0.pkl", g1_char_model)
    assert "path_nodes" in raw.misc_data and "loss" in raw.misc_data

    opt = check_g1_motion_file(root / "boxes" / "opt" / "boxes_0_0_opt.pkl", g1_char_model)
    num_frames = opt.motion_data.root_pos.shape[0]
    assert len(opt.misc_data["hf_mask_inds"]) == num_frames
    assert len(opt.misc_data["opt:body_constraints"]) == NUM_BODIES

    flipped = check_g1_motion_file(root / "boxes" / "opt" / "flipped" / "boxes_0_0_opt_flipped.pkl", g1_char_model)
    assert len(flipped.misc_data["hf_mask_inds"]) == num_frames
    assert np.allclose(flipped.motion_data.root_pos[:, 1], -opt.motion_data.root_pos[:, 1], atol=1e-5)
    assert (root / "boxes" / "opt" / "log" / "log_boxes_0_0.txt").is_file()


def test_kin_gen_file_mode_reads_only_gen_terrains(pipeline_env, trained_mdm_dir, g1_char_model):
    root = pipeline_env
    seed_everything(2)
    parc_2.mdm_procgen(kin_gen_config(root, trained_mdm_dir, "terrains", only_gen=True, num_new_motions=2))
    terrain_files = sorted((root / "terrains" / "raw").glob("*.pkl"))
    assert [f.name for f in terrain_files] == ["terrains_0.pkl", "terrains_1.pkl"]
    for terrain_file in terrain_files:  # terrain-only MS files
        data = file_io_helper.load_ms_file(str(terrain_file))
        assert data.motion_data is None
        assert tuple(data.terrain_data.hf.shape) == (16, 16)
        assert data.misc_data["path_nodes"].shape[-1] == 3

    terrain_list = root / "terrains.yaml"
    terrain_list.write_text(yaml.safe_dump({"terrains": [str(f) for f in terrain_files]}))
    parc_2.mdm_procgen(kin_gen_config(root, trained_mdm_dir, "file", procgen_mode="FILE",
                                      input_terrain_path=str(terrain_list), use_opt=False))
    data = check_g1_motion_file(root / "file" / "raw" / "file_0_0.pkl", g1_char_model)
    assert "min_point_offset" in data.misc_data
    assert not (root / "file" / "opt").exists()
