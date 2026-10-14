"""Lane C1: stage script logic that runs without training (CPU).

- parc_1: when a cached sampler.pkl is reused or rebuilt (develop 932d1f4);
- parc_0: the optional char_file reaches every stage config (develop 10c074c);
- parc_4: no create_dataset_config and a missing output_dir (upstream NameError and
  write-before-mkdir);
- parc_2 FILE mode: input terrains as MS files or legacy {terrain, path_nodes} pickles.
"""
import importlib.util
import pathlib
import pickle
import sys
import types

import pytest
import torch
import yaml

import parc.util.file_io_helper as file_io_helper
import parc.util.path_loader as path_loader
import parc.util.terrain_util as terrain_util
import scripts.parc_0_setup_iter as parc_0
import scripts.parc_1_train_gen as parc_1
import scripts.parc_2_kin_gen as parc_2

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "data" / "configs"
G1_CHAR_FILE = "data/assets/unitree_g1/g1_mocap_29dof.xml"
HUMANOID_CHAR_FILE = "data/assets/humanoid.xml"
LEGACY_TERRAIN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "dataio_legacy_terrain_only.pkl"


@pytest.fixture(autouse=True)
def repo_cwd(monkeypatch, tmp_path):
    # the configs use repo-relative paths; path_loader needs an existing DATA_DIR
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))


# ---------------------------------------------------------------- parc_1 sampler cache

class FakeSampler:
    """Stands in for MDMHeightfieldContactMotionSampler (pickled to sampler.pkl)."""
    num_built = 0

    def __init__(self, cfg):
        FakeSampler.num_built += 1
        self._device = cfg["device"]
        self.data = torch.arange(3.0)

    def get_seq_len(self):
        return 15


@pytest.fixture
def fake_sampler(monkeypatch):
    FakeSampler.num_built = 0
    monkeypatch.setattr(parc_1, "MDMHeightfieldContactMotionSampler", FakeSampler)
    return FakeSampler


def test_sampler_rebuild_reason():
    assert parc_1.sampler_rebuild_reason(types.SimpleNamespace(_device="cuda:0"), "cuda:0") is None
    assert parc_1.sampler_rebuild_reason(types.SimpleNamespace(_device="cpu"), "cpu") is None
    reason = parc_1.sampler_rebuild_reason(types.SimpleNamespace(_device="cuda:1"), "cuda:0")
    assert "cuda:1" in reason and "cuda:0" in reason
    assert parc_1.sampler_rebuild_reason(types.SimpleNamespace(_device="cpu"), "cuda:0") is not None
    assert parc_1.sampler_rebuild_reason(object(), "cpu") is not None


def test_sampler_cache_is_reused_on_the_same_device(fake_sampler, tmp_path):
    config = {"device": "cpu", "sampler_save_filepath": str(tmp_path / "new_dir" / "sampler.pkl")}
    sampler = parc_1.load_or_create_sampler(config)
    assert fake_sampler.num_built == 1 and (tmp_path / "new_dir" / "sampler.pkl").is_file()

    sampler = parc_1.load_or_create_sampler(config)
    assert fake_sampler.num_built == 1  # loaded, not rebuilt
    assert isinstance(sampler, FakeSampler) and sampler._device == "cpu"
    assert torch.equal(sampler.data, torch.arange(3.0))


def test_sampler_built_on_another_device_is_rebuilt(fake_sampler, tmp_path):
    sampler_path = tmp_path / "sampler.pkl"
    config = {"device": "cpu", "sampler_save_filepath": str(sampler_path)}
    parc_1.load_or_create_sampler(dict(config, device="cuda:3"))  # cached as if built on cuda:3
    assert fake_sampler.num_built == 1

    sampler = parc_1.load_or_create_sampler(config)
    assert fake_sampler.num_built == 2 and sampler._device == "cpu"

    parc_1.load_or_create_sampler(config)  # the rebuilt sampler replaced the cache
    assert fake_sampler.num_built == 2


def test_unreadable_sampler_cache_is_rebuilt(fake_sampler, tmp_path):
    sampler_path = tmp_path / "sampler.pkl"
    sampler_path.write_bytes(b"not a pickle")
    sampler = parc_1.load_or_create_sampler({"device": "cpu", "sampler_save_filepath": str(sampler_path)})
    assert fake_sampler.num_built == 1 and isinstance(sampler, FakeSampler)


# ---------------------------------------------------------------- parc_0 char_file

def setup_config(tmp_path, **overrides):
    config = path_loader.load_config(CONFIG_DIR / "parc_0_setup_iter_config_g1.yaml")
    config.update(output_dir=str(tmp_path / "iter_1"),
                  iter_start_dataset_path=str(tmp_path / "iter_1" / "start_motions.yaml"),
                  # Lane A adds data/configs/parc_3_tracker_g1.yaml; the humanoid tracker
                  # config (humanoid env) shows the env char_file being replaced
                  input_tracker_config_path=str(CONFIG_DIR / "parc_3_tracker_default.yaml"),
                  kin_gen_num_batches_of_motions=2)
    config.update(overrides)
    return config


def written_configs(out_dir):
    def load(rel_path):
        return yaml.safe_load((out_dir / rel_path).read_text())
    configs = {
        "mdm": load("p1_train_gen/mdm_config.yaml"),
        "kin_gen": [load("p2_kin_gen/{}/kin_gen_config.yaml".format(name)) for name in ("boxes_0_49", "boxes_50_99")],
        "tracker": load("p3_tracker/tracker.yaml"),
        "phys_record": load("p4_phys_record/phys_record.yaml"),
        "datasets": [load(stage + "/create_dataset_config.yaml")
                     for stage in ("p1_train_gen", "p3_tracker", "p4_phys_record")],
    }
    return configs


def test_setup_iter_writes_char_file_into_every_stage(tmp_path):
    parc_0.setup_iter(setup_config(tmp_path))
    out_dir = tmp_path / "iter_1"
    configs = written_configs(out_dir)

    assert configs["mdm"]["char_file"] == G1_CHAR_FILE
    assert all(c["opt"]["char_model"] == G1_CHAR_FILE for c in configs["kin_gen"])
    assert all(c["char_filepath"] == G1_CHAR_FILE for c in configs["datasets"])

    env_copy = out_dir / "p3_tracker" / "input_dm_env.yaml"
    assert configs["tracker"]["env_config"] == str(env_copy)
    env = yaml.safe_load(env_copy.read_text())
    assert env["env"]["char_file"] == G1_CHAR_FILE
    source_env = path_loader.load_config(CONFIG_DIR / "tracker_config" / "dm_env_default.yaml")
    assert source_env["env"]["char_file"] == HUMANOID_CHAR_FILE  # only the copy changed
    env["env"]["char_file"] = HUMANOID_CHAR_FILE
    assert env == source_env

    # the G1 kin gen settings from the setup config
    for kin_gen in configs["kin_gen"]:
        assert (kin_gen["procgen_mode"], kin_gen["first_heading_mode"], kin_gen["save_name"]) == ("BOXES", "auto", "boxes")
        assert kin_gen["mdm_model_path"] == str(out_dir / "p1_train_gen" / "checkpoints")
    assert configs["phys_record"]["env_file"] == str(out_dir / "p3_tracker" / "dm_env.yaml")


def test_setup_iter_without_char_file_keeps_the_stage_char_files(tmp_path):
    config = setup_config(tmp_path)
    del config["char_file"]
    config["input_mdm_config_path"] = str(CONFIG_DIR / "parc_1_train_gen_default.yaml")
    parc_0.setup_iter(config)
    out_dir = tmp_path / "iter_1"
    configs = written_configs(out_dir)

    assert configs["mdm"]["char_file"] == HUMANOID_CHAR_FILE  # from parc_1_train_gen_default.yaml
    assert all(c["char_filepath"] == G1_CHAR_FILE for c in configs["datasets"])  # from the G1 dataset config
    assert configs["tracker"]["env_config"] == "data/configs/tracker_config/dm_env_default.yaml"
    assert not (out_dir / "p3_tracker" / "input_dm_env.yaml").exists()


def test_setup_iter_rejects_a_missing_char_file(tmp_path):
    with pytest.raises(AssertionError, match="not a valid path"):
        parc_0.setup_iter(setup_config(tmp_path, char_file="data/assets/no_such_robot.xml"))


# ---------------------------------------------------------------- parc_4 setup

def load_parc_4_with_fake_run_tracker(monkeypatch):
    # parc_4 imports scripts.run_tracker, which imports IsaacGym; load parc_4's source
    # as a separate module against a stub instead
    calls = []
    fake_run_tracker = types.ModuleType("scripts.run_tracker")
    fake_run_tracker.main = lambda argv: calls.append(argv)
    import scripts
    monkeypatch.setitem(sys.modules, "scripts.run_tracker", fake_run_tracker)
    monkeypatch.setattr(scripts, "run_tracker", fake_run_tracker, raising=False)
    spec = importlib.util.spec_from_file_location("parc_4_under_test", REPO_ROOT / "scripts" / "parc_4_phys_record.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_phys_record_uses_dataset_file_and_creates_output_dir(monkeypatch, tmp_path):
    parc_4, calls = load_parc_4_with_fake_run_tracker(monkeypatch)
    dataset_file = tmp_path / "motions.yaml"
    dataset_file.write_text(yaml.safe_dump({"motions": [{"file": "a.pkl", "weight": 1.0},
                                                        {"file": "b.pkl", "weight": 1.0}]}))
    env_file = tmp_path / "dm_env.yaml"
    env_file.write_text(yaml.safe_dump({"env": {"char_file": G1_CHAR_FILE, "dm": {}}}))
    output_dir = tmp_path / "p4_phys_record"  # does not exist yet
    config = {"device": "cpu", "output_dir": str(output_dir), "model_file": str(tmp_path / "model.pt"),
              "agent_file": str(tmp_path / "agent_config.yaml"), "env_file": str(env_file),
              "dataset_file": str(dataset_file)}  # no create_dataset_config

    parc_4.record_motions(config)

    record_env = yaml.safe_load((output_dir / "record_env.yaml").read_text())
    assert record_env["env"]["dm"]["motion_file"] == str(dataset_file)
    assert record_env["env"]["output_motion_dir"] == str(output_dir / "recorded_motions")
    assert len(calls) == 1
    argv = " ".join(calls[0])
    assert "--mode record" in argv and "--num_envs 2" in argv
    assert "--env_config {}".format(output_dir / "record_env.yaml") in argv


# ---------------------------------------------------------------- parc_2 FILE-mode terrains

def test_kin_gen_loads_legacy_and_ms_terrain_files(tmp_path):
    terrains, paths = parc_2._load_input_terrains(str(LEGACY_TERRAIN_FIXTURE))
    legacy_terrain, legacy_path = terrains[0], paths[0]
    assert type(legacy_terrain) is terrain_util.SubTerrain  # develop's util.terrain_util class, remapped
    assert legacy_terrain.hf.device == torch.device("cpu")  # pickled on the GPU
    assert legacy_terrain.hf_maxmin.shape == legacy_terrain.hf.shape + (2,)
    assert legacy_path.device == torch.device("cpu") and legacy_path.shape[-1] == 3

    # the MS terrain-only file parc_2 writes in only_gen mode
    ms_path = tmp_path / "terrain_0.pkl"
    file_io_helper.save_ms_file(str(ms_path), terrain=legacy_terrain, path_nodes=legacy_path)
    terrain_list = tmp_path / "terrains.yaml"
    terrain_list.write_text(yaml.safe_dump({"terrains": [str(ms_path), str(LEGACY_TERRAIN_FIXTURE)]}))

    terrains, paths = parc_2._load_input_terrains(str(terrain_list))
    assert len(terrains) == len(paths) == 2
    assert torch.equal(terrains[0].hf, legacy_terrain.hf) and torch.equal(terrains[1].hf, legacy_terrain.hf)
    assert torch.allclose(terrains[0].min_point, legacy_terrain.min_point)
    assert torch.allclose(terrains[0].dxdy, legacy_terrain.dxdy)
    assert torch.equal(terrains[0].dims, legacy_terrain.dims)
    assert torch.allclose(paths[0], legacy_path)

    not_a_terrain = tmp_path / "not_a_terrain.pkl"
    not_a_terrain.write_bytes(pickle.dumps({"frames": [1, 2, 3]}))
    with pytest.raises(ValueError, match="legacy terrain pickle"):
        parc_2._load_input_terrains(str(not_a_terrain))
