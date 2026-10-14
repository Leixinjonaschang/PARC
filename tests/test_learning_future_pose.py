"""Tests for the future-pose observation encoder and predictor (porting plan, Lane B tasks B2-B6).

CPU only: the agents run on tests/fixtures/learning_fake_env.FakeEnv, which reports the G1 observation
contract (doc/porting_plan.md, Appendix A):

    raw (2363):     char_obs [0,266) | tar_obs [266,1670) | tar_contacts [1670,1886) | char_contacts [1886,1922) | hf [1922,2363)
    encoded (775):  char_obs [0,266) | future_pose latent [266,298) | char_contacts [298,334) | hf [334,775)
    predictor input (743): char_obs | char_contacts | hf
"""
import copy
import importlib
import json
import os
import subprocess
import sys
import types

import pytest
import torch
import yaml

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import learning_fake_env as fake  # noqa: E402

import parc.motion_tracker.learning.agent_builder as agent_builder  # noqa: E402
import parc.motion_tracker.learning.dm_ppo_agent as dm_ppo_agent  # noqa: E402
import parc.motion_tracker.learning.dm_ppo_fp_agent as dm_ppo_fp_agent  # noqa: E402
import parc.motion_tracker.learning.dm_ppo_fp_model as dm_ppo_fp_model  # noqa: E402
import parc.motion_tracker.learning.dm_ppo_model as dm_ppo_model  # noqa: E402
import parc.motion_tracker.learning.obs_encoder as obs_encoder  # noqa: E402
import parc.motion_tracker.learning.ppo_agent as ppo_agent  # noqa: E402
import parc.motion_tracker.learning.ppo_model as ppo_model  # noqa: E402
import parc.util.wandb_logger as wandb_logger  # noqa: E402

G1_AGENT_FILE = os.path.join(fake.TRACKER_CONFIG_DIR, "dm_agent_g1.yaml")
DEFAULT_AGENT_FILE = os.path.join(fake.TRACKER_CONFIG_DIR, "dm_agent_default.yaml")
DEVELOP_GEN_C_KEYS_FILE = os.path.join(FIXTURE_DIR, "learning_develop_genC_keys.json")

CHAR_OBS = slice(0, 266)
TAR_OBS_AND_CONTACTS = slice(266, 1886)
TAR_CONTACTS = slice(1670, 1886)
CHAR_CONTACTS = slice(1886, 1922)
HF = slice(1922, 2363)
ENC_CHAR_OBS = slice(0, 266)
ENC_LATENT = slice(266, 298)
ENC_CHAR_CONTACTS = slice(298, 334)
ENC_HF = slice(334, 775)


@pytest.fixture
def no_wandb(monkeypatch):
    # With use_wandb: false, WandbLogger calls wandb.init(mode="offline"), which writes ./wandb.
    monkeypatch.setattr(wandb_logger.wandb, "init", lambda *args, **kwargs: None)
    return


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    # agent_builder resolves config paths through path_loader, which needs an existing DATA_DIR.
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))
    return tmp_path


def g1_model_config():
    return fake.load_tracker_config("dm_agent_g1.yaml")["model"]


def small_g1_config(**predictor_overrides):
    # dm_agent_g1.yaml with small nets and short iterations
    config = fake.make_small_config("dm_agent_g1.yaml")
    config["model"]["future_pose_predictor"].update(predictor_overrides)
    return config


def build_fp_agent(config=None, env=None, seed=0):
    torch.manual_seed(seed)
    config = small_g1_config() if config is None else config
    env = fake.FakeEnv(num_envs=4) if env is None else env
    return dm_ppo_fp_agent.FuturePoseDMPPOAgent(config=config, env=env, device="cpu")


def build_model(model_config=None, seed=0):
    torch.manual_seed(seed)
    model_config = g1_model_config() if model_config is None else model_config
    return dm_ppo_fp_model.FuturePoseDMPPOModel(model_config, fake.FakeEnv(num_envs=4))


def make_obs(num, seed=0):
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn([num, fake.G1_OBS_DIM], generator=generator)
    contacts = slice(TAR_CONTACTS.start, CHAR_CONTACTS.stop)
    obs[:, contacts] = (obs[:, contacts] > 0).float()
    return obs


def write_yaml(path, config):
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return str(path)


def snapshot(module):
    return {key: val.detach().clone() for key, val in module.state_dict().items()}


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def param_ids(params):
    return set(id(param) for param in params)


def optimizer_param_ids(opt):
    return set(id(param) for group in opt._optimizer.param_groups for param in group["params"])


def rollout(agent):
    # One iteration of experience, as at the start of BaseAgent._train_iter.
    agent._curr_obs, agent._curr_info = agent.get_env().reset()
    agent._init_iter()
    agent.eval()
    agent.set_mode(ppo_agent.AgentMode.TRAIN)
    agent._rollout_train(agent._steps_per_iter)
    return


# ObservationEncoder

def test_encoder_maps_g1_obs_to_775_dims():
    encoder = obs_encoder.ObservationEncoder(g1_model_config(), fake.FakeEnv())
    output_shapes = [(key, info["shape"]) for key, info in encoder.get_output_obs_shapes().items()]
    assert output_shapes == [("char_obs", (266,)), ("future_pose", (32,)), ("char_contacts", (36,)), ("hf", (441,))]
    assert encoder.get_output_dim() == 775
    assert encoder.get_mock_obs_space().shape == (775,)

    obs = make_obs(8)
    encoded = encoder(obs)
    assert encoded.shape == (8, 775)
    assert torch.equal(encoded[:, ENC_CHAR_OBS], obs[:, CHAR_OBS])
    assert torch.equal(encoded[:, ENC_CHAR_CONTACTS], obs[:, CHAR_CONTACTS])
    assert torch.equal(encoded[:, ENC_HF], obs[:, HF])

    latent = encoded[:, ENC_LATENT]
    assert torch.equal(latent, encoder._encoders["future_pose"](obs[:, TAR_OBS_AND_CONTACTS]))
    assert torch.allclose(latent.norm(dim=-1), torch.ones(8), atol=1e-6)


def test_encoder_overrides_replace_parts():
    encoder = obs_encoder.ObservationEncoder(g1_model_config(), fake.FakeEnv())
    obs = make_obs(8)
    override = torch.zeros([8, 32])
    encoded = encoder(obs, overrides={"future_pose": override})
    assert torch.equal(encoded[:, ENC_LATENT], override)
    assert torch.equal(encoded[:, ENC_HF], obs[:, HF])


def test_encoder_rejects_obs_space_size_mismatch():
    env = fake.FakeEnv(obs_dim=fake.G1_OBS_DIM + 1)
    with pytest.raises(ValueError, match="sum to 2363 dims, but the observation space has 2364"):
        obs_encoder.ObservationEncoder(g1_model_config(), env)


@pytest.mark.parametrize("encoders, match", [
    ({"e": {"type": "transformer", "latent_dim": 8, "input_keys": ["tar_obs"]}}, "Unsupported type"),
    ({"e": {"type": "mlp", "latent_dim": 8, "input_keys": []}}, "no input_keys"),
    ({"e": {"type": "mlp", "latent_dim": 8, "input_keys": ["tar_pose"]}}, "not an observation key"),
    ({"e": {"type": "mlp", "latent_dim": 8, "input_keys": ["tar_obs"]},
      "f": {"type": "mlp", "latent_dim": 8, "input_keys": ["tar_obs", "hf"]}}, "already encoded"),
    ({"hf": {"type": "mlp", "latent_dim": 8, "input_keys": ["tar_obs"]}}, "does not encode"),
])
def test_encoder_rejects_bad_configs(encoders, match):
    with pytest.raises(ValueError, match=match):
        obs_encoder.ObservationEncoder({"obs_encoders": encoders}, fake.FakeEnv())


def test_encoder_without_encoders_passes_everything_through():
    encoder = obs_encoder.ObservationEncoder({}, fake.FakeEnv())
    obs = make_obs(4)
    assert encoder.get_output_dim() == fake.G1_OBS_DIM
    assert torch.equal(encoder(obs), obs)


def test_encoder_input_keys_default_to_its_name():
    # Generation-B checkpoints (doc/porting_plan.md, Appendix F) used an encoder named tar_obs.
    config = {"obs_encoders": {"tar_obs": {"type": "mlp", "latent_dim": 32, "hidden_dims": [64]}}}
    encoder = obs_encoder.ObservationEncoder(config, fake.FakeEnv())
    assert encoder.get_encoder_input_keys("tar_obs") == ["tar_obs"]
    assert list(encoder.get_output_obs_shapes().keys()) == ["char_obs", "tar_obs", "tar_contacts", "char_contacts", "hf"]
    assert encoder.get_output_dim() == 266 + 32 + 216 + 36 + 441


# FuturePoseDMPPOModel

def test_model_nets_and_predictor_sizes():
    model = build_model()
    assert model._actor_layers[0].in_features == 775
    assert model._critic_layers[0].in_features == 775
    assert model._predictor_input_keys == ["char_obs", "char_contacts", "hf"]
    assert model._predictor.fc[0].in_features == 743
    assert model._predictor.fc[-1].out_features == 32
    assert isinstance(model._predictor.fc[1], torch.nn.ReLU) # develop used the model's activation

    obs = make_obs(8)
    expected_input = torch.cat([obs[:, CHAR_OBS], obs[:, CHAR_CONTACTS], obs[:, HF]], dim=-1)
    assert torch.equal(model._get_predictor_input(obs), expected_input)


def test_model_can_be_deep_copied():
    # FakeEnv refuses deepcopy, like the IsaacGym env: the model must not keep a reference to it.
    model = build_model()
    clone = copy.deepcopy(model)
    assert states_equal(snapshot(clone), snapshot(model))


def test_predictor_test_mode_changes_only_the_actor():
    model = build_model()
    obs = make_obs(16)
    with torch.no_grad():
        actions = model.eval_actor(obs).mode
        vals = model.eval_critic(obs)

        model.set_predictor_test_mode(True)
        test_actions = model.eval_actor(obs).mode
        test_vals = model.eval_critic(obs)
        pred_latent = model._predictor(model._get_predictor_input(obs))
        pred_obs = model.obs_encoder(obs, overrides={"future_pose": pred_latent})
        expected_test_actions = ppo_model.PPOModel.eval_actor(model, pred_obs).mode

        model.set_predictor_test_mode(False)
        actions_again = model.eval_actor(obs).mode

    assert test_actions.shape == actions.shape == (16, fake.G1_ACTION_SIZE)
    assert not torch.allclose(test_actions, actions)
    assert torch.equal(test_actions, expected_test_actions)
    assert torch.equal(test_vals, vals)
    assert torch.equal(actions_again, actions)


@pytest.mark.parametrize("net_key, net_name", [("actor_net", "dm_vit_small"), ("critic_net", "dm_cnn_mlp")])
def test_model_rejects_dm_vit_and_cnn_nets(net_key, net_name):
    model_config = g1_model_config()
    model_config[net_key] = net_name
    with pytest.raises(ValueError, match="not supported"):
        build_model(model_config)


def test_model_without_target_encoder_has_no_predictor():
    model_config = g1_model_config()
    model_config["obs_encoders"] = {"tar": model_config["obs_encoders"]["future_pose"]}
    del model_config["future_pose_predictor"]
    model = build_model(model_config)

    assert not model.has_predictor()
    assert model.get_predictor_params() == []
    assert not any(key.startswith("_predictor.") for key in model.state_dict())
    with pytest.raises(RuntimeError):
        model.set_predictor_test_mode(True)
    with pytest.raises(RuntimeError):
        model.compute_predictor_loss(make_obs(4))


def test_model_rejects_unknown_explicit_target_encoder():
    model_config = g1_model_config()
    model_config["future_pose_predictor"]["target_encoder"] = "tar_obs"
    with pytest.raises(ValueError, match="target_encoder 'tar_obs' is not an obs encoder"):
        build_model(model_config)


def test_generation_b_predictor_target():
    # Generation B: encoder tar_obs on [tar_obs]; its predictor saw every other key, tar_contacts included.
    model_config = g1_model_config()
    model_config["obs_encoders"] = {"tar_obs": {"type": "mlp", "latent_dim": 32, "hidden_dims": [512, 512, 256]}}
    model_config["future_pose_predictor"]["target_encoder"] = "tar_obs"
    model = build_model(model_config)
    assert model._predictor_input_keys == ["char_obs", "tar_contacts", "char_contacts", "hf"]
    assert model._predictor.fc[0].in_features == 266 + 216 + 36 + 441


# FuturePoseDMPPOAgent

def test_predictor_learns_without_touching_encoder_actor_or_critic():
    agent = build_fp_agent(small_g1_config(learning_rate=1.0e-3))
    agent.train()
    norm_obs = agent._obs_norm.normalize(make_obs(64))

    predictor_ids = param_ids(agent._model.get_predictor_params())
    other_params = [(name, param) for name, param in agent.named_parameters() if id(param) not in predictor_ids]
    assert any(name.startswith("_model.obs_encoder.") for name, _ in other_params)
    others_before = {name: param.detach().clone() for name, param in other_params}
    predictor_before = snapshot(agent._model._predictor)

    losses = [agent._update_predictor(norm_obs).item() for _ in range(300)]

    assert losses[0] / losses[-1] >= 5.0, losses[::50]
    assert not states_equal(snapshot(agent._model._predictor), predictor_before)
    for param in agent._model.get_predictor_params():
        assert param.grad is None
    for name, param in other_params:
        assert torch.equal(param, others_before[name]), name
        assert param.grad is None, name


def test_compute_loss_takes_one_predictor_step():
    agent = build_fp_agent()
    rollout(agent)
    agent._build_train_data()
    agent.train()
    batch = agent._exp_buffer.sample(8)
    predictor_before = snapshot(agent._model._predictor)

    info = agent._compute_loss(batch)

    assert info["predictor_loss"].dim() == 0 and not info["predictor_loss"].requires_grad
    assert not states_equal(snapshot(agent._model._predictor), predictor_before)
    # Cleared, so that the main step's gradient clipping over model.parameters() does not see them.
    assert all(param.grad is None for param in agent._model.get_predictor_params())


def test_enabled_predictor_has_its_own_optimizer():
    agent = build_fp_agent()
    main_ids = optimizer_param_ids(agent._optimizer)
    predictor_ids = optimizer_param_ids(agent._predictor_optimizer)
    trainable_ids = param_ids(param for param in agent.parameters() if param.requires_grad)

    assert predictor_ids == param_ids(agent._model.get_predictor_params())
    assert main_ids.isdisjoint(predictor_ids)
    assert main_ids | predictor_ids == trainable_ids

    opt = agent._predictor_optimizer._optimizer
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 1e-4
    assert opt.param_groups[0]["weight_decay"] == 0.0


def test_train_iter_updates_the_predictor_when_enabled():
    agent = build_fp_agent()
    predictor_before = snapshot(agent._model._predictor)
    agent._curr_obs, agent._curr_info = agent.get_env().reset()
    agent._init_train()
    info = agent._train_iter()

    assert torch.isfinite(info["predictor_loss"])
    assert not states_equal(snapshot(agent._model._predictor), predictor_before)


def test_disabled_predictor_is_built_but_not_trained():
    agent = build_fp_agent(small_g1_config(enabled=False))
    assert agent._model.has_predictor() # kept, so the checkpoint layout does not depend on the flag
    assert agent._predictor_optimizer is None
    assert optimizer_param_ids(agent._optimizer).isdisjoint(param_ids(agent._model.get_predictor_params()))

    predictor_before = snapshot(agent._model._predictor)
    critic_before = snapshot(agent._model._critic_layers)
    agent._curr_obs, agent._curr_info = agent.get_env().reset()
    agent._init_train()
    info = agent._train_iter()

    assert "predictor_loss" not in info
    assert states_equal(snapshot(agent._model._predictor), predictor_before)
    assert not states_equal(snapshot(agent._model._critic_layers), critic_before) # an update did happen


def test_enabled_must_be_a_bool():
    with pytest.raises(ValueError, match="enabled must be true or false"):
        build_fp_agent(small_g1_config(enabled="false"))


def test_nan_predictor_loss_raises_before_stepping():
    agent = build_fp_agent()
    norm_obs = make_obs(8)
    norm_obs[0, CHAR_OBS.start] = float("nan") # a predictor input
    predictor_before = snapshot(agent._model._predictor)

    with pytest.raises(FloatingPointError, match="predictor_loss"):
        agent._update_predictor(norm_obs)
    assert states_equal(snapshot(agent._model._predictor), predictor_before)


def test_nan_in_training_data_exits_with_nonzero_status():
    script = "\n".join([
        "import sys",
        "sys.path.insert(0, {})".format(repr(FIXTURE_DIR)),
        "import torch",
        "import learning_fake_env as fake",
        "import parc.motion_tracker.learning.dm_ppo_fp_agent as dm_ppo_fp_agent",
        "env = fake.FakeEnv(num_envs=4)",
        "agent = dm_ppo_fp_agent.FuturePoseDMPPOAgent(fake.make_small_config('dm_agent_g1.yaml'), env, 'cpu')",
        "agent._curr_obs, agent._curr_info = env.reset()",
        "agent._init_train()",
        "env.nan_obs = True",
        "agent._train_iter()",
    ])
    result = subprocess.run([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=300)
    assert result.returncode != 0
    assert "FloatingPointError: NaN in obs" in result.stderr


def test_train_model_logs_predictor_loss(tmp_path, no_wandb):
    agent = build_fp_agent()
    log_file = str(tmp_path / "log.txt")
    agent.train_model(max_samples=32, out_model_file=str(tmp_path / "model.pt"), int_output_dir="",
                      log_file=log_file, logger_type="wandb")

    assert agent._iter == 2
    assert "Predictor_Loss" in agent._logger.log_headers
    assert agent._logger.get_current_val("Predictor_Loss") > 0.0
    with open(log_file, "r") as f:
        assert "Predictor_Loss" in f.read()


# Checkpoints

def test_save_and_load_round_trip(tmp_path):
    agent = build_fp_agent(seed=0)
    model_file = str(tmp_path / "model.pt")
    agent.save(model_file)

    loaded = build_fp_agent(seed=1)
    loaded.load(model_file)
    assert states_equal(snapshot(loaded), snapshot(agent))
    loaded.set_predictor_test_mode(True)


def test_load_tolerates_missing_predictor_weights(tmp_path):
    # e.g. a generation-B1 checkpoint, trained before the predictor existed
    agent = build_fp_agent(seed=0)
    state_dict = {key: val for key, val in agent.state_dict().items() if not key.startswith("_model._predictor.")}
    model_file = str(tmp_path / "model.pt")
    torch.save(state_dict, model_file)

    loaded = build_fp_agent(seed=1)
    predictor_before = snapshot(loaded._model._predictor)
    loaded.load(model_file)

    loaded_state = snapshot(loaded)
    assert all(torch.equal(loaded_state[key], val) for key, val in state_dict.items())
    assert states_equal(snapshot(loaded._model._predictor), predictor_before)
    with pytest.raises(RuntimeError, match="no future-pose predictor weights"):
        loaded.set_predictor_test_mode(True)


@pytest.mark.parametrize("change", ["drop_critic_bias", "add_key"])
def test_load_is_strict_outside_the_predictor(tmp_path, change):
    agent = build_fp_agent()
    state_dict = agent.state_dict()
    if change == "drop_critic_bias":
        del state_dict["_model._critic_out.bias"]
    else:
        state_dict["_model._extra"] = torch.zeros(1)
    model_file = str(tmp_path / "model.pt")
    torch.save(state_dict, model_file)

    with pytest.raises(RuntimeError, match="does not match the model"):
        build_fp_agent().load(model_file)


@pytest.mark.parametrize("enabled", [True, False])
def test_state_dict_matches_develop_generation_c(data_dir, enabled):
    # D2: develop's generation-C checkpoints must load. The fixture holds {state_dict key: shape} of
    # develop's agent with develop's dm_agent_g1.yaml on the same observation contract
    # (regenerate with tests/fixtures/learning_make_develop_genC_keys.py).
    with open(DEVELOP_GEN_C_KEYS_FILE, "r") as f:
        fixture = json.load(f)
    assert fixture["obs_shapes"] == dict((key, list(shape)) for key, shape, _ in fake.G1_OBS_SHAPES)

    config = fake.load_tracker_config("dm_agent_g1.yaml")
    config["model"]["future_pose_predictor"]["enabled"] = enabled
    agent = agent_builder.build_agent(write_yaml(data_dir / "agent.yaml", config), fake.FakeEnv(num_envs=4), "cpu")

    port_shapes = dict((key, list(val.shape)) for key, val in agent.state_dict().items())
    assert port_shapes == fixture["state_dict_shapes"]


# Configs and dispatch

def test_shipped_g1_config():
    with open(G1_AGENT_FILE, "r") as f:
        config = yaml.safe_load(f)
    assert config["agent_name"] == "DM_PPO"
    assert config["model"]["obs_encoders"] == {
        "future_pose": {"type": "mlp", "latent_dim": 32, "hidden_dims": [512, 512, 256],
                        "input_keys": ["tar_obs", "tar_contacts"]}}
    predictor_config = config["model"]["future_pose_predictor"]
    assert predictor_config["enabled"] is True # decision D1
    assert predictor_config["hidden_dims"] == [512, 512, 256]
    assert isinstance(predictor_config["learning_rate"], float) and predictor_config["learning_rate"] == 1e-4


@pytest.mark.parametrize("lr_text, expected", [("1e-4", 1e-4), ("1.0e-4", 1e-4), ("3e-5", 3e-5)])
def test_yaml_learning_rate_strings_become_floats(data_dir, lr_text, expected):
    with open(G1_AGENT_FILE, "r") as f:
        text = f.read()
    assert "learning_rate: 1.0e-4" in text
    text = text.replace("learning_rate: 1.0e-4", "learning_rate: " + lr_text)
    if lr_text == "1e-4":
        assert yaml.safe_load(text)["model"]["future_pose_predictor"]["learning_rate"] == "1e-4" # a string
    agent_file = data_dir / "agent.yaml"
    agent_file.write_text(text)

    agent = agent_builder.build_agent(str(agent_file), fake.FakeEnv(num_envs=4), "cpu")
    assert isinstance(agent._predictor_lr, float) and agent._predictor_lr == expected
    assert agent._predictor_optimizer._optimizer.param_groups[0]["lr"] == expected


def test_default_config_builds_upstream_dm_ppo_agent(data_dir):
    agent = agent_builder.build_agent(DEFAULT_AGENT_FILE, fake.FakeEnv(num_envs=4), "cpu")
    assert type(agent) is dm_ppo_agent.DMPPOAgent
    assert type(agent._model) is dm_ppo_model.DMPPOModel

    # Upstream's model: fc_3layers_2048units ([2048, 1024, 512]) actor and critic on the 2363-dim
    # observation, a 29-dim action mean and a scalar value. The FIXED log-std is not trainable.
    layers = [(fake.G1_OBS_DIM, 2048), (2048, 1024), (1024, 512)]
    num_layer_params = sum(n_in * n_out + n_out for n_in, n_out in layers)
    expected = 2 * num_layer_params + (512 * fake.G1_ACTION_SIZE + fake.G1_ACTION_SIZE) + (512 + 1)
    assert agent.calc_num_params() == expected == 14944286


@pytest.mark.parametrize("agent_name, obs_encoders, expected", [
    ("DM_PPO", None, dm_ppo_agent.DMPPOAgent),
    ("DM_PPO", {}, dm_ppo_agent.DMPPOAgent),
    ("DM_PPO", {"future_pose": {"latent_dim": 32}}, dm_ppo_fp_agent.FuturePoseDMPPOAgent),
    ("DM_PPO_FP", None, dm_ppo_fp_agent.FuturePoseDMPPOAgent),
    ("PPO", {"future_pose": {"latent_dim": 32}}, ppo_agent.PPOAgent),
])
def test_agent_class_dispatch(agent_name, obs_encoders, expected):
    model_config = {"actor_net": "fc_2layers_1024units"}
    if obs_encoders is not None:
        model_config["obs_encoders"] = obs_encoders
    assert agent_builder.get_agent_class({"agent_name": agent_name, "model": model_config}) is expected


def test_unknown_agent_name_raises():
    with pytest.raises(ValueError, match="Unsupported agent"):
        agent_builder.get_agent_class({"agent_name": "SAC", "model": {}})


# scripts/run_tracker.py --test_predictor

@pytest.fixture
def run_tracker(monkeypatch):
    # scripts.run_tracker imports env_builder, which imports IsaacGym (and IsaacGym must be imported
    # before torch). Swap in a builder that returns a FakeEnv.
    import parc.motion_tracker.envs as envs_package
    import scripts

    stub = types.ModuleType("parc.motion_tracker.envs.env_builder")
    stub.build_env = lambda env_file, num_envs, device, visualize, *args, **kwargs: fake.FakeEnv(num_envs=num_envs)
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(envs_package, "env_builder", stub, raising=False)

    old_module = sys.modules.pop("scripts.run_tracker", None)
    module = importlib.import_module("scripts.run_tracker")
    yield module

    if old_module is None:
        sys.modules.pop("scripts.run_tracker", None)
        if hasattr(scripts, "run_tracker"):
            delattr(scripts, "run_tracker")
    else:
        sys.modules["scripts.run_tracker"] = old_module
        scripts.run_tracker = old_module


def run_tracker_argv(tmp_path, agent_file, mode, extra_args=()):
    return ["run_tracker.py", "--mode", mode, "--env_config", "fake_env.yaml", "--agent_config", str(agent_file),
            "--num_envs", "4", "--device", "cpu", "--visualize", "False", "--rand_seed", "0",
            "--test_episodes", "4", "--max_samples", "16",
            "--out_model_file", str(tmp_path / "output" / "model.pt"),
            "--log_file", str(tmp_path / "output" / "log.txt")] + list(extra_args)


def test_run_tracker_test_predictor_after_load(data_dir, run_tracker, monkeypatch):
    agent_file = write_yaml(data_dir / "agent.yaml", small_g1_config())
    model_file = str(data_dir / "trained.pt")
    trained = build_fp_agent(seed=3)
    trained.save(model_file)

    tested = dict()
    original_test = run_tracker.test

    def test_and_record(agent, test_episodes):
        tested["state"] = snapshot(agent)
        tested["predictor_test_mode"] = agent._model.get_predictor_test_mode()
        tested["result"] = original_test(agent, test_episodes)
        return tested["result"]

    monkeypatch.setattr(run_tracker, "test", test_and_record)
    run_tracker.main(run_tracker_argv(data_dir, agent_file, "test",
                                      ["--model_file", model_file, "--test_predictor", "True"]))

    assert tested["predictor_test_mode"] is True
    assert states_equal(tested["state"], snapshot(trained))
    assert tested["result"]["num_eps"] >= 4


def test_run_tracker_rejects_test_predictor_in_train_mode(data_dir, run_tracker):
    agent_file = write_yaml(data_dir / "agent.yaml", small_g1_config())
    with pytest.raises(ValueError, match="only supported in test and record modes"):
        run_tracker.main(run_tracker_argv(data_dir, agent_file, "train", ["--test_predictor", "True"]))


def test_run_tracker_rejects_test_predictor_without_predictor(data_dir, run_tracker):
    agent_file = write_yaml(data_dir / "agent.yaml", fake.make_small_config("dm_agent_default.yaml"))
    with pytest.raises(ValueError, match="needs an agent config with obs_encoders"):
        run_tracker.main(run_tracker_argv(data_dir, agent_file, "test", ["--test_predictor", "True"]))
