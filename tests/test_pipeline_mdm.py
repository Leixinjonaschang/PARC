"""Lane C1: MDM changes for G1 (CPU unless marked).

- per-body heightfield SDF (develop d8f2c17) equals the old concatenated computation,
  and lowers peak GPU memory (CUDA only);
- the G1 config builds an MDM with hf-collision point samples for all 36 bodies;
- classifier-free guidance works with the expand()ed flag tensors gen_util passes
  (torch >= 2 rejects the old in-place writes);
- load_checkpoint(use_ema_model=True) uses the EMA weights when the checkpoint has them.
"""
import pathlib

import pytest
import torch

import parc.motion_generator.mdm as mdm_mod
import parc.util.path_loader as path_loader
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util
from parc.motion_generator.diffusion_util import MDMCustomGuidance, MDMKeyType

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
G1_TRAIN_GEN_CONFIG = REPO_ROOT / "data" / "configs" / "parc_1_train_gen_g1.yaml"
SEQ_LEN = 15  # 0.5 s at 30 fps; parc_1 sets it from the sampler
NUM_G1_BODIES = 36

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def data_dir(monkeypatch, tmp_path):
    # path_loader only needs DATA_DIR to be an existing directory
    monkeypatch.setenv("PARC_DATA_DIR", str(tmp_path))


def g1_mdm_config(device="cpu", small=True):
    cfg = path_loader.load_config(G1_TRAIN_GEN_CONFIG)
    cfg.update(device=device, use_wandb=False, seq_len=SEQ_LEN,
               char_file=str(REPO_ROOT / cfg["char_file"]))
    if small:
        # only the network size; the losses and features stay as in the G1 config
        cfg.update(diffusion_timesteps=21, d_model=64, num_heads=4, d_hid=64, num_layers=1,
                   in_mlp_layers=[64], out_mlp_layers=[64])
    return cfg


@pytest.fixture(scope="module")
def small_g1_mdm(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PARC_DATA_DIR", str(tmp_path_factory.mktemp("data_dir")))
        torch.manual_seed(0)
        return mdm_mod.MDM(g1_mdm_config())


def concatenated_point_hf_sdf(model, body_pos, body_rot, body_points, hfs, base_z):
    # MDM.compute_point_hf_sdf before d8f2c17: all bodies' points in one points_hf_sdf call
    batch_size = body_pos.shape[0]
    localized_points = []
    for b in range(model._kin_char_model.get_num_joints()):
        points = body_points[b].unsqueeze(0).unsqueeze(0)
        points = torch_util.quat_rotate(body_rot[..., b, :].unsqueeze(2), points) + body_pos[..., b, :].unsqueeze(2)
        localized_points.append(points.view(batch_size, -1, 3))
    localized_points = torch.cat(localized_points, dim=1)
    negative_dims = torch.tensor([model._num_x_neg, model._num_y_neg], dtype=torch.float32, device=model._device)
    min_box_center = (-model._dxdy * negative_dims).unsqueeze(0).expand(batch_size, 2)
    return terrain_util.points_hf_sdf(localized_points, hfs.squeeze(1), min_box_center, model._dxdy, base_z=base_z)


def random_body_state(model, batch_size, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    body_pos = 0.6 * torch.randn(batch_size, SEQ_LEN, NUM_G1_BODIES, 3, generator=gen)
    body_rot = torch.nn.functional.normalize(torch.randn(batch_size, SEQ_LEN, NUM_G1_BODIES, 4, generator=gen), dim=-1)
    hfs = 2.0 * torch.rand(batch_size, 1, model._grid_dim_x, model._grid_dim_y, generator=gen) - 1.0
    return body_pos.to(device), body_rot.to(device), hfs.to(device)


# ---------------------------------------------------------------- G1 config and point samples

def test_g1_mdm_has_hf_collision_points_for_every_body(small_g1_mdm):
    model = small_g1_mdm
    assert model._use_hf_collision_loss and model._use_target_loss
    points = model._char_point_samples
    assert len(points) == NUM_G1_BODIES == model._kin_char_model.get_num_joints()
    for b, body_points in enumerate(points):
        assert body_points.dim() == 2 and body_points.shape[1] == 3, b
        assert body_points.shape[0] >= 1, b
        assert body_points.device == torch.device("cpu")
        assert torch.isfinite(body_points).all(), b


def test_full_g1_config_builds_develop_sized_mdm():
    model = mdm_mod.MDM(g1_mdm_config(small=False))
    assert model._diffusion_timesteps == 1001
    assert (model._d_model, model._num_heads, model._d_hid, model._num_layers) == (1024, 16, 1024, 6)
    assert model._use_hf_collision_loss and model._use_target_loss
    # root pos 3 + root rot 3 + 35 joint positions + 29 hinge dofs + 36 contacts
    assert model._motion_frame_dof == 3 + 3 + 35 * 3 + 29 + 36
    assert len(model._char_point_samples) == NUM_G1_BODIES


# ---------------------------------------------------------------- per-body SDF

def test_per_body_sdf_equals_concatenated_sdf(small_g1_mdm):
    model = small_g1_mdm
    body_pos, body_rot, hfs = random_body_state(model, batch_size=2, device="cpu")
    base_z = model._min_h - 5.0

    pos_new = body_pos.clone().requires_grad_(True)
    sdf_new = model.compute_point_hf_sdf(pos_new, body_rot, model._char_point_samples, hfs, base_z=base_z)
    pos_old = body_pos.clone().requires_grad_(True)
    sdf_old = concatenated_point_hf_sdf(model, pos_old, body_rot, model._char_point_samples, hfs, base_z)

    num_points = sum(p.shape[0] for p in model._char_point_samples)
    assert sdf_new.shape == sdf_old.shape == (2, SEQ_LEN * num_points)
    assert torch.allclose(sdf_new, sdf_old)
    assert (sdf_new < 0.0).any() and (sdf_new > 0.0).any()  # some points penetrate, some do not

    # the hf-collision loss and its gradient are unchanged too
    for sdf, pos in ((sdf_new, pos_new), (sdf_old, pos_old)):
        (0.5 * torch.sum(torch.square(torch.clamp(sdf, max=0.0)))).backward()
    assert torch.allclose(pos_new.grad, pos_old.grad)


@needs_cuda
def test_per_body_sdf_lowers_peak_gpu_memory():
    model = mdm_mod.MDM(g1_mdm_config(device="cuda:0"))
    body_pos, body_rot, hfs = random_body_state(model, batch_size=8, device="cuda:0")
    base_z = model._min_h - 5.0

    def peak_memory(fn):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        pos = body_pos.clone().requires_grad_(True)
        sdf = fn(pos)
        (0.5 * torch.sum(torch.square(torch.clamp(sdf, max=0.0)))).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - start, sdf.detach()

    old_peak, sdf_old = peak_memory(lambda pos: concatenated_point_hf_sdf(
        model, pos, body_rot, model._char_point_samples, hfs, base_z))
    new_peak, sdf_new = peak_memory(lambda pos: model.compute_point_hf_sdf(
        pos, body_rot, model._char_point_samples, hfs, base_z=base_z))
    assert torch.allclose(sdf_new, sdf_old, atol=1e-5)
    assert new_peak < 0.9 * old_peak, (new_peak, old_peak)


# ---------------------------------------------------------------- classifier-free guidance

def build_generation_conds(model, batch_size, gen):
    # as gen_util.gen_mdm_motion builds them: the flags are expand()ed one-element tensors
    device = model._device
    hf = 2.0 * torch.rand(batch_size, 1, model._grid_dim_x, model._grid_dim_y, generator=gen) - 1.0
    target = torch.nn.functional.normalize(torch.randn(batch_size, 1, 2, generator=gen), dim=-1)
    guidance = MDMCustomGuidance()
    guidance.obs_cfg_scale = 0.65
    return {
        MDMKeyType.OBS_KEY: model.normalize_hf(hf),
        MDMKeyType.OBS_FLAG_KEY: torch.tensor([True], device=device).expand(batch_size),
        MDMKeyType.TARGET_KEY: target,
        MDMKeyType.TARGET_FLAG_KEY: torch.tensor([True], device=device).expand(batch_size),
        MDMKeyType.PREV_STATE_KEY: torch.randn(batch_size, model._num_prev_states, model._motion_frame_dof, generator=gen),
        MDMKeyType.PREV_STATE_NOISE_IND_KEY: torch.tensor([True], device=device).expand(batch_size),
        MDMKeyType.PREV_STATE_FLAG_KEY: torch.tensor([True], device=device).expand(batch_size),
        MDMKeyType.GUIDANCE_PARAMS: guidance,
    }


def test_cfg_predict_x0_accepts_expanded_flags(small_g1_mdm):
    model = small_g1_mdm
    model._denoise_model.eval()
    batch_size = 3
    gen = torch.Generator().manual_seed(1)
    conds = build_generation_conds(model, batch_size, gen)
    x_t = torch.randn(batch_size, SEQ_LEN, model._motion_frame_dof, generator=gen)
    t = torch.full((batch_size, 1, 1), 7, dtype=torch.int64)
    original_flags = {k: conds[k] for k in (MDMKeyType.PREV_STATE_NOISE_IND_KEY, MDMKeyType.PREV_STATE_FLAG_KEY)}

    key_padding_mask = model._denoise_model.create_key_padding_mask(batch_size, model._device)
    cond_tokens = model._denoise_model.embed_conds(conds, key_padding_mask)

    # reference: the pre-fix code (in-place flag writes) on materialized flags
    ref_conds = dict(conds)
    for key in original_flags:
        ref_conds[key] = conds[key].clone()
    ref_x = x_t.clone()
    with torch.no_grad():
        ref_conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY][:] = True
        ref_conds[MDMKeyType.PREV_STATE_FLAG_KEY][:] = True
        x_0 = model._denoise_model.fast_forward(ref_x, ref_conds, t, cond_tokens, key_padding_mask)
        ref_conds[MDMKeyType.PREV_STATE_NOISE_IND_KEY][:] = False
        ref_conds[MDMKeyType.PREV_STATE_FLAG_KEY][:] = False
        x_0_no_prev_state = model._denoise_model.forward(ref_x, ref_conds, t)
        expected = x_0 + 0.65 * (x_0_no_prev_state - x_0)

    predicted = model.predict_x0(x_t.clone(), conds, t, cond_tokens, key_padding_mask)

    assert torch.allclose(predicted, expected)
    for key, flag in original_flags.items():
        assert conds[key].shape == (batch_size,) and not conds[key].any()  # left False, as before
        assert flag.all()  # the caller's tensors are not written to


# ---------------------------------------------------------------- EMA weights

def test_load_checkpoint_uses_ema_weights_when_saved(small_g1_mdm, tmp_path):
    model = small_g1_mdm
    with torch.no_grad():
        for param in model._ema_denoise_model.parameters():
            param.add_(1.0)
    ema_state = {k: v.clone() for k, v in model._ema_denoise_model.state_dict().items()}
    regular_state = {k: v.clone() for k, v in model._denoise_model.state_dict().items()}
    path = tmp_path / "model.ckpt"
    model.save_checkpoint(str(path))

    def loaded_state(use_ema_model):
        loaded = mdm_mod.MDM.load_checkpoint(str(path), device="cpu", use_ema_model=use_ema_model)
        return loaded._denoise_model.state_dict()

    def same(state_a, state_b):
        return state_a.keys() == state_b.keys() and all(torch.equal(state_a[k], state_b[k]) for k in state_a)

    assert same(loaded_state(use_ema_model=True), ema_state)
    assert same(loaded_state(use_ema_model=False), regular_state)

    # a checkpoint without EMA weights keeps the trained weights instead of the
    # untrained EMA copy made by the constructor
    checkpoint = torch.load(str(path), map_location="cpu")
    del checkpoint["ema_model"]
    torch.save(checkpoint, str(path))
    assert same(loaded_state(use_ema_model=True), regular_state)
