"""torch_util.exp_map_to_axis_angle: finite gradients at zero, unchanged results elsewhere (P1a.3)."""
import math

import torch

import parc.util.torch_util as torch_util


def old_exp_map_to_axis_angle(exp_map):
    # upstream before the port: identical except for the unclamped division
    min_theta = 1e-5
    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / angle_exp
    angle = torch_util.normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    axis = torch.where(mask.unsqueeze(-1), axis, default_axis)
    return axis, angle


def random_exp_maps(num, max_angle, seed):
    gen = torch.Generator().manual_seed(seed)
    direction = torch_util.normalize(torch.randn(num, 3, generator=gen))
    theta = torch.rand(num, generator=gen) * max_angle
    return direction * theta.unsqueeze(-1)


def test_zero_exp_map_has_finite_gradient():
    weights = torch.randn(4, generator=torch.Generator().manual_seed(1))
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    # repeat: TorchScript switches to an optimized graph after the first calls
    for _ in range(4):
        exp_map = torch.zeros(5, 3)
        exp_map[1] = torch.tensor([1e-7, -2e-7, 3e-7])  # nonzero but below min_theta
        exp_map.requires_grad_(True)
        quat = torch_util.exp_map_to_quat(exp_map)
        (grad,) = torch.autograd.grad((quat * weights).sum(), exp_map)
        assert torch.isfinite(grad).all()
        assert torch.equal(quat.detach(), identity.expand(5, 4))


def test_matches_old_formula_up_to_4pi():
    exp_map = random_exp_maps(4096, 4.0 * math.pi, seed=0)
    axis, angle = torch_util.exp_map_to_axis_angle(exp_map)
    old_axis, old_angle = old_exp_map_to_axis_angle(exp_map)

    quat = torch_util.exp_map_to_quat(exp_map)
    old_quat = torch_util.axis_angle_to_quat(old_axis, old_angle)
    assert torch.all(torch.abs(torch.sum(quat * old_quat, dim=-1)) > 1.0 - 1e-6)
    assert torch.allclose(torch.norm(axis, dim=-1), torch.ones(4096), rtol=0, atol=1e-6)
    assert torch.all(torch.abs(angle) <= math.pi + 1e-6)


def test_axis_is_unit_for_angles_between_pi_and_2pi():
    # develop's variant clamped the wrapped (negative) angle and returned axes of length ~1e5 here
    theta = torch.linspace(math.pi + 0.01, 2.0 * math.pi - 0.01, 64)
    exp_map = torch_util.normalize(torch.tensor([0.3, -0.5, 0.8])) * theta.unsqueeze(-1)
    axis, angle = torch_util.exp_map_to_axis_angle(exp_map)
    assert torch.allclose(torch.norm(axis, dim=-1), torch.ones(64), rtol=0, atol=1e-6)
    assert torch.allclose(angle, theta - 2.0 * math.pi, rtol=0, atol=1e-5)


def test_gradient_unchanged_away_from_zero():
    exp_map = random_exp_maps(512, 4.0 * math.pi, seed=2)
    exp_map = exp_map[torch.norm(exp_map, dim=-1) > 1e-3]
    weights = torch.randn(exp_map.shape[0], 4, generator=torch.Generator().manual_seed(3))

    new_in = exp_map.clone().requires_grad_(True)
    (new_grad,) = torch.autograd.grad((torch_util.exp_map_to_quat(new_in) * weights).sum(), new_in)

    old_in = exp_map.clone().requires_grad_(True)
    old_quat = torch_util.axis_angle_to_quat(*old_exp_map_to_axis_angle(old_in))
    (old_grad,) = torch.autograd.grad((old_quat * weights).sum(), old_in)

    assert torch.isfinite(new_grad).all()
    torch.testing.assert_close(new_grad, old_grad, rtol=1e-5, atol=1e-6)
