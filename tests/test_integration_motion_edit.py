"""P3 integration: motion_edit_lib.create_terrain_for_motion for motions away from the xy origin (CPU).

The grid is anchored at the first frame's root and used to be sized from the absolute root
positions (as if that root were at the origin), so motions elsewhere got a grid that was too big
or did not cover them.
"""
import os
import sys

import pytest
import torch

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import motiontools_synth as synth  # noqa: E402

import parc.util.motion_edit_lib as medit_lib  # noqa: E402
import parc.util.motion_util as motion_util  # noqa: E402

DX = 0.1
PADDING = 1.0


@pytest.fixture(scope="module")
def humanoid():
    return synth.load_char("humanoid")


def make_frames(start_xy):
    # forward 1.2 m, then back past the start; sideways drift
    phase = torch.linspace(0.0, 1.5 * torch.pi, 40)
    root_pos = torch.zeros(40, 3)
    root_pos[:, 0] = 1.2 * torch.sin(phase)
    root_pos[:, 1] = -0.7 * torch.sin(0.5 * phase)
    root_pos[:, 2] = 0.9
    root_pos[:, 0:2] += torch.tensor(start_xy)
    return motion_util.MotionFrames(root_pos=root_pos)


def upstream_grid(root_pos, dx, padding):
    # the previous sizing, correct when the first root is at the xy origin
    num_padding = int(round(padding / dx))
    num_pos_x = int(abs(round(torch.max(root_pos[:, 0]).item() / dx))) + num_padding
    num_neg_x = int(abs(round(torch.min(root_pos[:, 0]).item() / dx))) + num_padding
    num_pos_y = int(abs(round(torch.max(root_pos[:, 1]).item() / dx))) + num_padding
    num_neg_y = int(abs(round(torch.min(root_pos[:, 1]).item() / dx))) + num_padding
    dims = (num_neg_x + num_pos_x + 1, num_neg_y + num_pos_y + 1)
    min_point = (root_pos[0, 0].item() - dx * num_neg_x, root_pos[0, 1].item() - dx * num_neg_y)
    return dims, min_point


def test_terrain_for_motion_at_the_origin_is_unchanged(humanoid):
    frames = make_frames((0.0, 0.0))
    terrain = medit_lib.create_terrain_for_motion(frames, humanoid, char_points=None, dx=DX, padding=PADDING)
    dims, min_point = upstream_grid(frames.root_pos, DX, PADDING)
    assert tuple(terrain.hf.shape) == dims == (45, 28)
    assert terrain.min_point.tolist() == pytest.approx(min_point, abs=1e-6)


@pytest.mark.parametrize("start_xy", [(5.3, -2.7), (-40.0, 12.0)])
def test_terrain_for_motion_away_from_the_origin(humanoid, start_xy):
    at_origin = medit_lib.create_terrain_for_motion(make_frames((0.0, 0.0)), humanoid, None, dx=DX, padding=PADDING)
    frames = make_frames(start_xy)
    terrain = medit_lib.create_terrain_for_motion(frames, humanoid, char_points=None, dx=DX, padding=PADDING)

    # the same grid, moved with the motion
    assert terrain.hf.shape == at_origin.hf.shape
    assert terrain.min_point.tolist() == pytest.approx((at_origin.min_point + torch.tensor(start_xy)).tolist(),
                                                       abs=1e-4)
    # covers the motion with the padding on every side, and the first root is on a grid point
    root_xy = frames.root_pos[:, 0:2]
    margin = PADDING - DX
    assert (terrain.min_point <= root_xy.min(dim=0)[0] - margin).all()
    assert (terrain.get_max_point() >= root_xy.max(dim=0)[0] + margin).all()
    first_ij = (root_xy[0] - terrain.min_point) / terrain.dxdy
    assert torch.allclose(first_ij, torch.round(first_ij), atol=1e-3)
    assert not terrain.hf.any()
