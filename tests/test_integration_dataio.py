"""P3 integration: MS files with optional sections left out (CPU)."""
import numpy as np
import pytest
import torch

import parc.util.file_io as file_io
import parc.util.motion_util as motion_util


def ms_motion_data(num_frames=5, num_joints=14, num_bodies=15, contacts=True):
    joint_rot = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    joint_rot[..., 3] = 1.0
    root_rot = np.zeros((num_frames, 4), dtype=np.float64)
    root_rot[:, 3] = 1.0
    return file_io.MSMotionData(root_pos=np.arange(num_frames * 3, dtype=np.float64).reshape(num_frames, 3),
                                root_rot=root_rot, joint_rot=joint_rot,
                                body_contacts=np.ones((num_frames, num_bodies), dtype=np.float64) if contacts else None,
                                fps=30, loop_mode="CLAMP")


@pytest.mark.parametrize("contacts", [True, False])
def test_motion_frames_from_ms_motion_data(contacts):
    frames = motion_util.MotionFrames.from_ms_motion_data(ms_motion_data(contacts=contacts), device="cpu")
    assert frames.root_pos.dtype == torch.float32 and tuple(frames.root_pos.shape) == (5, 3)
    assert frames.root_rot.dtype == torch.float32 and tuple(frames.joint_rot.shape) == (5, 14, 4)
    if contacts:
        assert frames.contacts.dtype == torch.float32 and tuple(frames.contacts.shape) == (5, 15)
    else:
        assert frames.contacts is None  # body_contacts is optional in MS files
