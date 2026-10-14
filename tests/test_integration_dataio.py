"""P3 integration: MS files with optional sections left out (CPU).

- MotionFrames.from_ms_motion_data with body_contacts None;
- create_dataset skips files without terrain or motion data with a warning (as it skips empty
  classes, D10) instead of crashing.
"""
import os
import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, FIXTURE_DIR)
import motiontools_synth as synth  # noqa: E402

import parc.util.create_dataset as create_dataset  # noqa: E402
import parc.util.file_io as file_io  # noqa: E402
import parc.util.file_io_helper as file_io_helper  # noqa: E402
import parc.util.motion_util as motion_util  # noqa: E402


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


def test_create_dataset_skips_files_without_terrain_or_motion(tmp_path):
    char_model = synth.load_char("humanoid")
    frames, terrain = synth.make_motion(char_model)
    no_contacts = frames.get_copy("cpu")
    no_contacts.contacts = None
    walk_dir, jump_dir = tmp_path / "motions" / "walk", tmp_path / "motions" / "jump"
    walk_dir.mkdir(parents=True)
    jump_dir.mkdir(parents=True)
    save = file_io_helper.save_ms_file
    save(str(walk_dir / "with_terrain.pkl"), motion_frames=frames, fps=30, loop_mode="CLAMP", terrain=terrain)
    save(str(walk_dir / "no_contacts.pkl"), motion_frames=no_contacts, fps=30, loop_mode="CLAMP", terrain=terrain)
    save(str(walk_dir / "no_terrain.pkl"), motion_frames=frames, fps=30, loop_mode="CLAMP")
    save(str(jump_dir / "terrain_only.pkl"), terrain=terrain)  # the only file of its class
    skipped = {path: path.read_bytes() for path in (walk_dir / "no_terrain.pkl", jump_dir / "terrain_only.pkl")}

    save_path = tmp_path / "dataset.yaml"
    with pytest.warns(UserWarning) as record:
        create_dataset.create_dataset_yaml(
            folder_paths=[tmp_path / "motions"], save_path=save_path, char_filepath=str(synth.CHAR_FILES["humanoid"]),
            compute_preprocessing_data=True, override_old_hf_mask_inds=False, cut_some_classes_in_half=False,
            motion_classes_to_cut_in_half=[], max_terrain_dim_x=1001, max_terrain_dim_y=1001, min_num_frames=16)

    files = sorted(pathlib.Path(m["file"]).name for m in yaml.safe_load(save_path.read_text())["motions"])
    assert files == ["no_contacts.pkl", "with_terrain.pkl"]
    messages = [str(w.message) for w in record]
    assert any("no_terrain.pkl, it has no terrain" in m for m in messages)
    assert any("terrain_only.pkl, it has no motion data" in m for m in messages)
    assert any("motion class 'jump' has no usable motions" in m for m in messages)
    for path, data in skipped.items():
        assert path.read_bytes() == data  # not re-saved
    # the motions it kept were re-saved with their hf_mask_inds
    misc_data = file_io.load_ms_file(str(walk_dir / "no_contacts.pkl")).misc_data
    assert len(misc_data[file_io_helper.HF_MASK_INDS_KEY]) == frames.root_pos.shape[0]
