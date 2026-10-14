"""Helpers shared by scripts/run_optimize_motions.py and scripts/run_simple_augment_motions.py.

Paths in the motion-opt configs and motion lists may use $DATA_DIR (see
parc/util/path_loader.py). Motions are read from MS files; legacy (develop-era) motion
pickles must be converted first with scripts/convert_legacy_motions.py.
"""
import os
from typing import Optional, Tuple

import torch

import parc.anim.kin_char_model as kin_char_model
import parc.util.file_io as file_io
import parc.util.file_io_helper as file_io_helper
import parc.util.geom_util as geom_util
import parc.util.motion_util as motion_util
import parc.util.path_loader as path_loader
import parc.util.terrain_util as terrain_util


def load_config(config_path) -> dict:
    return path_loader.load_config(path_loader.resolve_path(config_path))


def resolve(path) -> str:
    return str(path_loader.resolve_path(path))


def fetch_motion_files(motion_file):
    """Returns ([motion file], [weight]) for one MS file or a motions yaml (`motions: [{file, weight}]`)."""
    ext = os.path.splitext(str(motion_file))[1]
    if ext == ".yaml":
        motion_config = load_config(motion_file)
        motion_files = []
        motion_weights = []
        for motion_entry in motion_config["motions"]:
            curr_weight = motion_entry.get("weight", 1.0)
            assert curr_weight >= 0
            motion_files.append(resolve(motion_entry["file"]))
            motion_weights.append(curr_weight)
    else:
        motion_files = [resolve(motion_file)]
        motion_weights = [1.0]
    return motion_files, motion_weights


def load_char_model(char_file, device) -> kin_char_model.KinCharModel:
    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(resolve(char_file))
    return char_model


def load_motion(filepath, char_model: kin_char_model.KinCharModel, device
                ) -> Tuple[motion_util.MotionFrames, Optional[terrain_util.SubTerrain], dict, int]:
    """Loads an MS motion file as (MotionFrames [num_frames], terrain or None, misc_data, fps).

    Contacts are per contact body (zeros if the file has none). Raises
    file_io.MSFileFormatError for legacy files (the message says how to convert them) and
    ValueError if the motion does not fit ``char_model``.
    """
    file_data = file_io_helper.load_ms_file(resolve(filepath), device=device)
    motion_data = file_data.motion_data
    if motion_data is None:
        raise ValueError("{}: no motion data (a terrain-only MS file?)".format(filepath))

    num_frames = motion_data.root_pos.shape[0]
    num_joints = char_model.get_num_joints() - 1
    if motion_data.joint_rot.shape[1] != num_joints:
        raise ValueError("{}: the motion has {} joint rotations but the character model has {} non-root joints; "
                         "is char_model the character this motion was made for?".format(
                             filepath, motion_data.joint_rot.shape[1], num_joints))

    contacts = motion_data.body_contacts
    num_contact_bodies = char_model.get_num_contact_bodies()
    if contacts is None:
        contacts = torch.zeros(size=[num_frames, num_contact_bodies], dtype=torch.float32, device=device)
    elif contacts.shape[-1] != num_contact_bodies:
        raise ValueError("{}: the motion has contacts for {} bodies but the character model has {} contact bodies".format(
            filepath, contacts.shape[-1], num_contact_bodies))

    frames = motion_util.MotionFrames(root_pos=motion_data.root_pos, root_rot=motion_data.root_rot,
                                      joint_rot=motion_data.joint_rot, contacts=contacts)

    terrain = None
    if file_data.terrain_data is not None:
        terrain = terrain_util.SubTerrain.from_ms_terrain_data(file_data.terrain_data, device=device)

    misc_data = file_data.misc_data if file_data.misc_data is not None else dict()
    return frames, terrain, misc_data, motion_data.fps


def get_body_points(char_model: kin_char_model.KinCharModel, char_point_sample_cfg: dict):
    return geom_util.get_char_point_samples(
        char_model,
        sphere_num_subdivisions=char_point_sample_cfg["sphere_num_subdivisions"],
        box_num_slices=char_point_sample_cfg["box_num_slices"],
        box_dim_x=char_point_sample_cfg["box_dim_x"],
        box_dim_y=char_point_sample_cfg["box_dim_y"],
        capsule_num_circle_points=char_point_sample_cfg["capsule_num_circle_points"],
        capsule_num_sphere_subdivisons=char_point_sample_cfg["capsule_num_sphere_subdivisions"],
        capsule_num_cylinder_slices=char_point_sample_cfg["capsule_num_cylinder_slices"])
