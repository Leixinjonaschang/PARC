"""Utilities for saving and loading motion data.

This module provides helpers to convert between the motion dictionary
format used throughout the project and the serialized on-disk
representation.  The on-disk format is a dictionary with the following
structure::

    {
        "motion_data": {
            "root_position": np.ndarray,
            "root_rotation": np.ndarray,
            "joint_dof": np.ndarray,
            "body_contacts": np.ndarray or None,
            "fps": int,
            "loop_mode": str,
        },
        "terrain_data": {
            "hf": np.ndarray,
            "min_point": np.ndarray,
            "dx": float,
            "hf_maxmin": np.ndarray,
        },
        "misc_data": {
            ... additional metadata ...
        },
    }

Each top level value is individually pickled before the container
itself is saved.  This makes it possible to load only a subset of the
stored data without having to unpickle the entire payload.

The dictionary format is used to make the motion data easily transferrable to other projects.

misc_data is a dictionary designed to hold a bunch of various miscellaneous data that is not core to the motion-terrain,
but can be helpful to store in the file for various visualization and optimization reasons.
These won't be as portable to other repos, so these are pickled separately from the motion and terrain data.

hf_mask_inds: Optional[List[np.ndarray]] # n * [various, 2]
min_point_offset: Optional[np.ndarray] # [2]
path_nodes: Optional[np.ndarray] # [num_nodes, 3]
opt:body_constraints: Optional[List]
obs
obs_shapes
"""
import os
import pickle
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)

import numpy as np

import parc.util.pickle_compat as pickle_compat

MOTION_DATA_KEY = "motion_data"
TERRAIN_DATA_KEY = "terrain_data"
MISC_DATA_KEY = "misc_data"
MS_CONTAINER_KEYS = (MOTION_DATA_KEY, TERRAIN_DATA_KEY, MISC_DATA_KEY)

# Top-level keys of develop-era motion pickles (motion_edit_lib.save_motion_data
# and robot_motion_preprocess): flat dicts, not MS containers.
LEGACY_MOTION_KEYS = ("frames", "terrain", "contacts", "fps", "loop_mode")
LEGACY_CONVERT_HINT = (
    "convert with scripts/convert_legacy_motions.py, e.g. "
    "python -m scripts.convert_legacy_motions --inputs <legacy dir> --outputs <new dir> --char-file <char xml>"
)


class MSFileFormatError(ValueError):
    """The file is not a readable MS container (for example a legacy motion pickle)."""


def is_ms_container(obj) -> bool:
    return isinstance(obj, dict) and any(key in obj for key in MS_CONTAINER_KEYS)


def is_legacy_motion_dict(obj) -> bool:
    return (isinstance(obj, dict) and not is_ms_container(obj)
            and any(key in obj for key in LEGACY_MOTION_KEYS))


@dataclass
class MSMotionData:
    root_pos: np.ndarray # [n, 3]
    root_rot: np.ndarray # [n, 4] quat (x, y, z, w)
    joint_rot: np.ndarray # [n, n_joints, 4] quat (x, y, z, w)
    body_contacts: Optional[np.ndarray] # [n, n_body]
    fps: int
    loop_mode: str

@dataclass
class MSTerrainData:
    hf: np.ndarray # [m, n] heightfield
    hf_maxmin: np.ndarray # [m, n, 2] heightfield max/min bounds (for data aug)
    min_point: np.ndarray # [2]
    dx: float

@dataclass
class MSFileData:
    motion_data: Optional[MSMotionData]
    terrain_data: Optional[MSTerrainData]
    misc_data: Optional[MutableMapping[str, Any]]

def save_ms_file(data: MSFileData, filepath):
    
    ext = os.path.splitext(filepath)[1]

    if ext == ".pkl":
        container = {
            MOTION_DATA_KEY: None if data.motion_data is None else pickle.dumps(asdict(data.motion_data)),
            TERRAIN_DATA_KEY: None if data.terrain_data is None else pickle.dumps(asdict(data.terrain_data)),
            MISC_DATA_KEY: None if data.misc_data is None else pickle.dumps(data.misc_data)
        }
        with open(filepath, "wb") as f:
            pickle.dump(container, f)
    elif ext == ".txt":
        container = {
            MOTION_DATA_KEY: pickle.dumps(asdict(data.motion_data)),
            TERRAIN_DATA_KEY: pickle.dumps(asdict(data.terrain_data)),
            MISC_DATA_KEY: pickle.dumps(data.misc_data)
        }
        with open(filepath, "w") as f:
            f.write(str(asdict(data)))
    else:
        assert False, "unknown extension"
    return
    
def _load_section(container, key, filepath):
    payload = container.get(key)
    if payload is None:
        return None
    try:
        return pickle_compat.loads(payload)
    except Exception as exc:
        raise MSFileFormatError("{}: cannot unpickle the '{}' section ({}: {})".format(
            filepath, key, type(exc).__name__, exc)) from exc


def load_ms_file(filepath) -> MSFileData:
    # pickle_compat remaps develop-era module paths and numpy._core, and loads
    # tensors pickled on any device onto the CPU.
    with open(filepath, "rb") as f:
        container = pickle_compat.load(f)

    if not is_ms_container(container):
        if is_legacy_motion_dict(container):
            raise MSFileFormatError("{}: legacy motion format (keys {}), not an MS file; {}".format(
                filepath, sorted(map(str, container.keys())), LEGACY_CONVERT_HINT))
        if isinstance(container, dict):
            found = "keys {}".format(sorted(map(str, container.keys())))
        else:
            found = "a {}".format(type(container).__name__)
        raise MSFileFormatError("{}: not an MS motion file (expected a dict with keys {}, found {})".format(
            filepath, list(MS_CONTAINER_KEYS), found))

    motion_data = _load_section(container, MOTION_DATA_KEY, filepath)
    terrain_data = _load_section(container, TERRAIN_DATA_KEY, filepath)
    misc_data = _load_section(container, MISC_DATA_KEY, filepath)

    if motion_data is not None:
        motion_data = MSMotionData(**motion_data)

    if terrain_data is not None:
        terrain_data = MSTerrainData(**terrain_data)

    return MSFileData(motion_data=motion_data, terrain_data=terrain_data, misc_data=misc_data)