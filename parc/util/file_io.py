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

MOTION_DATA_KEY = "motion_data"
TERRAIN_DATA_KEY = "terrain_data"
MISC_DATA_KEY = "misc_data"

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
    
def load_ms_file(filepath) -> MSFileData:
    with open(filepath, "rb") as f:
        container = pickle.load(f)

    try:
        motion_data = pickle.loads(container[MOTION_DATA_KEY])
    except:
        print("Error loading motion data")
        motion_data = None

    try:
        terrain_data = pickle.loads(container[TERRAIN_DATA_KEY])
    except:
        terrain_data = None

    try:
        misc_data = pickle.loads(container[MISC_DATA_KEY])
    except:
        misc_data = None

    if motion_data is not None:
        motion_data = MSMotionData(**motion_data)

    if terrain_data is not None:
        terrain_data = MSTerrainData(**terrain_data)

    return MSFileData(motion_data=motion_data, terrain_data=terrain_data, misc_data=misc_data)