from dataclasses import asdict
from typing import Any, List, MutableMapping, Optional, Union

import numpy as np
import torch

import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.file_io as file_io
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util

HF_MASK_INDS_KEY = "hf_mask_inds"
BODY_CONSTRAINTS_KEY = "opt:body_constraints"
MIN_POINT_OFFSET_KEY = "min_point_offset"
PATH_NODES_KEY = "path_nodes"
OBS_KEY = "obs"
OBS_SHAPES_KEY = "obs_shapes"
CAM_PARAMS_KEY = "cam_params"

def numpify_body_constraints(obj):
    if isinstance(obj, moopt.BodyConstraint):
        objdict = asdict(obj)
        for key, value in objdict.items():
            if isinstance(value, torch.Tensor):
                objdict[key] = value.cpu().numpy()
        return objdict
    elif isinstance(obj, list):
        return [numpify_body_constraints(x) for x in obj]
    else:
        assert False
    
def torchify_body_constraints(obj, device):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, torch.Tensor):
                obj[key] = torch.from_numpy(value).to(dtype=torch.float32, device=device)
        body_constraint = moopt.BodyConstraint(**obj)
        return body_constraint
    elif isinstance(obj, list):
        return [torchify_body_constraints(x, device) for x in obj]
    elif isinstance(obj, moopt.BodyConstraint):
        if isinstance(obj.constraint_point, torch.Tensor):
            obj.constraint_point = obj.constraint_point.to(device=device)
        elif isinstance(obj.constraint_point, np.ndarray):
            obj.constraint_point = torch.from_numpy(obj.constraint_point).to(device=device)
        else:
            assert False
    else:
        assert False

def numpify_hf_mask_inds(hf_mask_inds):
    np_hf_mask_inds = []
    for t in range(len(hf_mask_inds)):
        if isinstance(hf_mask_inds[t], np.ndarray):
            np_hf_mask_inds.append(hf_mask_inds[t])
        elif isinstance(hf_mask_inds[t], torch.Tensor):
            np_hf_mask_inds.append(hf_mask_inds[t].cpu().numpy())
        else:
            assert False, "incorrect hf_mask_inds type"

    return np_hf_mask_inds

def numpify_torch_tensor(val):
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    elif isinstance(val, np.ndarray):
        return val
    else:
        assert False, "unhandled type"

def save_ms_file_simple(filepath,
                        motion_frames: motion_util.MotionFrames,
                        terrain: terrain_util.SubTerrain,
                        ms_file_data: file_io.MSFileData):
    
    if motion_frames is not None:
        motion_frames.assert_num_dims(2)
        ms_file_data.motion_data.root_pos = motion_frames.root_pos.cpu().numpy()
        ms_file_data.motion_data.root_rot = motion_frames.root_rot.cpu().numpy()
        ms_file_data.motion_data.joint_rot = motion_frames.joint_rot.cpu().numpy()
        ms_file_data.motion_data.body_contacts = motion_frames.contacts.cpu().numpy()
    else:
        motion_data = None

    if terrain is not None:
        if isinstance(terrain.hf, torch.Tensor):
            terrain = terrain.numpy_copy()
        ms_file_data.terrain_data = file_io.MSTerrainData(hf = terrain.hf,
                                             hf_maxmin = terrain.hf_maxmin,
                                             min_point = terrain.min_point,
                                             dx = terrain.dxdy[0])
    
    file_io.save_ms_file(ms_file_data, filepath)
    
    return

def save_ms_file(filepath,
                 motion_frames: Optional[motion_util.MotionFrames],
                 fps: Optional[int],
                 loop_mode: Optional[str],
                 terrain: Optional[terrain_util.SubTerrain],
                 ## MISC DATA
                 hf_mask_inds = Union[List[torch.Tensor], List[np.ndarray], None],
                 body_constraints = Optional[List[moopt.BodyConstraint]],
                 min_point_offset = Union[torch.Tensor, np.ndarray, None],
                 path_nodes = Union[torch.Tensor, np.ndarray, None],
                 obs = None,
                 obs_shapes = None,
                 cam_params = None,
                 extra_misc_data: Optional[MutableMapping[str, Any]] = None
                 ):
    # Save helper which makes sure all torch tensors are converted to numpy first
    
    if motion_frames is not None:
        motion_frames.assert_num_dims(2)
        motion_data = file_io.MSMotionData(root_pos = motion_frames.root_pos.cpu().numpy(),
                                        root_rot = motion_frames.root_rot.cpu().numpy(),
                                        joint_rot = motion_frames.joint_rot.cpu().numpy(),
                                        body_contacts = motion_frames.contacts.cpu().numpy(),
                                        fps = fps,
                                        loop_mode = loop_mode)
    else:
        motion_data = None

    if terrain is not None:
        if isinstance(terrain.hf, torch.Tensor):
            terrain = terrain.numpy_copy()
        terrain_data = file_io.MSTerrainData(hf = terrain.hf,
                                             hf_maxmin = terrain.hf_maxmin,
                                             min_point = terrain.min_point,
                                             dx = terrain.dxdy[0])
    else:
        terrain_data = None
    
    misc_data = dict()

    if hf_mask_inds is not None:
        misc_data[HF_MASK_INDS_KEY] = numpify_hf_mask_inds(hf_mask_inds)

    if body_constraints is not None:
        misc_data[BODY_CONSTRAINTS_KEY] = numpify_body_constraints(body_constraints)

    if min_point_offset is not None:
        misc_data[MIN_POINT_OFFSET_KEY] = numpify_torch_tensor(min_point_offset)

    if path_nodes is not None:
        misc_data[PATH_NODES_KEY] = numpify_torch_tensor(path_nodes)

    if obs is not None:
        misc_data[OBS_KEY] = obs

    if obs_shapes is not None:
        misc_data[OBS_SHAPES_KEY] = obs_shapes

    if cam_params is not None:
        misc_data[CAM_PARAMS_KEY] = cam_params

    if extra_misc_data is not None:
        misc_data.update(extra_misc_data)

    file_data = file_io.MSFileData(motion_data = motion_data,
                                   terrain_data = terrain_data,
                                   misc_data = misc_data)
    file_io.save_ms_file(data=file_data, filepath=filepath)

    return

def load_ms_file(filepath: str, device="cpu") -> file_io.MSFileData:
    # Load helper which makes sure all tensors/arrays are converted to torch tensors on the target device
    file_data = file_io.load_ms_file(filepath)

    motion_data = file_data.motion_data
    if motion_data is not None:
        motion_data.root_pos = torch.from_numpy(motion_data.root_pos).to(device=device, dtype=torch.float32)
        motion_data.root_rot = torch.from_numpy(motion_data.root_rot).to(device=device, dtype=torch.float32)
        motion_data.joint_rot = torch.from_numpy(motion_data.joint_rot).to(device=device, dtype=torch.float32)
        if motion_data.body_contacts is not None:
            motion_data.body_contacts = torch.from_numpy(motion_data.body_contacts).to(device=device, dtype=torch.float32)

    terrain_data = file_data.terrain_data
    if terrain_data is not None:
        terrain_data.hf = torch.from_numpy(terrain_data.hf).to(device=device, dtype=torch.float32)
        terrain_data.hf_maxmin = torch.from_numpy(terrain_data.hf_maxmin).to(device=device, dtype=torch.float32)

    misc_data = file_data.misc_data
    if misc_data is not None:
        if HF_MASK_INDS_KEY in misc_data:
            hf_mask_inds = misc_data[HF_MASK_INDS_KEY]
            th_hf_mask_inds = []
            for t in range(len(hf_mask_inds)):
                if isinstance(hf_mask_inds[t], np.ndarray):
                    th_hf_mask_inds.append(torch.from_numpy(hf_mask_inds[t]).to(dtype=torch.int64, device=device))
                elif isinstance(hf_mask_inds[t], torch.Tensor):
                    th_hf_mask_inds.append(hf_mask_inds[t].to(dtype=torch.int64, device=device))
                else:
                    assert False, "incorrect hf_mask_inds type"
            misc_data[HF_MASK_INDS_KEY] = th_hf_mask_inds

        if BODY_CONSTRAINTS_KEY in misc_data:
           misc_data[BODY_CONSTRAINTS_KEY] = torchify_body_constraints(misc_data[BODY_CONSTRAINTS_KEY], device=device)

        if MIN_POINT_OFFSET_KEY in misc_data:
            min_point_offset = misc_data[MIN_POINT_OFFSET_KEY]
            if isinstance(min_point_offset, np.ndarray):
                misc_data[MIN_POINT_OFFSET_KEY] = torch.from_numpy(min_point_offset).to(dtype=torch.float32, device=device)
            elif isinstance(min_point_offset, torch.Tensor):
                misc_data[MIN_POINT_OFFSET_KEY] = min_point_offset.to(dtype=torch.float32, device=device)
            else:
                assert False, "incorrect min_point_offset type"

        if PATH_NODES_KEY in misc_data:
            path_nodes = misc_data[PATH_NODES_KEY]
            if isinstance(path_nodes, np.ndarray):
                misc_data[PATH_NODES_KEY] = torch.from_numpy(path_nodes).to(dtype=torch.float32, device=device)
            elif isinstance(path_nodes, torch.Tensor):
                misc_data[PATH_NODES_KEY] = path_nodes.to(dtype=torch.float32, device=device)
            else:
                assert False, "incorrect path_nodes type"

    return file_data