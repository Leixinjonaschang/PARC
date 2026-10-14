import copy
import pathlib
import pickle
import warnings
from typing import Optional

import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.file_io_helper as file_io_helper
import parc.util.geom_util as geom_util
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util

# A library of functions that edits motion data
# for simplicity assume everything is a torch tensor (so we can use other torch util functions)

class MotionData:
    def __init__(self, motion_data, device="cpu"):
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        self._data = motion_data
        self._device = device
        if "frames" in self._data and not isinstance(self._data["frames"], torch.Tensor):
            self._data["frames"] = torch.tensor(self._data["frames"], dtype=torch.float32, device=device)
            
        if "contacts" in self._data and not isinstance(self._data["contacts"], torch.Tensor):
            self._data["contacts"] = torch.tensor(self._data["contacts"], dtype=torch.float32, device=device)

        if "terrain" in self._data:
            if self._data["terrain"] is None:
                del self._data["terrain"]
            else:
                self._data["terrain"].update_old()
                self._data["terrain"].to_torch(device)

        if "path_nodes" in self._data:
            if isinstance(self._data["path_nodes"], np.ndarray):
                self._data["path_nodes"] = torch.tensor(self._data["path_nodes"], dtype=torch.float32, device=device)
            elif isinstance(self._data["path_nodes"], torch.Tensor):
                self._data["path_nodes"] = self._data["path_nodes"].to(device=device)
            else:
                assert False, "path_nodes format is wrong"

        # fix fps
        if "fps" not in self._data:
            self._data["fps"] = 30
        if "loop_mode" not in self._data:
            self._data["loop_mode"] = "CLAMP"
        if self.get_fps() > 29 and self.get_fps() < 31:
            self.set_fps(30)

        if self.has_opt_body_constraints():
            self._data["opt:body_constraints"] = file_io_helper.torchify_body_constraints(self._data["opt:body_constraints"], device)
        return
    
    def set_hf_mask_inds_device(self, device):
        for t in range(len(self._data["hf_mask_inds"])):
            self._data["hf_mask_inds"][t] = self._data["hf_mask_inds"][t].to(device=device)
        return
    
    def set_device(self, device):
        self._device = device
        for key in self._data:
            if key == "fps" or key == "loop_mode" or "target_xy":
                continue

            if key == "hf_mask_inds":
                self.set_hf_mask_inds_device(device)

            elif key == "terrain":
                self._data[key].set_device(device)
            else:
                print(key, device)
                self._data[key] = self._data[key].to(device=device)
        return

    def get_fps(self):
        return self._data["fps"]
    
    def set_fps(self, fps):
        self._data["fps"] = int(fps)
        return

    def get_loop_mode(self):
        return self._data["loop_mode"]
    
    def get_frames(self):
        return self._data["frames"]
    
    def set_frames(self, motion_frames):
        self._data["frames"] = motion_frames
        return
    
    def has_contacts(self):
        return "contacts" in self._data

    def get_contacts(self):
        assert "contacts" in self._data
        return self._data["contacts"]
    
    def set_contacts(self, contacts):
        self._data["contacts"] = contacts
        return
    
    def set_hf_mask_inds(self, hf_mask_inds):
        self._data["hf_mask_inds"] = hf_mask_inds
        return
    
    def has_hf_mask_inds(self):
        return "hf_mask_inds" in self._data
    
    def get_hf_mask_inds(self):
        assert "hf_mask_inds" in self._data
        return self._data["hf_mask_inds"]

    def has_terrain(self):
        return "terrain" in self._data

    def get_terrain(self) -> terrain_util.SubTerrain:
        assert "terrain" in self._data
        return self._data["terrain"]
    
    def set_terrain(self, terrain):
        self._data["terrain"] = terrain
        return

    def remove_terrain(self):
        del self._data["terrain"]
        return

    def has_opt_body_constraints(self):
        return "opt:body_constraints" in self._data

    def get_opt_body_constraints(self):
        assert "opt:body_constraints" in self._data
        return self._data["opt:body_constraints"]
    
    def set_opt_body_constraints(self, body_constraints):
        self._data["opt:body_constraints"] = body_constraints
        return
    
    def remove_opt_body_constraints(self):
        del self._data["opt:body_constraints"]
        return

    def save_to_file(self, motion_filepath, verbose=True):
        kwargs = {}
        if "min_point_offset" in self._data:
            kwargs["min_point_offset"] = self._data["min_point_offset"]

        if "path_nodes" in self._data:
            kwargs["path_nodes"] = self._data["path_nodes"]

        if "hf_mask_inds" in self._data:
            kwargs["hf_mask_inds"] = self._data["hf_mask_inds"]

        if verbose:
            print("Saving to", motion_filepath)

        contacts = None if not self.has_contacts() else self.get_contacts()
        terrain = None if not self.has_terrain() else self.get_terrain()


        save_motion_data(motion_filepath=motion_filepath, 
                         motion_frames=self.get_frames(),
                         contact_frames=contacts,
                         terrain=terrain,
                         fps=self.get_fps(),
                         loop_mode=self.get_loop_mode(),
                         **kwargs)
        return

def load_motion_file(motion_filepath, device="cpu", convert_to_class=True):
    warnings.warn(
        "load_motion_file is deprecated and will be removed in a future release. "
        "Use new_function() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ext = pathlib.Path(motion_filepath).suffix
    assert ext == ".pkl"
    with open(motion_filepath, "rb") as filestream:
        motion_data = pickle.load(filestream)
    if not convert_to_class:
        return motion_data
    return MotionData(motion_data, device=device)

def save_motion_data(motion_filepath, motion_frames, contact_frames, 
                     terrain: Optional[terrain_util.SubTerrain], fps: int, loop_mode: str,
                     **kwargs):
    warnings.warn(
        "save_motion_data is deprecated and will be removed in a future release. "
        "Use new_function() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    data = dict()

    if isinstance(motion_frames, torch.Tensor):
        motion_frames = motion_frames.cpu().numpy().astype(np.float32)

    if motion_frames is not None:
        data["frames"] = motion_frames

    if contact_frames is not None:
        if isinstance(contact_frames, torch.Tensor):
            contact_frames = contact_frames.cpu().numpy().astype(np.float32)
        data["contacts"] = contact_frames

    if terrain is not None:
        if isinstance(terrain.hf, torch.Tensor):
            terrain = terrain.numpy_copy()

        data["terrain"] = terrain

    if fps is not None:
        data["fps"] = fps

    if loop_mode is not None:
        data["loop_mode"] = loop_mode

    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.cpu().numpy()
        elif key == "opt:body_constraints":
            save_data = moopt.body_constraint_to_numpy_recursive(value)
            data[key] = save_data
        else:
            data[key] = value

    with open(motion_filepath, "wb") as f:
        pickle.dump(data, f)
    return

def create_terrain_for_motion(motion_frames: motion_util.MotionFrames,
                              char_model: kin_char_model.KinCharModel,
                              char_points,
                              dx=0.1,
                              padding=5.0):
    
    num_padding = int(round(padding/dx))

    root_pos = motion_frames.root_pos

    max_root_x = torch.max(root_pos[:, 0]).item()
    max_root_y = torch.max(root_pos[:, 1]).item()
    min_root_x = torch.min(root_pos[:, 0]).item()
    min_root_y = torch.min(root_pos[:, 1]).item()
    # first frame root xy is (0, 0)

    num_pos_x = int(abs(round(max_root_x / dx))) + num_padding
    num_neg_x = int(abs(round(min_root_x / dx))) + num_padding
    num_pos_y = int(abs(round(max_root_y / dx))) + num_padding
    num_neg_y = int(abs(round(min_root_y / dx))) + num_padding

    # Now we will deterministically construct a heightfield, so that we can use a heightfield
    # contact function to compute foot contacts and fix foot penetrations
    #hf, hf_mask = terrain_util.hf_from_motion(motion_frames, 
    #                                        min_height=0.0, ground_height=0.0, 
    #                                        dx=dx, char_model=char_model, canon_idx=0,
    #                                     num_neg_x=num_neg_x, num_pos_x=num_pos_x,
    #                                     num_neg_y=num_neg_y, num_pos_y=num_pos_y,
    #                                     floor_heights=platform_heights)
    # # 3x3 maxpool filter
    # #maxpool = torch.nn.MaxPool2d(kernel_size=[3,9], stride=1, padding=[1,4])
    # if maxpool_filter:
    #     character_box_size = 1.0#0.5
    #     kernel_size = int(character_box_size / dx)

    #     maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    #     hf = maxpool(hf.unsqueeze(dim=0)).squeeze(dim=0)
    #     hf = hf.squeeze(0)
    #     #hf_mask = maxpool(hf_mask.float().unsqueeze(0)).squeeze(0).squeeze(0).to(dtype=torch.bool)

    grid_dim_x = num_neg_x + num_pos_x + 1
    grid_dim_y = num_neg_y + num_pos_y + 1
    #low_ind_bound = torch.tensor([0, 0], dtype=torch.int64, device=device)
    #high_ind_bound = torch.tensor([grid_dim_x, grid_dim_y], dtype=torch.int64, device=device) - 1

    hf = torch.zeros(size=(grid_dim_x, grid_dim_y), dtype=torch.float32, device=root_pos.device)

    min_x = root_pos[0, 0] - dx * num_neg_x
    min_y = root_pos[0, 1] - dx * num_neg_y
    min_point = torch.stack([min_x, min_y])

    terrain = terrain_util.SubTerrain(x_dim=hf.shape[0], y_dim=hf.shape[1],
                                      dx=dx, dy=dx, min_x=min_point[0].item(), min_y=min_point[1].item(),
                                      device="cpu")
    terrain.hf = hf
    return terrain

def rotate_motion(motion_frames: motion_util.MotionFrames, 
                  rot_quat: torch.Tensor, 
                  origin: torch.Tensor):
    motion_frames.assert_num_dims(2)
    assert len(rot_quat.shape) == 1 and rot_quat.shape[0] == 4
    assert len(origin.shape) == 1 and origin.shape[0] == 3
    root_pos = motion_frames.root_pos
    root_rot = motion_frames.root_rot

    local_root_pos = root_pos - origin
    new_local_root_pos = torch_util.quat_rotate(rot_quat.unsqueeze(0), local_root_pos)
    new_root_pos = new_local_root_pos + origin

    new_root_rot = torch_util.quat_multiply(rot_quat, root_rot)

    new_motion_frames = motion_frames.get_copy()
    new_motion_frames.root_pos = new_root_pos
    new_motion_frames.root_rot = new_root_rot

    return new_motion_frames

def move_xy_root_to_origin(motion_frames: motion_util.MotionFrames):
    motion_frames.assert_num_dims(2)
    translation = -motion_frames.root_pos[0, 0:3]
    translation[2] = 0.0
    ret_frames = motion_frames.get_copy()
    ret_frames.root_pos += translation
    return ret_frames

def flip_rotation_about_XZ_plane(exp_map):
    # reflect vector about XZ plane
    exp_map[1] *= -1.0

    # change direction of rotation
    exp_map[:] *= -1.0
    return

def flip_quat_about_XZ_plane(quat):
    quat[1] *= -1.0
    quat[3] *= -1.0
    return

def flip_motion_about_XZ_plane(motion_frames: motion_util.MotionFrames, char_model: kin_char_model.KinCharModel):
    def swap(input_tensor, ind_1, ind_2):
        temp = copy.deepcopy(input_tensor[ind_1])
        input_tensor[ind_1] = input_tensor[ind_2]
        input_tensor[ind_2] = temp
        return

    def get_mirror_name(name, left_prefix="left_", right_prefix="right_"):
        if name.startswith(left_prefix):
            return right_prefix + name[len(left_prefix):]
        if name.startswith(right_prefix):
            return left_prefix + name[len(right_prefix):]
        return None

    def get_symmetric_pairs(names, indices=None):
        if indices is None:
            indices = range(len(names))
        name_to_idx = {n: i for n, i in zip(names, indices)}
        pairs = []
        for name, idx in name_to_idx.items():
            mirror_name = get_mirror_name(name)
            if mirror_name is None:
                continue
            if mirror_name not in name_to_idx:
                assert False, f"Missing symmetric counterpart for {name}"
            if name.startswith("left_"):
                pairs.append((idx, name_to_idx[mirror_name]))
        return pairs

    def get_joint_dof_slice(j):
        joint = char_model.get_joint(j)
        start = joint.dof_idx
        end = start + joint.get_dof_dim()
        return slice(start, end)

    joint_indices = list(range(1, char_model.get_num_joints()))
    joint_pairs = get_symmetric_pairs(
        [char_model.get_joint(j).name for j in joint_indices],
        indices=joint_indices,
    )
    contact_body_names = [
        char_model.get_contact_body_name(i)
        for i in range(char_model.get_num_contact_bodies())
    ]
    contact_pairs = get_symmetric_pairs(
        contact_body_names,
        indices=list(range(len(contact_body_names))),
    )
    contact_body_ids = [int(idx.item()) for idx in char_model.get_contact_body_ids()]

    new_motion_frames = motion_frames.get_copy()
    num_frames = new_motion_frames.root_pos.shape[0]
    new_motion_frames.assert_num_dims(2)
    for i in range(num_frames):
        root_pos = new_motion_frames.root_pos[i].clone()
        root_rot_exp = torch_util.quat_to_exp_map(new_motion_frames.root_rot[i])
        joint_rot = new_motion_frames.joint_rot[i].clone()

        root_pos[1] *= -1.0
        flip_rotation_about_XZ_plane(root_rot_exp)

        for b in range(joint_rot.shape[0]):
            flip_quat_about_XZ_plane(joint_rot[b])

        joint_dof = char_model.rot_to_dof(joint_rot)
        for left_idx, right_idx in joint_pairs:
            swap(joint_dof, get_joint_dof_slice(left_idx), get_joint_dof_slice(right_idx))

        new_motion_frames.root_pos[i] = root_pos
        new_motion_frames.root_rot[i] = torch_util.exp_map_to_quat(root_rot_exp)
        new_motion_frames.joint_rot[i] = char_model.dof_to_rot(joint_dof)

        if new_motion_frames.contacts is not None:
            contacts = new_motion_frames.contacts[i].clone()
            if contacts.shape[-1] == len(contact_body_ids):
                for left_idx, right_idx in contact_pairs:
                    swap(contacts, left_idx, right_idx)
            elif contacts.shape[-1] == char_model.get_num_bodies():
                for left_idx, right_idx in contact_pairs:
                    swap(contacts, contact_body_ids[left_idx], contact_body_ids[right_idx])
            new_motion_frames.contacts[i] = contacts

    new_motion_frames.body_pos = None
    new_motion_frames.body_rot = None

    return new_motion_frames
    
def compute_ground_plane_foot_contacts(motion_frames, char_model: kin_char_model.KinCharModel,
                                       contact_eps=0.04):
    device = char_model._device
    num_frames = motion_frames.shape[0]
    num_bodies = len(char_model._body_names)

    contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)

    num_frames = motion_frames.shape[0]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")

    key_ids = [lf_id, rf_id]

    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]

        # Assume only 1 geom for feet
        geom = char_model._geoms[body_id][0]
            
        geom_offset = geom._offset
        geom_dims = geom._dims

        box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)
        box_points_z = box_points[:, :, 2]
        
        # NOTE: we can extend this to grid heightfields in the future
        # will need to do parallel box/box collision detection
        gplane_contact = box_points_z - contact_eps < 0.0
        gplane_contact = torch.any(gplane_contact, dim=-1)
        
        contacts[:, body_id] = gplane_contact.to(dtype=torch.float32)

    return contacts

def compute_hf_foot_contacts_and_correct_pen(motion_frames, terrain: terrain_util.SubTerrain,
                             char_model: kin_char_model.KinCharModel, contact_eps=0.04):
    
    device = char_model._device
    num_frames = motion_frames.shape[0]
    num_bodies = len(char_model._body_names)

    contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)

    num_frames = motion_frames.shape[0]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")

    key_ids = [lf_id, rf_id]

    pen_correction_z = torch.zeros(size=(num_frames,))
    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]

        # Assume only 1 geom for feet
        geom = char_model._geoms[body_id][0]
            
        geom_offset = geom._offset
        geom_dims = geom._dims

        box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)
        
        # This method will check the grid indices of each of the box points,
        # then check if the points height is above the height of its grid index.
        # There are edge cases where this will give false positives, so fixing this is
        # a todo with ray box intersections

        grid_inds = terrain.get_grid_index(box_points[..., 0:2])
        hf = terrain.hf.unsqueeze(0).expand(num_frames, -1, -1)
        cell_heights = hf[torch.arange(0, num_frames, 1).unsqueeze(-1), grid_inds[..., 0], grid_inds[..., 1]]
        box_points_z = box_points[..., 2]
        
        floor_contact = torch.any(box_points_z < cell_heights + contact_eps, dim=-1)
        contacts[:, body_id] = floor_contact

        curr_pen_correction_z, _ = torch.min(box_points_z - cell_heights, dim=-1)
        pen_correction_z = torch.min(pen_correction_z, curr_pen_correction_z)

    updated_motion_frames = motion_frames.clone()
    updated_motion_frames[:, 2] -= pen_correction_z
    return updated_motion_frames, contacts

def compute_motion_terrain_hand_contacts(motion_frames, terrain: terrain_util.SubTerrain,
                          char_model: kin_char_model.KinCharModel, contact_eps=0.04):
    
    device = char_model._device
    num_frames = motion_frames.shape[0]
    num_bodies = len(char_model._body_names)

    contacts = torch.zeros(size=(num_frames, num_bodies), dtype=torch.float32, device=device)

    num_frames = motion_frames.shape[0]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    lh_id = char_model.get_body_id("left_hand")
    rh_id = char_model.get_body_id("right_hand")

    key_ids = [lh_id, rh_id]

    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        #key_body_rot = body_rot[:, body_id]

        geom = char_model._geoms[body_id][0]
        geom_offset = geom._offset
        geom_dims = geom._dims

        sd = terrain_util.points_hf_sdf(key_body_pos.unsqueeze(0), 
                                        terrain.hf.unsqueeze(0), 
                                        terrain.min_point.unsqueeze(0), 
                                        terrain.dxdy, 
                                        base_z=torch.min(terrain.hf).item() - 10.0,
                                        inverted=False,
                                        radius=geom_dims.item())
        
        contact = (sd[0] < contact_eps).to(dtype=torch.float32)
        contacts[:, body_id] = contact
    return contacts

def correct_foot_ground_pen(motion_frames, char_model: kin_char_model.KinCharModel,
                            ground_height = 0.0):
    num_frames = motion_frames.shape[0]
    root_pos, root_rot, joint_dof = motion_lib.extract_pose_data(motion_frames)
    root_rot_quat = torch_util.exp_map_to_quat(root_rot)
    joint_rot = char_model.dof_to_rot(joint_dof)

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")

    key_ids = [lf_id, rf_id]

    updated_motion_frames = motion_frames.clone()

    # Find the minimum foot box point with z < ground_height for all frames.
    # Then offset the motion by the negative of the minimum foot box points < ground_height

    min_point_z = torch.ones(size=(num_frames,)) * ground_height
    for body_id in key_ids:
        key_body_pos = body_pos[:, body_id]
        key_body_rot = body_rot[:, body_id]

        for geom in char_model._geoms[body_id]:
            
            geom_offset = geom._offset
            geom_dims = geom._dims

            box_points = geom_util.get_box_points_batch(key_body_pos, key_body_rot, geom_dims, geom_offset)

            box_points_z = box_points[:, :, 2]
            curr_geom_min_point_z, _ = torch.min(box_points_z, dim=1)
            min_point_z = torch.min(min_point_z, curr_geom_min_point_z)

    updated_motion_frames[:, 2] -= (min_point_z - ground_height)

    return updated_motion_frames

def search_for_matching_motion_frames(
        mlib_A: motion_lib.MotionLib,
        mlib_B: motion_lib.MotionLib,
        m_A_start_frame_idx: int,
        m_A_end_frame_idx: int,
        m_B_start_frame_idx: int,
        m_B_end_frame_idx: int):
    if mlib_A.num_motions() != 1 or mlib_B.num_motions() != 1:
        raise ValueError("search_for_matching_motion_frames expects MotionLib instances with a single motion loaded")

    fps_A = mlib_A.get_motion_fps(0)
    fps_B = mlib_B.get_motion_fps(0)
    if fps_A != fps_B:
        raise ValueError("Motion libraries must have the same FPS to compare frames")

    num_frames_A = mlib_A._motion_num_frames[0].item()
    num_frames_B = mlib_B._motion_num_frames[0].item()

    m_A_start_frame_idx = int(np.clip(m_A_start_frame_idx, 0, num_frames_A - 1))
    m_A_end_frame_idx = int(np.clip(m_A_end_frame_idx, 0, num_frames_A - 1))
    m_B_start_frame_idx = int(np.clip(m_B_start_frame_idx, 0, num_frames_B - 1))
    m_B_end_frame_idx = int(np.clip(m_B_end_frame_idx, 0, num_frames_B - 1))

    if m_A_start_frame_idx > m_A_end_frame_idx:
        m_A_start_frame_idx, m_A_end_frame_idx = m_A_end_frame_idx, m_A_start_frame_idx
    if m_B_start_frame_idx > m_B_end_frame_idx:
        m_B_start_frame_idx, m_B_end_frame_idx = m_B_end_frame_idx, m_B_start_frame_idx

    num_segment_frames_A = m_A_end_frame_idx - m_A_start_frame_idx + 1
    num_segment_frames_B = m_B_end_frame_idx - m_B_start_frame_idx + 1

    if num_segment_frames_A < 1 or num_segment_frames_B < 1:
        raise ValueError("Motion segment selections must contain at least one frame.")

    char_model_A = mlib_A._kin_char_model
    char_model_B = mlib_B._kin_char_model
    if char_model_A.get_num_joints() != char_model_B.get_num_joints():
        raise ValueError("Motion libraries must share the same character model to compare frames")

    device_A = mlib_A._device
    device_B = mlib_B._device

    frame_indices_A = torch.arange(
        m_A_start_frame_idx,
        m_A_end_frame_idx + 1,
        dtype=torch.float32,
        device=device_A,
    )
    frame_indices_B = torch.arange(
        m_B_start_frame_idx,
        m_B_end_frame_idx + 1,
        dtype=torch.float32,
        device=device_B,
    )

    motion_ids_A = torch.zeros_like(frame_indices_A, dtype=torch.long, device=device_A)
    motion_ids_B = torch.zeros_like(frame_indices_B, dtype=torch.long, device=device_B)

    motion_times_A = frame_indices_A / fps_A
    motion_times_B = frame_indices_B / fps_B

    root_pos_A, root_rot_A, root_vel_A, root_ang_vel_A, joint_rot_A, dof_vel_A, contacts_A = mlib_A.calc_motion_frame(
        motion_ids_A, motion_times_A
    )
    root_pos_B, root_rot_B, root_vel_B, root_ang_vel_B, joint_rot_B, dof_vel_B, contacts_B = mlib_B.calc_motion_frame(
        motion_ids_B, motion_times_B
    )

    body_pos_A, body_rot_A = char_model_A.forward_kinematics(root_pos_A, root_rot_A, joint_rot_A)
    body_pos_B, body_rot_B = char_model_B.forward_kinematics(root_pos_B, root_rot_B, joint_rot_B)

    if device_B != device_A:
        root_pos_B = root_pos_B.to(device_A)
        root_rot_B = root_rot_B.to(device_A)
        root_vel_B = root_vel_B.to(device_A)
        root_ang_vel_B = root_ang_vel_B.to(device_A)
        body_pos_B = body_pos_B.to(device_A)

    body_pos_A = body_pos_A.clone()
    body_pos_B = body_pos_B.clone()

    body_pos_A[:, :, 0:2] -= body_pos_A[:, 0:1, 0:2].clone()
    body_pos_B[:, :, 0:2] -= body_pos_B[:, 0:1, 0:2].clone()

    heading_inv_A = torch_util.calc_heading_quat_inv(root_rot_A)
    heading_inv_B = torch_util.calc_heading_quat_inv(root_rot_B)

    body_pos_A[:, :, 0:3] = torch_util.quat_rotate(heading_inv_A.unsqueeze(1), body_pos_A[:, :, 0:3])
    body_pos_B[:, :, 0:3] = torch_util.quat_rotate(heading_inv_B.unsqueeze(1), body_pos_B[:, :, 0:3])

    root_vel_A = torch_util.quat_rotate(heading_inv_A, root_vel_A)
    root_vel_B = torch_util.quat_rotate(heading_inv_B, root_vel_B)

    root_ang_vel_A = torch_util.quat_rotate(heading_inv_A, root_ang_vel_A)
    root_ang_vel_B = torch_util.quat_rotate(heading_inv_B, root_ang_vel_B)

    min_motion_match_err = float("inf")
    match_idx_A = m_A_start_frame_idx
    match_idx_B = m_B_start_frame_idx

    for i in range(num_segment_frames_A):
        body_pos_diff = torch.norm(body_pos_A[i:i + 1, 1:] - body_pos_B[:, 1:], dim=-1)
        body_pos_diff = torch.sum(body_pos_diff, dim=-1)

        root_vel_diff = torch.norm(root_vel_A[i] - root_vel_B, dim=-1)
        root_ang_vel_diff = torch.norm(root_ang_vel_A[i] - root_ang_vel_B, dim=-1)

        motion_match_err = body_pos_diff * 0.65 + root_vel_diff * 0.2 + root_ang_vel_diff * 0.15

        best_val, best_idx = torch.min(motion_match_err, dim=0)
        if best_val < min_motion_match_err:
            min_motion_match_err = best_val.item()
            match_idx_A = m_A_start_frame_idx + i
            match_idx_B = m_B_start_frame_idx + best_idx.item()

    match_time_A = torch.tensor([match_idx_A], dtype=torch.float32, device=device_A) / fps_A
    match_time_B = torch.tensor([match_idx_B], dtype=torch.float32, device=device_B) / fps_B

    motion_id_A = torch.zeros(1, dtype=torch.long, device=device_A)
    motion_id_B = torch.zeros(1, dtype=torch.long, device=device_B)

    root_pos_A, root_rot_A, *_ = mlib_A.calc_motion_frame(motion_id_A, match_time_A)
    root_pos_B, root_rot_B, *_ = mlib_B.calc_motion_frame(motion_id_B, match_time_B)

    if root_pos_B.device != root_pos_A.device:
        root_pos_B = root_pos_B.to(root_pos_A.device)
        root_rot_B = root_rot_B.to(root_rot_A.device)

    root_pos_A = root_pos_A.squeeze(0)
    root_rot_A = root_rot_A.squeeze(0)
    root_pos_B = root_pos_B.squeeze(0)
    root_rot_B = root_rot_B.squeeze(0)

    heading_A = torch_util.calc_heading(root_rot_A.unsqueeze(0))
    heading_B = torch_util.calc_heading(root_rot_B.unsqueeze(0))
    heading_diff = (heading_A - heading_B)[0]

    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=root_pos_A.dtype, device=root_pos_A.device)
    heading_diff_quat = torch_util.axis_angle_to_quat(z_axis, heading_diff.unsqueeze(0)).squeeze(0)

    rotated_root_pos_B = torch_util.quat_rotate(heading_diff_quat, root_pos_B)
    root_pos_diff = root_pos_A - rotated_root_pos_B
    root_pos_diff[2] = 0.0

    return match_idx_A, match_idx_B, heading_diff, root_pos_diff

def change_motion_fps(src_frames, src_fps, tar_fps, char_model: kin_char_model.KinCharModel):

    device = src_frames.device

    # assumes the first index is the number of motions index
    mlib = motion_lib.MotionLib(src_frames.unsqueeze(0), char_model, device, "motion_frames", loop_mode=motion_lib.LoopMode.CLAMP, fps=src_fps)

    tar_dt = 1.0 / tar_fps
    src_duration = mlib._motion_lengths[0].item()

    new_frames = []

    motion_ids = torch.tensor([0], dtype=torch.int64, device=device)

    t = 0.0
    while t < src_duration:

        t_th = torch.tensor([t], dtype=torch.float32, device=device)

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = mlib.calc_motion_frame(motion_ids, t_th)

        root_rot = torch_util.quat_to_exp_map(root_rot)
        joint_dof = mlib.joint_rot_to_dof(joint_rot)

        curr_frame = torch.cat([root_pos, root_rot, joint_dof], dim=-1).squeeze(0)
        new_frames.append(curr_frame)

        t += tar_dt


    new_frames = torch.stack(new_frames)

    return new_frames

def scale_motion_segment(motion_frames: motion_util.MotionFrames, 
                         scale: float, start_frame_idx: int, end_frame_idx: int):
    motion_frames.assert_num_dims(2)
    new_motion_frames = motion_frames.get_copy()
    root_pos = new_motion_frames.root_pos
    xy_disp = root_pos[end_frame_idx, 0:2] - root_pos[start_frame_idx, 0:2]

    new_xy_disp = scale * xy_disp

    xy_disp_ratio = torch.nan_to_num(new_xy_disp / xy_disp, nan=1.0)
    canon_xy = root_pos[start_frame_idx, 0:2].clone()
    root_pos[start_frame_idx:end_frame_idx+1, 0:2] -= canon_xy
    root_pos[start_frame_idx:end_frame_idx+1, 0:2] *= xy_disp_ratio
    root_pos[start_frame_idx:end_frame_idx+1, 0:2] += canon_xy

    root_pos[end_frame_idx+1:, 0:2] += new_xy_disp - xy_disp
    return new_motion_frames

def remove_hesitation_frames(motion_frames: motion_util.MotionFrames,
                             char_model: kin_char_model.KinCharModel,
                             hesitation_val = 0.15,
                             hesitation_min_seq_len = 4,
                             verbose=False):

    def find_consecutive_groups(int_set):
        if len(int_set) == 0:
            return []
        # Sort the set to process in ascending order
        sorted_list = sorted(int_set)
        
        # Initialize variables
        result = []
        current_group = [sorted_list[0]]
        
        # Iterate through the sorted list
        for i in range(1, len(sorted_list)):
            if sorted_list[i] == sorted_list[i - 1] + 1:
                # Add to the current group if consecutive
                current_group.append(sorted_list[i])
            else:
                # Otherwise, start a new group
                result.append(current_group)
                current_group = [sorted_list[i]]
        
        # Append the last group
        result.append(current_group)
        return result


    root_pos = motion_frames.root_pos
    root_rot = motion_frames.root_rot
    joint_rot = motion_frames.joint_rot

    # Find sequences of frames where the start and end pose are very similar
    # and there was minimal motion in between

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    # for each frame, do a brute force search to find similar future frames.
    num_frames = body_pos.shape[0]
    hesitation_frames = set()
    for i in range(num_frames):
        if i in hesitation_frames:
            continue
        for j in range(i+1, num_frames):
            body_pos_diff = body_pos[j] - body_pos[i]
            body_pos_dist = torch.linalg.norm(body_pos_diff)

            if body_pos_dist < hesitation_val:
                hesitation_frames.add(j)
    hesitation_groups = find_consecutive_groups(hesitation_frames)

    new_hesitation_groups = []
    hesitation_frames = []
    for group in hesitation_groups:
        if len(group) < hesitation_min_seq_len:
            continue
        new_hesitation_groups.append(group)
        hesitation_frames.extend(group)

    if verbose:
        print("HESITATION FRAMES")
        print(new_hesitation_groups)

    new_frame_idxs = []
    for i in range(num_frames):
        if i in hesitation_frames:
            continue
        new_frame_idxs.append(i)

    new_motion_frames = motion_frames.get_slice(new_frame_idxs)

    return new_motion_frames
