from typing import List

import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.util.file_io as file_io
import parc.util.torch_util as torch_util


class MotionFrames:
    def __init__(self, 
                 root_pos: torch.Tensor = None, 
                 root_rot: torch.Tensor = None,
                 joint_rot: torch.Tensor = None, 
                 body_pos: torch.Tensor = None, 
                 body_rot: torch.Tensor = None, 
                 contacts: torch.Tensor = None):
        self.root_pos = root_pos
        self.root_rot = root_rot
        self.joint_rot = joint_rot
        self.body_pos = body_pos
        self.body_rot = body_rot
        self.contacts = contacts
        return
    
    @classmethod
    def from_ms_motion_data(cls, motion_data: file_io.MSMotionData, device="cpu"):
        root_pos = motion_data.root_pos
        root_rot = motion_data.root_rot
        joint_rot = motion_data.joint_rot
        contacts = motion_data.body_contacts

        if isinstance(root_pos, np.ndarray):
            root_pos = torch.from_numpy(root_pos)
        if isinstance(root_rot, np.ndarray):
            root_rot = torch.from_numpy(root_rot)
        if isinstance(joint_rot, np.ndarray):
            joint_rot = torch.from_numpy(joint_rot)
        if isinstance(contacts, np.ndarray):
            contacts = torch.from_numpy(contacts)
        
        root_pos = root_pos.to(device=device, dtype=torch.float32)
        root_rot = root_rot.to(device=device, dtype=torch.float32)
        joint_rot = joint_rot.to(device=device, dtype=torch.float32)
        contacts = contacts.to(device=device, dtype=torch.float32)

        inst = cls(root_pos = root_pos,
                   root_rot = root_rot,
                   joint_rot = joint_rot,
                   contacts = contacts)
        return inst
    
    def init_blank_frames(self, char_model: kin_char_model.KinCharModel, history_length: int, batch_size=1):
        device = char_model._device
        num_joints = char_model.get_num_joints()
        self.root_pos = torch.zeros(size=[batch_size, history_length, 3], dtype=torch.float32, device=device)
        self.root_rot = torch.zeros(size=[batch_size, history_length, 4], dtype=torch.float32, device=device)
        self.root_rot[..., 3] = 1.0
        self.joint_rot = torch.zeros(size=[batch_size, history_length, num_joints-1, 4], dtype=torch.float32, device=device)
        self.joint_rot[..., 3] = 1.0
        self.body_pos = torch.zeros(size=[batch_size, history_length, num_joints, 3], dtype=torch.float32, device=device)
        self.body_rot = torch.zeros(size=[batch_size, history_length, num_joints, 4], dtype=torch.float32, device=device)
        self.contacts = torch.zeros(size=[batch_size, history_length, num_joints], dtype=torch.float32, device=device)
        return
    
    def get_legacy_mlib_format(self, char_model: kin_char_model.KinCharModel):
        mlib_motion_frames = torch.cat([self.root_pos, torch_util.quat_to_exp_map(self.root_rot),
                                        char_model.rot_to_dof(self.joint_rot)], dim=-1)
        mlib_contact_frames = self.contacts
        return mlib_motion_frames, mlib_contact_frames
    
    def get_slice(self, in_slice):
        if len(self.root_pos.shape) == 3: # has batch dimension
            ret = MotionFrames(
                root_pos = None if self.root_pos is None else self.root_pos[:, in_slice],
                root_rot = None if self.root_rot is None else self.root_rot[:, in_slice],
                joint_rot = None if self.joint_rot is None else self.joint_rot[:, in_slice],
                body_pos = None if self.body_pos is None else self.body_pos[:, in_slice],
                body_rot = None if self.body_rot is None else self.body_rot[:, in_slice],
                contacts = None if self.contacts is None else self.contacts[:, in_slice]
            )
        elif len(self.root_pos.shape) == 2: # no batch dimension
            ret = MotionFrames(
                root_pos = None if self.root_pos is None else self.root_pos[in_slice],
                root_rot = None if self.root_rot is None else self.root_rot[in_slice],
                joint_rot = None if self.joint_rot is None else self.joint_rot[in_slice],
                body_pos = None if self.body_pos is None else self.body_pos[in_slice],
                body_rot = None if self.body_rot is None else self.body_rot[in_slice],
                contacts = None if self.contacts is None else self.contacts[in_slice]
            )
        else:
            assert False, "tensors have unsupported shape: " + str(self.root_pos.shape)
        return ret
    
    def set_slice(self, other: 'MotionFrames', in_slice):
        if self.root_pos is not None and other.root_pos is not None:
            self.root_pos[..., in_slice, :] = other.root_pos
        if self.root_rot is not None and other.root_rot is not None:
            self.root_rot[..., in_slice, :] = other.root_rot
        if self.joint_rot is not None and other.joint_rot is not None:
            self.joint_rot[..., in_slice, :] = other.joint_rot
        if self.body_pos is not None and other.body_pos is not None:
            self.body_pos[..., in_slice, :] = other.body_pos
        if self.body_rot is not None and other.body_rot is not None:
            self.body_rot[..., in_slice, :] = other.body_rot
        if self.contacts is not None and other.contacts is not None:
            self.contacts[..., in_slice, :] = other.contacts
        return
    
    def unsqueeze(self, dim):
        ret = MotionFrames(
            root_pos = None if self.root_pos is None else self.root_pos.unsqueeze(dim),
            root_rot = None if self.root_rot is None else self.root_rot.unsqueeze(dim),
            joint_rot = None if self.joint_rot is None else self.joint_rot.unsqueeze(dim),
            body_pos = None if self.body_pos is None else self.body_pos.unsqueeze(dim),
            body_rot = None if self.body_rot is None else self.body_rot.unsqueeze(dim),
            contacts = None if self.contacts is None else self.contacts.unsqueeze(dim)
        )
        return ret
    
    def squeeze(self, dim):
        ret = MotionFrames(
            root_pos = None if self.root_pos is None else self.root_pos.squeeze(dim),
            root_rot = None if self.root_rot is None else self.root_rot.squeeze(dim),
            joint_rot = None if self.joint_rot is None else self.joint_rot.squeeze(dim),
            body_pos = None if self.body_pos is None else self.body_pos.squeeze(dim),
            body_rot = None if self.body_rot is None else self.body_rot.squeeze(dim),
            contacts = None if self.contacts is None else self.contacts.squeeze(dim)
        )
        return ret
    
    def expand_first_dim(self, b):
        ret = MotionFrames(
            root_pos = None if self.root_pos is None else self.root_pos.expand(b, *self.root_pos.shape[1:]),
            root_rot = None if self.root_rot is None else self.root_rot.expand(b, *self.root_rot.shape[1:]),
            joint_rot = None if self.joint_rot is None else self.joint_rot.expand(b, *self.joint_rot.shape[1:]),
            body_pos = None if self.body_pos is None else self.body_pos.expand(b, *self.body_pos.shape[1:]),
            body_rot = None if self.body_rot is None else self.body_rot.expand(b, *self.body_rot.shape[1:]),
            contacts = None if self.contacts is None else self.contacts.expand(b, *self.contacts.shape[1:])
        )
        return ret
    
    def get_idx(self, idx):
        ret = MotionFrames(
            root_pos = None if self.root_pos is None else self.root_pos[idx],
            root_rot = None if self.root_rot is None else self.root_rot[idx],
            joint_rot = None if self.joint_rot is None else self.joint_rot[idx],
            body_pos = None if self.body_pos is None else self.body_pos[idx],
            body_rot = None if self.body_rot is None else self.body_rot[idx],
            contacts = None if self.contacts is None else self.contacts[idx]
        )
        return ret
    
    def set_vals(self, other, ids):

        if self.root_pos is not None:
            self.root_pos[ids] = other.root_pos[ids].clone()
        if self.root_rot is not None:
            self.root_rot[ids] = other.root_rot[ids].clone()
        if self.joint_rot is not None:
            self.joint_rot[ids] = other.joint_rot[ids].clone()
        if self.body_pos is not None:
            self.body_pos[ids] = other.body_pos[ids].clone()
        if self.body_rot is not None:
            self.body_rot[ids] = other.body_rot[ids].clone()
        if self.contacts is not None:
            self.contacts[ids] = other.contacts[ids].clone()

        return
    
    def get_copy(self, new_device=None):
        if new_device is None:
            new_device = self.root_pos.device
        ret = MotionFrames(
            root_pos = None if self.root_pos is None else self.root_pos.clone().to(device=new_device),
            root_rot = None if self.root_rot is None else self.root_rot.clone().to(device=new_device),
            joint_rot = None if self.joint_rot is None else self.joint_rot.clone().to(device=new_device),
            body_pos = None if self.body_pos is None else self.body_pos.clone().to(device=new_device),
            body_rot = None if self.body_rot is None else self.body_rot.clone().to(device=new_device),
            contacts = None if self.contacts is None else self.contacts.clone().to(device=new_device)
        )
        return ret
    
    def set_device(self, device):
        if self.root_pos is not None:
            self.root_pos = self.root_pos.to(device=device)
        if self.root_rot is not None:
            self.root_rot = self.root_rot.to(device=device)
        if self.joint_rot is not None:
            self.joint_rot = self.joint_rot.to(device=device)
        if self.body_pos is not None:
            self.body_pos = self.body_pos.to(device=device)
        if self.body_rot is not None:
            self.body_rot = self.body_rot.to(device=device)
        if self.contacts is not None:
            self.contacts = self.contacts.to(device=device)
        return
    
    def assert_num_dims(self, num_dims):
        if self.root_pos is not None:
            assert len(self.root_pos.shape) == num_dims
        if self.root_rot is not None:
            assert len(self.root_rot.shape) == num_dims
        if self.joint_rot is not None:
            assert len(self.joint_rot.shape) == num_dims + 1 # extra dim for joints
        if self.body_pos is not None:
            assert len(self.body_pos.shape) == num_dims + 1
        if self.body_rot is not None:
            assert len(self.body_rot.shape) == num_dims + 1
        if self.contacts is not None:
            assert len(self.contacts.shape) == num_dims # extra dim for joints, -1 dim since last dim is 0d
        return

def cat_motion_frames(motion_frames_list: List[MotionFrames]):
    # concat on the sequence dimension
    
    if len(motion_frames_list[0].root_pos.shape) == 3:
        num_dims = 3
        cat_dim = 1
    elif len(motion_frames_list[0].root_pos.shape) == 2:
        num_dims = 2
        cat_dim = 0
    else:
        assert False

    root_pos = [] if motion_frames_list[0].root_pos is not None else None
    root_rot = [] if motion_frames_list[0].root_rot is not None else None
    joint_rot = [] if motion_frames_list[0].joint_rot is not None else None
    body_pos = [] if motion_frames_list[0].body_pos is not None else None
    body_rot = [] if motion_frames_list[0].body_rot is not None else None
    contacts = [] if motion_frames_list[0].contacts is not None else None

    for elem in motion_frames_list:
        elem.assert_num_dims(num_dims)
        if root_pos is not None:
            root_pos.append(elem.root_pos)

        if root_rot is not None:
            root_rot.append(elem.root_rot)

        if joint_rot is not None:
            joint_rot.append(elem.joint_rot)

        if body_pos is not None:
            body_pos.append(elem.body_pos)

        if body_rot is not None:
            body_rot.append(elem.body_rot)

        if contacts is not None:
            contacts.append(elem.contacts)

    if root_pos is not None:
        root_pos = torch.cat(root_pos, dim=cat_dim)

    if root_rot is not None:
        root_rot = torch.cat(root_rot, dim=cat_dim)

    if joint_rot is not None:
        joint_rot = torch.cat(joint_rot, dim=cat_dim)

    if body_pos is not None:
        body_pos = torch.cat(body_pos, dim=cat_dim)

    if body_rot is not None:
        body_rot = torch.cat(body_rot, dim=cat_dim)

    if contacts is not None:
        contacts = torch.cat(contacts, dim=cat_dim)
        
    ret = MotionFrames(root_pos, root_rot, joint_rot, body_pos, body_rot, contacts)
    return ret

def motion_frames_from_mlib_format(mlib_motion_frames, char_model: kin_char_model.KinCharModel,
                                   contacts=None):

    root_pos = mlib_motion_frames[..., 0:3]
    root_rot = torch_util.exp_map_to_quat(mlib_motion_frames[..., 3:6])
    joint_rot = char_model.dof_to_rot(mlib_motion_frames[..., 6:])

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)


    if contacts is not None:
            ret = MotionFrames(
            root_pos=root_pos,
            root_rot=root_rot,
            joint_rot=joint_rot,
            body_pos=body_pos,
            body_rot=body_rot,
            contacts=contacts
        )
    else:
        ret = MotionFrames(
            root_pos=root_pos,
            root_rot=root_rot,
            joint_rot=joint_rot,
            body_pos=body_pos,
            body_rot=body_rot
        )
    return ret