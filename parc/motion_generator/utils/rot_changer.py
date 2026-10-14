import enum

import torch

import parc.anim.kin_char_model as kin_char_model
import parc.util.torch_util as torch_util


class RotationType(enum.Enum):
    ROT_MAT = 0
    ROT_6D = 1
    EXP_MAP = 2
    AXIS_ANGLE = 3
    QUAT = 4
    DEFAULT = 5

RotTypeToDimMap = {
    RotationType.ROT_MAT: 9,
    RotationType.ROT_6D: 6,
    RotationType.EXP_MAP: 3,
    RotationType.AXIS_ANGLE: 3,
    RotationType.QUAT: 4
}

class RotChanger:
    def __init__(self, rot_type: RotationType, char_model: kin_char_model.KinCharModel):
        self.rot_type = rot_type
        self.char_model = char_model
        return
    
    def convert_quat_to_rot_type(self, q: torch.Tensor):
        if self.rot_type == RotationType.ROT_MAT:
            return torch_util.quat_to_matrix(q)
        elif self.rot_type == RotationType.ROT_6D:
            return torch_util.quat_to_tan_norm(q)
        elif self.rot_type == RotationType.EXP_MAP or self.rot_type == RotationType.DEFAULT:
            return torch_util.quat_to_exp_map(q)
        elif self.rot_type == RotationType.AXIS_ANGLE:
            axis, angle = torch_util.quat_to_axis_angle(self.rot_imported)
            return torch.cat([axis, angle], dim=-1)
        elif self.rot_type == RotationType.QUAT:
            return q
        else:
            assert False
    
    def get_root_rot_dim(self):
        if self.rot_type == RotationType.DEFAULT:
            return 3
        else:
            return RotTypeToDimMap[self.rot_type]

    def get_joint_rot_dim(self):
        if self.rot_type == RotationType.DEFAULT:
            return self.char_model.get_dof_size()
        else:
            return self.char_model.get_num_non_root_joints() * RotTypeToDimMap[self.rot_type]
    
    def convert_joint_quats_to_rot_type(self, q: torch.Tensor):
        if self.rot_type == RotationType.DEFAULT:
            return self.char_model.rot_to_dof(q)
        else:
            rots = self.convert_quat_to_rot_type(q)
            batch_size, seq_len, num_joints, d = rots.shape
            rots = rots.view(batch_size, seq_len, -1)
            return rots

    def convert_rot_type_to_quat(self, rot: torch.Tensor):
        if self.rot_type == RotationType.ROT_MAT:
            return torch_util.matrix_to_quat(rot)
        elif self.rot_type == RotationType.ROT_6D:
            return torch_util.tan_norm_to_quat(rot)
        elif self.rot_type == RotationType.EXP_MAP or self.rot_type == RotationType.DEFAULT:
            return torch_util.exp_map_to_quat(rot)
        elif self.rot_type == RotationType.AXIS_ANGLE:
            axis, angle = rot[...,:3], rot[..., 3:]
            return torch_util.axis_angle_to_quat(axis, angle)
        elif self.rot_type == RotationType.QUAT:
            return rot
        else:
            assert False
        return
    
    def convert_joint_rots_to_quat(self, joint_rots):
        if self.rot_type == RotationType.DEFAULT:
            return self.char_model.dof_to_rot(joint_rots)
        else:
            if len(joint_rots.shape) == 3:
                batch_size, seq_len, d = joint_rots.shape
                joint_rots = joint_rots.view(batch_size, seq_len, -1, RotTypeToDimMap[self.rot_type])
            rots = self.convert_rot_type_to_quat(joint_rots)
            return rots
    
    # @classmethod
    # def get_rot_rpr_dim(cls, rot_rpr_type: RotationType):
    #     return RotTypeToDimMap[rot_rpr_type]

    # def from_rotmat(self, rot):
    #     return torch_util.matrix_to_exp_map(rot)

    # def from_rot6d(self, rot):
    #     rotmat = torch_util.tan_norm_to_matrix(rot)
    #     return torch_util.matrix_to_exp_map(rotmat)

    # def from_axis_angle(self, rot):
    #     axis, angle = rot[...,:3], rot[...,3:]
    #     return torch_util.axis_angle_to_exp_map(axis, angle)

    # def from_quat(self, rot):
    #     return torch_util.quat_to_exp_map(rot)

    # def from_exp_map(self, rot):
    #     return torch_util.exp_map_to_quat(rot)
        

    # def to_rotmat(self):
    #     rot_quat = torch_util.exp_map_to_quat(self.rot_imported)
    #     return torch_util.quat_to_matrix(rot_quat)
    
    # def to_rot6d(self):
    #     rot_quat = torch_util.exp_map_to_quat(self.rot_imported)
    #     return torch_util.quat_to_tan_norm(rot_quat)

    # def to_axis_angle(self):
    #     axis, angle = torch_util.quat_to_axis_angle(self.rot_imported)
    #     return torch.cat([axis, angle], dim=-1)
    
    # def to_quat(self):
    #     return self.quat
    
    # def to_exp_map(self):
    #     return torch_util.quat_to_exp_map()