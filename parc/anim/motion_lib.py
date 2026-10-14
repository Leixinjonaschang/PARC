import copy
import enum
import os
from typing import TYPE_CHECKING

import torch

from parc.util import file_io_helper, path_loader, terrain_util, torch_util
from parc.util.motion_util import MotionFrames

if TYPE_CHECKING:
    from parc.anim.kin_char_model import KinCharModel

class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1

def extract_pose_data(frame):
    root_pos = frame[..., 0:3]
    root_rot = frame[..., 3:6]
    joint_dof = frame[..., 6:]
    return root_pos, root_rot, joint_dof

class MotionLib():
    def __init__(self, char_model: 'KinCharModel' = None, device=None, contact_info=False):
        if char_model is None:
            raise ValueError("kin_char_model must be provided")
        if device is None:
            raise ValueError("device must be provided")

        self._device = torch.device(device)
        self._kin_char_model = char_model
        self._contact_info = contact_info
        return

    @classmethod
    def from_file(cls, motion_file, char_model: 'KinCharModel', device, contact_info=False):
        motion_file = path_loader.resolve_path(motion_file)
        inst = cls(char_model, device, contact_info=contact_info)
        inst._load_motion_file(motion_file)
        return inst

    @classmethod
    def from_frames(cls, frames: MotionFrames, char_model: 'KinCharModel', device,
                    loop_mode: LoopMode, fps: float, contact_info=False):
        inst = cls(char_model, device, contact_info=contact_info)
        inst._load_motion_frames(frames, loop_mode, fps)
        return inst

    def num_motions(self):
        return self._motion_lengths.shape[0]

    def get_total_length(self):
        return torch.sum(self._motion_lengths).item()
    
    def sample_motions(self, n, motion_weights=None):
        if motion_weights is None:
            motion_weights = self._motion_weights
        motion_ids = torch.multinomial(motion_weights, num_samples=n, replacement=True)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]
    
    def get_motion_num_frames(self, motion_ids):
        return self._motion_num_frames[motion_ids]
    
    def get_motion_loop_mode(self, motion_ids):
        return self._motion_loop_modes[motion_ids]
    
    def get_motion_loop_mode_enum(self, motion_id):
        return LoopMode(self._motion_loop_modes[motion_id].item())
    
    def get_motion_fps(self, motion_id):
        return self._motion_fps[motion_id].item()
    
    def calc_motion_phase(self, motion_ids, times):
        motion_len = self._motion_lengths[motion_ids]
        loop_mode = self._motion_loop_modes[motion_ids]
        phase = calc_phase(times=times, motion_len=motion_len, loop_mode=loop_mode)
        return phase

    def calc_motion_frame(self, motion_ids, motion_times):
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        
        root_pos0 = self._frame_root_pos[frame_idx0]
        root_pos1 = self._frame_root_pos[frame_idx1]
        
        root_rot0 = self._frame_root_rot[frame_idx0]
        root_rot1 = self._frame_root_rot[frame_idx1]

        root_vel = self._frame_root_vel[frame_idx0]
        root_ang_vel = self._frame_root_ang_vel[frame_idx0]

        joint_rot0 = self._frame_joint_rot[frame_idx0]
        joint_rot1 = self._frame_joint_rot[frame_idx1]

        dof_vel = self._frame_dof_vel[frame_idx0]

        blend_unsq = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsq) * root_pos0 + blend_unsq * root_pos1
        root_rot = torch_util.slerp(root_rot0, root_rot1, blend)
        
        joint_rot = torch_util.slerp(joint_rot0, joint_rot1, blend_unsq)

        root_pos_offset = self._calc_loop_offset(motion_ids, motion_times)
        root_pos += root_pos_offset

        ret = [root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel]

        if self._contact_info:
            contacts = (1.0 - blend_unsq) * self._frame_contacts[frame_idx0] + blend_unsq * self._frame_contacts[frame_idx1]
            ret.append(contacts)

        return tuple(ret)

    def joint_rot_to_dof(self, joint_rot):
        joint_dof = self._kin_char_model.rot_to_dof(joint_rot)
        return joint_dof
    
    def _load_motion_frames(self, motion_frames: MotionFrames, loop_mode, fps):
        if not isinstance(motion_frames, MotionFrames):
            raise TypeError("motion_frames must be a MotionFrames instance")

        frames = motion_frames.get_copy(self._device)

        if frames.root_pos is None or frames.root_rot is None or frames.joint_rot is None:
            raise ValueError("MotionFrames must contain root_pos, root_rot, and joint_rot tensors")

        root_pos = frames.root_pos
        root_rot = frames.root_rot
        joint_rot = frames.joint_rot

        if root_pos.dim() == 2:
            root_pos = root_pos.unsqueeze(0)
        if root_rot.dim() == 2:
            root_rot = root_rot.unsqueeze(0)
        if joint_rot.dim() == 3:
            joint_rot = joint_rot.unsqueeze(0)

        if root_pos.dim() != 3 or root_rot.dim() != 3 or joint_rot.dim() != 4:
            raise ValueError("Unexpected MotionFrames tensor shapes")

        num_motions = root_pos.shape[0]
        num_frames = root_pos.shape[1]

        if root_rot.shape[0] != num_motions or root_rot.shape[1] != num_frames:
            raise ValueError("root_rot must have the same batch and frame dimensions as root_pos")
        if joint_rot.shape[0] != num_motions or joint_rot.shape[1] != num_frames:
            raise ValueError("joint_rot must have the same batch and frame dimensions as root_pos")

        root_pos = root_pos.to(dtype=torch.float32)
        root_rot = torch_util.quat_normalize(root_rot.to(dtype=torch.float32))
        joint_rot = torch_util.quat_normalize(joint_rot.to(dtype=torch.float32))

        if self._contact_info:
            if frames.contacts is None:
                contacts = torch.zeros(size=[num_frames, self._kin_char_model.get_num_bodies()], dtype=torch.float32, device=self._device)
            else:
                contacts = frames.contacts
            if contacts.dim() == 2:
                contacts = contacts.unsqueeze(0)
            if contacts.shape[0] != num_motions or contacts.shape[1] != num_frames:
                raise ValueError("contacts must have the same batch and frame dimensions as root_pos")
            contacts = contacts.to(device=self._device, dtype=torch.float32)
            self._frame_contacts = contacts.reshape(-1, contacts.shape[-1])
        else:
            self._frame_contacts = None

        self._motion_fps = torch.full((num_motions,), fps, dtype=torch.float32, device=self._device)
        self._motion_dt = torch.full((num_motions,), 1.0 / fps, dtype=torch.float32, device=self._device)

        motion_length = 1.0 / fps * (num_frames - 1)
        self._motion_num_frames = torch.full((num_motions,), num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.full((num_motions,), motion_length, dtype=torch.float32, device=self._device)

        if isinstance(loop_mode, LoopMode):
            loop_tensor = torch.full((num_motions,), loop_mode.value, dtype=torch.int, device=self._device)
        elif torch.is_tensor(loop_mode):
            if loop_mode.numel() == 1:
                loop_tensor = torch.full((num_motions,), int(loop_mode.item()), dtype=torch.int, device=self._device)
            else:
                loop_tensor = loop_mode.to(device=self._device, dtype=torch.int)
        elif isinstance(loop_mode, (list, tuple)):
            loop_tensor = torch.tensor(loop_mode, dtype=torch.int, device=self._device)
        else:
            loop_tensor = torch.full((num_motions,), int(loop_mode), dtype=torch.int, device=self._device)

        if loop_tensor.shape[0] != num_motions:
            raise ValueError("loop_mode must provide one value per motion")

        self._motion_loop_modes = loop_tensor

        self._frame_root_pos = root_pos
        self._frame_root_rot = root_rot
        self._frame_joint_rot = joint_rot
        self._motion_root_pos_delta = self._frame_root_pos[:, -1] - self._frame_root_pos[:, 0]
        self._motion_root_pos_delta[..., -1] = 0.0

        self._frame_root_vel = torch.zeros_like(self._frame_root_pos)
        self._frame_root_vel[..., :-1, :] = fps * (self._frame_root_pos[..., 1:, :] - self._frame_root_pos[..., :-1, :])
        if num_frames > 1:
            self._frame_root_vel[..., -1, :] = self._frame_root_vel[..., -2, :]

        self._frame_root_ang_vel = torch.zeros_like(self._frame_root_pos)
        root_drot = torch_util.quat_diff(self._frame_root_rot[..., :-1, :], self._frame_root_rot[..., 1:, :])
        self._frame_root_ang_vel[..., :-1, :] = fps * torch_util.quat_to_exp_map(root_drot)
        if num_frames > 1:
            self._frame_root_ang_vel[..., -1, :] = self._frame_root_ang_vel[..., -2, :]

        self._frame_dof_vel = self._kin_char_model.compute_frame_dof_vel(self._frame_joint_rot, fps)


        # must reararange the self._frame_X vars so that their shape goes from
        # (num_motions, num_frames, dof) -> (num_motions * num_frames, dof)
        self._frame_root_pos = self._frame_root_pos.view(-1, 3)
        self._frame_root_rot = self._frame_root_rot.view(-1, 4)
        self._frame_joint_rot = self._frame_joint_rot.view(-1, self._frame_joint_rot.shape[-2], self._frame_joint_rot.shape[-1])
        self._frame_root_vel = self._frame_root_vel.view(-1, 3)
        self._frame_root_ang_vel = self._frame_root_ang_vel.view(-1, 3)
        self._frame_dof_vel = self._frame_dof_vel.view(-1, self._frame_dof_vel.shape[-1])


        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)

        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)

        self._motion_weights = torch.ones(size=[num_motions], dtype=torch.float32, device=self._device)

        self._terrains = [None]
        self._motion_names = [""]
        self._hf_mask_inds = [None]
        return

    @property
    def _motion_frames(self):
        root_rot_exp = torch_util.quat_to_exp_map(self._frame_root_rot)
        joint_dof = self.joint_rot_to_dof(self._frame_joint_rot)
        return torch.cat([self._frame_root_pos, root_rot_exp, joint_dof], dim=-1)

    def _load_motion_file(self, motion_file):
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        self._motion_loop_modes = []
        self._motion_root_pos_delta = []
        self._motion_files = []
        
        self._frame_root_pos = []
        self._frame_root_rot = []
        self._frame_root_vel = []
        self._frame_root_ang_vel = []
        self._frame_joint_rot = []
        self._frame_dof_vel = []

        if self._contact_info:
            self._frame_contacts = []
        else:
            self._frame_contacts = None

        self._terrains = []
        self._motion_names = []
        self._hf_mask_inds = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print_iter = num_motion_files < 1000 or f % 500 == 0
            if print_iter:
                print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, str(curr_file)))
            
            curr_file_data = file_io_helper.load_ms_file(curr_file, device=self._device)
            curr_motion_data = curr_file_data.motion_data

            fps = curr_motion_data.fps
            loop_mode = LoopMode[curr_motion_data.loop_mode].value
            dt = 1.0 / fps
            root_pos = curr_motion_data.root_pos
            root_rot = curr_motion_data.root_rot
            joint_rot = curr_motion_data.joint_rot
            num_frames = root_pos.shape[0]
            assert num_frames == root_rot.shape[0] == joint_rot.shape[0]

            curr_weight = motion_weights[f]
            motion_name = os.path.basename(os.path.splitext(curr_file)[0])

            # ensure we have unique motion names
            assert motion_name not in self._motion_names, motion_name + " is a repeat. full path: " + curr_file
            self._motion_names.append(motion_name)

            if print_iter:
                print("fps =", fps)
                print("loop_mode =", loop_mode)
                print("num frames =", num_frames)

            curr_len = 1.0 / fps * (num_frames - 1)

            root_pos_delta = root_pos[-1] - root_pos[0]
            root_pos_delta[..., -1] = 0.0

            root_vel = torch.zeros_like(root_pos)
            root_vel[..., :-1, :] = fps * (root_pos[..., 1:, :] - root_pos[..., :-1, :])
            root_vel[..., -1, :] = root_vel[..., -2, :]
            
            root_ang_vel = torch.zeros_like(root_pos)
            root_drot = torch_util.quat_diff(root_rot[..., :-1, :], root_rot[..., 1:, :])
            root_ang_vel[..., :-1, :] = fps * torch_util.quat_to_exp_map(root_drot)
            root_ang_vel[..., -1, :] = root_ang_vel[..., -2, :]

            dof_vel = self._kin_char_model.compute_frame_dof_vel(joint_rot, dt)

            self._motion_weights.append(curr_weight)
            self._motion_fps.append(fps)
            self._motion_dt.append(dt)
            self._motion_num_frames.append(num_frames)
            self._motion_lengths.append(curr_len)
            self._motion_loop_modes.append(loop_mode)
            self._motion_root_pos_delta.append(root_pos_delta)
            self._motion_files.append(curr_file)
            
            #self._motion_frames.append(frames)
            self._frame_root_pos.append(root_pos)
            self._frame_root_rot.append(root_rot)
            self._frame_root_vel.append(root_vel)
            self._frame_root_ang_vel.append(root_ang_vel)
            self._frame_joint_rot.append(joint_rot)
            self._frame_dof_vel.append(dof_vel)

            if self._contact_info:
                if curr_motion_data.body_contacts is not None:
                    contacts = curr_motion_data.body_contacts
                else:
                    num_joints = self._kin_char_model.get_num_joints()
                    contacts = torch.zeros(size=[num_frames, num_joints], dtype=torch.float32, device=self._device)

                self._frame_contacts.append(contacts)

            terrain_data = curr_file_data.terrain_data
            if terrain_data is not None:
                terrain = terrain_util.SubTerrain.from_ms_terrain_data(terrain_data=terrain_data, device=self._device)
                self._terrains.append(terrain)
            else:
                self._terrains.append(None)

            misc_data = curr_file_data.misc_data
            if misc_data is not None:
                if file_io_helper.HF_MASK_INDS_KEY in misc_data:
                    self._hf_mask_inds.append(misc_data[file_io_helper.HF_MASK_INDS_KEY])
                else:
                    self._hf_mask_inds.append(None)
            else:
                self._hf_mask_inds.append(None)

        if self._contact_info:
            self._frame_contacts = torch.cat(self._frame_contacts, dim=0)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float32, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float32, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float32, device=self._device)
        self._motion_loop_modes = torch.tensor(self._motion_loop_modes, dtype=torch.int, device=self._device)
        
        self._motion_root_pos_delta = torch.stack(self._motion_root_pos_delta, dim=0)
        
        self._frame_root_pos = torch.cat(self._frame_root_pos, dim=0)
        self._frame_root_rot = torch.cat(self._frame_root_rot, dim=0)
        self._frame_root_vel = torch.cat(self._frame_root_vel, dim=0)
        self._frame_root_ang_vel = torch.cat(self._frame_root_ang_vel, dim=0)
        self._frame_joint_rot = torch.cat(self._frame_joint_rot, dim=0)
        self._frame_dof_vel = torch.cat(self._frame_dof_vel, dim=0)
        
        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)
        
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            motion_files = []
            motion_weights = []

            motion_config = path_loader.load_config(motion_file)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(self, motion_ids, times):
        num_frames = self._motion_num_frames[motion_ids]
        frame_start_idx = self._motion_start_idx[motion_ids]
        
        phase = self.calc_motion_phase(motion_ids, times)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0
        
        frame_idx0 += frame_start_idx
        frame_idx1 += frame_start_idx

        return frame_idx0, frame_idx1, blend
    
    def _calc_loop_offset(self, motion_ids, times):
        loop_mode = self._motion_loop_modes[motion_ids]
        wrap_mask = (loop_mode == LoopMode.WRAP.value)

        wrap_motion_ids = motion_ids[wrap_mask]
        times = times[wrap_mask]

        motion_len = self._motion_lengths[wrap_motion_ids]
        root_pos_deltas = self._motion_root_pos_delta[wrap_motion_ids]

        phase = times / motion_len
        phase = torch.floor(phase)
        phase = phase.unsqueeze(-1)
        
        root_pos_offset = torch.zeros((motion_ids.shape[0], 3), device=self._device)
        root_pos_offset[wrap_mask] = phase * root_pos_deltas

        return root_pos_offset
    
    def calc_motion_frame_dofs(self, motion_ids, motion_times):

        if not self._contact_info:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self.calc_motion_frame(motion_ids, motion_times)
            root_rot = torch_util.quat_to_exp_map(root_rot)
            joint_rot = self.joint_rot_to_dof(joint_rot)
            return torch.cat([root_pos, root_rot, joint_rot], dim=-1)
        else:
            root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self.calc_motion_frame(motion_ids, motion_times)
            root_rot = torch_util.quat_to_exp_map(root_rot)
            joint_rot = self.joint_rot_to_dof(joint_rot)
        
            return torch.cat([root_pos, root_rot, joint_rot, contacts], dim=-1)
        
    def maxpool_contacts(self, kernel_size):
        print("maxpooling contacts")
        # NOTE: this assumes there is only 1 motion in this motion lib
        return torch.max_pool1d(self._frame_contacts.unsqueeze(0), 
                                kernel_size=kernel_size,
                                stride=1,
                                padding=kernel_size//2).squeeze(0)
    
    def get_motion_names(self):
        assert hasattr(self, "_motion_names")
        return self._motion_names
    
    def get_frames_for_id(self, id, compute_fk = False):
        num_frames = self._motion_num_frames[id]
        start_frame_idx = self._motion_start_idx[id]
        motion_slice = slice(start_frame_idx, start_frame_idx+num_frames)
        root_pos = self._frame_root_pos[motion_slice]
        root_rot = self._frame_root_rot[motion_slice]
        joint_rot = self._frame_joint_rot[motion_slice]

        if compute_fk:
            body_pos, body_rot = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        else:
            body_pos = None
            body_rot = None

        if self._contact_info:
            contacts = self._frame_contacts[motion_slice]
        else:
            contacts = None

        return MotionFrames(root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot, 
                            body_pos = body_pos, body_rot=body_rot, contacts=contacts)
    
    def clone(self, device):

        new_mlib = copy.deepcopy(self)
        new_mlib._device = device

        for attr, value in vars(new_mlib).items():
            if isinstance(value, torch.Tensor):
                setattr(new_mlib, attr, value.to(device))
        new_mlib._kin_char_model = new_mlib._kin_char_model.get_copy(device)
        return new_mlib


@torch.jit.script
def calc_phase(times, motion_len, loop_mode):
    phase = times / motion_len
        
    loop_wrap_mask = (loop_mode == LoopMode.WRAP.value)
    phase_wrap = phase[loop_wrap_mask]
    phase_wrap = phase_wrap - torch.floor(phase_wrap)
    phase[loop_wrap_mask] = phase_wrap
        
    phase = torch.clip(phase, 0.0, 1.0)

    return phase