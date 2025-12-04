import enum
import numpy as np
import os
import pickle
import torch
import yaml
import copy
from util.logger import Logger
import util.torch_util as torch_util
from anim.kin_char_model import KinCharModel

import sys
try:
    import numpy.core
    sys.modules['numpy._core'] = np.core
    if hasattr(np.core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
except ImportError:
    pass

class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1

def extract_pose_data(frame):
    root_pos = frame[..., 0:3]
    root_rot = frame[..., 3:6]
    joint_dof = frame[..., 6:]
    return root_pos, root_rot, joint_dof

class MotionLib():
    def __init__(self, motion_input, kin_char_model: KinCharModel, device, init_type="motion_file", 
                 loop_mode = None, fps = None, contact_info=False, contacts=None):
        self._device = device
        self._kin_char_model = kin_char_model
        self._contact_info = contact_info # TODO: add this to other load motion functions

        self._hf_mask_inds = None

        if init_type == "diffusion_file": # bool to note that we are loading a diffusion file
            self._load_diffusion_file(motion_input)
        elif init_type == "motion_file":
            self._load_motions(motion_input)
        elif init_type == "motion_frames": # NOTE: expects shape: (num motions, num_frames, dofs)
            if contact_info:
                self._load_motion_frames(motion_input, loop_mode, fps, frame_contacts = contacts)
            else:
                self._load_motion_frames(motion_input, loop_mode, fps)
            
        return

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
    
    def get_motion_loop_mode(self, motion_ids):
        return self._motion_loop_modes[motion_ids]
    
    def get_motion_loop_mode_enum(self, motion_id):
        return LoopMode(self._motion_loop_modes[motion_id].item())
    
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
    
    def get_motion_frame(self, motion_ids, frame_idxs):

        motion_start_idx = self._motion_start_idx[motion_ids]
        frame_idxs = motion_start_idx + frame_idxs

        root_pos = self._frame_root_pos[frame_idxs]
        root_rot = self._frame_root_rot[frame_idxs]
        root_vel = self._frame_root_vel[frame_idxs]
        root_ang_vel = self._frame_root_ang_vel[frame_idxs]
        joint_rot = self._frame_joint_rot[frame_idxs]
        dof_vel = self._frame_dof_vel[frame_idxs]

        ret = [root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel]
        if self._contact_info:
            contacts = self._frame_contacts[frame_idxs]
            ret.append(contacts)

        return tuple(ret)

    def joint_rot_to_dof(self, joint_rot):
        joint_dof = self._kin_char_model.rot_to_dof(joint_rot)
        return joint_dof
    
    def _load_motion_frames(self, motion_frames, loop_mode, fps, frame_contacts=None):
        if len(motion_frames.shape) == 2:
            motion_frames = motion_frames.unsqueeze(0)
        if frame_contacts is not None:
            if len(frame_contacts.shape) == 2:
                frame_contacts = frame_contacts.unsqueeze(0)
            assert len(motion_frames.shape) == len(frame_contacts.shape) == 3, frame_contacts.shape
        # note, not including motion weights because this lib is not meant to be randomly sampled from
        # motion_frames.shape = (num_motions, num_frames, dofs)
        num_motions = motion_frames.shape[0]
        if self._contact_info and frame_contacts is not None:
            self._frame_contacts = frame_contacts # assuming this is already a tensor
            self._frame_contacts = self._frame_contacts.reshape(-1, frame_contacts.shape[-1])

        self._motion_fps = fps * torch.ones(size=(num_motions,), dtype=torch.float32, device=self._device)
        self._motion_dt = 1.0/fps * torch.ones(size=(num_motions,), dtype=torch.float32, device=self._device)
        
        # assume all motions have the same num frames and length
        num_frames = motion_frames[0].shape[0]
        motion_length = 1.0 / fps * (num_frames - 1)
        self._motion_num_frames = num_frames * torch.ones(size=(num_motions,), dtype=torch.long, device=self._device)
        self._motion_lengths = motion_length * torch.ones(size=(num_motions,), dtype=torch.float32, device=self._device)
        
        loop_mode = loop_mode.value
        self._motion_loop_modes = loop_mode * torch.ones(size=(num_motions,), dtype=torch.int, device=self._device)
        
        self._motion_frames = motion_frames
        # frame_joint_rot has an extra dim because it adds a quat dim (4) for each joint
        self._frame_root_pos, self._frame_root_rot, self._frame_joint_rot = self._extract_frame_data(motion_frames)
        self._motion_root_pos_delta = self._frame_root_pos[:, -1] - self._frame_root_pos[:, 0]
        self._motion_root_pos_delta[..., -1] = 0.0

        self._frame_root_vel = torch.zeros_like(self._frame_root_pos)
        self._frame_root_vel[..., :-1, :] = fps * (self._frame_root_pos[..., 1:, :] - self._frame_root_pos[..., :-1, :])
        self._frame_root_vel[..., -1, :] = self._frame_root_vel[..., -2, :] # since we cant actually know the last frame vel
        
        self._frame_root_ang_vel = torch.zeros_like(self._frame_root_pos)
        root_drot = torch_util.quat_diff(self._frame_root_rot[..., :-1, :], self._frame_root_rot[..., 1:, :])
        self._frame_root_ang_vel[..., :-1, :] = fps * torch_util.quat_to_exp_map(root_drot)
        self._frame_root_ang_vel[..., -1, :] = self._frame_root_ang_vel[..., -2, :]

        self._frame_dof_vel = self._kin_char_model.compute_frame_dof_vel(self._frame_joint_rot, fps)
        

        # TODO: must reararange the self._frame_X vars so that their shape goes from
        # (num_motions, num_frames, dof) -> (num_motions * num_frames, dof)
        self._frame_root_pos = self._frame_root_pos.view(-1, 3)
        self._frame_root_rot = self._frame_root_rot.view(-1, 4)
        self._frame_joint_rot = self._frame_joint_rot.view(-1, self._frame_joint_rot.shape[-2], self._frame_joint_rot.shape[-1])
        self._frame_root_vel = self._frame_root_vel.view(-1, 3)
        self._frame_root_ang_vel = self._frame_root_ang_vel.view(-1, 3)
        self._frame_dof_vel = self._frame_dof_vel.view(-1, self._frame_dof_vel.shape[-1])


        # need to make sure motion_frames has all the frames stacked along one dimension?
        self._motion_frames = torch.reshape(self._motion_frames, shape=(-1, self._motion_frames.shape[-1]))

        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)

        self._motion_weights = torch.ones(size=[num_motions], dtype=torch.float32, device=self._device)
        return
    
    def _load_motions(self, motion_file):
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        self._motion_loop_modes = []
        self._motion_root_pos_delta = []
        self._motion_files = []

        self._motion_extras = []
        
        self._motion_frames = []
        self._frame_root_pos = []
        self._frame_root_rot = []
        self._frame_root_vel = []
        self._frame_root_ang_vel = []
        self._frame_joint_rot = []
        self._frame_dof_vel = []

        if self._contact_info:
            self._frame_contacts = []

        self._terrains = []
        self._motion_names = []
        self._hf_mask_inds = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print_iter = num_motion_files < 1000 or f % 500 == 0
            if print_iter:
                print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            
            with open(curr_file, "rb") as filestream:
                curr_motion = pickle.load(filestream)
                fps = 30 if "fps" not in curr_motion else curr_motion["fps"]
                if isinstance(fps, np.ndarray):
                    fps = fps.item()
                loop_mode = "CLAMP" if "loop_mode" not in curr_motion else curr_motion["loop_mode"]
                if "frames" in curr_motion:
                    frames = curr_motion["frames"]
                    if len(frames.shape) == 3 and frames.shape[0] == 1:
                        print("extra dim found in frames")
                        frames = np.squeeze(frames)
                else:
                    frames = np.zeros(shape=[3, 6 + self._kin_char_model.get_dof_size()], dtype=np.float32)
                curr_weight = motion_weights[f]

                motion_name = os.path.basename(os.path.splitext(curr_file)[0])
                assert motion_name not in self._motion_names # ensure we have unique motion names
                self._motion_names.append(motion_name)

                
                
                

                loop_mode = LoopMode[loop_mode].value
                dt = 1.0 / fps

                num_frames = frames.shape[0]
                if print_iter:
                    print("fps =", fps)
                    print("loop_mode =", loop_mode)
                    print("num frames =", num_frames)

                # if num_frames < 3:
                #     frames = np.tile(frames, reps=[3, 1])
                #     num_frames = 3
                curr_len = 1.0 / fps * (num_frames - 1)

                root_pos, root_rot, joint_rot = self._extract_frame_data(frames)
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
                
                self._motion_frames.append(frames)
                self._frame_root_pos.append(root_pos)
                self._frame_root_rot.append(root_rot)
                self._frame_root_vel.append(root_vel)
                self._frame_root_ang_vel.append(root_ang_vel)
                self._frame_joint_rot.append(joint_rot)
                self._frame_dof_vel.append(dof_vel)

                if self._contact_info:
                    if "contacts" in curr_motion:
                        contacts = curr_motion["contacts"]
                        contacts = torch.tensor(contacts, dtype=torch.float32, device=self._device)
                        if len(contacts.shape) == 3 and contacts.shape[0] == 1:
                            contacts = torch.squeeze(contacts)
                    else:
                        num_joints = self._kin_char_model.get_num_joints()
                        contacts = torch.zeros(size=[num_frames, num_joints], dtype=torch.float32, device=self._device)

                    self._frame_contacts.append(contacts)

                if "extra" in curr_motion:
                    self._motion_extras.append(curr_motion["extra"])
                    print(curr_motion["extra"])
                else:
                    self._motion_extras.append(None)

                if "terrain" in curr_motion:
                    terrain = curr_motion["terrain"]
                    terrain.update_old()
                    terrain.to_torch(self._device)
                    self._terrains.append(terrain)
                else:
                    self._terrains.append(None)

                if "hf_mask_inds" in curr_motion:
                    hf_mask_inds = curr_motion["hf_mask_inds"]
                    if hf_mask_inds is None:
                        self._hf_mask_inds.append(None)
                    else:
                        for t in range(len(hf_mask_inds)):
                            hf_mask_inds[t] = hf_mask_inds[t].to(device=self._device)
                        self._hf_mask_inds.append(hf_mask_inds)
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
        
        self._motion_frames = np.concatenate(self._motion_frames, axis=0)
        self._motion_frames = torch.tensor(self._motion_frames, dtype=torch.float32, device=self._device)
        
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

            with open(motion_file, 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

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

    def _extract_frame_data(self, frame):
        root_pos, root_rot, joint_dof = extract_pose_data(frame)
        # TODO: check if the type is tensor already
        # In that case, use copy constructor
        if not torch.is_tensor(frame):
            root_pos = torch.tensor(root_pos, dtype=torch.float32, device=self._device)
            root_rot = torch.tensor(root_rot, dtype=torch.float32, device=self._device)
            joint_dof = torch.tensor(joint_dof, dtype=torch.float32, device=self._device)
        else:
            root_pos = torch.clone(root_pos)
            root_rot = torch.clone(root_rot)
            joint_dof = torch.clone(joint_dof)

        root_rot_quat = torch_util.exp_map_to_quat(root_rot)

        joint_rot = self._kin_char_model.dof_to_rot(joint_dof)
        joint_rot = torch_util.quat_pos(joint_rot)

        return root_pos, root_rot_quat, joint_rot
    
    def get_frame_data(self, motion_id, frame_start, frame_end):
        assert frame_start < frame_end, "frame_start must be less than frame_end"
        assert motion_id < self.num_motions(), "motion_id out of range"

        start_idx = self._motion_start_idx[motion_id] + frame_start
        end_idx = self._motion_start_idx[motion_id] + frame_end

        root_pos = self._frame_root_pos[start_idx:end_idx]
        root_rot = self._frame_root_rot[start_idx:end_idx]
        joint_rot = self._frame_joint_rot[start_idx:end_idx]

        ret = [root_pos, root_rot, joint_rot]
        if self._contact_info:
            contacts = self._frame_contacts[start_idx:end_idx]
            ret.append(contacts)

        return tuple(ret)

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
        # TODO: do this for each motion using motion_num_frames
        return torch.max_pool1d(self._frame_contacts.unsqueeze(0), 
                                kernel_size=kernel_size,
                                stride=1,
                                padding=kernel_size//2).squeeze(0)
    
    def get_motion_names(self):
        assert hasattr(self, "_motion_names")
        return self._motion_names
    
    def get_frames_for_id(self, id):
        num_frames = self._motion_num_frames[id]
        start_frame_idx = self._motion_start_idx[id]
        motion_slice = slice(start_frame_idx, start_frame_idx+num_frames)
        root_pos = self._frame_root_pos[motion_slice]
        root_rot = self._frame_root_rot[motion_slice]
        joint_rot = self._frame_joint_rot[motion_slice]

        body_pos, body_rot = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)

        return root_pos, root_rot, joint_rot, body_pos, body_rot
    
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
