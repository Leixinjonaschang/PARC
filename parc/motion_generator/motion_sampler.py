import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.torch_util as torch_util
from parc.anim.motion_lib import LoopMode


class MotionSampler:
    def __init__(self, cfg):
        self._device = cfg['device']
        self._seq_len = None
        self._fps = None
        char_file = cfg['char_file']
        self._kin_char_model = kin_char_model.KinCharModel(self._device)
        self._kin_char_model.load_char_file(char_file)

        self._motion_lib_file = cfg['motion_lib_file']
        if isinstance(self._motion_lib_file, str):
            self._mlib = motion_lib.MotionLib.from_file(motion_file=self._motion_lib_file,
                                                        char_model=self._kin_char_model,
                                                        device = self._device,
                                                        contact_info = True) # TODO: make config var, do ablations on using and not using contact info
        elif isinstance(self._motion_lib_file, motion_lib.MotionLib):
            self._mlib = self._motion_lib_file
        else:
            assert False
        
        print("num_motions =", self._mlib.num_motions())
        return
    
    def check_init(self):
        assert self._seq_len is not None
        assert self._fps is not None
    
    def get_seq_len(self):
        return self._seq_len

    def _sample_motion_start_times(self, motion_ids, seq_duration):
        # The duration is there to make sure we don't sample start times for clips
        # that don't have enough motion data time left to fill

        motion_full_lengths = self._mlib.get_motion_length(motion_ids)
        if self._random_start_times:
            # Logic for sampling start times (to prevent sampling frames out of bounds for CLAMP clips)
            motion_start_times_loop = torch.rand_like(motion_full_lengths) * motion_full_lengths #(motion_full_lengths - motion_len)
            motion_start_times_clamp = torch.rand_like(motion_full_lengths) * (motion_full_lengths - seq_duration)
            motion_loop_modes = self._mlib.get_motion_loop_mode(motion_ids)
            motion_start_times = torch.where(motion_loop_modes == LoopMode.WRAP.value, motion_start_times_loop, motion_start_times_clamp)
        else:
            motion_start_times = torch.zeros_like(motion_full_lengths)
        
        return motion_start_times

    def _extract_motion_data(self, motion_ids, motion_start_times):
        # This function samples clips from the motion lib assuming they are either vault or running clips.
        # This function returns all motion data in global coordinates.

        # 1) For each motion id, sample start times that are within their maximum length
        num_samples = motion_ids.shape[0]

        # 2) Put all frames we want to sample into flat tensors of size (num_samples x num_frames)
        # (this is a motion_lib requirement)
        motion_times = motion_start_times.unsqueeze(-1) + self._motion_times
        motion_times = motion_times.flatten()
        repeated_motion_ids = motion_ids.unsqueeze(-1).expand(-1, self._seq_len).flatten()

        # 3) Get motion data from motion_lib
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = self._mlib.calc_motion_frame(repeated_motion_ids, motion_times=motion_times)

        # 4) Reshape motion data
        # motion data is currently in the form (num_samples x num_frames) x (num_dof)
        # We reshape it into (num_samples) x (num_frames) x (num_dof)
        root_rot = torch_util.quat_to_exp_map(root_rot)
        joint_rot = self._mlib.joint_rot_to_dof(joint_rot)
        motion_samples = torch.cat((root_pos, root_rot, joint_rot), dim=1)
        motion_samples = torch.reshape(motion_samples, shape=(num_samples, self._seq_len, self._num_dof))

        # same thing for contact data
        # it is initially (num_samples x num_frames) x (num_rb)
        contacts = torch.reshape(contacts, shape=(num_samples, self._seq_len, self._num_rb))

        return motion_samples, contacts