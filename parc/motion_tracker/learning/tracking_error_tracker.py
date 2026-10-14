import torch

import parc.motion_tracker.envs.base_env as base_env


class TrackingErrorTracker():
    def __init__(self, num_envs, device):
        self._device = device
        self._episodes = 0

        self._mean_root_pos_err = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_root_rot_err = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_body_rot_err = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_body_pos_err = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_root_vel_err = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_root_ang_vel_err = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_dof_vel_err = torch.zeros([1], device=device, dtype=torch.float32)

        self._root_pos_err_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._root_rot_err_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._body_rot_err_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._body_pos_err_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._root_vel_err_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._root_ang_vel_err_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._dof_vel_err_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)

        self._ep_len_buf = torch.zeros([num_envs], device=device, dtype=torch.long)

        return

    def get_mean_root_pos_err(self):
        return self._mean_root_pos_err
    
    def get_mean_root_rot_err(self):
        return self._mean_root_rot_err
    
    def get_mean_body_rot_err(self):
        return self._mean_body_rot_err
    
    def get_mean_body_pos_err(self):
        return self._mean_body_pos_err
    
    def get_mean_root_vel_err(self):
        return self._mean_root_vel_err
    
    def get_mean_root_ang_vel_err(self):
        return self._mean_root_ang_vel_err
    
    def get_mean_dof_vel_err(self):
        return self._mean_dof_vel_err

    def reset(self):
        self._episodes = 0

        self._mean_root_pos_err[:] = 0.0
        self._mean_root_rot_err[:] = 0.0
        self._mean_body_rot_err[:] = 0.0
        self._mean_body_pos_err[:] = 0.0
        self._mean_root_vel_err[:] = 0.0
        self._mean_root_ang_vel_err[:] = 0.0
        self._mean_dof_vel_err[:] = 0.0

        self._root_pos_err_buf[:] = 0.0
        self._root_rot_err_buf[:] = 0.0
        self._body_rot_err_buf[:] = 0.0
        self._body_pos_err_buf[:] = 0.0
        self._root_vel_err_buf[:] = 0.0
        self._root_ang_vel_err_buf[:] = 0.0
        self._dof_vel_err_buf[:] = 0.0

        self._ep_len_buf[:] = 0
        return

    def update(self, tracking_error, done):
        assert(tracking_error.shape[0] == self._root_pos_err_buf.shape[0])
        assert(done.shape[0] == self._root_pos_err_buf.shape[0])

        self._root_pos_err_buf += tracking_error[:, 0]
        self._root_rot_err_buf += tracking_error[:, 1]
        self._body_pos_err_buf += tracking_error[:, 2]
        self._body_rot_err_buf += tracking_error[:, 3]
        self._dof_vel_err_buf += tracking_error[:, 4]
        self._root_vel_err_buf += tracking_error[:, 5]
        self._root_ang_vel_err_buf += tracking_error[:, 6]
        
        self._ep_len_buf += 1

        reset_mask = done != base_env.DoneFlags.NULL.value
        reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
        num_resets = len(reset_ids)

        if (num_resets > 0):
            new_count = self._episodes + num_resets
            w_new = float(num_resets) / new_count
            w_old = float(self._episodes) / new_count

            root_pos_err_per_ts = self._root_pos_err_buf[reset_ids] / self._ep_len_buf[reset_ids]
            root_rot_err_per_ts = self._root_rot_err_buf[reset_ids] / self._ep_len_buf[reset_ids]
            body_pos_err_per_ts = self._body_pos_err_buf[reset_ids] / self._ep_len_buf[reset_ids]
            body_rot_err_per_ts = self._body_rot_err_buf[reset_ids] / self._ep_len_buf[reset_ids]
            dof_vel_err_per_ts = self._dof_vel_err_buf[reset_ids] / self._ep_len_buf[reset_ids]
            root_vel_err_per_ts = self._root_vel_err_buf[reset_ids] / self._ep_len_buf[reset_ids]
            root_ang_vel_err_per_ts = self._root_ang_vel_err_buf[reset_ids] / self._ep_len_buf[reset_ids]

            self._mean_root_pos_err = w_new * torch.mean(root_pos_err_per_ts) + w_old * self._mean_root_pos_err
            self._mean_root_rot_err = w_new * torch.mean(root_rot_err_per_ts) + w_old * self._mean_root_rot_err
            self._mean_body_pos_err = w_new * torch.mean(body_pos_err_per_ts) + w_old * self._mean_body_pos_err
            self._mean_body_rot_err = w_new * torch.mean(body_rot_err_per_ts) + w_old * self._mean_body_rot_err
            self._mean_dof_vel_err = w_new * torch.mean(dof_vel_err_per_ts) + w_old * self._mean_dof_vel_err
            self._mean_root_vel_err = w_new * torch.mean(root_vel_err_per_ts) + w_old * self._mean_root_vel_err
            self._mean_root_ang_vel_err = w_new * torch.mean(root_ang_vel_err_per_ts) + w_old * self._mean_root_ang_vel_err

            self._episodes += num_resets

            self._root_pos_err_buf[reset_ids] = 0.0
            self._root_rot_err_buf[reset_ids] = 0.0
            self._body_pos_err_buf[reset_ids] = 0.0
            self._body_rot_err_buf[reset_ids] = 0.0
            self._dof_vel_err_buf[reset_ids] = 0.0
            self._root_vel_err_buf[reset_ids] = 0.0
            self._root_ang_vel_err_buf[reset_ids] = 0.0
            
            self._ep_len_buf[reset_ids] = 0

        return
    