import os
import time

import numpy as np
import torch

import parc.motion_tracker.envs.base_env as base_env
import parc.motion_tracker.learning.base_agent as base_agent
import parc.motion_tracker.learning.dm_ppo_model as dm_ppo_model
import parc.motion_tracker.learning.normalizer as normalizer
import parc.motion_tracker.learning.ppo_agent as ppo_agent
import parc.motion_tracker.learning.rl_util as rl_util
import parc.motion_tracker.learning.tracking_error_tracker as tracking_error_tracker
import parc.util.torch_util as torch_util


class DMPPOAgent(ppo_agent.PPOAgent):

    NAME = "DM_PPO"

    def __init__(self, config, env, device):
        self._env = env
        super().__init__(config, env, device)

        if self._env._report_tracking_error:
            self._test_tracking_error_tracker = tracking_error_tracker.TrackingErrorTracker(
                self._env._num_envs,
                device=self._device)
        return
    
    def _init_train(self):
        super()._init_train()
        if self._env._report_tracking_error:
            self._test_tracking_error_tracker.reset()
        return
    
    def _load_params(self, config):
        super()._load_params(config)

        self._config = config
        return
    
    def _build_model(self, config):
        model_config = config["model"]
        self._model = dm_ppo_model.DMPPOModel(model_config, self._env)
        return
    
    def _build_normalizers(self):
        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)

        obs_dim = obs_space.shape[0]

        # normalizable_obs = self._env._compute_obs(normalizable_obs_only=True)
        # normalizable_obs_dim = normalizable_obs.shape[1]
        # if normalizable_obs_dim < obs_dim:
        #     non_norm_indices = slice(normalizable_obs_dim, obs_dim)
        # else:
        #     non_norm_indices = None

        obs_shapes = self._env._compute_obs(ret_obs_shapes=True)
        non_norm_indices = []
        curr_dim = 0
        for key in obs_shapes:
            use_normalizer = obs_shapes[key]["use_normalizer"]
            shape = obs_shapes[key]["shape"]

            if len(shape) >= 2:
                flat_dim = shape[0] * shape[1]
            else:
                flat_dim = shape[0]

            if not use_normalizer:
                non_norm_indices.append(torch.arange(curr_dim, curr_dim + flat_dim, 1, dtype=torch.int64, device=self._device))
            curr_dim += flat_dim
        if len(non_norm_indices) > 0:
            non_norm_indices = torch.cat(non_norm_indices, dim=0)    
        else:
            non_norm_indices = None

        clip = self._config["norm_obs_clip"]
        print("Obs normalizer clip:", clip)
        self._obs_norm = normalizer.Normalizer(obs_space.shape, device=self._device, dtype=obs_dtype,
                                               non_norm_indices=non_norm_indices,
                                               clip=clip) # heightmap obs, tar contacts 3*15 and contacts 15
        self._a_norm = self._build_action_normalizer()
        return
    
    def hard_reset_envs(self):
        self._curr_obs, self._curr_info = self._env.reset()
        return
    
    def test_model(self, num_episodes):
        self.eval()
        self.set_mode(base_agent.AgentMode.TEST)


        print("******************** TESTING MODEL ********************")
        self.hard_reset_envs()
        test_info = self._rollout_test(num_episodes)
        print("******************** FINISHED TESTING MODEL ********************")
        return test_info
    
    def _rollout_train(self, num_steps):
        for i in range(num_steps):
            action, action_info = self._decide_action(self._curr_obs, self._curr_info)
            self._record_data_pre_step(self._curr_obs, self._curr_info, action, action_info)

            next_obs, r, done, next_info = self._step_env(action)
            self._train_return_tracker.update(next_info, done)
            self._record_data_post_step(next_obs, r, done, next_info)

            self._curr_obs, self._curr_info = self._reset_done_envs(done)
            self._exp_buffer.inc()

        return
    
    def step(self):
        action, action_info = self._decide_action(self._curr_obs, self._curr_info)
        next_obs, r, done, next_info = self._step_env(action)
        return next_obs, r, done, next_info, action, action_info
    
    def _rollout_test(self, num_episodes):
        self._test_return_tracker.reset()
        if self._env._report_tracking_error:
            self._test_tracking_error_tracker.reset()

        if (num_episodes == 0):
            test_info = {
                "mean_return": 0.0,
                "mean_ep_len": 0.0,
                "num_eps": 0
            }
        else:
            num_envs = self.get_num_envs()
            # minimum number of episodes to collect per env
            # this is mitigate bias in the return estimate towards shorter episodes
            min_eps_per_env = int(np.ceil(num_episodes / num_envs))

            while True:
                action, action_info = self._decide_action(self._curr_obs, self._curr_info)

                next_obs, r, done, next_info = self._step_env(action)
                self._test_return_tracker.update(next_info, done)

                if "tracking_error" in next_info and self._env._report_tracking_error:
                    tracking_error = next_info["tracking_error"]
                    self._test_tracking_error_tracker.update(tracking_error, done)
            
                self._curr_obs, self._curr_info = self._reset_done_envs(done)
            
                eps_per_env = self._test_return_tracker.get_eps_per_env()
                if (torch.all(eps_per_env > min_eps_per_env - 1)):
                    break
        
            test_return = self._test_return_tracker.get_mean_return()
            test_ep_len = self._test_return_tracker.get_mean_ep_len()
            test_info = {
                "mean_return": test_return.item(),
                "mean_ep_len": test_ep_len.item(),
                "num_eps": self._test_return_tracker.get_episodes()
            }

            if "tracking_error" in next_info:
                test_mean_root_pos_tracking_err = self._test_tracking_error_tracker.get_mean_root_pos_err()
                test_mean_root_rot_tracking_err = self._test_tracking_error_tracker.get_mean_root_rot_err()
                test_mean_body_pos_tracking_err = self._test_tracking_error_tracker.get_mean_body_pos_err()
                test_mean_body_rot_tracking_err = self._test_tracking_error_tracker.get_mean_body_rot_err()
                test_mean_dof_vel_err = self._test_tracking_error_tracker.get_mean_dof_vel_err()
                test_mean_root_vel_err = self._test_tracking_error_tracker.get_mean_root_vel_err()
                test_mean_root_ang_vel_err = self._test_tracking_error_tracker.get_mean_root_ang_vel_err()

                test_info["test_mean_root_pos_tracking_err"] = test_mean_root_pos_tracking_err.item()
                test_info["test_mean_root_rot_tracking_err"] = test_mean_root_rot_tracking_err.item()
                test_info["test_mean_body_pos_tracking_err"] = test_mean_body_pos_tracking_err.item()
                test_info["test_mean_body_rot_tracking_err"] = test_mean_body_rot_tracking_err.item()
                test_info["test_mean_dof_vel_tracking_err"] = test_mean_dof_vel_err.item()
                test_info["test_mean_root_vel_tracking_err"] = test_mean_root_vel_err.item()
                test_info["test_mean_root_ang_vel_tracking_err"] = test_mean_root_ang_vel_err.item()
        return test_info
    
    def _train_iter(self):
        info = super()._train_iter()

        for key in self._train_return_tracker._mean_returns:
            info[key] = self._train_return_tracker.get_specific_mean_return(key).item()
        

        return info
    
    def train_model(self, max_samples, out_model_file, int_output_dir, log_file, logger_type):
        start_time = time.time()

        self._curr_obs, self._curr_info = self._env.reset()

        # TODO: use the logger type keyword?
        self._logger = self._build_logger(log_file)
        self._init_train()

        while self._sample_count < max_samples:
            

            train_info = self._train_iter()
            
            output_iter = (self._iter % self._iters_per_output == 0)
            if (output_iter):
                test_info = self.test_model(self._test_episodes)
                extra_log_info = self._env.get_extra_log_info()
                for collection in extra_log_info:
                    for k, v in extra_log_info[collection].items():
                        self._logger.log(k, v, collection=collection, quiet=True)
                self._env.post_test_update()
            
            self._sample_count = self._update_sample_count()
            self._log_train_info(train_info, test_info, start_time)
            self._logger.print_log()

            

            if (output_iter):
                self._logger.write_log()
                
                self._train_return_tracker.reset()
                #self._curr_obs, self._curr_info = self._env.reset()
                self.hard_reset_envs()

            checkpoint_iter = (self._iter % self._iters_per_checkpoint == 0)
            if (checkpoint_iter):
                self._output_train_model(self._iter, out_model_file, int_output_dir)
            
            self._iter += 1

        return

    def _log_train_info(self, train_info, test_info, start_time):
        super()._log_train_info(train_info, test_info, start_time)
        return
    
    def _build_exp_buffer(self, config):
        super()._build_exp_buffer(config)

        buffer_length = self._get_exp_buffer_length()
        batch_size = self.get_num_envs()

        timestep_buffer = torch.zeros(size=[buffer_length, batch_size], dtype=torch.int, device=self._device)
        self._exp_buffer.add_buffer("timestep", timestep_buffer)

        ep_num_buffer = torch.zeros(size=[buffer_length, batch_size], dtype=torch.int, device=self._device)
        self._exp_buffer.add_buffer("ep_num", ep_num_buffer)

        compute_time_buffer = torch.zeros(size=[buffer_length, batch_size], dtype=torch.float32, device=self._device)
        self._exp_buffer.add_buffer("compute_time", compute_time_buffer)
        
        prev_contact_force_buffer = torch.zeros(size=[buffer_length, batch_size, 15, 3], dtype=torch.float32, device=self._device)
        self._exp_buffer.add_buffer("prev_char_contact_forces", prev_contact_force_buffer)

        next_contact_force_buffer = torch.zeros(size=[buffer_length, batch_size, 15, 3], dtype=torch.float32, device=self._device)
        self._exp_buffer.add_buffer("next_char_contact_forces", next_contact_force_buffer)

        env_id_buffer = torch.zeros(size=[buffer_length, batch_size], dtype=torch.int64, device=self._device)
        self._exp_buffer.add_buffer("env_id", env_id_buffer)

        self._env_ids = torch.arange(0, batch_size, 1, device=self._device, dtype=torch.int64)
        return
    
    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)


        self._exp_buffer.record("prev_char_contact_forces", info["char_contact_forces"])
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record("timestep", next_info["timestep"])
        self._exp_buffer.record("ep_num", next_info["ep_num"])

        num_envs = self.get_num_envs()
        compute_time = next_info["compute_time"] * torch.ones([num_envs], dtype=torch.float32, device=self._device)
        self._exp_buffer.record("compute_time", compute_time)
        self._exp_buffer.record("next_char_contact_forces", next_info["char_contact_forces"])

        self._exp_buffer.record("env_id", self._env_ids.detach().clone())
        return
    
    def _build_train_data(self):
        self.eval()
        
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")
        
        norm_next_obs = self._obs_norm.normalize(next_obs)

        ## FOR TRANSFORMER and CNN ##
        # split eval critic into smaller batches because transformer can't take it all in memory
        # next_vals = torch.zeros(size=norm_next_obs.shape[0:2], dtype=torch.float32, device=self._device)
        # for i in range(norm_next_obs.shape[0]):
        #     curr_next_vals = self._model.eval_critic(norm_next_obs[i])
        #     curr_next_vals = curr_next_vals.squeeze(-1).detach()
        #     next_vals[i] = curr_next_vals

        ## FOR MLP ##
        next_vals = self._model.eval_critic(norm_next_obs)
        next_vals = next_vals.squeeze(-1).detach()

        val_min, val_max = self._compute_val_bound()
        next_vals = torch.clamp(next_vals, val_min, val_max)

        succ_val = self._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = self._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, self._discount, self._td_lambda)

        norm_obs = self._obs_norm.normalize(obs)

        ## FOR TRANSFORMER and CNN ##
        # vals = torch.zeros(size=norm_obs.shape[0:2], dtype=torch.float32, device=self._device)
        # for i in range(norm_obs.shape[0]):
        #     curr_vals = self._model.eval_critic(norm_obs[i])
        #     curr_vals = curr_vals.squeeze(-1).detach()
        #     vals[i] = curr_vals

        ## FOR MLP ##
        vals = self._model.eval_critic(norm_obs)
        vals = vals.squeeze(-1).detach()


        adv = new_vals - vals
        
        rand_action_mask = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        rand_action_adv = adv_flat[rand_action_mask]
        adv_std, adv_mean = torch.std_mean(rand_action_adv)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)
        
        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)
        
        adv_std, adv_mean = torch.std_mean(rand_action_adv)

        info = {
            "adv_mean": adv_mean,
            "adv_std": adv_std
        }
        return info
    

    def _output_train_model(self, iter, out_model_file, int_output_dir):
        super()._output_train_model(iter, out_model_file, int_output_dir)

        if (int_output_dir != "") and self._env.has_dm_envs():
            int_fail_rates_file = os.path.join(int_output_dir, "fail_rates_{:010d}.pt".format(iter))     
            torch.save(self._env.get_dm_env()._motion_id_fail_rates.cpu(), int_fail_rates_file)
        return