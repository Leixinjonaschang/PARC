import os
import torch
import learning.ppo_agent as ppo_agent
import learning.base_agent as base_agent
import util.torch_util as torch_util
import learning.normalizer as normalizer
# import envs.ig_terrain_runner_env as terrain_runner
import learning.dm_ppo_model as dm_ppo_model
import util.tb_logger as tb_logger
import util.wandb_logger as wandb_logger
import util.mp_util as mp_util
import learning.rl_util as rl_util
import envs.base_env as base_env
import envs.ig_parkour.ig_parkour_env as ig_parkour_env
from learning.dm_ppo_return_tracker import DMPPOReturnTracker
import numpy as np
import copy
import time
from learning.base_agent import AgentMode
import learning.tracking_error_tracker as tracking_error_tracker
import random

class DMPPOAgent(ppo_agent.PPOAgent):

    NAME = "DM_PPO"

    def __init__(self, config, env: ig_parkour_env.IGParkourEnv, device):

        self._is_terrain_runner = env.NAME == "ig_terrain_runner" or env.NAME == "ig_parkour"
        self._has_target_task = env.NAME == "ig_terrain_runner" or \
            env.NAME == "ig_deepmimic_terrain" or \
            env.NAME == "ig_parkour"
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

    def _check_nan(self, tensor, name, exit_on_nan=True):
        if torch.isnan(tensor).any():
            count = torch.isnan(tensor).sum().item()
            print(f"CRITICAL ERROR: NaN in {name} - count: {count}/{tensor.numel()}")
            if exit_on_nan:
                print("Exiting program due to NaN detection.")
                exit()
            return True
        return False
    
    def _load_params(self, config):
        super()._load_params(config)

        self._config = config
        return
    
    def _build_logger(self, log_file):
        project_name = self._config["project_name"]
        exp_name = self._config["exp_name"]
        wandb_config = copy.deepcopy(self._env._config)
        wandb_config["agent"] = self._config
        use_wandb = self._config["use_wandb"]


        if use_wandb:
            log = wandb_logger.WandbLogger(project_name=project_name, exp_name=exp_name, connect_to_wandb=use_wandb,
                                        param_config=wandb_config)
        else:
            log = tb_logger.TBLogger()
        #log = tb_logger.TBLogger(project_name=project_name, exp_name=exp_name, wandb_host=use_wandb, config=wandb_config)
        log.set_step_key("Samples")
        if (mp_util.is_root_proc()):
            log.configure_output_file(log_file)
        return log
    
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
        if self._is_terrain_runner:
            self._env.apply_hard_reset()
            
        self._curr_obs, self._curr_info = self._env.reset()

        return
    
    def set_predictor_test_mode(self, enabled: bool):
        if hasattr(self._model, "set_predictor_test_mode"):
            self._model.set_predictor_test_mode(enabled)
        return

    def test_model(self, num_episodes):
        self.eval()
        self.set_mode(base_agent.AgentMode.TEST)

        # Enable action debugging
        self._debug_action_print = True
        if hasattr(self._env, '_debug_action_print'):
            self._env._debug_action_print = True

        print("******************** TESTING MODEL ********************")
        print("Action debugging enabled - will print first 10 steps")
        self.hard_reset_envs()
        test_info = self._rollout_test(num_episodes)
        print("******************** FINISHED TESTING MODEL ********************")
        
        # Disable action debugging
        self._debug_action_print = False
        if hasattr(self._env, '_debug_action_print'):
            self._env._debug_action_print = False
        
        return test_info
    
    def _build_return_tracker(self):
        self._train_return_tracker = DMPPOReturnTracker(self.get_num_envs(), self._device, self._has_target_task)
        self._test_return_tracker = DMPPOReturnTracker(self.get_num_envs(), self._device, self._has_target_task)
        return
    
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

    def _compute_loss(self, batch):
        info = super()._compute_loss(batch)
        
        if "norm_obs" in batch:
            norm_obs = batch["norm_obs"]
        else:
            norm_obs = self._obs_norm.normalize(batch["obs"])
            
        predictor_info = self._model.train_future_pose_predictor(norm_obs)
        info.update(predictor_info)
        
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

        if self._is_terrain_runner:
            self._logger.log("replan timer", self._env.get_replan_time_buf().item())
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
        
        num_char_bodies = self._env._kin_char_model.get_num_joints()
        prev_contact_force_buffer = torch.zeros(size=[buffer_length, batch_size, num_char_bodies, 3], dtype=torch.float32, device=self._device)
        self._exp_buffer.add_buffer("prev_char_contact_forces", prev_contact_force_buffer)

        next_contact_force_buffer = torch.zeros(size=[buffer_length, batch_size, num_char_bodies, 3], dtype=torch.float32, device=self._device)
        self._exp_buffer.add_buffer("next_char_contact_forces", next_contact_force_buffer)

        env_id_buffer = torch.zeros(size=[buffer_length, batch_size], dtype=torch.int64, device=self._device)
        self._exp_buffer.add_buffer("env_id", env_id_buffer)

        self._env_ids = torch.arange(0, batch_size, 1, device=self._device, dtype=torch.int64)
        
        if self._is_terrain_runner:
            replan_timer_buffer = torch.zeros(size=[buffer_length, batch_size], dtype=torch.float32, device=self._device)
            self._exp_buffer.add_buffer("replan_timer", replan_timer_buffer)
            
            replan_counter_buffer = torch.zeros(size=[buffer_length, batch_size], dtype=torch.int64, device=self._device)
            self._exp_buffer.add_buffer("replan_counter", replan_counter_buffer)
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


        if self._is_terrain_runner:
            replan_timer = self._env.get_replan_time_buf().repeat(self.get_num_envs())
            self._exp_buffer.record("replan_timer", replan_timer)

            replan_counter = self._env.get_replan_counter()
            self._exp_buffer.record("replan_counter", replan_counter)
        return
    
    def _build_train_data(self):
        self.eval()
        
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")
        
        norm_next_obs = self._obs_norm.normalize(next_obs)

        # 检查关键变量的 NaN
        self._check_nan(obs, "obs")
        self._check_nan(norm_next_obs, "norm_next_obs")
        self._check_nan(r, "reward")
        self._check_nan(done, "done")
        self._check_nan(rand_action_mask, "rand_action_mask")

        ## FOR MLP ##
        next_vals = self._model.eval_critic(norm_next_obs)
        next_vals = next_vals.squeeze(-1).detach()
        self._check_nan(next_vals, "next_vals")

        val_min, val_max = self._compute_val_bound()
        next_vals = torch.clamp(next_vals, val_min, val_max)

        succ_val = self._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = self._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, self._discount, self._td_lambda)
        self._check_nan(new_vals, "new_vals (TD returns)")

        norm_obs = self._obs_norm.normalize(obs)

        ## FOR MLP ##
        vals = self._model.eval_critic(norm_obs)
        vals = vals.squeeze(-1).detach()
        self._check_nan(vals, "vals (critic output)")

        adv = new_vals - vals
        self._check_nan(adv, "raw advantage")
        
        rand_action_mask = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        rand_action_adv = adv_flat[rand_action_mask]
        
        if rand_action_adv.numel() == 0:
            print("ERROR: No random actions in batch!")
            rand_action_adv = adv_flat
        
        # 简化优势函数统计
        adv_std, adv_mean = torch.std_mean(rand_action_adv)
        adv_std = torch.clamp_min(adv_std, 1e-5)
        
        norm_adv = (adv - adv_mean) / adv_std
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)
        
        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)
        
        adv_std, adv_mean = torch.std_mean(rand_action_adv)

        info = {
            "adv_mean": adv_mean,
            "adv_std": adv_std
        }
        return info
    

    def record_motions(self):
        self.eval()
        self.set_mode(base_agent.AgentMode.TEST)
        # This script records motions from each agent

        # For now, lets just get recording 1 motion per env
        self._env.set_rand_reset(False)
        self._env.set_demo_mode(True)
        self._env.set_rand_root_pos_offset_scale(0.0)
        self._env._episode_length = 1000.0
        #self._env._enable_early_termination = False

        # set this to false to save disk space
        record_obs = True
        

        # TODO: also keep track of which agents failed
        def record_motions_helper(name_suffix: str, prev_successful_motions = None):
            self._curr_obs, self._curr_info = self._env.reset()
            #test_info = self._rollout_test(num_episodes)

            self._env.build_agent_states_dict(name_suffix, record_obs=record_obs)
            self._env.write_agent_states()

            if prev_successful_motions is not None:
                for env_id in range(len(prev_successful_motions)):
                    self._env.set_writing_env_state(env_id, not prev_successful_motions[env_id])

            while True:
                action, action_info = self._decide_action(self._curr_obs, self._curr_info)

                next_obs, r, done, next_info = self._step_env(action)

                #self._env.write_agent_states()
                self._curr_obs, self._curr_info = self._reset_done_envs(done)

                if not self._env.is_writing_agent_states():
                    print("done writing agent states")
                    break
        
        # TEMPORARY: RECORDING FAILURE VIDEOS
        #self._env._enable_early_termination = False
        #self._env._record_ref = True

        #self._env.get_dm_env().set_motion_start_time_fraction(torch.tensor([0.5], dtype=torch.float32, device=self._device).expand(self._env.get_num_envs()))
        record_motions_helper(name_suffix="_dm")

        successful_motions = copy.deepcopy(self._env.get_env_success_states())

        # TEMPORARY
        # while (sum(successful_motions) == 0):
        #     rand_x = random.random() * 0.2 - 0.1
        #     rand_y = random.random() * 0.2 - 0.1
        #     root_pos_offset = torch.tensor([rand_x, rand_y, 0.0], dtype=torch.float32, device=self._device)
        #     root_pos_offset = root_pos_offset.unsqueeze(0).expand(size=[self._env._num_envs, -1])
        #     self._env.get_dm_env().set_root_pos_offset(root_pos_offset)
        #     record_motions_helper(name_suffix="_dm")
        #     successful_motions = copy.deepcopy(self._env.get_env_success_states())
        #exit()

        
        # try at different starting time fractions
        possible_start_time_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

        # keep track of number of successful motions
        num_successful_motions = []
        num_successful_motions.append(sum(successful_motions))

        

        counter = 0
        num_envs = self.get_num_envs()
        start_time_fraction_per_motion_id = [None] * num_envs
        for id in range(num_envs):
            if successful_motions[id]:
                start_time_fraction_per_motion_id[id] = 0.0

        while not all(successful_motions) and counter < len(possible_start_time_fractions):
            start_time_fraction = possible_start_time_fractions[counter]
            
            
            for id in range(num_envs):
                motion_length = (1.0 - start_time_fraction) * self._env.get_dm_env()._motion_lib.get_motion_length(id).item()
                if motion_length < 2.0:
                    successful_motions[id] = True # don't record motions that will be too short, they will likely be uninteresting
            start_time_fraction_tensor = start_time_fraction * torch.ones(size=[num_envs], dtype=torch.float32, device=self._device)
            self._env.get_dm_env().set_motion_start_time_fraction(start_time_fraction_tensor)
            record_motions_helper(name_suffix="_dm", prev_successful_motions=successful_motions)

            counter += 1

            new_successful_motions = copy.deepcopy(self._env.get_env_success_states())

            num_successful_motions.append(sum(new_successful_motions))
            for id in range(num_envs):
                successful_motions[id] = successful_motions[id] or new_successful_motions[id]

                if new_successful_motions[id]:
                    start_time_fraction_per_motion_id[id] = start_time_fraction

        # TODO: also try something that sets the successful end_time fraction?
        # hard to say because if a motion doesn't complete successfully, it likely
        # looks bad well before the end

        print("Successful motions at 0 percent start time:", num_successful_motions[0])
        print("Success rate:", num_successful_motions[0] / len(successful_motions))


        print("len num_successful_motions:", len(num_successful_motions))

        for i in range(len(num_successful_motions) - 1):
            print(i)
            start_time_fraction = possible_start_time_fractions[i]
            print("Successful motions at", start_time_fraction, "percent start time:", num_successful_motions[i+1])

        print("Total successfull motions:", sum(num_successful_motions))
        print("Total success rate:", sum(num_successful_motions) / len(successful_motions))


        exit()
        # TODO: save the ref motions using the start time fractions
        mlib = self._env.get_dm_env()._motion_lib
        motion_save_folder_path = "output/_motions/successful_ref_motions/"
        os.makedirs(motion_save_folder_path, exist_ok=True)
        for id in range(num_envs):
            motion_length = mlib._motion_lengths[id].item()
            start_time_fraction = start_time_fraction_per_motion_id[id]
            if start_time_fraction is None:
                continue
            start_time = start_time_fraction * motion_length
            fps = mlib._motion_fps[id].item()
            start_frame = int(np.floor(start_time * fps))

            motion_start_idx = mlib._motion_start_idx[id].item()
            motion_end_idx = motion_start_idx + mlib._motion_num_frames[id].item()
            motion_frames = mlib._motion_frames[motion_start_idx + start_frame:motion_end_idx]
            contact_frames = mlib._frame_contacts[motion_start_idx + start_frame:motion_end_idx]
            terrain = mlib._terrains[id]

            import zmotion_editing_tools.motion_edit_lib as medit_lib
            motion_og_name = mlib.get_motion_names()[id]
            motion_save_path = motion_save_folder_path + motion_og_name + "_success_ref.pkl"
            medit_lib.save_motion_data(motion_filepath = motion_save_path,
                                       motion_frames = motion_frames,
                                       contact_frames = contact_frames,
                                       terrain = terrain,
                                       fps = mlib._motion_fps[id].item(),
                                       loop_mode = "CLAMP")

        exit()

        # TODO: keep track of motions that were not able to track, and search for a start time
        # where they do track

        num_augments_per_motion = 2
        radius = 0.35
        max_heading_angle_degrees = 35.0
        max_start_time_fraction = 0.5
        min_start_time_fraction = 0.0
        num_envs = self._env.get_num_envs()
        for i in range(num_augments_per_motion):
            
            root_pos_offset = 2.0 * torch.rand(size=[num_envs, 3], dtype=torch.float32, device=self._device) - 1.0
            root_pos_offset *= radius
            root_pos_offset[..., 2] = 0.0

            root_heading_offset = 2.0 * torch.rand(size=[num_envs], dtype=torch.float32, device=self._device) - 1.0
            root_heading_offset *= max_heading_angle_degrees * torch.pi / 180.0
            z_axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=self._device).expand(num_envs, -1)
            root_rot_offset = torch_util.axis_angle_to_quat(z_axis, root_heading_offset)

            start_time_fraction = torch.rand(size=[num_envs], dtype=torch.float32, device=self._device) * \
                (max_start_time_fraction - min_start_time_fraction) + min_start_time_fraction

            self._env.get_dm_env().set_root_pos_offset(root_pos_offset)
            self._env.get_dm_env().set_root_rot_offset(root_rot_offset)
            self._env.get_dm_env().set_motion_start_time_fraction(start_time_fraction)

            name_suffix = "_dm_aug" + str(i)
            record_motions_helper(name_suffix)
        return
    
    def load(self, in_file):
        super().load(in_file)

        # TODO: make config param
        # if True:
        #     model_config = self._config["model"]
        #     self._model._build_critic(model_config, self._env)
        #     self._model._critic_layers = self._model._critic_layers.to(device=self._device)
        #     self._model._critic_out = self._model._critic_out.to(device=self._device)
        return
    
    def test_model2(self, num_episodes):
        self.eval()
        self.set_mode(AgentMode.TEST)
        

        num_procs = mp_util.get_num_procs()
        num_eps_proc = int(np.ceil(num_episodes / num_procs))

        self._actor_mean_output = None
        self._actor_std_output = None

        relu_layers = []
        self._actor_hidden_layers = []
        self._utility = []
        self._activations = []
        for l in range(len(self._model._actor_layers)):
            layer = self._model._actor_layers[l]
            if isinstance(layer, torch.nn.ReLU):
                relu_layers.append(layer)
            if isinstance(layer, torch.nn.Linear):
                self._actor_hidden_layers.append(layer)
                self._utility.append(torch.zeros(size=[layer.out_features], dtype=torch.float32, device=self._device))
                self._activations.append(torch.zeros(size=[layer.out_features], dtype=torch.float32, device=self._device))
        self._mean_net_acts = torch.zeros(size=[self._model._action_dist._mean_net.out_features], dtype=torch.float32, device=self._device)
        self._intermediate_outputs = []

        # compute stable rank of actor and critic
        for l in range(len(self._actor_hidden_layers)):
            layer = self._actor_hidden_layers[l]
            weight_mat = layer.weight.detach().clone().to(dtype=torch.float64)
            U, S, Vh = torch.linalg.svd(weight_mat)

            S = S.cpu()
            q = S.shape[0]
            s_sum = torch.sum(S)
            curr_s_sum = 0.0
            stable_rank = 0
            for k in range(q):
                curr_s_sum += S[k]
                val = curr_s_sum / s_sum
                if val > 0.99:
                    stable_rank = k
                    break
            print("Max possible rank of layer", l, ":", q)
            print("Stable rank of layer", l, ":", stable_rank)

            print("Average weight magnitude of layer", l, ":", torch.mean(torch.abs(weight_mat)).item())
        
        weight_mat = self._model._action_dist._mean_net.weight.detach().clone().to(dtype=torch.float64)
        U, S, Vh = torch.linalg.svd(weight_mat)

        S = S.cpu()
        q = S.shape[0]
        s_sum = torch.sum(S)
        curr_s_sum = 0.0
        stable_rank = 0
        for k in range(q):
            curr_s_sum += S[k]
            val = curr_s_sum / s_sum
            if val > 0.99:
                stable_rank = k
                break
        print("Max possible rank of mean_net:", q)
        print("Stable rank of mean_net:", stable_rank)
        print("Average weight magnitude of mean_net:", torch.mean(torch.abs(weight_mat)).item())
        

        def save_output(module, input, output):
            self._intermediate_outputs.append(output.detach().clone())
            return
        
        def clear_and_save_output(module, input, output):
            self._intermediate_outputs.clear()
            self._intermediate_outputs.append(output.detach().clone())
            return
        
        def save_output_and_update_utility(module, input, output):
            self._intermediate_outputs.append(output.detach().clone())
            return
        
        def save_mean_net_output_and_update_util(module, input, output):
            self._actor_mean_output = output.detach().clone()

            num_layers = len(self._intermediate_outputs)
            assert num_layers == len(self._utility)
            eta = 0.99
            self._mean_net_acts = self._mean_net_acts * eta + torch.abs(self._actor_mean_output) * (1.0 - eta)
            for l in range(num_layers):
                layer_act = torch.mean(self._intermediate_outputs[l], dim=0)
                layer_util = self._utility[l]

                self._utility[l] = layer_util * eta

                if l < num_layers - 1:
                    next_weights = self._actor_hidden_layers[l+1].weight
                    sum_outgoing_w = torch.sum(torch.abs(next_weights), dim=0)
                else:
                    next_weights = self._model._action_dist._mean_net.weight
                    sum_outgoing_w = torch.sum(torch.abs(next_weights), dim=0)
                self._utility[l] += (1.0 - eta) * torch.abs(layer_act) * sum_outgoing_w

                self._activations[l] = self._activations[l] * eta
                self._activations[l] += (1.0 - eta) * torch.abs(layer_act)

                print("layer", l)
                #print("max utility:", self._utility[l].max())
                # TODO: change to calculating based on average number of units with > 0 activation
                threshold = 0.01
                normalized_activations = self._activations[l]# / self._activations[l].mean()
                num_dormant = torch.count_nonzero(normalized_activations < threshold).item()
                percent_dormant = num_dormant / normalized_activations.shape[0] * 100.0
                print("Percentage dormant units:", percent_dormant, "%")
                print("Number of dormant units:", num_dormant, "/", normalized_activations.shape[0])
                print("Activation mean:", self._activations[l].mean().item())
                print("Activation std:", self._activations[l].std().item())
                print("Activation max:", self._activations[l].max().item())
                print("Activation min:", self._activations[l].min().item())
                print("Utility mean:", self._utility[l].mean().item())
                print("Utility std:", self._utility[l].std().item())
                print("Utility max:", self._utility[l].max().item())
                print("Utility min:", self._utility[l].min().item())

            num_dormant = torch.count_nonzero(self._mean_net_acts < threshold).item()
            percent_dormant = num_dormant / self._mean_net_acts.shape[0] * 100.0
            print("Mean Net Percentage dormant units:", percent_dormant, "%")
            print("Mean Net Number of dormant units:", num_dormant, "/", self._mean_net_acts.shape[0])
            print("Mean Net Activation mean:", self._mean_net_acts.mean().item())
            print("Mean Net Activation std:", self._mean_net_acts.std().item())
            print("Mean Net Activation max:", self._mean_net_acts.max().item())
            print("Mean Net Activation min:", self._mean_net_acts.min().item())
            return
        self._model._action_dist._mean_net.register_forward_hook(save_mean_net_output_and_update_util)

        # for name, param in self._model.named_parameters():
        #     #print(name +":", param)
        #     print(name)
        #     print("mean:", param.mean())
        #     print("std:", param.std())
        #     print("abs max:", param.abs().max())
        #     param.register_forward_hook(save_output)
        #named_layers = dict(self._model.named_modules())
        #print(named_layers)
            #torch.nn.modules.module.register_module_forward_hook()

        

        for l in range(len(relu_layers)):
            layer = relu_layers[l]
            if l == 0: # TODO: make more general
                layer.register_forward_hook(clear_and_save_output)
            elif l == len(relu_layers) - 1:
                layer.register_forward_hook(save_output_and_update_utility)
            else:
                layer.register_forward_hook(save_output)

        # for l in range(len(self._model._critic_layers)):
        #     layer = self._model._critic_layers[l]
        #     if isinstance(layer, torch.nn.ReLU):
        #         layer.register_forward_hook(save_output)


        self._curr_obs, self._curr_info = self._env.reset()
        test_info = self._rollout_test(num_eps_proc)



        return test_info
    
    def _output_train_model(self, iter, out_model_file, int_output_dir):
        super()._output_train_model(iter, out_model_file, int_output_dir)

        if (int_output_dir != "") and self._env.has_dm_envs():
            int_fail_rates_file = os.path.join(int_output_dir, "fail_rates_{:010d}.pt".format(iter))     
            torch.save(self._env.get_dm_env()._motion_id_fail_rates.cpu(), int_fail_rates_file)
        return
    
    def eval_mode(self):
        self.eval()
        self.set_mode(AgentMode.TEST)
        return