import enum
import os

import gym
import isaacgym.gymapi as gymapi
import isaacgym.gymtorch as gymtorch
import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motion_tracker.envs.base_env as base_env
import parc.motion_tracker.envs.ig_env as ig_env
import parc.util.torch_util as torch_util


class CameraMode(enum.Enum):
    still = 0
    track = 1

class ControlMode(enum.Enum):
    pd = 0
    vel = 1
    torque = 2
    pd_exp = 3
    pd_1d = 4

class IGCharEnv(ig_env.IGEnv):
    def __init__(self, config, num_envs, device, visualize):
        env_config = config["env"]
        self._env_spacing = env_config['env_spacing']
        self._global_obs = env_config["global_obs"]
        self._root_height_obs = env_config.get("root_height_obs", True)

        self._camera_mode = CameraMode[env_config["camera_mode"]]
        self._camera_env_id = 0

        super().__init__(config=config, num_envs=num_envs, device=device,
                         visualize=visualize)
        return

    def get_reward_bounds(self):
        return (0.0, 1.0)

    def _parse_init_pose(self, init_pose, device):
        if (init_pose is not None):
            init_pose = torch.tensor(init_pose, device=device)
        else:
            dof_size = self._kin_char_model.get_dof_size()
            init_pose = torch.zeros(6 + dof_size, dtype=torch.float32, device=device)
            
        init_root_pos, init_root_rot, init_dof_pos = motion_lib.extract_pose_data(init_pose)
        self._init_root_pos = init_root_pos
        self._init_root_rot = torch_util.exp_map_to_quat(init_root_rot)
        self._init_dof_pos = init_dof_pos
        return

    def _build_envs(self, config):
        self._char_handles = []

        terrain_type = config["env"].get("terrain", "ground_plane")
        if terrain_type == "ground_plane":
            self._build_ground_plane(config)
            self._use_heightmap = False
        elif terrain_type == "heightmap":
            self._build_heightmap(config)
            self._use_heightmap = True
        else:
            assert False
        self._load_char_asset(config)
        
        env_config = config["env"]
        init_pose = env_config.get("init_pose", None)
        self._parse_init_pose(init_pose, self._device)
        
        super()._build_envs(config)

        self._check_char_model()
        
        return

    def _build_env(self, env_id, env_ptr, config):
        char_handle = self._build_character(env_id, env_ptr, config)
        if (env_id == 0):
            self._print_actor_prop(env_ptr, char_handle)

        self._char_handles.append(char_handle)
        
        if self._enable_dof_force_sensors():
            self._gym.enable_actor_dof_force_sensors(env_ptr, char_handle)
        return 
    
    def _load_char_asset(self, config):
        char_file = config["env"]["char_file"]
        self._char_control_mode = ControlMode[config["env"]["control_mode"]]

        asset_options = self._build_char_asset_options(config)
        char_file_dir = os.path.dirname(char_file)
        char_file_filename = os.path.basename(char_file)
        self._char_asset = self._gym.load_asset(self._sim, char_file_dir, char_file_filename, asset_options)

        self._build_kin_char_model(char_file)
        return

    def _build_character(self, env_id, env_ptr, config):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        char_handle = self._gym.create_actor(env_ptr, self._char_asset, start_pose, "character", col_group, col_filter, segmentation_id)
        
        drive_mode = self._control_mode_to_drive_mode(self._char_control_mode)
        dof_prop = self._gym.get_asset_dof_properties(self._char_asset)
        dof_prop["driveMode"] = drive_mode
        
        if (self._char_control_mode == ControlMode.pd):
            pass
        elif (self._char_control_mode == ControlMode.vel):
            dof_prop["stiffness"] = 0.0
        elif (self._char_control_mode == ControlMode.torque):
            dof_prop["stiffness"] = 0.0
            dof_prop["damping"] = 0.0
        elif (self._char_control_mode == ControlMode.pd_exp):
            dof_prop["stiffness"] = 0.0
            dof_prop["damping"] = 0.0
        elif (self._char_control_mode == ControlMode.pd_1d):
            dof_prop["stiffness"] = 0.0
            dof_prop["damping"] = 0.0
        else:
            assert(False), "Unsupported control mode: {}".format(self._char_control_mode)
        
        self._gym.set_actor_dof_properties(env_ptr, char_handle, dof_prop)

        return char_handle

    def _build_char_asset_options(self, config):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0

        drive_mode = self._control_mode_to_drive_mode(self._char_control_mode)
        asset_options.default_dof_drive_mode = drive_mode
        return asset_options

    def _get_env_spacing(self):
        return self._env_spacing

    def _build_kin_char_model(self, char_file):
        if not hasattr(self, "_kin_char_model"):
            _, file_ext = os.path.splitext(char_file)
            if (file_ext == ".xml"):
                char_model = kin_char_model.KinCharModel(self._device)
            else:
                print("Unsupported character file format: {:s}".format(file_ext))
                assert(False)

            self._kin_char_model = char_model
            self._kin_char_model.load_char_file(char_file)
        return
    
    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        env_handle = self._envs[0]
        char_handle = self._get_char_actor_handle()
        num_envs = self.get_num_envs()

        actors_per_env = self._get_actors_per_env()
        actor_root_state = self._root_state.view([num_envs, actors_per_env, self._root_state.shape[-1]])
        self._char_root_pos = actor_root_state[:, 0, 0:3]
        self._char_root_rot = actor_root_state[:, 0, 3:7]
        self._char_root_vel = actor_root_state[:, 0, 7:10]
        self._char_root_ang_vel = actor_root_state[:, 0, 10:13]
        
        dofs_per_env = self._dof_state.shape[0] // num_envs
        num_char_dofs = self._gym.get_actor_dof_count(env_handle, char_handle)
        dof_state = self._dof_state.view([num_envs, dofs_per_env, 2])
        self._char_dof_pos = dof_state[..., :num_char_dofs, 0]
        self._char_dof_vel = dof_state[..., :num_char_dofs, 1]
        
        # Note: NEVER use these tensors in observations calculations
        # they are not updated immediately during episode resets, and are not valid
        # until the second timestep after a reset
        bodies_per_env = self._rigid_body_state.shape[0] // num_envs
        num_char_bodies = self._gym.get_actor_rigid_body_count(env_handle, char_handle)
        rigid_body_state = self._rigid_body_state.view([num_envs, bodies_per_env, self._rigid_body_state.shape[-1]])
        self._char_rigid_body_pos = rigid_body_state[..., :num_char_bodies, 0:3]
        self._char_rigid_body_rot = rigid_body_state[..., :num_char_bodies, 3:7]
        self._char_rigid_body_vel = rigid_body_state[..., :num_char_bodies, 7:10]
        self._char_rigid_body_ang_vel = rigid_body_state[..., :num_char_bodies, 10:13]
        
        self._char_contact_forces = self._contact_forces.view([num_envs, bodies_per_env, 3])[..., :num_char_bodies, :]

        if self._enable_dof_force_sensors():
            self._char_dof_forces = self._dof_forces.view([num_envs, dofs_per_env])[..., :num_char_dofs]
        
        action_space = self._build_action_space()
        self._action_bound_low = torch.tensor(action_space.low, device=self._device)
        self._action_bound_high = torch.tensor(action_space.high, device=self._device)
        self._action_buffer = torch.zeros([num_envs, dofs_per_env], device=self._device, dtype=torch.float32)
        a_dim = self._action_bound_low.shape[-1]
        self._char_action_buffer = self._action_buffer[..., 0:a_dim]
        
        key_bodies = config["env"].get("key_bodies", [])
        print('asda',key_bodies)
        self._key_body_ids = self._build_body_ids_tensor(key_bodies)
        
        if (self._char_control_mode == ControlMode.pd_exp
            or self._char_control_mode == ControlMode.pd_1d):
            self._build_pd_exp_tensors()
        
        return
    
    def _check_char_model(self):
        # checks to make sure the kinematic model is consistent with the simulation model
        sim_body_names = self._gym.get_actor_rigid_body_names(self._envs[0], self._get_char_actor_handle())
        kin_body_names = self._kin_char_model.get_body_names()

        for sim_name, kin_name in zip(sim_body_names, kin_body_names):
            assert(sim_name == kin_name)

        return

    def _build_pd_exp_tensors(self):
        dof_prop = self._gym.get_asset_dof_properties(self._char_asset)
        kp = dof_prop["stiffness"]
        kd = dof_prop["damping"]
        self._pd_exp_kp = torch.tensor(kp, device=self._device, dtype=torch.float32)
        self._pd_exp_kd = torch.tensor(kd, device=self._device, dtype=torch.float32)

        env_handle = self._envs[0]
        char_handle = self._get_char_actor_handle()
        actuator_props = self._gym.get_actor_actuator_properties(env_handle, char_handle)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        torque_lim = np.array(motor_efforts)
        self._pd_exp_torque_lim = torch.tensor(torque_lim, device=self._device, dtype=torch.float32)

        num_envs = self.get_num_envs()
        self._pd_exp_tar = torch.zeros([num_envs, torque_lim.shape[-1]], device=self._device, dtype=torch.float32)
        
        # check to make sure that pd_1d is only used for 1D joints
        if (self._char_control_mode == ControlMode.pd_1d):
            num_joints = self._kin_char_model.get_num_joints()
            for j in range(1, num_joints):
                j_dim = self._kin_char_model.get_joint_dof_dim(j)
                assert(j_dim == 1), "pd_1d only supports 1D joints"

        return
    
    def _build_action_space(self):
        if (self._char_control_mode == ControlMode.pd):
            low, high = self._build_action_bounds_pd()
        elif (self._char_control_mode == ControlMode.vel):
            low, high = self._build_action_bounds_vel()
        elif (self._char_control_mode == ControlMode.torque):
            low, high = self._build_action_bounds_torque()
        elif (self._char_control_mode == ControlMode.pd_exp):
            low, high = self._build_action_bounds_pd()
        elif (self._char_control_mode == ControlMode.pd_1d):
            low, high = self._build_action_bounds_pd()
        else:
            assert(False), "Unsupported control mode: {}".format(self._char_control_mode)

        action_space = gym.spaces.Box(low=low, high=high)
        return action_space
    
    def _build_body_ids_tensor(self, body_names):
        env_ptr = self._envs[0]
        char_handle = self._get_char_actor_handle()
        body_ids = []

        for body_name in body_names:
            body_id = self._gym.find_actor_rigid_body_handle(env_ptr, char_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = torch.tensor(body_ids, device=self._device, dtype=torch.long)
        return body_ids
    
    def _print_actor_prop(self, env_ptr, char_handle):
        num_dofs = self._gym.get_actor_dof_count(env_ptr, char_handle)
        rb_props = self._gym.get_actor_rigid_body_properties(env_ptr, char_handle)
        total_mass = sum(rb.mass for rb in rb_props)
        char_info = "Char properties\n\tDoFs: {:d}\n\tMass: {:.3f} kg\n".format(num_dofs, total_mass)
        print(char_info)
        return
    
    def _control_mode_to_drive_mode(self, mode):
        if (mode == ControlMode.pd):
            drive_mode = gymapi.DOF_MODE_POS
        elif (mode == ControlMode.vel):
            drive_mode = gymapi.DOF_MODE_VEL
        elif (mode == ControlMode.torque):
            drive_mode = gymapi.DOF_MODE_EFFORT  
        elif (mode == ControlMode.pd_exp):
            drive_mode = gymapi.DOF_MODE_EFFORT 
        elif (mode == ControlMode.pd_1d):
            drive_mode = gymapi.DOF_MODE_EFFORT 
        else:
            assert(False), "Unsupported control mode: {}".format(mode)
        return drive_mode
    
    def _build_action_bounds_pd(self):
        env_handle = self._envs[0]
        char_handle = self._get_char_actor_handle()
        dof_prop = self._gym.get_actor_dof_properties(env_handle, char_handle)
        dof_low = dof_prop["lower"]
        dof_high = dof_prop["upper"]
        
        low = np.zeros(dof_high.shape)
        high = np.zeros(dof_high.shape)

        num_joints = self._kin_char_model.get_num_joints()
        for j in range(1, num_joints):
            curr_joint = self._kin_char_model.get_joint(j)
            j_dof_dim = curr_joint.get_dof_dim()

            if (j_dof_dim > 0):
                if (j_dof_dim == 3): # 3D spherical j
                    # spherical joints are modeled as exponential maps
                    # so the bounds are computed a bit differently from revolute joints
                    j_low = curr_joint.get_joint_dof(dof_low)
                    j_high = curr_joint.get_joint_dof(dof_high)
                    j_low = np.max(np.abs(j_low))
                    j_high = np.max(np.abs(j_high))
                    curr_scale = max([j_low, j_high])
                    curr_scale = 1.2 * curr_scale

                    curr_low = -curr_scale
                    curr_high = curr_scale
                else:
                    j_low = curr_joint.get_joint_dof(dof_low)
                    j_high = curr_joint.get_joint_dof(dof_high)

                    curr_mid = 0.5 * (j_high + j_low)
                    curr_scale = 0.7 * (j_high - j_low)
                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                curr_joint.set_joint_dof(curr_low, low)
                curr_joint.set_joint_dof(curr_high, high)

        return low, high

    def _build_action_bounds_vel(self):
        action_size = int(self._char_dof_pos.shape[-1])
        low = -2.0 * np.pi * np.ones([action_size])
        high = 2.0 * np.pi * np.ones([action_size])
        return low, high

    def _build_action_bounds_torque(self):
        env_handle = self._envs[0]
        char_handle = self._get_char_actor_handle()
        actuator_props = self._gym.get_actor_actuator_properties(env_handle, char_handle)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        low = -np.array(motor_efforts)
        high = np.array(motor_efforts)
        return low, high

    def _pre_physics_step(self, actions):
        self._apply_action(actions)
        super()._pre_physics_step(actions)
        return
    
    def _step_sim(self):
        self._apply_forces()
        super()._step_sim()
        return

    def _get_char_actor_handle(self):
        return self._char_handles[0]
        
    def _apply_forces(self):
        if (self._char_control_mode == ControlMode.torque):
            action_tensor = gymtorch.unwrap_tensor(self._action_buffer)
            self._gym.set_dof_actuation_force_tensor(self._sim, action_tensor)

        elif (self._char_control_mode == ControlMode.pd_exp):
            self._gym.refresh_dof_state_tensor(self._sim)
            torque_tensor = self._calc_pd_exp_torque()
            self._char_action_buffer[:] = torque_tensor
            action_tensor = gymtorch.unwrap_tensor(self._action_buffer)
            self._gym.set_dof_actuation_force_tensor(self._sim, action_tensor)

        elif (self._char_control_mode == ControlMode.pd_1d):
            self._gym.refresh_dof_state_tensor(self._sim)
            torque_tensor = self._calc_pd_1d_torque()
            self._char_action_buffer[:] = torque_tensor
            action_tensor = gymtorch.unwrap_tensor(self._action_buffer)
            self._gym.set_dof_actuation_force_tensor(self._sim, action_tensor)

        return
    
    def _calc_pd_exp_torque(self):
        sim_dof = self._char_dof_pos
        sim_dof_vel = self._char_dof_vel
        tar_dof = self._pd_exp_tar
        sim_joint_rot = self._kin_char_model.dof_to_rot(sim_dof)
        tar_joint_rot = self._kin_char_model.dof_to_rot(tar_dof)

        diff_dof = self._kin_char_model.compute_dof_vel(sim_joint_rot, tar_joint_rot, 1.0)
        torque = self._pd_exp_kp * diff_dof - self._pd_exp_kd * sim_dof_vel
        torque = torch.clip(torque, -self._pd_exp_torque_lim, self._pd_exp_torque_lim)

        return torque

    def _calc_pd_1d_torque(self):
        sim_dof = self._char_dof_pos
        sim_dof_vel = self._char_dof_vel
        tar_dof = self._pd_exp_tar

        torque = self._pd_exp_kp * (tar_dof - sim_dof) - self._pd_exp_kd * sim_dof_vel
        torque = torch.clip(torque, -self._pd_exp_torque_lim, self._pd_exp_torque_lim)

        return torque

    def _update_reward(self):
        self._reward_buf[:] = compute_reward(self._char_root_pos)
        return

    def _update_done(self):
        self._done_buf[:] = compute_done(self._done_buf, self._time_buf, 
                                         self._episode_length)
        return
    
    def _compute_obs(self, env_ids=None):
        if (env_ids is None):
            root_pos = self._char_root_pos
            root_rot = self._char_root_rot
            root_vel = self._char_root_vel
            root_ang_vel = self._char_root_ang_vel
            dof_pos = self._char_dof_pos
            dof_vel = self._char_dof_vel
        else:
            root_pos = self._char_root_pos[env_ids]
            root_rot = self._char_root_rot[env_ids]
            root_vel = self._char_root_vel[env_ids]
            root_ang_vel = self._char_root_ang_vel[env_ids]
            dof_pos = self._char_dof_pos[env_ids]
            dof_vel = self._char_dof_vel[env_ids]

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        if (self._has_key_bodies()):
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        obs = compute_char_obs(root_pos=root_pos,
                               root_rot=root_rot, 
                               root_vel=root_vel,
                               root_ang_vel=root_ang_vel,
                               joint_rot=joint_rot,
                               dof_vel=dof_vel,
                               key_pos=key_pos,
                               global_obs=self._global_obs,
                               root_height_obs=self._root_height_obs)
        return obs
    
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            self._reset_char(env_ids)
        return

    def _reset_char(self, env_ids):
        self._char_rigid_body_vel[env_ids] = 0.0
        self._char_rigid_body_ang_vel[env_ids] = 0.0
        
        self._char_root_pos[env_ids, :] = self._init_root_pos
        self._char_root_rot[env_ids, :] = self._init_root_rot
        self._char_root_vel[env_ids, :] = 0.0
        self._char_root_ang_vel[env_ids, :] = 0.0
        
        self._char_dof_pos[env_ids, :] = self._init_dof_pos
        self._char_dof_vel[env_ids, :] = 0.0

        char_handle = self._get_char_actor_handle()
        self._actors_need_reset[env_ids, char_handle] = True
        return

    def _apply_action(self, actions):
        clip_action = torch.minimum(torch.maximum(actions, self._action_bound_low), self._action_bound_high)
        self._char_action_buffer[:] = clip_action

        if (self._char_control_mode == ControlMode.pd):
            action_tensor = gymtorch.unwrap_tensor(self._action_buffer)
            self._gym.set_dof_position_target_tensor(self._sim, action_tensor)
        elif (self._char_control_mode == ControlMode.vel):
            action_tensor = gymtorch.unwrap_tensor(self._action_buffer)
            self._gym.set_dof_velocity_target_tensor(self._sim, action_tensor)
        elif (self._char_control_mode == ControlMode.torque):
            pass
        elif (self._char_control_mode == ControlMode.pd_exp):
            self._pd_exp_tar[:] = actions
        elif (self._char_control_mode == ControlMode.pd_1d):
            self._pd_exp_tar[:] = actions
        else:
            assert(False), "Unsupported drive mode: {}".format(self._char_drive_mode)
        return

    def _has_key_bodies(self):
        return len(self._key_body_ids) > 0

    def _init_camera(self):
        #self._gym.refresh_actor_root_state_tensor(self._sim)
        char_pos = self._char_root_pos[0].cpu().numpy()
        
        cam_pos = gymapi.Vec3(char_pos[0], char_pos[1] - 5.0, 3.0)
        cam_target = gymapi.Vec3(char_pos[0], char_pos[1], 0.0)
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)
        self._cam_prev_char_pos = char_pos
        return

    def _update_camera(self):
        if (self._camera_mode is CameraMode.still):
            pass
        elif (self._camera_mode is CameraMode.track):
            #self._gym.refresh_actor_root_state_tensor(self._sim)
            # the rl loop should handle this
            root_pos = (self._char_root_pos[self._camera_env_id] + self._env_offsets[self._camera_env_id]).cpu().numpy()
        
            cam_trans = self._gym.get_viewer_camera_transform(self._viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
            cam_delta = cam_pos - self._cam_prev_char_pos

            new_cam_target = gymapi.Vec3(root_pos[0], root_pos[1], 1.0)
            new_cam_pos = gymapi.Vec3(root_pos[0] + cam_delta[0], 
                                      root_pos[1] + cam_delta[1], 
                                      cam_pos[2])

            self._gym.viewer_camera_look_at(self._viewer, None, new_cam_pos, new_cam_target)

            self._cam_prev_char_pos[:] = root_pos
        else:
            assert(False), "Unsupported camera mode {}".format(self._camera_mode)
        return



#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def convert_to_local_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    heading_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                            heading_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos

@torch.jit.script
def convert_to_local_root_body_pos(root_rot, body_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    root_inv_rot = torch_util.quat_conjugate(root_rot)
    root_rot_expand = root_inv_rot.unsqueeze(-2)
    root_rot_expand = root_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_root_rot_expand = root_rot_expand.reshape(root_rot_expand.shape[0] * root_rot_expand.shape[1], 
                                                   root_rot_expand.shape[2])
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_root_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos

@torch.jit.script
def compute_char_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    
    if (global_obs):
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel
    else:
        local_root_rot = torch_util.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)


    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1], joint_rot.shape[2]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1] * joint_rot_obs_flat.shape[-1]])

    obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]

    if (len(key_pos) > 0):
        root_pos_expand = root_pos.unsqueeze(-2)
        key_pos = key_pos - root_pos_expand
        if (not global_obs):
            #key_pos = convert_to_local_body_pos(root_rot, key_pos)
            #key_pos = convert_to_local_body_pos(original_root_rot, key_pos)
            heading_rot_expand = heading_rot.unsqueeze(-2)
            heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
            flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                                    heading_rot_expand.shape[2])
            flat_body_pos = key_pos.reshape(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
            flat_local_body_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_body_pos)
            key_pos = flat_local_body_pos.reshape(key_pos.shape[0], key_pos.shape[1], key_pos.shape[2])

        key_pos_flat = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2]])
        obs = obs + [key_pos_flat]

    if (root_height_obs):
        root_h = root_pos[:, 2:3]
        obs = [root_h] + obs
    
    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_reward(root_pos):
    # type: (Tensor) -> Tensor
    r = torch.ones_like(root_pos[..., 0])
    return r


@torch.jit.script
def compute_done(done_buf, time, ep_len):
    # type: (Tensor, Tensor, float) -> Tensor
    timeout = time >= ep_len

    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    done[timeout] = base_env.DoneFlags.TIME.value
    return done