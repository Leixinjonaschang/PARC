import abc
import re
import sys

import gym
import isaacgym.gymapi as gymapi
import isaacgym.gymtorch as gymtorch
import isaacgym.gymutil as gymutil
import numpy as np
import torch

import parc.motion_tracker.envs.base_env as base_env
import parc.util.torch_util as torch_util
from parc.util.logger import Logger


class IGEnv(base_env.BaseEnv):
    NAME = "isaacgym"

    def __init__(self, config, num_envs, device, visualize):
        super().__init__(visualize=visualize)
        
        self._viewer = None
        self._num_envs = num_envs
        self._device = device
        self._enable_viewer_sync = True
        self._config = config

        env_config = config["env"]
        self._episode_length = env_config["episode_length"] # episode length in seconds
        print("Episode length:", self._episode_length)
        self._env_spacing = 5 if "env_spacing" not in env_config else env_config["env_spacing"]
        self._env_style = env_config.get("env_style", "square")
        self._build_sim(config)
        self._build_envs(config)
        self._gym.prepare_sim(self._sim)
        
        self._build_sim_tensors(config)
        self._build_data_buffers()

        self._action_space = self._build_action_space()

        if (self._visualize):
            self._build_viewer()
            self._init_camera()
            
        return
    
    def get_num_envs(self):
        return self._num_envs

    def reset(self, env_ids=None):
        if (env_ids is None): # note, not the same as passing in empty tensor []
            num_envs = self.get_num_envs()
            env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)

        self._reset_envs(env_ids)
        
        reset_env_ids = self._reset_sim_tensors()
        self._refresh_sim_tensors()
        self._update_observations(reset_env_ids)
        self._update_info(reset_env_ids)

        return self._obs_buf, self._info
    
    def step(self, action):
        # apply actions
        self._pre_physics_step(action)

        # step physics and render each frame
        self._physics_step()
        
        # to fix!
        if (self._device == "cpu"):
            self._gym.fetch_results(self._sim, True)
            
        if (self._viewer):
            self._update_camera()
            self._render()
        
        # compute observations, rewards, resets, ...
        self._post_physics_step()
        
        return self._obs_buf, self._reward_buf, self._done_buf, self._info
    
    def get_obs_space(self):
        obs = self._compute_obs()
        obs_shape = list(obs.shape[1:])
        obs_dtype = torch_util.torch_dtype_to_numpy(obs.dtype)
        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=obs_dtype,
        )
        return obs_space
    
    def _build_sim(self, config):
        self._gym = gymapi.acquire_gym()
        self._pysics_engine = gymapi.SimType.SIM_PHYSX
        
        env_config = config["env"]
        sim_freq = env_config.get("sim_freq", 60)
        control_freq = env_config.get("control_freq", 10)
        self._control_freq = control_freq
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"
        sim_timestep = 1.0 / sim_freq

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_params = self._parse_sim_params(config, sim_timestep)

        compute_device_id = self._get_device_idx()
        if self._visualize:
            graphics_device_id = compute_device_id
        else:
            graphics_device_id = -1

        self._sim = self._gym.create_sim(compute_device_id, graphics_device_id, self._pysics_engine, self._sim_params)
        assert(self._sim is not None), "Failed to create sim"

        return

    def _get_device_idx(self):
        re_idx = re.search(r"\d", self._device)
        if (re_idx is None):
            device_idx = 0
        else:
            num_idx = re_idx.start()
            device_idx = int(self._device[num_idx:])
        return device_idx

    def _parse_sim_params(self, config, sim_timestep):
        sim_params = gymapi.SimParams()
        sim_params.dt = sim_timestep
        sim_params.num_client_threads = 0
        
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        if "gravity_z" in config["env"]:
            sim_params.gravity.z = config["env"]["gravity_z"]
        else:
            sim_params.gravity.z = -9.81

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.num_subscenes = 0
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

        if ("gpu" in self._device or "cuda" in self._device):
            sim_params.physx.use_gpu = True
            sim_params.use_gpu_pipeline = True
        elif ("cpu" in self._device):
            sim_params.physx.use_gpu = False
            sim_params.use_gpu_pipeline = False
        else:
            assert(False), "Unsupported simulation device: {}".format(self._device)

        # if sim options are provided in cfg, parse them and update/override above:
        if "sim" in config:
            gymutil.parse_sim_config(config["sim"], sim_params)
        
        return sim_params

    def _build_envs(self, config):
        self._envs = []
        env_config = config["env"]

        env_spacing = self._get_env_spacing()
        if self._env_style == "square":
            num_env_per_row = int(np.sqrt(self._num_envs))
            lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
            upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
            self._env_offsets = torch.zeros(size = (self._num_envs,3), dtype=torch.float32, device=self._device)
            for i in range(self._num_envs):
                curr_col = i % num_env_per_row
                curr_row = i // num_env_per_row
                self._env_offsets[i, 0] = env_spacing * 2 * curr_col
                self._env_offsets[i, 1] = env_spacing * 2 * curr_row

            # create spawn boundaries
            self._spawn_max_x = self._env_spacing * 10.0
            self._spawn_min_x = -self._env_spacing * 10.0
            self._spawn_max_y = self._env_spacing * 10.0
            self._spawn_min_y = -self._env_spacing * 10.0
        elif self._env_style == "x_line":
            num_env_per_row = self._num_envs
            lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
            upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
            self._env_offsets = torch.zeros(size = (self._num_envs,3), dtype=torch.float32, device=self._device)
            self._env_offsets[:, 0] = torch.arange(0, self._num_envs, 1, dtype=torch.float32, device=self._device) * config["env"]["env_spacing"] * 2.0
        else:
            assert False
        
        for i in range(self._num_envs):
            Logger.print("Building {:d}/{:d} envs".format(i + 1, self._num_envs), end='\r')
            env_ptr = self._gym.create_env(self._sim, lower, upper, num_env_per_row)
            self._build_env(i, env_ptr, config)
            self._envs.append(env_ptr)

        Logger.print("\n")

        return

    
    @abc.abstractmethod
    def _build_env(self, env_id, env_ptr, config):
        return
    
    @abc.abstractmethod
    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._timestep_buf[env_ids] = 0
            self._time_buf[env_ids] = 0
            self._done_buf[env_ids] = base_env.DoneFlags.NULL.value
        return

    def _reset_sim_tensors(self):
        # note: need_reset_buf and actors_need_reset are same tensor, different views
        actor_ids = self._need_reset_buf.nonzero(as_tuple=False)
        actor_ids = actor_ids.type(torch.int32).flatten()

        if (len(actor_ids) > 0):
            self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                         gymtorch.unwrap_tensor(self._root_state),
                                                         gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
            if (self._dof_state is not None):
                has_dof = self._actor_dof_dims[actor_ids.type(torch.long)] > 0
                dof_actor_ids = actor_ids[has_dof]
                self._gym.set_dof_state_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(self._dof_state),
                                                      gymtorch.unwrap_tensor(dof_actor_ids), len(dof_actor_ids))
                
                dof_pos = self._dof_state[..., :, 0]
                dof_pos = dof_pos.contiguous()
                self._gym.set_dof_position_target_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(dof_pos),
                                                      gymtorch.unwrap_tensor(dof_actor_ids), len(dof_actor_ids))

            reset_env_ids = torch.sum(self._actors_need_reset, dim=-1).nonzero(as_tuple=False)
            reset_env_ids = reset_env_ids.flatten()
            self._actors_need_reset[:] = False
        else:
            reset_env_ids = []

        return reset_env_ids

    def _get_env_spacing(self):
        return self._env_spacing
    
    def _build_ground_plane(self, config):
        env_configs = config["env"]

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = env_configs["plane"]["static_friction"]
        plane_params.dynamic_friction = env_configs["plane"]["dynamic_friction"]
        plane_params.restitution = env_configs["plane"]["restitution"]
        if "z_offset" in env_configs["plane"]:
            plane_params.distance = env_configs["plane"]["z_offset"]
        self._gym.add_ground(self._sim, plane_params)
        return
    
    

    def _build_viewer(self):
        # subscribe to keyboard shortcuts
        self._viewer = self._gym.create_viewer(
            self._sim, gymapi.CameraProperties())
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_ESCAPE, "QUIT")
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_V, "toggle_viewer_sync")

        # set the camera position based on up axis
        sim_params = self._gym.get_sim_params(self._sim)
        if sim_params.up_axis == gymapi.UP_AXIS_Z:
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
        else:
            cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

        self._gym.viewer_camera_look_at(
                self._viewer, None, cam_pos, cam_target)

        return

    def _build_sim_tensors(self, config):
        root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        force_sensor_tensor = self._gym.acquire_force_sensor_tensor(self._sim)
        contact_force_tensor = self._gym.acquire_net_contact_force_tensor(self._sim)
        
        self._root_state = gymtorch.wrap_tensor(root_state_tensor)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor)
        
        if self._enable_dof_force_sensors():
            dof_force_tensor = self._gym.acquire_dof_force_tensor(self._sim)
            self._dof_forces = gymtorch.wrap_tensor(dof_force_tensor)

        return
            
    def _build_data_buffers(self):
        num_envs = self.get_num_envs()
        actors_per_env = self._get_actors_per_env()

        self._reward_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._done_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._timestep_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._time_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)

        self._need_reset_buf = torch.zeros(self._root_state.shape[0], device=self._device, dtype=torch.bool)
        self._actors_need_reset = self._need_reset_buf.view((num_envs, actors_per_env))
        
        obs_space = self.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_buf = torch.zeros([num_envs] + list(obs_space.shape), device=self._device, dtype=obs_dtype)

        self._actor_dof_dims = self._build_actor_dof_dims()

        self._info = dict()

        return

    def _build_actor_dof_dims(self):
        num_envs = self.get_num_envs()
        actors_per_env = self._get_actors_per_env()

        actor_dof_dims = torch.zeros(self._need_reset_buf.shape, device=self._device, dtype=torch.int)
        env_actor_dof_dims = actor_dof_dims.view([num_envs, actors_per_env])

        for e in range(num_envs):
            env_handle = self._envs[e]
            for a in range(actors_per_env):
                num_dofs = self._gym.get_actor_dof_count(env_handle, a)
                env_actor_dof_dims[e, a] = num_dofs

        return actor_dof_dims

    @abc.abstractmethod
    def _build_action_space(self):
        return
    
    def _get_actors_per_env(self):
        n = self._root_state.shape[0] // self.get_num_envs()
        return n
    
    def _pre_physics_step(self, actions):
        return
    
    def _physics_step(self):
        for i in range(self._sim_steps):
            self._step_sim()
        return

    def _step_sim(self):
        self._gym.simulate(self._sim)
        return
    
    def _post_physics_step(self):
        self._refresh_sim_tensors()
        
        self._update_time()
        self._update_misc()
        self._update_observations()
        self._update_info()
        self._update_reward()
        self._update_done()
        return

    def _refresh_sim_tensors(self):
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

        self._gym.refresh_force_sensor_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        
        if self._enable_dof_force_sensors():
            self._gym.refresh_dof_force_tensor(self._sim)
        return
    
    def _update_time(self, num_steps=1):
        self._timestep_buf += num_steps
        self._time_buf[:] = self._timestep * self._timestep_buf
        return

    def _update_misc(self):
        return

    def _update_observations(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            obs = self._compute_obs(env_ids)
            if (env_ids is None):
                self._obs_buf[:] = obs
            else:
                self._obs_buf[env_ids] = obs
        return

    def _enable_dof_force_sensors(self):
        return False
    
    @abc.abstractmethod
    def _update_reward(self):
        return
    
    @abc.abstractmethod
    def _update_done(self):
        return

    def _update_info(self, env_ids=None):
        return
    
    @abc.abstractmethod
    def _compute_obs(env_ids=None):
        return

    def _render(self, sync_frame_time=False, clear_lines=True):
        # check for window closed
        if (self._gym.query_viewer_has_closed(self._viewer)):
            sys.exit()

        # check for keyboard events
        for evt in self._gym.query_viewer_action_events(self._viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self._enable_viewer_sync = not self._enable_viewer_sync

        # fetch results
        if (self._device != "cpu"):
            self._gym.fetch_results(self._sim, True)
                
        # step graphics
        if (self._enable_viewer_sync):
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, True)
        else:
            self._gym.poll_viewer_events(self._viewer)
        
        if clear_lines:
            self._gym.clear_lines(self._viewer)

        return

    def _init_camera(self):
        return

    def _update_camera(self):
        return

    def _get_vis_col_group(self):
        return self.get_num_envs()
    
    def _draw_point(self, env, pt, color, size=0.01):
        if isinstance(pt, torch.Tensor):
            pt = pt.cpu().detach().numpy()

        if isinstance(color, torch.Tensor):
            color = color.cpu().detach().numpy()

        vertices = []
        for i in range(3):
            axis = np.zeros(3)
            axis[i] = size
            vertices.append(pt)
            vertices.append(pt + axis)
            vertices.append(pt)
            vertices.append(pt - axis)

        vertices = np.array(vertices, dtype=np.float32)
        colors = np.broadcast_to(color, (6, 3)).astype(np.float32)
        self._gym.add_lines(self._viewer, env, 6, vertices, colors)
        return

    def _draw_line(self, env, p1, p2, color):
        if isinstance(p1, torch.Tensor):
            p1 = p1.cpu().detach().numpy()

        if isinstance(p2, torch.Tensor):
            p2 = p2.cpu().detach().numpy()

        if isinstance(color, torch.Tensor):
            color = color.cpu().detach().numpy()

        vertices = np.array([p1, p2], dtype=np.float32)
        colors = np.array([color], dtype=np.float32)
        self._gym.add_lines(self._viewer, env, 1, vertices, colors)
        return
    
    def _draw_flag(self, env, point, height, color):
        if isinstance(point, torch.Tensor):
            point = point.cpu().detach().numpy()
        # draw a flag at the point
        p1 = [point[0], point[1], point[2]]
        p2 = [point[0], point[1], point[2] + height]
        self._draw_line(env, p1, p2, color)
        self._draw_point(env, p1, color, 0.1)
        self._draw_point(env, p2, color, 0.1)
        return