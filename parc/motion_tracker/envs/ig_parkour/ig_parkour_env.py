import os
import pickle
import time
from collections import OrderedDict

import gym
import isaacgym.gymapi as gymapi
import isaacgym.gymtorch as gymtorch
import numpy as np
import torch

import parc.motion_tracker.envs.base_env as base_env
import parc.motion_tracker.envs.ig_char_env as ig_char_env
import parc.motion_tracker.envs.ig_parkour.dm_env as dm_env
import parc.motion_tracker.envs.ig_parkour.mgdm_dm_util as mgdm_dm_util
import parc.util.geom_util as geom_util
import parc.util.ig_util as ig_util
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util
from parc.util.logger import Logger

SIM_CHAR_IDX = 0
REF_CHAR_IDX = 1

EVENT_PAUSE = "PAUSE"
EVENT_STEP = "STEP"
PRINT_CHAR_STATE = "PRINT_CHAR_STATE"
RESET_ALL_ENVS = "RESET_ALL_ENVS"
RAND_RESET = "RAND_RESET"
WRITE_MOTION_FILE_FROM_AGENT = "WRITE_MOTION_FILE_FROM_AGENT"
EVENT_SWITCH_CAMERA = "EVENT_SWITCH_CAMERA"
EVENT_HARD_RESET = "EVENT_HARD_RESET"
EVENT_VIEW_DEBUG_VISUALS = "EVENT_VIEW_DEBUG_VISUALS"
EVENT_TYPE_COMMANDS = "EVENT_TYPE_COMMANDS"
EVENT_UP = "EVENT_UP"
EVENT_DOWN = "EVENT_DOWN"
EVENT_LEFT = "EVENT_LEFT"
EVENT_RIGHT = "EVENT_RIGHT"

class IGParkourEnv(ig_char_env.IGCharEnv):
    NAME = "ig_parkour"

    def __init__(self, config, num_envs, device, visualize):
        self._start_compute_time = time.time()
        env_config = config["env"]

        self._num_envs = num_envs
        self._device = device
        self._visualize = visualize

        self._bypass_record_fail = False
        self._fraction_dm_envs = env_config["fraction_dm_envs"]
        self._num_dm_envs = min(int(self._fraction_dm_envs * num_envs), num_envs)

        self._contact_detection_eps = env_config["contact_detection_eps"]
        self._output_motion_dir = env_config.get("output_motion_dir", "output/_motions/recorded_motions/")
        os.makedirs(self._output_motion_dir, exist_ok=True)
        print("making output motion dir:", self._output_motion_dir)
        self._never_done = env_config.get("never_done", False)
        
        # for deepmimic
        self._enable_early_termination = env_config["enable_early_termination"]
        self._termination_height = torch.tensor(env_config["termination_height"], dtype=torch.float32, device=device)
        self._pose_termination = env_config.get("pose_termination", False)
        self._pose_termination_dist = torch.tensor(env_config["pose_termination_dist"], dtype=torch.float32, device=device) #env_config.get("pose_termination_dist", 1.0)
        self._tar_obs_steps = env_config.get("tar_obs_steps", [1])
        self._tar_obs_steps = torch.tensor(self._tar_obs_steps, device=device, dtype=torch.int)
        self._rand_reset = env_config.get("rand_reset", True)

        self._ref_char_offset = torch.tensor(env_config["ref_char_offset"], device=device, dtype=torch.float)

        self._use_contact_info = env_config["use_contact_info"]
        if self._use_contact_info:
            self._contact_weights = env_config["contact_weights"]
            self._contact_weights = torch.tensor(self._contact_weights, dtype=torch.float32, device=device)

        self._debug_visuals = env_config["debug_visuals"]
        self._debug_visual_env_ids = [0, num_envs-1]
        if num_envs > 2:
            self._debug_visual_env_ids.append(1)


        self._enable_tar_obs = env_config.get("enable_tar_obs", True)
        self._global_root_height_obs = env_config["global_root_height_obs"]
        self._track_root = env_config["track_root"]
        self._track_root_h = env_config["track_root_h"]
        self._root_pos_termination_dist = env_config["root_pos_termination_dist"]
        self._root_rot_termination_angle = env_config["root_rot_termination_angle"]

        # reward weights
        self._pose_w = env_config["pose_w"]
        self._vel_w = env_config["vel_w"]
        self._root_pos_w = env_config["root_pos_w"]
        self._root_vel_w = env_config["root_vel_w"]
        self._key_pos_w = env_config["key_pos_w"]
        total_w = self._pose_w + self._vel_w + self._root_pos_w + self._root_vel_w + self._key_pos_w
        self._pose_w = self._pose_w / total_w
        self._vel_w = self._vel_w / total_w
        self._root_pos_w = self._root_pos_w / total_w
        self._root_vel_w = self._root_vel_w / total_w
        self._key_pos_w = self._key_pos_w / total_w

        self._report_tracking_error = env_config.get("report_tracking_error", False)

        self._use_heightmap = True


        ray_points_behind = env_config["ray_points_behind"]
        ray_points_ahead = env_config["ray_points_ahead"]
        ray_num_left = env_config["ray_num_left"]
        ray_num_right = env_config["ray_num_right"]
        ray_dx = env_config["ray_dx"]
        ray_angle = env_config["ray_angle"]
        self._ray_xy_points = geom_util.get_xy_points_cone(
                                    center=torch.zeros(size=(2,), dtype=torch.float32, device=device),
                                    dx=ray_dx,
                                    num_neg=ray_points_behind,
                                    num_pos=ray_points_ahead,
                                    num_rays_neg=ray_num_left,
                                    num_rays_pos=ray_num_right,
                                    angle_between_rays=ray_angle)
        
        num_points = self._ray_xy_points.shape[0]
        self._ray_hfs = torch.zeros(size=(num_envs, num_points), dtype=torch.float32, device=device)

        self._build_kin_char_model(env_config["char_file"])
        if self.has_dm_envs():
            self._dm_env = dm_env.DeepMimicEnv(config, self._num_dm_envs, device, visualize, self._kin_char_model)

        super().__init__(config=config, num_envs=num_envs, device=device,
                         visualize=visualize)
        
        self._paused = False
        if self._visualize:
            self._setup_IO(config, num_envs, device)
            
            
        self.set_write_agent_states_flag(env_config.get("write_agent_states", False))

        if self.is_writing_agent_states():
            self.build_agent_states_dict()

        self._demo_mode = env_config["demo_mode"]
        if self._demo_mode:
            # self._enable_early_termination = False
            # self._episode_length = 1000.0
            # self._max_replans = 1000
            print("DEMO MODE ENABLED")
        return
    
    def _setup_IO(self, config, num_envs, device):

        self._paused = config["env"].get("start_paused", False)
        print("Initial pause state: ", self._paused)

        print("*****Keyboard commands*****")
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_H, EVENT_HARD_RESET
        )
        print("H: hard reset envs")
        self._hard_reset_all_envs_flag = False

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_P, EVENT_PAUSE
        )
        print("P: pause")
        
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_D, EVENT_STEP
        )
        print("D: step")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_C, PRINT_CHAR_STATE
        )
        print("C: print char state")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_R, RESET_ALL_ENVS
        )
        print("R: Reset all envs")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_T, RAND_RESET
        )
        print("T: enable/disable rand reset")
        

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_W, WRITE_MOTION_FILE_FROM_AGENT
        )
        print("W: write remaining char states of current episode to motion file")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_Q, EVENT_SWITCH_CAMERA
        )
        print("Q: Switch env for camera to track")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_V, EVENT_VIEW_DEBUG_VISUALS
        )
        print("V: view debug visuals")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_UP, EVENT_UP
        )
        print("UP: cursor up")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_DOWN, EVENT_DOWN
        )
        print("DOWN: cursor down")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_LEFT, EVENT_LEFT
        )
        print("LEFT: cursor left")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_RIGHT, EVENT_RIGHT
        )
        print("RIGHT: cursor right")

        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_0, EVENT_TYPE_COMMANDS
        )
        print("0: Typing commands")

        print("Writing motion file key sequence: P -> R -> D -> D -> W -> P")

        self._reset_all_envs_flag = False
        return
    
    def _IO(self):
        # check for keyboard events
        for evt in self._gym.query_viewer_action_events(self._viewer):
            stop = self._process_event(evt)
            if stop:
                return stop
        return False

    def _process_event(self, evt):
        if evt.action == EVENT_PAUSE and evt.value > 0:
            self._paused = not self._paused
            print("Paused: ", self._paused)
            if not self._paused: 
                # write the current state before sim step
                self.write_agent_states()
        if evt.action == EVENT_STEP and evt.value > 0:
            print("Step")
            return True
        if evt.action == PRINT_CHAR_STATE and evt.value > 0:
            char_states, char_contacts = self._get_char_state(0)
            print(char_states)
            print(char_contacts)
        if evt.action == RESET_ALL_ENVS and evt.value > 0:
            self._reset_all_envs_flag = True
            print("Resetting all envs because reset button was pressed")
        if evt.action == RAND_RESET and evt.value > 0:
            self.set_rand_reset()
        if evt.action == WRITE_MOTION_FILE_FROM_AGENT and evt.value > 0:
            record_obs = self.has_dm_envs()
            self.build_agent_states_dict(record_obs=record_obs)
            self._bypass_record_fail = True
        if evt.action == EVENT_SWITCH_CAMERA and evt.value > 0:
            new_camera_env_id = -1
            while new_camera_env_id < 0 or new_camera_env_id > self._num_envs-1:
                new_camera_env_id = input("insert env id: ")
                print("inputted: ", new_camera_env_id)
                if new_camera_env_id.isdigit():
                   new_camera_env_id = int(new_camera_env_id)
                else:
                    new_camera_env_id = -1 
            
            self._camera_env_id = new_camera_env_id
            print("env that camera is following:", self._camera_env_id)
        if evt.action == EVENT_HARD_RESET and evt.value > 0:
            print("MANUAL HARD RESET ON ALL ENVS")
            self._hard_reset_all_envs_flag = True
        if evt.action == EVENT_VIEW_DEBUG_VISUALS and evt.value > 0:
            self._debug_visuals = not self._debug_visuals
        if evt.action == EVENT_TYPE_COMMANDS and evt.value > 0:
            command_str = input("Type command: ")
            self.process_command_str(command_str)

        return False
    
    def process_command_str(self, command_str: str):
        if command_str == "pose termination":
            self._pose_termination = not self._pose_termination
            print("pose termination set to:", self._pose_termination)
        elif command_str == "early termination":
            self._enable_early_termination = not self._enable_early_termination
            print("enable early termination set to:", self._enable_early_termination)
        elif command_str == "dm demo mode":
            print("dm demo mode set to:", self.get_dm_env().set_demo_mode())
        elif command_str == "ref char offset":
            root_offset = input("type ref char offset: ")
            root_offset = root_offset.split(sep=" ")
            if len(root_offset) >= 1:
                x_offset = float(root_offset[0])
                self._ref_char_offset[0] = x_offset
            else:
                x_offset = 0.0
            if len(root_offset) >= 2:
                y_offset = float(root_offset[1])
                self._ref_char_offset[1] = y_offset
            else:
                y_offset = 0.0
            if len(root_offset) >= 3:
                z_offset = float(root_offset[2])
                self._ref_char_offset[2] = z_offset
            else:
                z_offset = 0.0
        elif command_str == "ep len":
            new_ep_length = float(input("type episode length: "))
            self._episode_length = new_ep_length
            

        elif command_str == "reset start time fraction":
            reset_time_fraction = input("type reset time fraction: ")
            if reset_time_fraction.replace('.','',1).isdigit():
                print("setting reset time fraction to:", reset_time_fraction)

                reset_time_fraction = torch.ones(size=[self._num_envs], dtype=torch.float32, device=self._device) * float(reset_time_fraction)
                self.set_reset_motion_start_time_fraction(reset_time_fraction)
            else:
                print("invalid input")
        elif command_str == "reset root offset":
            root_offset = input("type root offset: ")
            root_offset = root_offset.split(sep=" ")
            if len(root_offset) >= 1:
                x_offset = float(root_offset[0])
            else:
                x_offset = 0.0
            if len(root_offset) >= 2:
                y_offset = float(root_offset[1])
            else:
                y_offset = 0.0
            if len(root_offset) >= 3:
                z_offset = float(root_offset[2])
            else:
                z_offset = 0.0

            if self.has_dm_envs():
                root_pos_offset = torch.tensor([x_offset, y_offset, z_offset], dtype=torch.float32, device=self._device)
                root_pos_offset = root_pos_offset.unsqueeze(0).expand(size=[self._num_dm_envs, -1])
                self.get_dm_env().set_root_pos_offset(root_pos_offset)
        elif command_str == "reset heading offset":
            heading_offset = input("type heading offset (degrees): ")
            heading_offset = float(heading_offset) * torch.pi / 180.0

            if self.has_dm_envs():
                z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self._device)
                if not isinstance(heading_offset, torch.Tensor):
                    heading_offset = torch.tensor([heading_offset], dtype=torch.float32, device=self._device)
                heading_quat = torch_util.axis_angle_to_quat(z_axis, heading_offset)
                heading_quat = heading_quat.expand(size=[self._num_dm_envs, -1])
                self.get_dm_env().set_root_rot_offset(heading_quat)
        elif command_str == "motion id":
            new_motion_id = int(input("Input motion id: "))
            new_motion_id = max(min(new_motion_id, self._dm_env._motion_lib.num_motions() - 1), 0)
            print("setting motion id to:", new_motion_id)
            self._dm_env._selected_motion_id = new_motion_id
            self._dm_env._one_motion_mode = True
        elif command_str == "clear motion id":
            self._dm_env._one_motion_mode = False
        elif command_str == "never done":
            self._never_done = not self._never_done
        elif command_str == "motion name":

            if self.has_dm_envs():
                motion_name = self.get_dm_env().get_env_motion_name(self._camera_env_id)
                print("motion name:", motion_name)
        else:
            print("invalid command")

        return
    
    def _build_envs(self, config):
        self._load_char_asset(config)

        self._ref_char_handles = []
        self._char_handles = []
        init_pose = config["env"].get("init_pose", None)
        self._parse_init_pose(init_pose, self._device)
        self._build_terrains(config["env"])
        self._envs = []
        env_spacing = self._get_env_spacing()
        num_env_per_row = int(np.sqrt(self._num_envs))
        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self._env_offsets = torch.zeros(size = (self._num_envs,3), dtype=torch.float32, device=self._device)
        for i in range(self._num_envs):
            curr_col = i % num_env_per_row
            curr_row = i // num_env_per_row
            self._env_offsets[i, 0] = env_spacing * 2 * curr_col
            self._env_offsets[i, 1] = env_spacing * 2 * curr_row

        for i in range(self._num_envs):
            Logger.print("Building {:d}/{:d} envs".format(i + 1, self._num_envs), end='\r')
            env_ptr = self._gym.create_env(self._sim, lower, upper, num_env_per_row)
            self._build_env(i, env_ptr, config)
            self._envs.append(env_ptr)

        Logger.print("\n")
        return

    def _build_env(self, env_id, env_ptr, config):
        super()._build_env(env_id, env_ptr, config)

        if (self._visualize):
            ref_char_handle = self._build_ref_character(env_id, env_ptr, config)
            self._ref_char_handles.append(ref_char_handle)
        return 
    
    def _build_ref_character(self, env_id, env_ptr, config):
        col = [0.5, 0.9, 0.1]

        vis_col_group = self._get_vis_col_group()
        col_group = vis_col_group + env_id
        col_filter = 0
        segmentation_id = 0

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        char_handle = self._gym.create_actor(env_ptr, self._char_asset, start_pose, "ref character", col_group, col_filter, segmentation_id)
        
        dof_prop = self._gym.get_asset_dof_properties(self._char_asset)
        dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT
        dof_prop["stiffness"] = 0.0
        dof_prop["damping"] = 0.0
        self._gym.set_actor_dof_properties(env_ptr, char_handle, dof_prop)

        num_bodies = self._gym.get_asset_rigid_body_count(self._char_asset)
        for j in range(num_bodies):
            self._gym.set_rigid_body_color(env_ptr, char_handle, j, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(col[0], col[1], col[2]))
        return char_handle
    
    def _build_ref_char_sim_tensors(self):
        env_handle = self._envs[0]
        char_handle = self._get_char_actor_handle()
        num_envs = self.get_num_envs()
        
        actors_per_env = self._get_actors_per_env()
        actor_root_state = self._root_state.view([num_envs, actors_per_env, self._root_state.shape[-1]])
        
        ref_char_handle = self._get_ref_char_actor_handle()
        self._ref_char_root_pos = actor_root_state[:, ref_char_handle, 0:3]
        self._ref_char_root_rot = actor_root_state[:, ref_char_handle, 3:7]
        self._ref_char_root_vel = actor_root_state[:, ref_char_handle, 7:10]
        self._ref_char_root_ang_vel = actor_root_state[:, ref_char_handle, 10:13]
        
        dofs_per_env = self._dof_state.shape[0] // num_envs
        num_char_dofs = self._gym.get_actor_dof_count(env_handle, char_handle)
        dof_state = self._dof_state.view([num_envs, dofs_per_env, 2])
        self._ref_char_dof_pos = dof_state[..., ref_char_handle * num_char_dofs:(ref_char_handle + 1) * num_char_dofs, 0]
        self._ref_char_dof_vel = dof_state[..., ref_char_handle * num_char_dofs:(ref_char_handle + 1) * num_char_dofs, 1]
        
        return
    
    def _reset_ref_char(self, env_ids):
        self._ref_char_root_pos[env_ids] = self._ref_root_pos[env_ids]
        self._ref_char_root_rot[env_ids] = self._ref_root_rot[env_ids]
        self._ref_char_root_vel[env_ids] = 0.0#self._ref_root_vel[env_ids]
        self._ref_char_root_ang_vel[env_ids] = 0.0#self._ref_root_ang_vel[env_ids]
        
        self._ref_char_dof_pos[env_ids] = self._ref_dof_pos[env_ids]
        self._ref_char_dof_vel[env_ids] = 0.0#self._ref_dof_vel[env_ids]
        
        self._ref_char_root_pos[env_ids, :] += self._ref_char_offset
        
        #ref_char_handle = self._get_ref_char_actor_handle()
        #self._actors_need_reset[env_ids, ref_char_handle] = True
        self._actors_need_reset[env_ids, REF_CHAR_IDX] = True
        return
    
    def _get_ref_char_actor_handle(self):
        return self._ref_char_handles[0]

    def _build_terrains(self, env_config):
        if self.has_dm_envs():
            dm_terrain_save_path = env_config["dm"].get("terrain_save_path", None)
            if dm_terrain_save_path is not None and os.path.exists(dm_terrain_save_path):
                dm_verts, dm_tris = self.get_dm_env().load_terrain(dm_terrain_save_path)
            else:
                x_offset = 0.0
                y_offset = 0.0
                dm_verts, dm_tris = self.get_dm_env().build_terrain(env_config, dm_terrain_save_path, 
                                                        x_offset, y_offset)
            
            print("Assembling DeepMimic triangle mesh...")
            dm_trimesh_start_time = time.time()
            total_num_verts = 0
            all_dm_verts = []
            all_dm_tris = []
            for i in range(len(dm_verts)):
                assert len(dm_verts[i]) == self.get_dm_env()._terrains_per_motion
                for j in range(len(dm_verts[i])):
                    verts = dm_verts[i][j]
                    tris = dm_tris[i][j] + total_num_verts
                    
                    all_dm_verts.append(verts)
                    all_dm_tris.append(tris)
                    total_num_verts += verts.shape[0]

            all_dm_verts = np.concatenate(all_dm_verts, axis=0)
            all_dm_tris = np.concatenate(all_dm_tris, axis=0)
            dm_trimesh_end_time = time.time()
            dm_trimesh_time = dm_trimesh_end_time - dm_trimesh_start_time
            print("Finished assembling DeepMimic triangle mesh in", dm_trimesh_time, "seconds.")
            ig_util.add_trimesh_to_gym(all_dm_verts, all_dm_tris, self._sim, self._gym)
        return

    def _refresh_obs_hfs(self):
        if not self._use_heightmap:
            return
        
        char_root_pos_xyz = self._get_global_xyz_pos(self._char_root_pos[:, 0:3])
        char_heading = torch_util.calc_heading(self._char_root_rot)
        
        if self.has_dm_envs():
            dm_char_root_pos_xyz = self._get_dm_slice(char_root_pos_xyz)
            dm_char_heading = self._get_dm_slice(char_heading)
            self.get_dm_env()._refresh_obs_hfs(dm_char_root_pos_xyz, dm_char_heading)

        if self._visualize and self._debug_visuals:
            self._draw_local_hf()

        return
    
    def _draw_local_hf(self):
        red = [1.0, 0.0, 0.0]
        blue = [0.0, 0.0, 1.0]
        if self._camera_env_id < self._num_dm_envs:
            xyz_points = self.get_dm_env()._ray_xyz_points[self._camera_env_id]
            for i in range(0, xyz_points.shape[0], 1):
                point = xyz_points[i]

                # draw in env 0 because xy_points already has the offsets
                self._draw_point(self._envs[0], point, red, size = 0.04)
        return
    
    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        self._init_camera()

        self._num_rbs = self._kin_char_model.get_num_joints()
        self._num_contact_rbs = self._kin_char_model.get_num_contact_bodies()

        num_envs = self.get_num_envs()

        self._ref_root_pos = torch.zeros_like(self._char_root_pos)
        self._ref_root_rot = torch.zeros_like(self._char_root_rot)
        self._ref_root_vel = torch.zeros_like(self._char_root_vel)
        self._ref_root_ang_vel = torch.zeros_like(self._char_root_ang_vel)
        self._ref_body_pos = torch.zeros_like(self._char_rigid_body_pos)
        self._ref_joint_rot = torch.zeros_like(self._char_rigid_body_rot[..., 1:, :])
        self._ref_dof_pos = torch.zeros_like(self._char_dof_pos) 
        self._ref_dof_vel = torch.zeros_like(self._char_dof_vel)

        if self._use_contact_info:
            self._ref_contacts = torch.zeros(size=(num_envs, self._num_contact_rbs), device=self._device, dtype=torch.float32)
        
        env_config = config["env"]
        contact_bodies = env_config.get("contact_bodies", [])
        print("contact bodies:", contact_bodies)
        self._contact_body_ids = self._build_body_ids_tensor(contact_bodies)

        joint_err_w = env_config.get("joint_err_w", None)
        self._parse_joint_err_weights(joint_err_w)
        
        if (self._visualize):
            self._build_ref_char_sim_tensors()

        self._give_sim_tensor_views()
        return
    
    def _give_sim_tensor_views(self):
        if self.has_dm_envs():
            self.get_dm_env().get_sim_tensor_views(ref_root_pos = self._get_dm_slice(self._ref_root_pos),
                                                   ref_root_rot = self._get_dm_slice(self._ref_root_rot),
                                                   ref_root_vel = self._get_dm_slice(self._ref_root_vel),
                                                   ref_root_ang_vel = self._get_dm_slice(self._ref_root_ang_vel),
                                                   ref_body_pos = self._get_dm_slice(self._ref_body_pos),
                                                   ref_joint_rot = self._get_dm_slice(self._ref_joint_rot),
                                                   ref_dof_pos = self._get_dm_slice(self._ref_dof_pos),
                                                   ref_dof_vel = self._get_dm_slice(self._ref_dof_vel),
                                                   ref_contacts = self._get_dm_slice(self._ref_contacts),
                                                   char_root_pos = self._get_dm_slice(self._char_root_pos),
                                                   char_root_rot = self._get_dm_slice(self._char_root_rot),
                                                   char_root_vel = self._get_dm_slice(self._char_root_vel),
                                                   char_root_ang_vel = self._get_dm_slice(self._char_root_ang_vel),
                                                   char_dof_pos = self._get_dm_slice(self._char_dof_pos),
                                                   char_dof_vel = self._get_dm_slice(self._char_dof_vel),
                                                   char_contact_forces = self._get_dm_slice(self._char_contact_forces),
                                                   char_rigid_body_pos = self._get_dm_slice(self._char_rigid_body_pos),
                                                   char_rigid_body_vel = self._get_dm_slice(self._char_rigid_body_vel),
                                                   char_rigid_body_ang_vel = self._get_dm_slice(self._char_rigid_body_ang_vel),
                                                   )
        return
    
    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        self._refresh_obs_hfs()
        return
    
    def _build_data_buffers(self):
        # This is called after sim tensors are built
        num_envs = self.get_num_envs()
        actors_per_env = self._get_actors_per_env()

        self._reward_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._done_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._timestep_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._time_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)

        self._ep_num_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int64)

        # need this before replan(), since replan sets the actors_need_reset buffer
        self._need_reset_buf = torch.zeros(self._root_state.shape[0], device=self._device, dtype=torch.bool)
        self._actors_need_reset = self._need_reset_buf.view((num_envs, actors_per_env))

        self._actor_dof_dims = self._build_actor_dof_dims()
        self._info = dict()

        self._give_data_buffer_views()

        obs_space = self.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_buf = torch.zeros([num_envs] + list(obs_space.shape), device=self._device, dtype=obs_dtype)

        if self._visualize:
            self._update_ref_char()

        return
    
    def _give_data_buffer_views(self):
        if self.has_dm_envs():
            self.get_dm_env().get_data_buffer_views(reward_buf = self._get_dm_slice(self._reward_buf),
                                                    done_buf = self._get_dm_slice(self._done_buf),
                                                    time_buf = self._get_dm_slice(self._time_buf),
                                                    timestep_buf = self._get_dm_slice(self._timestep_buf),
                                                    actors_need_reset = self._get_dm_slice(self._actors_need_reset),
                                                    env_offsets = self._get_dm_slice(self._env_offsets),
                                                    key_body_ids = self._key_body_ids,
                                                    ray_xy_points = self._ray_xy_points,
                                                    ray_hfs = self._get_dm_slice(self._ray_hfs))
        return

    def _get_char_contact_state(self, env_ids, eps=1e-5):
        if env_ids is None:
            char_contacts = torch.norm(self._char_contact_forces, dim=-1)
        else:
            char_contacts = torch.norm(self._char_contact_forces[env_ids], dim=-1)
        char_contacts = char_contacts > eps
        char_contacts = char_contacts.to(dtype=torch.float32)
        return char_contacts
    
    def _get_char_state(self, env_ids, eps=1e-5, concat=False, ref=False):
        if ref:
            root_pos = self._ref_root_pos[env_ids]
            root_rot = torch_util.quat_to_exp_map(self._ref_root_rot[env_ids])
            joint_dof = self._ref_dof_pos[env_ids]
            char_contacts = self._ref_contacts[env_ids]
        else:
            root_pos = self._char_root_pos[env_ids]
            root_rot = torch_util.quat_to_exp_map(self._char_root_rot[env_ids])
            joint_dof = self._char_dof_pos[env_ids]
            char_contacts = torch.norm(self._char_contact_forces[env_ids], dim=-1)
            char_contacts = char_contacts > eps
            char_contacts = char_contacts.to(dtype=torch.float32)

        char_state = torch.cat([root_pos, root_rot, joint_dof], dim=-1)

        ret = [char_state, char_contacts]
        
        if concat:
            return torch.cat(ret, dim=-1)
        else:
            return tuple(ret)
    
    def _post_physics_step(self):
        super()._post_physics_step()
        
        if self._visualize:
            if self._reset_all_envs_flag:
                self._reset_all_envs_flag = False
                self._done_buf[:] = base_env.DoneFlags.FAIL.value

        self.write_agent_states()
        return
    
    def save_agent_states_to_file(self, env_id, output_motion_name = None):
        motion_frames = torch.stack(self._dm_agent_motion[env_id]["frames"])
        contact_frames = torch.stack(self._dm_agent_motion[env_id]["contacts"])

        if self._record_obs:
            obs = torch.stack(self._dm_agent_motion[env_id]["obs"])
            self._dm_agent_motion[env_id]["obs"] = obs.cpu().numpy()

        # make sure appropriate root position transforms are applied for dm env
        if env_id < self._num_dm_envs:
            motion_frames[:, 0:2] = self._get_global_xy_pos(motion_frames[:, 0:2], env_id)

        if self._use_heightmap:
            # slice a terrain around the current motion, and also localize it so motion starts at origin
            if env_id < self._num_dm_envs:
                terrain = self._dm_env._terrain

            padding = round(1.0 // terrain.dxdy[0].item()) * terrain.dxdy[0].item()
            print("padding:", padding)
            sliced_terrain, localized_root_pos = terrain_util.slice_terrain_around_motion(
                motion_frames[..., 0:3], terrain, padding=padding)
            motion_frames = motion_frames.clone()
            motion_frames[..., 0:3] = localized_root_pos
            self._dm_agent_motion[env_id]["terrain"] = sliced_terrain.numpy_copy()

        
        self._dm_agent_motion[env_id]["frames"] = motion_frames.cpu().numpy()
        self._dm_agent_motion[env_id]["contacts"] = contact_frames.cpu().numpy()

        if output_motion_name is None:
            output_motion_name = "dm_motion_" + str(env_id).zfill(3)

        output_filepath = os.path.join(self._output_motion_dir, output_motion_name + ".pkl")
        with open(output_filepath, 'wb') as file:
            pickle.dump(self._dm_agent_motion[env_id], file)
            print("wrote motion data to", output_filepath)

        print("num frames =", self._dm_agent_motion[env_id]["frames"].shape[0])
        return
    
    def set_write_agent_states_flag(self, val):
        self._write_agent_states_flag = val
        return
    
    def is_writing_agent_states(self):
        return self._write_agent_states_flag
    
    def is_writing_env_state(self, env_id):
        return self._writing_env_state[env_id]
    
    def set_writing_env_state(self, env_id, val):
        self._writing_env_state[env_id] = val
        return
    
    def set_env_success_state(self, env_id, val):
        self._env_success_state[env_id] = val
        return
    
    def get_env_success_states(self):
        return self._env_success_state
    
    def write_agent_states(self):
        if self.is_writing_agent_states():
            self.set_write_agent_states_flag(False)

            for env_id in range(0, self._num_envs):
                if not self.is_writing_env_state(env_id):
                    continue
                self.set_write_agent_states_flag(True) # keep the flag on as long as one env is still writing states
                
                ref = hasattr(self, "_record_ref") and self._record_ref is True
                char_states, char_contacts = self._get_char_state(env_id, ref=ref)
                self._dm_agent_motion[env_id]["frames"].append(char_states.detach().clone())
                self._dm_agent_motion[env_id]["contacts"].append(char_contacts.detach().clone())
                
                if self._record_obs:
                    self._dm_agent_motion[env_id]["obs"].append(self._obs_buf[env_id].detach().clone())

                if self._done_buf[env_id] == base_env.DoneFlags.FAIL.value:
                    # update_done() is called before this, so this is recording the last timestep
                    
                    self.set_writing_env_state(env_id, False)

                    if self.has_dm_envs() and env_id < self._num_dm_envs:
                        motion_length = self.get_dm_env().get_env_motion_length(env_id).item()
                        curr_motion_time = self.get_dm_env().get_env_motion_time(env_id).item()
                        motion_name = self.get_dm_env().get_env_motion_name(env_id)

                        if not self._bypass_record_fail:
                            if curr_motion_time < motion_length - self._timestep * 2.0:
                                print("env", env_id, "failed to track motion", motion_name)
                                continue # if it failed before time is up, then it didn't successfully track the motion
                            
                        self.set_env_success_state(env_id, True)
                        output_motion_name = motion_name + self._save_motion_name_suffix
                        self.save_agent_states_to_file(env_id, output_motion_name)
                    else:
                        self.save_agent_states_to_file(env_id)
        return

    def _update_time(self):
        super()._update_time()
        return
    
    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)

        if self.has_dm_envs():
            self.get_dm_env().pre_physics_step()
        return
    
    def reset(self, env_ids=None):
        if (env_ids is None): # note, not the same as passing in empty tensor []
            num_envs = self.get_num_envs()
            env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)
        
        if self.has_dm_envs():
            dm_env_ids = self._extract_dm_env_ids(env_ids)
            self.get_dm_env().reset(dm_env_ids)
        
        if self._visualize:
            self._reset_ref_char(env_ids)

        reset_env_ids = self._reset_sim_tensors()
        self._refresh_sim_tensors()
        self._update_observations(reset_env_ids)
        self._update_info(reset_env_ids)

        if len(reset_env_ids) > 0:
            self._ep_num_buf[reset_env_ids] += 1

        return self._obs_buf, self._info
    
    def step(self, action):
        self._IO()
        while self._paused:
            if (self._viewer):
                self._update_camera()
                self._render(clear_lines=False)
                time.sleep(1.0/120.0) # to avoid cpu from overworking
            if self._IO():
                break
        return super().step(action)
    
    def _compute_obs(self, env_ids=None, ret_obs_shapes=False):
        if (env_ids is None):
            dm_env_ids = None
            root_pos = self._char_root_pos
            root_rot = self._char_root_rot
            root_vel = self._char_root_vel
            root_ang_vel = self._char_root_ang_vel
            dof_pos = self._char_dof_pos
            dof_vel = self._char_dof_vel
            if self._use_heightmap:
                hf = self._ray_hfs
        else:
            dm_env_ids = self._extract_dm_env_ids(env_ids)
            root_pos = self._char_root_pos[env_ids]
            root_rot = self._char_root_rot[env_ids]
            root_vel = self._char_root_vel[env_ids]
            root_ang_vel = self._char_root_ang_vel[env_ids]
            dof_pos = self._char_dof_pos[env_ids]
            dof_vel = self._char_dof_vel[env_ids]

            if self._use_heightmap:
                hf = self._ray_hfs[env_ids]
            
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        if (self._has_key_bodies()):
            body_pos, _ = self._kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        ## get tar obs
        tar_root_pos = []
        tar_root_rot = []
        tar_joint_rot = []
        tar_key_pos = []
        tar_contacts = []

        if self.has_dm_envs() and (dm_env_ids is None or len(dm_env_ids) > 0):
            dm_tar_root_pos, dm_tar_root_rot, dm_tar_joint_rot, dm_tar_key_pos, dm_tar_contacts = \
                self.get_dm_env().compute_tar_obs(self._tar_obs_steps, dm_env_ids)
            tar_root_pos.append(dm_tar_root_pos)
            tar_root_rot.append(dm_tar_root_rot)
            tar_joint_rot.append(dm_tar_joint_rot)
            tar_key_pos.append(dm_tar_key_pos)
            tar_contacts.append(dm_tar_contacts)
            
        tar_root_pos = torch.cat(tar_root_pos, dim=0)
        tar_root_rot = torch.cat(tar_root_rot, dim=0)
        tar_joint_rot = torch.cat(tar_joint_rot, dim=0)
        tar_key_pos = torch.cat(tar_key_pos, dim=0)
        tar_contacts = torch.cat(tar_contacts, dim=0)

        obs_dict = mgdm_dm_util.compute_deepmimic_obs(
            root_pos=root_pos, 
            root_rot=root_rot, 
            root_vel=root_vel, 
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_obs=self._global_obs,
            root_height_obs=self._global_root_height_obs,
            enable_tar_obs=self._enable_tar_obs,
            tar_root_pos=tar_root_pos,
            tar_root_rot=tar_root_rot,
            tar_joint_rot=tar_joint_rot,
            tar_key_pos=tar_key_pos)
        
        obs = []
        obs_shapes = OrderedDict()
        for key in obs_dict:
            curr_obs = obs_dict[key]

            if ret_obs_shapes: # Want original unflattened shapes
                obs_shapes[key] = {
                    "use_normalizer": True,
                    "shape": curr_obs.shape[1:]
                }

            if key == "tar_obs":
                curr_obs = torch.reshape(curr_obs, [curr_obs.shape[0], curr_obs.shape[1] * curr_obs.shape[2]])
            obs.append(curr_obs)

        #if not normalizable_obs_only:
        if self._use_contact_info:
            if self._enable_tar_obs:
                tar_contacts_flat = tar_contacts.reshape(tar_contacts.shape[0], tar_contacts.shape[1] * tar_contacts.shape[2])
                obs.append(tar_contacts_flat)

            # also include the character's own contact obs
            char_contacts = self._get_char_contact_state(env_ids)
            obs.append(char_contacts)

            if ret_obs_shapes:
                if self._enable_tar_obs:
                    obs_shapes["tar_contacts"] = {
                        "use_normalizer": False,
                        "shape": tar_contacts.shape[1:]
                    }
                obs_shapes["char_contacts"] = {
                    "use_normalizer": False,
                    "shape": char_contacts.shape[1:],
                }

        # also include local heightfield
        if self._use_heightmap:
            obs.append(hf)

            if ret_obs_shapes:
                obs_shapes["hf"] = {
                    "use_normalizer": False,
                    "shape": self._ray_hfs.shape[1:]
                }

        if ret_obs_shapes:
            print("OBS SHAPES")
            for key in obs_shapes:
                print(key, obs_shapes[key])
            return obs_shapes

        obs = torch.cat(obs, dim=-1)
        
        return obs

    def _update_done(self):
        if self.has_dm_envs():
            self.get_dm_env().update_done(termination_height=self._termination_height,
                                          episode_length=self._episode_length,
                                          contact_body_ids=self._contact_body_ids,
                                          pose_termination=self._pose_termination,
                                          pose_termination_dist=self._pose_termination_dist,
                                          global_obs = self._global_obs,
                                          enable_early_termination=self._enable_early_termination,
                                          track_root=self._track_root,
                                          root_pos_termination_dist=self._root_pos_termination_dist,
                                          root_rot_termination_angle=self._root_rot_termination_angle)
            
        if self._never_done:
            self._done_buf[:] = base_env.DoneFlags.NULL.value
        return
    
    def _update_reward(self):
        joint_rot = self._kin_char_model.dof_to_rot(self._char_dof_pos)
        if (self._has_key_bodies()):
            key_pos = self._char_rigid_body_pos[..., self._key_body_ids, :]
            ref_key_pos = self._ref_body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)
            ref_key_pos = key_pos

        track_root_h = self._track_root_h
        track_root = self._track_root

        comp_r = mgdm_dm_util.compute_deepmimic_reward(
            root_pos=self._char_root_pos,
            root_rot=self._char_root_rot,
            root_vel=self._char_root_vel,
            root_ang_vel=self._char_root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=self._char_dof_vel,
            key_pos=key_pos,
            tar_root_pos=self._ref_root_pos,
            tar_root_rot=self._ref_root_rot,
            tar_root_vel=self._ref_root_vel,
            tar_root_ang_vel=self._ref_root_ang_vel,
            tar_joint_rot=self._ref_joint_rot,
            tar_dof_vel=self._ref_dof_vel,
            tar_key_pos=ref_key_pos,
            joint_rot_err_w=self._joint_err_w,
            dof_err_w=self._dof_err_w,
            track_root_h=track_root_h,
            track_root=track_root)
        
        pose_r, vel_r, root_pos_r, root_vel_r, key_pos_r = comp_r[:, 0], comp_r[:, 1], comp_r[:, 2], comp_r[:, 3], comp_r[:, 4]
        
        self._info["rewards"] = dict()
        self._info["rewards"]["pose_r"] = pose_r
        self._info["rewards"]["vel_r"] = vel_r
        self._info["rewards"]["root_pos_r"] = root_pos_r
        self._info["rewards"]["root_vel_r"] = root_vel_r
        self._info["rewards"]["key_pos_r"] = key_pos_r

        
        deepmimic_r = self._pose_w * pose_r \
            + self._vel_w * vel_r \
            + self._root_pos_w * root_pos_r \
            + self._root_vel_w * root_vel_r \
            + self._key_pos_w * key_pos_r
        
        if self._use_contact_info:
            char_contact_forces = self._char_contact_forces[:, self._kin_char_model.get_contact_body_ids()]
            comp_penalty = mgdm_dm_util.compute_contact_reward(tar_contacts=self._ref_contacts,
                                      contact_forces=char_contact_forces,
                                      contact_weights=self._contact_weights)
            
            contact_penalty = torch.mean(comp_penalty, dim=-1)
            self._info["rewards"]["contact_penalty"] = contact_penalty
            deepmimic_r += contact_penalty

        self._reward_buf[:] = deepmimic_r

        self._info["rewards"]["total_r"] = self._reward_buf.clone()

        if self._use_contact_info:
            if self._visualize and self._debug_visuals:
                # COLOR IN CONTACT OF FOOT
                high_color = gymapi.Vec3(1.0, 0.0, 0.0)
                for rb_index in range(15): # TODO: modify for contact rbs
                    low_color = gymapi.Vec3(1.0, 1.0, 1.0)
                    force_mag = torch.norm(self._char_contact_forces[self._camera_env_id, rb_index])
                    color = low_color + (high_color - low_color) * force_mag
                    self._gym.set_rigid_body_color(self._envs[self._camera_env_id], 
                                                   self._char_handles[self._camera_env_id], 
                                                   rb_index, gymapi.MESH_VISUAL_AND_COLLISION, color)

                    # COLOR IN REF FOOT CONTACT (for first env only)
                    low_color = gymapi.Vec3(0.0, 1.0, 0.0)
                    contact_value = self._ref_contacts[self._camera_env_id, rb_index]
                    color = low_color + (high_color - low_color) * contact_value
                    self._gym.set_rigid_body_color(self._envs[self._camera_env_id], 
                                                   self._ref_char_handles[self._camera_env_id], 
                                                   rb_index, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        if self._report_tracking_error:
            if self.has_dm_envs():
                char_joint_rot = self._kin_char_model.dof_to_rot(self._char_dof_pos)


                char_body_pos, char_body_rot = self._kin_char_model.forward_kinematics(self._char_root_pos,
                                                                                       self._char_root_rot,
                                                                                       char_joint_rot)
                
                ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos,
                                                                                     self._ref_root_rot,
                                                                                     self._ref_joint_rot)
                #dm_env.
                tracking_error = mgdm_dm_util.compute_tracking_error(
                    root_pos=self._char_root_pos,
                    root_rot=self._char_root_rot,
                    body_rot=char_body_rot,
                    body_pos=char_body_pos,
                    tar_root_pos=self._ref_root_pos,
                    tar_root_rot=self._ref_root_rot,
                    tar_body_rot=ref_body_rot,
                    tar_body_pos=ref_body_pos,
                    root_vel=self._char_root_vel,
                    root_ang_vel=self._char_root_ang_vel,
                    dof_vel=self._char_dof_vel,
                    tar_root_vel=self._ref_root_vel,
                    tar_root_ang_vel=self._ref_root_ang_vel,
                    tar_dof_vel=self._ref_dof_vel
                    )
                self._info["tracking_error"] = tracking_error
        return

    def _update_misc(self):
        super()._update_misc()
        self._update_ref_motion()


        if (self._visualize):
            self._update_ref_char()

        return
    
    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)

        self._info["timestep"] = self._timestep_buf.clone().detach()
        compute_time = time.time() - self._start_compute_time
        self._info["ep_num"] = self._ep_num_buf.detach().clone()
        self._info["compute_time"] = compute_time
        self._info["char_contact_forces"] = self._char_contact_forces.detach().clone()
        return

    def _update_ref_motion(self):
        if self.has_dm_envs():
            self.get_dm_env()._update_ref_motion()
        return

    def _update_ref_char(self):
        self._ref_char_root_pos[:] = self._ref_root_pos
        self._ref_char_root_rot[:] = self._ref_root_rot
        self._ref_char_root_vel[:] = 0
        self._ref_char_root_ang_vel[:] = 0
        self._ref_char_dof_pos[:] = self._ref_dof_pos
        self._ref_char_dof_vel[:] = 0
        self._ref_char_root_pos += self._ref_char_offset
        self._actors_need_reset[:, REF_CHAR_IDX] = True
        return
    
    def _parse_joint_err_weights(self, joint_err_w):
        num_joints = self._kin_char_model.get_num_joints()

        if (joint_err_w is None):
            self._joint_err_w = torch.ones(num_joints - 1, device=self._device, dtype=torch.float32)
        else:
            self._joint_err_w = torch.tensor(joint_err_w, device=self._device, dtype=torch.float32)

        assert(self._joint_err_w.shape[-1] == num_joints - 1)
        
        dof_size = self._kin_char_model.get_dof_size()
        self._dof_err_w = torch.zeros(dof_size, device=self._device, dtype=torch.float32)

        for j in range(1, num_joints):
            dof_dim = self._kin_char_model.get_joint_dof_dim(j)
            if (dof_dim > 0):
                curr_w = self._joint_err_w[j - 1]
                dof_idx = self._kin_char_model.get_joint_dof_idx(j)
                self._dof_err_w[dof_idx:dof_idx + dof_dim] = curr_w
        return
    
    def build_agent_states_dict(self, name_suffix="", record_obs=False):
        if not record_obs:
            self._dm_agent_motion = [{
                "fps": int(self._control_freq),
                "loop_mode": "CLAMP",
                "frames": [],
                "contacts": []
            } for _ in range(self._num_envs)]
        else:
            env_ids = torch.tensor([0], dtype=torch.int64, device=self._device)
            obs_shapes = self._compute_obs(env_ids, ret_obs_shapes=True)

            self._dm_agent_motion = [{
                "fps": int(self._control_freq),
                "loop_mode": "CLAMP",
                "frames": [],
                "contacts": [],
                "obs": [],
                "obs_shapes": obs_shapes
            } for _ in range(self._num_envs)]
        self._record_obs = record_obs
        self.set_write_agent_states_flag(True)
        self._writing_env_state = [True] * self._num_envs
        self._env_success_state = [False] * self._num_envs
        self._save_motion_name_suffix = name_suffix
        print("recording agent motion..")
        return
    
    def _get_global_xy_pos(self, env_pos, env_ids=None):
        if env_ids is None:
            return env_pos + self._env_offsets[:, 0:2]
        else:
            return env_pos + self._env_offsets[env_ids, 0:2]
        
    def _get_global_xyz_pos(self, env_pos, env_ids=None):
        if env_ids is None:
            return env_pos + self._env_offsets[:, 0:3]
        else:
            return env_pos + self._env_offsets[env_ids, 0:3]
        
    def _extract_dm_env_ids(self, env_ids):
        return env_ids[env_ids < self._num_dm_envs]
        
    def _get_dm_slice(self, data):
        assert data.shape[0] == self._num_envs
        return data[:self._num_dm_envs]
    
    def has_dm_envs(self):
        return self._num_dm_envs > 0
    
    def get_dm_env(self):
        return self._dm_env
    
    def get_extra_log_info(self):
        extra_log_info = {}
        if self.has_dm_envs():
            extra_log_info.update(self._dm_env.get_extra_log_info())
        return extra_log_info
    
    def post_test_update(self):
        if self.has_dm_envs():
            self._dm_env.post_test_update()
        return
    
    def set_rand_reset(self, val=None):
        if val is None:
            val = not self._rand_reset
        self._rand_reset = val
        if self.has_dm_envs():
            self.get_dm_env()._rand_reset = val

        print("Setting rand reset to:", self._rand_reset)
        return
    
    def set_demo_mode(self, val=None):
        if self.has_dm_envs():
            self.get_dm_env().set_demo_mode(val)
        return
    
    def set_rand_root_pos_offset_scale(self, val):
        if self.has_dm_envs():
            self.get_dm_env().set_rand_root_pos_offset_scale(val)
        return
    
    def set_reset_motion_start_time_fraction(self, val):
        if self.has_dm_envs():
            self.get_dm_env().set_motion_start_time_fraction(val)
        return

    def set_output_motion_dir(self, path):
        self._output_motion_dir = path
        return