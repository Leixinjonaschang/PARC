import platform

if platform.system() == "Linux":
    import parc.motion_tracker.envs.env_builder as env_builder
    import parc.motion_tracker.learning.agent_builder as agent_builder

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

import parc.anim.motion_lib as motion_lib
import parc.motion_generator.gen_util as gen_util
import parc.motionscope.include.global_header as g
import parc.motionscope.polyscope_util as ps_util
import parc.motionscope.ps_mdm_util as ps_mdm_util
import parc.motion_synthesis.procgen.mdm_path as mdm_path
import parc.util.misc_util as misc_util
import parc.util.motion_util as motion_util
import parc.util.torch_util as torch_util
from parc.motionscope.include.singleton import SingletonClass


class IsaacGymManager(SingletonClass):
    env = None
    agent = None
    device = None
    paused = True
    ps_char = None
    recorded_frames = []
    is_recording = False
    is_closed_loop_generating = False
    replan_time = 1.0
    num_replan_loops = 4
    start_time_fraction = 0.0

    def start_isaac_gym(self, env_file, num_envs=1, device="cuda:0", visualize=False):
        if platform.system() == "Linux":
            self.env = env_builder.build_env(env_file, num_envs=num_envs, device=device, visualize=visualize)
            self.device = device
        return
    
    def load_agent(self, agent_file, model_file):
        if platform.system() == "Linux":
            self.agent = agent_builder.build_agent(agent_file, self.env, self.device)
            self.agent.load(model_file)
            self.agent.eval_mode()
            self.agent.reset()
        return
    
    def create_motionscope_character(self):
        char_model = self.env._kin_char_model.get_copy(g.MainVars().device)
        self.ps_char = ps_mdm_util.MDMCharacterPS("isaac_gym_sim_char", color=[0.9, 0.9, 1.0], char_model=char_model)

        ## OBSERVATIONS ##
        self.obs_shapes = self.env._compute_obs(ret_obs_shapes=True)
        self.obs_slices = dict()

        curr_idx = 0
        for key in self.obs_shapes:
            print(key)
            print(self.obs_shapes[key])

            flat_shape = self.obs_shapes[key]["shape"].numel()
            self.obs_slices[key] = slice(curr_idx, curr_idx + flat_shape)
            curr_idx += flat_shape

        self.num_tar_obs = self.obs_shapes["tar_obs"]["shape"][0]
        self.ps_tar_char_meshes = []
        self.tar_char_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for i in range(self.num_tar_obs):
            ps_char_meshes = ps_util.create_char_mesh("ig_tar_obs_" + str(i), color=self.tar_char_color, transparency=0.3, char_model=char_model)
            self.ps_tar_char_meshes.append(ps_char_meshes)
        return
    
    def is_ready(self):
        return self.env != None and self.agent != None
    
    def reset(self):
        self.recorded_frames.clear()
        self.env.reset()
        if self.is_recording:
            self.record_char_state()
        return

    def reset_to_time(self, time: float):
        self.env.get_dm_env()._rand_reset = False
        self.env.get_dm_env()._motion_start_time_fraction[:] = time
        self.reset()
        self.env.get_dm_env()._motion_start_time_fraction[:] = 0.0
        self.env.get_dm_env()._rand_reset = True
        return
    
    def reset_to_frame_0(self):
        self.env.get_dm_env()._rand_reset = False
        self.reset()
        self.env.get_dm_env()._rand_reset = True
        return
    
    def step(self):
        self.agent.step()

        ## VISUALIZATION CODE ##
        if self.ps_char is not None:
            global_root_pos = self.env._char_root_pos[0].cpu()
            global_root_rot = self.env._char_root_rot[0].cpu()

            sim_dof = self.env._char_dof_pos[0].cpu()
            joint_rot = self.ps_char.char_model.dof_to_rot(sim_dof)

            terrain = g.TerrainMeshManager().get_active_terrain(require=True)
            shadow_height = terrain.get_hf_val_from_points(global_root_pos[0:2])
            self.ps_char.forward_kinematics(root_pos=global_root_pos, root_rot=global_root_rot, joint_rot=joint_rot, shadow_height=shadow_height)

            global_heading_quat = torch_util.calc_heading_quat(global_root_rot)
            
            obs = self.env._obs_buf.cpu()

            tar_obs = obs[0, self.obs_slices["tar_obs"]].reshape(self.num_tar_obs, -1)

            tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos = misc_util.inverse_tar_obs(tar_obs)

            for idx in range(self.num_tar_obs):
                root_pos = torch_util.quat_rotate(global_heading_quat, tar_root_pos[idx]) + global_root_pos
                root_rot = torch_util.quat_multiply(global_heading_quat, tar_root_rot[idx])
                joint_rot = tar_joint_rot[idx]
                body_pos, body_rot = self.ps_char.char_model.forward_kinematics(root_pos, root_rot, joint_rot)
                ps_util.update_char_motion_mesh(body_pos, body_rot, self.ps_tar_char_meshes[idx], self.ps_char.char_model)

                # for body_id in range(0, char_model.get_num_joints()):
                #     body_contact = tar_contacts[round_frame_idx, i, body_id].item()
                #     ig_obs.ps_tar_char_meshes[i][body_id].set_color(red * body_contact + ig_obs.tar_char_color * (1.0 - body_contact))
        
        if self.is_recording:
            self.record_char_state()

            if self.is_closed_loop_generating:
                dm_env = self.env.get_dm_env()
                motion_time = dm_env._get_motion_times(0)

                if motion_time > self.replan_time:
                    target_world_pos = g.MainVars().mouse_world_pos.to(device=self.env._device)
                    if len(target_world_pos.shape) == 1:
                        target_world_pos = target_world_pos.unsqueeze(0)
                    prev_frames = self.recorded_frames[-2:]
                    prev_frames = motion_util.cat_motion_frames(prev_frames)

                    total_gen_motion_frames = []
                    for i in range(self.num_replan_loops):
                        gen_motion_frames = gen_util.gen_mdm_motion(
                            target_world_pos = target_world_pos,
                            prev_frames = prev_frames,
                            terrain = self.env.get_dm_env()._terrain,
                            mdm_model = g.g_mdm_model,
                            char_model = self.env._kin_char_model,
                            mdm_settings = g.MDMSettings().gen_settings,
                            target_dir = None)
                        
                        total_gen_motion_frames.append(gen_motion_frames)
                        num_gen_frames = gen_motion_frames.root_pos.shape[1]
                        prev_frames = gen_motion_frames.get_slice(slice(num_gen_frames-2, num_gen_frames))
                        
                    total_gen_motion_frames = motion_util.cat_motion_frames(total_gen_motion_frames)
                    
                    
                    dm_env._mlib = motion_lib.MotionLib.from_frames(frames=total_gen_motion_frames,
                                                                    char_model=dm_env._kin_char_model,
                                                                    device=dm_env._device,
                                                                    loop_mode=dm_env._mlib.get_motion_loop_mode_enum(0),
                                                                    fps=dm_env._mlib.get_motion_fps(0),
                                                                    contact_info=dm_env._mlib._contact_info)
                    
                    self._curr_frame = 2 # start rendering the earliest generated future frame

                    dm_env._motion_time_offsets[0] = 1.0 / 30.0
                    dm_env._time_buf[0] = 0.0
                    dm_env._timestep_buf[0] = 0
        return
    
    def loop_function(self):
        self.step()
        return
    
    def compute_critic_value(self):
        norm_obs = self.agent._obs_norm.normalize(self.agent._curr_obs)
        critic_val = self.agent._model.eval_critic(norm_obs)

        return critic_val.item()
    
    def record_char_state(self):
        root_pos = self.env._char_root_pos[0].clone()
        root_rot = self.env._char_root_rot[0].clone()
        joint_dof = self.env._char_dof_pos[0].clone()
        joint_rot = self.env._kin_char_model.dof_to_rot(joint_dof)
        char_contacts = torch.norm(self.env._char_contact_forces[0], dim=-1)
        char_contacts = char_contacts > 0.001
        char_contacts = char_contacts.to(dtype=torch.float32)

        curr_frame = motion_util.MotionFrames(root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot, contacts=char_contacts)

        self.recorded_frames.append(curr_frame.unsqueeze(0).unsqueeze(0))
        return
    
    def create_motion_from_recorded_frames(self):

        if len(self.recorded_frames) < 2:
            print("not enough recorded frames")
            return
        
        motion_frames = motion_util.cat_motion_frames(self.recorded_frames)
        motion_frames.set_device(g.MainVars().device)

        g.MotionManager().make_new_motion(motion_frames, "dm_recorded_motion", motion_fps=30, vis_fps=5)
        return

def isaac_gym_gui():

    if not IsaacGymManager().is_ready():
        if psim.Button("Start Isaac Gym"):
            # env_file = "../tests/parkour/parkour_dataset_v_22_exp001/dm_env_TEASER_TERRAIN_mxu.yaml"
            # env_file = "../tests/parkour/parkour_dataset_v_22_exp001/dm_env_long_terrain_x_axis.yaml"
            # env_file = "../Data/shortcut_model_demo/simple_parc/dm_env_simple_parc.yaml"

            env_file = "data/terrains/dm_env_civilization.yaml"
            IsaacGymManager().start_isaac_gym(env_file, visualize=False)

            # model_path = "../tests/parkour/parkour_dataset_v_22_exp001/output/model.pt"
            # model_path = "../tests/parkour/parkour_dataset_v_20_exp001/output2/checkpoints/model_0000098800.pt"
            # agent_file = "../tests/parkour/parkour_dataset_v_22_exp001/ppo_agent.yaml"

            #model_path = "../tests/parc/april272025/iter_4/p3_tracker/model.pt"
            #agent_file = "../tests/parc/april272025/iter_4/p3_tracker/agent_config.yaml"
            
            tracker_model_path = "../tests/parc/april272025/iter_5_light/p3_tracker/model.pt"
            tracker_agent_path = "../tests/parc/april272025/iter_5_light/p3_tracker/agent_config.yaml"
            
            IsaacGymManager().load_agent(tracker_agent_path, tracker_model_path)

            IsaacGymManager().create_motionscope_character()
            IsaacGymManager().reset_to_frame_0()

    if IsaacGymManager().is_ready():
        if psim.Button("Step"):
            IsaacGymManager().step()

        if psim.Button("Reset"):
            IsaacGymManager().reset()

        changed, IsaacGymManager().start_time_fraction = psim.SliderFloat("Start time fraction", IsaacGymManager().start_time_fraction, v_min=0.0, v_max=1.0)

        if psim.Button("Reset to time"):
            IsaacGymManager().reset_to_time(IsaacGymManager().start_time_fraction)
    
        if psim.Button("Reset to frame 0"):
            IsaacGymManager().reset_to_frame_0()

        if psim.Button("Transfer current motion to IG parkour env"):

            mlib = g.MotionManager().get_curr_motion().mlib
            mlib = mlib.clone(IsaacGymManager().env._device)

            IsaacGymManager().env.get_dm_env()._mlib = mlib

        if psim.Button("Compute value at current frame"):
            IsaacGymManager().compute_critic_value()

        if psim.TreeNode("Physics-based motion GUI"):
            changed, IsaacGymManager().is_recording = psim.Checkbox("Record IG char states", IsaacGymManager().is_recording)

            changed, IsaacGymManager().replan_time = psim.InputFloat("Replan time", IsaacGymManager().replan_time)
            changed, IsaacGymManager().num_replan_loops = psim.InputInt("Num replan loops", IsaacGymManager().num_replan_loops)
            changed, g.MDMSettings().gen_settings.ddim_stride = psim.InputInt("DDIM Stride", g.MDMSettings().gen_settings.ddim_stride)

            if psim.Button("Make motion from recorded frames"):
                IsaacGymManager().create_motion_from_recorded_frames()

            if IsaacGymManager().is_recording:
                if len(IsaacGymManager().recorded_frames) > 2 and psim.Button("Motion Gen with current state"):

                    prev_frames = IsaacGymManager().recorded_frames[-2:]
                    prev_frames = g.motion_util.cat_motion_frames(prev_frames)

                    path_nodes = g.PathPlanningSettings().path_nodes.to(device=IsaacGymManager().device)

                    mdm_path_settings = g.PathPlanningSettings().mdm_path_settings
                    mdm_gen_settings = g.MDMSettings().gen_settings

                    gen_motion_frames, done = mdm_path.generate_frames_along_path(prev_frames=prev_frames,
                                                             path_nodes_xyz=path_nodes,
                                                             terrain=IsaacGymManager().env.get_dm_env()._terrain,
                                                             char_model=IsaacGymManager().env._kin_char_model,
                                                             mdm_model=g.g_mdm_model,
                                                             mdm_settings=mdm_gen_settings,
                                                             path_settings=mdm_path_settings,
                                                             verbose=False)
                    
                    mlib_motion_frames, mlib_contact_frames = gen_motion_frames.get_mlib_format(IsaacGymManager().env._kin_char_model)

                    mlib_motion_frames = mlib_motion_frames.to(device=g.MainVars().device).squeeze(0)
                    mlib_contact_frames = mlib_contact_frames.to(device=g.MainVars().device).squeeze(0)
                    print(mlib_motion_frames.shape, mlib_contact_frames.shape)
                    g.MotionManager().make_new_motion(mlib_motion_frames, mlib_contact_frames, "closed_loop_gen_motion",
                                                      motion_fps=30, vis_fps=5, new_color = [0.2, 0.8, 0.2])
                    
                changed, IsaacGymManager().is_closed_loop_generating = psim.Checkbox("Is closed loop generating", IsaacGymManager().is_closed_loop_generating)

            psim.TreePop()

        changed, IsaacGymManager().paused = psim.Checkbox("Paused", IsaacGymManager().paused)

        if not IsaacGymManager().paused:
            IsaacGymManager().loop_function()

            
        if psim.TreeNode("Extra Info"):
            motion_time = IsaacGymManager().env.get_dm_env()._get_motion_times(0)
            psim.TextUnformatted("Motion time: " + str(motion_time.item()) + " s")

            critic_val = IsaacGymManager().compute_critic_value()
            psim.TextUnformatted("Critic val: " + str(critic_val))
            psim.TreePop()
            
    return