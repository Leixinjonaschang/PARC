import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

import parc.motionscope.include.global_header as g
import parc.motion_synthesis.procgen.mdm_path as mdm_path
from parc.motion_generator.kin_controller import KinematicController
import parc.util.torch_util as torch_util


def kin_controller_gui():

    curr_motion = g.MotionManager().get_curr_motion()
    path_planning_settings = g.PathPlanningSettings()
    main_vars = g.MainVars()

    if psim.Button("Create controller"):
        initial_frames = curr_motion.mlib.get_frames_for_id(0).unsqueeze(0)
        g.g_kin_controller = KinematicController(
            mdm_model = g.g_mdm_model,
            char_model = curr_motion.char.char_model,
            initial_frames = initial_frames)
    
    controller = g.g_kin_controller

    if controller is not None:
        assert isinstance(controller, KinematicController)
        main_vars.paused = True
        psim.TextUnformatted("Current frame: " + str(controller._curr_frame))
        changed, controller._paused = psim.Checkbox("Paused", controller._paused)
        changed, controller._replan_period = psim.InputInt("Replan period", controller._replan_period)


        if path_planning_settings.path_nodes is None:
            target_world_pos = main_vars.mouse_world_pos
        else:
            num_path_nodes = path_planning_settings.path_nodes.shape[0]
            closest_node_idx = mdm_path.get_closest_path_node_idx(curr_motion.char.get_body_pos(0).unsqueeze(0), path_planning_settings.path_nodes)
            next_node_idx = torch.clamp(closest_node_idx + path_planning_settings.mdm_path_settings.next_node_lookahead, max=num_path_nodes-1)
            target_world_pos = path_planning_settings.path_nodes[next_node_idx].squeeze(0) # final shape: [3]


        terrain = g.TerrainMeshManager().get_active_terrain(require=True)
        curr_motion_frames = controller.get_curr_motion_frame()

        controller.step(target_world_pos = target_world_pos,
                        terrain = terrain,
                        mdm_settings = g.MDMSettings().gen_settings)#,
                        #target_dir = target_heading_dir)

        curr_motion_frames = controller.get_curr_motion_frame()

        ind = terrain.get_grid_index(curr_motion_frames.root_pos[0, 0:2])
        z = terrain.hf[ind[0], ind[1]].item()

        curr_motion.char.forward_kinematics(
            root_pos = curr_motion_frames.root_pos[0],
            root_rot = curr_motion_frames.root_rot[0],
            joint_rot = curr_motion_frames.joint_rot[0],
            shadow_height = z
        )

        curr_motion.char.update_local_hf(terrain)

        contacts = curr_motion_frames.contacts[0].cpu()
        curr_motion.char.update_contact_colours_one_frame(contacts)

        if psim.Button("Remove Controller"):
            g.g_kin_controller = None
