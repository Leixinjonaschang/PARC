import time

import polyscope.imgui as psim
import torch

import parc.motionscope.include.global_header as g
import parc.motion_synthesis.procgen.astar as astar
import parc.motion_synthesis.procgen.mdm_path as mdm_path
import parc.util.motion_util as motion_util


def path_planning_gui():
    main_vars = g.MainVars()
    settings = g.PathPlanningSettings()
    astar_settings = g.PathPlanningSettings().astar_settings
    mdm_path_settings = g.PathPlanningSettings().mdm_path_settings
    curr_motion = g.MotionManager().get_curr_motion()
    terrain = g.TerrainMeshManager().get_active_terrain(require=True)

    changed, settings.extend_path_mode = psim.Checkbox("Extend path mode", settings.extend_path_mode)
    changed, settings.manual_placement_mode = psim.Checkbox("Manual Placement Mode", settings.manual_placement_mode)

    if psim.TreeNode("A* config"):
        changed, astar_settings.max_z_diff = psim.InputFloat("max z diff", astar_settings.max_z_diff)
        changed, astar_settings.max_jump_xy_dist = psim.InputFloat("max jump xy dist", astar_settings.max_jump_xy_dist)
        changed, astar_settings.max_jump_z_diff = psim.InputFloat("max jump z diff", astar_settings.max_jump_z_diff)
        changed, astar_settings.min_jump_z_diff = psim.InputFloat("min jump z diff", astar_settings.min_jump_z_diff)
        changed, astar_settings.w_z = psim.InputFloat("w_z", astar_settings.w_z)
        changed, astar_settings.w_xy = psim.InputFloat("w_xy", astar_settings.w_xy)
        changed, astar_settings.w_bumpy = psim.InputFloat("w_bumpy", astar_settings.w_bumpy)
        changed, astar_settings.max_bumpy = psim.InputFloat("max_bumpy", astar_settings.max_bumpy)
        changed, astar_settings.uniform_cost_max = psim.InputFloat("uniform cost max", astar_settings.uniform_cost_max)
        changed, astar_settings.uniform_cost_min = psim.InputFloat("uniform cost min", astar_settings.uniform_cost_min)
        changed, astar_settings.min_start_end_xy_dist = psim.InputFloat("min start end xy dist", astar_settings.min_start_end_xy_dist)
        changed, astar_settings.max_cost = psim.InputFloat("max cost", astar_settings.max_cost)
        psim.TreePop()

    if psim.TreeNode("MDM gen along path config"):
        changed, mdm_path_settings.next_node_lookahead = psim.InputInt("Next node lookahead (For target dir)", mdm_path_settings.next_node_lookahead)
        changed, mdm_path_settings.rewind_num_frames = psim.InputInt("Rewind num frames (before generating mdm)", mdm_path_settings.rewind_num_frames)
        changed, mdm_path_settings.end_of_path_buffer = psim.InputInt("End of path buffer (for determining end of mdm gen)", mdm_path_settings.end_of_path_buffer)
        changed, mdm_path_settings.max_motion_length = psim.InputFloat("Max motion length", mdm_path_settings.max_motion_length)
        changed, mdm_path_settings.path_batch_size = psim.InputInt("Path batch size", mdm_path_settings.path_batch_size)
        changed, mdm_path_settings.mdm_batch_size = psim.InputInt("MDM batch size", mdm_path_settings.mdm_batch_size)
        changed, mdm_path_settings.top_k = psim.InputInt("MDM top k", mdm_path_settings.top_k)
        changed, mdm_path_settings.w_target = psim.InputFloat("w_target", mdm_path_settings.w_target)
        changed, mdm_path_settings.w_contact = psim.InputFloat("w_contact", mdm_path_settings.w_contact)
        changed, mdm_path_settings.w_pen = psim.InputFloat("w_pen", mdm_path_settings.w_pen)
        changed, settings.use_prev_frames = psim.Checkbox("Generate from prev frames", settings.use_prev_frames)
        psim.TreePop()

    if psim.Button("Random start/end points"):
        start_grid_ind, end_grid_ind = astar.pick_random_start_end_nodes_on_edges(terrain=terrain,
                                                                                  min_dist=astar_settings.min_start_end_xy_dist)
        g.PathPlanningSettings().clear_waypoints()
        g.PathPlanningSettings().place_waypoint(start_grid_ind)
        g.PathPlanningSettings().place_waypoint(end_grid_ind)


    if psim.Button("Construct A* Graph"):
        g.PathPlanningSettings().nav_graph = astar.construct_navigation_graph(terrain = terrain,
                                                                              max_z_diff = g.PathPlanningSettings().astar_settings.max_z_diff,
                                                                              max_jump_xy_dist = g.PathPlanningSettings().astar_settings.max_jump_xy_dist,
                                                                              max_jump_z_diff = g.PathPlanningSettings().astar_settings.max_jump_z_diff,
                                                                              min_jump_z_diff = g.PathPlanningSettings().astar_settings.min_jump_z_diff)

    if psim.Button("A*") and g.PathPlanningSettings().nav_graph is not None:
        g.PathPlanningSettings().run_astar()
        

    if psim.Button("clear waypoints"):
        g.PathPlanningSettings().clear_waypoints()

    if settings.path_nodes is not None:

        if psim.Button("clear path nodes"):
            g.PathPlanningSettings().clear_path_nodes()

        if psim.Button("Generate Path from current motion and terrain"):

            path_nodes = []

            root_pos = curr_motion.mlib._frame_root_pos
            num_frames = root_pos.shape[0]

            path_nodes.append(root_pos[0].clone())

            for i in range(1, num_frames):
                curr_root_pos = root_pos[i]

                if torch.linalg.norm(curr_root_pos - path_nodes[-1]) > g.g_mdm_model._dx:
                    path_nodes.append(curr_root_pos.clone())

            terrain_dx = terrain.dxdy[0].item()
            terrain_dy = terrain.dxdy[1].item()
            for i in range(len(path_nodes)):

                node_pos = path_nodes[i]
                node_pos[0] = round(node_pos[0].item() / terrain_dx) * terrain_dx
                node_pos[1] = round(node_pos[1].item() / terrain_dy) * terrain_dy

            def remove_duplicates(tensor_list):
                unique_tensors = []
                for tensor in tensor_list:
                    if not any(torch.equal(tensor, unique_tensor) for unique_tensor in unique_tensors):
                        unique_tensors.append(tensor)
                return unique_tensors
            
            path_nodes = remove_duplicates(path_nodes)

            path_nodes = torch.stack(path_nodes)
            hf_z = terrain.get_hf_val_from_points(path_nodes[..., 0:2])
            path_nodes[..., 2] = hf_z.clone()
            g.PathPlanningSettings().path_nodes = path_nodes
            g.PathPlanningSettings().visualize_path_nodes("path_from_motion", path_nodes)
 

        if psim.Button("Gen MDM motion at path start"):
            gen_motion_frames = mdm_path.gen_mdm_motion_at_path_start(path_nodes_xyz=g.PathPlanningSettings().path_nodes,
                                                             terrain=terrain,
                                                             char_model=curr_motion.char.char_model,
                                                             mdm_model=g.g_mdm_model,
                                                             mdm_settings=g.MDMSettings().gen_settings,
                                                             prev_frames=None,
                                                             batch_size=1)


            g.MotionManager().make_new_motion(motion_frames=gen_motion_frames,
                              new_motion_name="ASTARMDM",
                              motion_fps=g.g_mdm_model._sequence_fps, 
                              vis_fps=10)

        if psim.Button("Generate Frames along path from current motion"):
            num_prev_states = g.g_mdm_model._num_prev_states
            start_frame = curr_motion.mlib._motion_num_frames[0].item() - 1 - g.PathPlanningSettings().mdm_path_settings.rewind_num_frames
            start_time = start_frame / curr_motion.mlib._motion_fps[0].item()
            curr_motion.set_to_time(start_time)
            gen_motion_frames, _ = mdm_path.generate_frames_along_path(prev_frames=curr_motion.char.motion_frames.unsqueeze(0),
                                                                        path_nodes_xyz=g.PathPlanningSettings().path_nodes,
                                                                        terrain=terrain,
                                                                        char_model=curr_motion.char.char_model,
                                                                        mdm_model=g.g_mdm_model,
                                                                        mdm_settings=g.MDMSettings().gen_settings,
                                                                        path_settings=mdm_path_settings)

            new_frames, new_contacts = gen_motion_frames.get_mlib_format(curr_motion.char.char_model)
            new_frames = new_frames.squeeze(0)
            new_contacts = new_contacts.squeeze(0)
            new_frames = torch.cat([curr_motion.mlib._motion_frames[:start_frame], new_frames[num_prev_states:]], dim=0)
            new_contacts = torch.cat([curr_motion.mlib._frame_contacts[:start_frame], new_contacts[num_prev_states:]], dim=0)
            g.MotionManager().make_new_motion(motion_frames=new_frames, 
                              contact_frames=new_contacts, 
                              new_motion_name="ASTARMDM",
                              motion_fps=g.g_mdm_model._sequence_fps, 
                              vis_fps=10)
            
        
        if psim.Button("Generate Frames from start until end of path"):
            start_time = time.time()

            if g.PathPlanningSettings().use_prev_frames:
                prev_frames = curr_motion.char.motion_frames
            else:
                prev_frames = None

            mdm_motion_frames, sliced_terrains, info = mdm_path.generate_frames_until_end_of_path(
                path_nodes=g.PathPlanningSettings().path_nodes,
                terrain=terrain,
                char_model=curr_motion.char.char_model,
                mdm_model=g.g_mdm_model,
                prev_frames=prev_frames,
                mdm_path_settings=mdm_path_settings,
                mdm_gen_settings=g.MDMSettings().gen_settings)
            
            end_time = time.time()
            batch_size = len(mdm_motion_frames)#.root_pos.shape[0]
            print("Finished generating", batch_size, "motions in:", end_time - start_time, "seconds.")
            print("losses:", info["losses"])
            print("contact losses:", info["contact_losses"])
            print("pen losses:", info["pen_losses"])

            batch_size = g.PathPlanningSettings().mdm_path_settings.top_k

            fps = curr_motion.mlib._motion_fps[0].item()
            curr_frame_idx = main_vars.motion_time * fps
            curr_frame_idx = int(round(curr_frame_idx))

            for i in range(batch_size):
                motion_frames = mdm_motion_frames[i]
                motion_frames.set_device(g.MainVars().device)
            

                if g.MDMSettings().append_mdm_motion_to_prev_motion:
                    motion_frames = motion_util.cat_motion_frames([
                        curr_motion.mlib.get_frames_for_id(0).get_slice(slice(0, curr_frame_idx+1)).unsqueeze(0),
                        motion_frames.get_slice(slice(curr_motion.char._history_length, motion_frames.root_pos.shape[1]))])


                g.MotionManager().make_new_motion(
                    motion_frames=motion_frames,
                    new_motion_name="ASTARMDM" + str(i).zfill(len(str(batch_size))),
                    motion_fps=g.g_mdm_model._sequence_fps,
                    vis_fps=5)

    return