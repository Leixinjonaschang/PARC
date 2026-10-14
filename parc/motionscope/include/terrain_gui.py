import time

import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

import parc.motionscope.include.global_header as g
import parc.util.geom_util as geom_util
import parc.util.motion_edit_lib as medit_lib
import parc.util.terrain_util as terrain_util


########## TERRAIN EDITING GUI ##########
def terrain_editing_gui():
    main_vars = g.MainVars()
    settings = g.TerrainEditorSettings()
    curr_motion = g.MotionManager().get_curr_motion()
    terrain_manager = g.TerrainMeshManager()
    active_entry = terrain_manager.get_active_entry()
    active_name = terrain_manager.get_active_terrain_name()
    terrain_names = terrain_manager.get_registered_terrain_names()

    any_visible = False
    for name in terrain_names:
        entry = terrain_manager.get_entry(name)
        if entry is None or entry.hf_ps_mesh is None:
            continue
        if entry.hf_ps_mesh.is_enabled():
            any_visible = True
            break

    toggle_label = "Hide all terrains" if any_visible else "Show all terrains"
    if psim.Button(toggle_label):
        target_visibility = not any_visible
        terrain_manager.set_all_terrain_meshes_enabled(target_visibility)
        settings.viewing_terrain = target_visibility

    if psim.TreeNode("Loaded terrains"):
        for name in terrain_names:
            entry = terrain_manager.get_entry(name)
            if entry is None:
                continue

            if psim.TreeNode(name):
                if psim.Button(f"Remove##terrain_remove_{name}"):
                    terrain_manager.remove_terrain(name)
                    active_name = terrain_manager.get_active_terrain_name()
                    active_entry = terrain_manager.get_active_entry()
                    psim.TreePop()
                    continue

                is_active = name == active_name
                changed_selected, _ = psim.Checkbox("selected", is_active)
                if changed_selected:
                    terrain_manager.set_active_terrain(name)
                    active_name = name
                    active_entry = terrain_manager.get_active_entry()
                    if active_entry is not None and active_entry.terrain is not None and curr_motion is not None:
                        active_terrain = active_entry.terrain
                        curr_motion.update_transforms(
                            shadow_height=curr_motion.char.get_hf_below_root(active_terrain)
                        )
                        curr_motion.char.update_local_hf(active_terrain)

                mesh = entry.hf_ps_mesh
                visible = mesh is not None and mesh.is_enabled()
                changed_visible, visible = psim.Checkbox("visible", visible)
                if changed_visible:
                    terrain_manager.set_terrain_mesh_enabled(visible, terrain_name=name)

                psim.TreePop()
        psim.TreePop()

    active_entry = terrain_manager.get_active_entry()
    if active_entry is None or active_entry.terrain is None:
        psim.TextUnformatted("No active terrain loaded.")
        return

    terrain = active_entry.terrain

    if hasattr(settings, "last_sdf_stats") and settings.last_sdf_stats is not None:
        sdf_stats = settings.last_sdf_stats
        if "message" in sdf_stats:
            psim.TextUnformatted(sdf_stats["message"])
        else:
            num_points = sdf_stats.get("num_points", 0)
            min_sdf = sdf_stats.get("min", 0.0)
            max_sdf = sdf_stats.get("max", 0.0)
            min_pen = sdf_stats.get("min_penetration", 0.0)
            num_pen = sdf_stats.get("num_penetrating", 0)
            query_range = sdf_stats.get("max_distance", 0.0)
            psim.TextUnformatted(
                f"SDF samples: {num_points} | min: {min_sdf:.5f} | max: {max_sdf:.5f} | min penetration: {min_pen:.5f} | penetrating points: {num_pen} | query max dist: {query_range:.2f}"
            )

    if psim.TreeNode("Terrain Info"):

        shape_str = "Shape: " + np.array2string(terrain.dims.cpu().numpy())
        psim.TextUnformatted(shape_str)

        min_xy_str = "Min X,Y: " + np.array2string(terrain.min_point.cpu().numpy())
        psim.TextUnformatted(min_xy_str)

        selected_grid_ind_str = "Curr grid ind: " + np.array2string(main_vars.selected_grid_ind.cpu().numpy())
        psim.TextUnformatted(selected_grid_ind_str)

        curr_grid_ind_point = terrain.get_point(main_vars.selected_grid_ind)
        z = terrain.get_hf_val(main_vars.selected_grid_ind)
        curr_grid_ind_xyz = np.array([curr_grid_ind_point[0].item(), curr_grid_ind_point[1].item(), z])
        curr_grid_ind_point_str = "Curr grid point X,Y, Z: " + np.array2string(curr_grid_ind_xyz, precision=6)
        psim.TextUnformatted(curr_grid_ind_point_str)
        psim.TreePop()

    edge_width = g.TerrainMeshManager().hf_ps_mesh.get_edge_width()
    changed, view_edges = psim.Checkbox("View terrain edges", edge_width > 0.0)
    if changed:
        g.TerrainMeshManager().hf_ps_mesh.set_edge_width(1.0 if view_edges else 0.0)


    changed, main_vars.mouse_size = psim.InputInt("mouse size", main_vars.mouse_size)
    if changed:
        main_vars.mouse_size = min(main_vars.mouse_size, 10)
        main_vars.mouse_size = max(main_vars.mouse_size, 0)
        g.update_mouse_ball_ps_meshes(main_vars.mouse_size)
    changed, settings.height = psim.InputFloat("terrain height", settings.height)

    
    opened = psim.BeginCombo("terrain edit mode", settings.edit_modes[settings.curr_edit_mode])
    if opened:
        for i, item in enumerate(settings.edit_modes):
            is_selected = (i == settings.curr_edit_mode)
            if psim.Selectable(item, is_selected)[0]:
                settings.curr_edit_mode = i

            # Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if is_selected:
                psim.SetItemDefaultFocus()
        psim.EndCombo()

    changed, settings.viewing_terrain = psim.Checkbox("View terrain", settings.viewing_terrain)
    if changed:
        g.TerrainMeshManager().hf_ps_mesh.set_enabled(settings.viewing_terrain)
    changed, settings.viewing_max = psim.Checkbox("View max heightfield bounds", settings.viewing_max)
    if changed:
        g.TerrainMeshManager().hf_max_mesh.set_enabled(settings.viewing_max)
    changed, settings.viewing_min = psim.Checkbox("View min heightfield bounds", settings.viewing_min)
    if changed:
        g.TerrainMeshManager().hf_min_mesh.set_enabled(settings.viewing_min)

    if psim.Button("Set all to height"):
        if settings.edit_modes[settings.curr_edit_mode] == "heightfield":
            terrain.hf[...] = settings.height
            terrain_manager.soft_rebuild()
            curr_motion.char.update_local_hf(terrain)

        if settings.edit_modes[settings.curr_edit_mode] == "max":
            terrain.hf_maxmin[..., 0] = settings.height
            terrain_manager.build_ps_max_mesh()
        if settings.edit_modes[settings.curr_edit_mode] == "min":
            terrain.hf_maxmin[..., 1] = settings.height
            terrain_manager.build_ps_min_mesh()

    if psim.Button("Compute extra vals for terrain"):
        char_point_samples = geom_util.get_char_point_samples(curr_motion.char.char_model)
        hf_mask_inds, new_terrain = terrain_util.compute_hf_extra_vals(
            curr_motion.mlib.get_frames_for_id(0),
            terrain,
            curr_motion.char.char_model,
            char_point_samples)

        curr_motion.mlib._hf_mask_inds = [hf_mask_inds]

        terrain = new_terrain
        terrain_manager.update_active_terrain(new_terrain)

    if psim.TreeNode("Build New Flat Terrain"):
        changed, settings.dx = psim.InputFloat("dx", settings.dx)
        changed, settings.new_terrain_dim_x = psim.InputInt("New terrain dim x", settings.new_terrain_dim_x)
        changed, settings.new_terrain_dim_y = psim.InputInt("New terrain dim y", settings.new_terrain_dim_y)
        if psim.Button("Build New Flat Terrain"):
            terrain = terrain_util.SubTerrain("terrain",
                                              x_dim = settings.new_terrain_dim_x,
                                              y_dim = settings.new_terrain_dim_y,
                                              dx = settings.dx,
                                              dy = settings.dx,
                                              min_x = 0.0,
                                              min_y = 0.0,
                                              device=main_vars.device)
            terrain_manager.update_active_terrain(terrain)
        psim.TreePop()

    if psim.TreeNode("Heuristic Terrain Fitting Algorithms"):

        changed, settings.terrain_padding = psim.InputFloat("terrain padding", settings.terrain_padding)
        if(psim.Button("Regenerate Terrain")):
            terrain = medit_lib.create_terrain_for_motion(curr_motion.mlib.get_frames_for_id(0),
                                                        curr_motion.char.char_model,
                                                        char_points = curr_motion.char.get_surface_point_samples(),
                                                        padding=settings.terrain_padding,
                                                        dx = settings.dx)

            # TODO: don't hard code thjs
            terrain.hf_maxmin[..., 0] = 3.0
            terrain.hf_maxmin[..., 1] = -3.0
            terrain.convert_mask_to_maxmin()
            terrain_manager.update_active_terrain(terrain)
            curr_motion.char.update_local_hf(terrain)

        if(psim.Button("Guess Terrain")):
            terrain_util.hf_from_motion_discrete_heights(curr_motion.mlib._motion_frames,
                                                            terrain,
                                                            curr_motion.char.char_model,
                                                            [0.0, 0.6])
            terrain_manager.soft_rebuild()

            curr_motion.char.update_local_hf(terrain)

        if(psim.Button("Guess Terrain 2")):
            new_terrain = terrain_util.build_contact_terrain_heuristic(
                motion=curr_motion.mlib.get_frames_for_id(0),
                char_model=curr_motion.char.char_model,
                body_point_samples=curr_motion.char.get_surface_point_samples(),
                contact_threshold=0.1,
                xy_padding=2.0,
                grid_dx=0.1,
                grid_dy=0.1
            )
            terrain_manager.register_terrain("guessed terrain", new_terrain)
            curr_motion.char.update_local_hf(terrain)
        psim.TreePop()

    if psim.TreeNode("Change Terrain Dimensions"):
        changed, settings.slice_min_i = psim.InputInt("Slice min i", settings.slice_min_i)
        changed, settings.slice_min_j = psim.InputInt("Slice min j", settings.slice_min_j)
        changed, settings.slice_max_i = psim.InputInt("Slice max i", settings.slice_max_i)
        changed, settings.slice_max_j = psim.InputInt("Slice max j", settings.slice_max_j)

        if psim.Button("Slice terrain"):
            terrain_util.slice_terrain(terrain,
                                       settings.slice_min_i, settings.slice_min_j,
                                       settings.slice_max_i, settings.slice_max_j)
            terrain_manager.rebuild()
        if psim.Button("Slice terrain around motion (and localize motion)"):
            root_pos = curr_motion.mlib._frame_root_pos
            sliced_terrain, localized_root_pos = terrain_util.slice_terrain_around_motion(
                root_pos, terrain)

            localized_motion = curr_motion.mlib.get_frames_for_id(0)
            localized_motion.root_pos = localized_root_pos

            g.MotionManager().make_new_motion(
                motion_frames=localized_motion,
                new_motion_name=curr_motion.name,
                motion_fps=curr_motion.mlib._motion_fps[0].item(),
                vis_fps=5)

            terrain = sliced_terrain
            terrain_manager.update_active_terrain(sliced_terrain)

        if psim.Button("Slice terrain around motion"):
            sliced_terrain = terrain_util.slice_terrain_around_motion(curr_motion.mlib._frame_root_pos, terrain, localize=False)

            terrain = sliced_terrain
            terrain_manager.update_active_terrain(sliced_terrain)

        if psim.Button("Pad terrain"):
            terrain.pad(1, 0.0)
            terrain_manager.rebuild()

        if psim.Button("Pad with min height"):
            min_h = torch.min(terrain.hf).item()
            terrain.pad(1, min_h)
            terrain_manager.rebuild()
        psim.TreePop()

    if psim.TreeNode("Terrain Editing Operations"):
        if psim.Button("Downsample Terrain"):
            terrain = terrain_util.downsample_terrain(terrain)

            # TODO: don't hard code thjs
            terrain.hf_maxmin[..., 0] = 3.0
            terrain.hf_maxmin[..., 1] = -3.0
            terrain_manager.update_active_terrain(terrain)
            curr_motion.char.update_local_hf(terrain)

        changed, settings.maxpool_size = psim.InputInt("maxpool size", settings.maxpool_size)
        if(psim.Button("Maxpool terrain")):
            terrain_util.maxpool_hf(terrain.hf, terrain.hf_maxmin, settings.maxpool_size)
            terrain_manager.soft_rebuild()
            curr_motion.char.update_local_hf(terrain)

        if psim.Button("Maxpool terrain x"):
            terrain_util.maxpool_hf_1d_x(terrain.hf, terrain.hf_maxmin, settings.maxpool_size)
            terrain_manager.soft_rebuild()
            curr_motion.char.update_local_hf(terrain)

        if psim.Button("Maxpool terrain y"):
            terrain_util.maxpool_hf_1d_y(terrain.hf, terrain.hf_maxmin, settings.maxpool_size)
            terrain_manager.soft_rebuild()
            curr_motion.char.update_local_hf(terrain)

        if psim.Button("Minpool Terrain"):
            terrain_util.minpool_hf(terrain.hf, terrain.hf_maxmin, settings.maxpool_size)
            terrain_manager.soft_rebuild()
            curr_motion.char.update_local_hf(terrain)

        if psim.Button("Detect sharp lines"):
            def detect_sharp_line(i, j):

                center_h = terrain.hf[i, j]
                
                test1 = center_h > terrain.hf[i-1, j] and center_h > terrain.hf[i+1, j]
                test2 = center_h < terrain.hf[i-1, j] and center_h < terrain.hf[i+1, j]
                test3 = center_h > terrain.hf[i, j-1] and center_h > terrain.hf[i, j+1]
                test4 = center_h < terrain.hf[i, j-1] and center_h < terrain.hf[i, j+1]

                return test1 or test2 or test3 or test4

            sharp_line_points = []
            for i in range(1, terrain.hf.shape[0]-1):
                for j in range(1, terrain.hf.shape[1]-1):
                    if detect_sharp_line(i, j):
                        xyz = terrain.get_xyz_point(torch.tensor([i, j], dtype=torch.int64)).numpy()
                        sharp_line_points.append(xyz)

            sharp_line_points = np.stack(sharp_line_points)

            ps.register_point_cloud("sharp line points", sharp_line_points, radius=0.01)
            #test = 0

        if psim.Button("Remove sharp lines"):
            terrain_util.remove_sharp_lines(terrain)
            terrain_manager.rebuild()

        if psim.Button("Flat maxpool 2x2"):

            terrain_util.flat_maxpool_2x2(terrain)

            terrain_manager.rebuild()

        if psim.Button("Flat maxpool 3x3"):

            terrain_util.flat_maxpool_3x3(terrain)

            terrain_manager.rebuild()

        if psim.Button("Flatten 4x4 path start/end nodes"):

            start_node_th = torch.from_numpy(g.PathPlanningSettings().start_node)
            end_node_th = torch.from_numpy(g.PathPlanningSettings().end_node)
            terrain_util.flatten_4x4_near_edge(terrain, start_node_th,
                                            terrain.get_hf_val(start_node_th).item())
            terrain_util.flatten_4x4_near_edge(terrain, end_node_th,
                                            terrain.get_hf_val(end_node_th).item())
            terrain_manager.rebuild()

        if psim.Button("Plot local hf"):
            hf_z = curr_motion.char.get_normalized_local_hf(3.0)
            hf_z = hf_z.transpose(1, 0).rot90(k=3)
            #plt.imshow(hf_z.transpose(1, 0), cmap='viridis', origin='lower')
            plt.imshow(hf_z, cmap='viridis', origin='lower')
            plt.show()

        if psim.Button("Plot global hf"):
            hf_z = terrain.hf.numpy()
            plt.imshow(hf_z.transpose(1, 0), cmap='viridis', origin='lower')
            plt.xticks([])
            plt.yticks([])
            filepath = "output/terrain" + str(time.time()) + ".png"
            plt.savefig(fname=filepath, bbox_inches="tight")
            plt.show()

        psim.TreePop()

    if psim.TreeNode("Procedural Generation"):
        if psim.TreeNode("Generate Boxes Config"):
            changed, settings.num_boxes = psim.InputInt("num boxes", settings.num_boxes)
            changed, settings.box_max_len = psim.InputInt("box max len", settings.box_max_len)
            changed, settings.box_min_len = psim.InputInt("box min len", settings.box_min_len)
            changed, settings.max_box_h = psim.InputFloat("box max height", settings.max_box_h)
            changed, settings.min_box_h = psim.InputFloat("box min height", settings.min_box_h)
            changed, settings.max_box_angle = psim.InputFloat("max box angle", settings.max_box_angle)
            changed, settings.min_box_angle = psim.InputFloat("min box angle", settings.min_box_angle)
            changed, settings.use_maxmin = psim.Checkbox("Use maxmin", settings.use_maxmin)
            psim.TreePop()
        
        if psim.Button("Generate Boxes"):

            terrain_util.add_boxes_to_hf2(terrain.hf,
                                        box_max_height=settings.max_box_h,
                                        box_min_height=settings.min_box_h,
                                        hf_maxmin= terrain.hf_maxmin if settings.use_maxmin else None,
                                        num_boxes=settings.num_boxes,
                                        box_max_len=settings.box_max_len,
                                        box_min_len=settings.box_min_len,
                                        max_angle=settings.max_box_angle,
                                        min_angle=settings.min_box_angle)

            terrain_manager.soft_rebuild()
            curr_motion.char.update_local_hf(terrain)

        if psim.TreeNode("Curvy Paths config"):
            changed, settings.num_terrain_paths = psim.InputInt("num terrain paths", settings.num_terrain_paths)
            changed, settings.path_max_height = psim.InputFloat("path max height", settings.path_max_height)
            changed, settings.path_min_height = psim.InputFloat("path min height", settings.path_min_height)
            changed, settings.floor_height = psim.InputFloat("floor height", settings.floor_height)
            psim.TreePop()
            
        if psim.Button("Generate Curvy Paths"):
            terrain_util.gen_paths_hf(terrain, num_paths = settings.num_terrain_paths, maxpool_size=settings.maxpool_size,
                                    floor_height=settings.floor_height,
                                    path_min_height=settings.path_min_height, path_max_height=settings.path_max_height)
            terrain_manager.rebuild()

        if psim.TreeNode("Staircases config"):
            changed, settings.min_stair_start_height = psim.InputFloat("min_stair_start_height", settings.min_stair_start_height)
            changed, settings.max_stair_start_height = psim.InputFloat("max_stair_start_height", settings.max_stair_start_height)
            changed, settings.min_step_height = psim.InputFloat("min_step_height", settings.min_step_height)
            changed, settings.max_step_height = psim.InputFloat("max_step_height", settings.max_step_height)
            changed, settings.num_stairs = psim.InputInt("num_stairs", settings.num_stairs)
            changed, settings.min_stair_thickness = psim.InputFloat("min_stair_thickness", settings.min_stair_thickness)
            changed, settings.max_stair_thickness = psim.InputFloat("max_stair_thickness", settings.max_stair_thickness)
            psim.TreePop()

        if psim.Button("Generate Staircases"):
            terrain.hf[...] = 0.0
            terrain_util.add_stairs_to_hf(terrain,
                                          min_stair_start_height=settings.min_stair_start_height,
                                          max_stair_start_height=settings.max_stair_start_height,
                                          min_step_height=settings.min_step_height,
                                          max_step_height=settings.max_step_height,
                                          num_stairs=settings.num_stairs,
                                          min_stair_thickness=settings.min_stair_thickness,
                                          max_stair_thickness=settings.max_stair_thickness)
            terrain_manager.rebuild()

        if psim.Button("Gen Cave"):

            terrain = terrain_util.generate_cave(25, 25, 5, device=g.MainVars().device)

            terrain_manager.update_active_terrain(terrain)

        psim.TreePop()
    return