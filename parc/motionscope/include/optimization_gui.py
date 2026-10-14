import copy

import polyscope as ps
import polyscope.imgui as psim
import scipy
import torch
import numpy as np

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motionscope.include.global_header as g
import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.geom_util as geom_util
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util
import parc.motionscope.polyscope_util as ps_util

g_opt_generator = None

def update_opt_vis_meshes(curr_motion_frames: motion_lib.MotionFrames,
                          settings,
                          char_model):
    if len(settings.opt_vis_meshes) == 0:
        return

    body_pos, body_rot = char_model.forward_kinematics(
        curr_motion_frames.root_pos,
        curr_motion_frames.root_rot,
        curr_motion_frames.joint_rot,
    )

    for vis_idx, char_meshes in zip(settings.opt_vis_frame_idxs, settings.opt_vis_meshes):
        if vis_idx >= curr_motion_frames.root_pos.shape[0]:
            continue
        ps_util.update_char_motion_mesh(body_pos[vis_idx], body_rot[vis_idx], char_meshes, char_model)

def motion_optimization_gui():
    global g_opt_generator
    settings = g.OptimizationSettings()
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()

    changed, settings.num_iters = psim.InputInt("num iters:", settings.num_iters)
    changed, settings.step_size = psim.InputFloat("optimization step size", settings.step_size)
    changed, settings.w_root_pos = psim.InputFloat("w_root_pos", settings.w_root_pos)
    changed, settings.w_root_rot = psim.InputFloat("w_root_rot", settings.w_root_rot)
    changed, settings.w_joint_rot = psim.InputFloat("w_joint_rot", settings.w_joint_rot)
    changed, settings.w_smoothness = psim.InputFloat("w_smoothness", settings.w_smoothness)
    changed, settings.w_penetration = psim.InputFloat("w_penetration", settings.w_penetration)
    changed, settings.w_contact = psim.InputFloat("w_contact", settings.w_contact)
    changed, settings.w_sliding = psim.InputFloat("w_sliding", settings.w_sliding)
    changed, settings.w_body_constraints = psim.InputFloat("w_body_constraints", settings.w_body_constraints)
    changed, settings.w_jerk = psim.InputFloat("w_jerk", settings.w_jerk)
    changed, settings.max_jerk = psim.InputFloat("max_jerk", settings.max_jerk)
    changed, settings.use_wandb = psim.Checkbox("Use wandb", settings.use_wandb)
    changed, settings.visualize_optimization = psim.Checkbox("Visualize optimization", settings.visualize_optimization)
    #changed, settings.auto_compute_body_constraints = psim.Checkbox("Auto compute body constraints", settings.auto_compute_body_constraints)
    if settings.body_constraints is not None:
        num_bodies = curr_motion.char.char_model.get_num_joints()
        assert len(settings.body_constraints) == num_bodies
        if psim.TreeNode("Body Contact Point Constraints"):
            for b in range(num_bodies):
                curr_body_name = curr_motion.char.char_model.get_body_name(b)
                if psim.TreeNode(curr_body_name):
                    num_constraints = len(settings.body_constraints[b])
                    info_str = "num constraints: " + str(num_constraints)
                    psim.TextUnformatted(info_str)
                    if num_constraints > 0:
                        for c_idx in range(num_constraints):
                            constraint = settings.body_constraints[b][c_idx]
                            if psim.TreeNode("constraint " + str(c_idx)):
                                changed, constraint.start_frame_idx = psim.InputInt("start frame idx", constraint.start_frame_idx)
                                changed, constraint.end_frame_idx = psim.InputInt("end frame idx", constraint.end_frame_idx)

                                np_constraint_pt = constraint.constraint_point.numpy()
                                changed3, np_constraint_pt = psim.InputFloat3("pt", np_constraint_pt)
                                if changed3:
                                    constraint.constraint_point = torch.tensor(np_constraint_pt, dtype=torch.float32, device=g.MainVars().device)
                                    g.OptimizationSettings().create_body_constraint_ps_mesh(
                                        b, constraint.start_frame_idx, constraint.end_frame_idx, constraint.constraint_point,
                                        curr_motion.char.char_model)
                                #psim.TextUnformatted(np.array2string(constraint.constraint_point.numpy()))

                                if psim.Button("Test sdf"):
                                    # NOTE: only works for single sphere bodies right now
                                    curr_body_rot = curr_motion.char.get_body_rot(b)
                                    offset = curr_motion.char.char_model.get_geoms(b)[0]._offset
                                    radius = curr_motion.char.char_model.get_geoms(b)[0]._dims
                                    curr_body_pos = curr_motion.char.get_body_pos(b)
                                    body_center = torch_util.quat_rotate(curr_body_rot, offset) + curr_body_pos
                                    print(constraint.constraint_point)
                                    print(body_center)
                                    print(radius)
                                    sd = geom_util.sdSphere(constraint.constraint_point, body_center, radius)
                                    print("signed distance:", sd.item())

                                psim.TreePop()
                    psim.TreePop()
            psim.TreePop()
    
    if psim.Button("Clear body constraints"):
        settings.clear_body_constraints()

    if psim.Button("Motion optimization"):
        # move everything to GPU before starting optimization
        
        # slice the terrain around the motion so that optimization is more efficient
        active_terrain = g.TerrainMeshManager().get_active_terrain(require=True)
        terrain = terrain_util.slice_terrain_around_motion(curr_motion.mlib._motion_frames,
                                                           active_terrain,
                                                           localize=False)
        

        # move everything to GPU before starting optimization
        if g.MainVars().device == "cpu" and torch.cuda.is_available():
            src_frames = curr_motion.mlib.get_frames_for_id(0)
            src_frames.set_device(device="cuda:0")
            terrain.to_torch(device="cuda:0")
            char_model = kin_char_model.KinCharModel("cuda:0")
            char_model.load_char_file(g.g_char_filepath)

            if settings.body_constraints is not None:
                body_constraints = []
                for b in range(len(settings.body_constraints)):
                    body_constraints.append([])
                    for c_idx in range(len(settings.body_constraints[b])):
                        curr_body_constraint = settings.body_constraints[b][c_idx]
                        body_constraints[b].append(copy.deepcopy(curr_body_constraint))
                        body_constraints[b][c_idx].constraint_point = body_constraints[b][c_idx].constraint_point.to(device="cuda:0")
            else:
                body_constraints = settings.body_constraints
                
        else:
            src_frames = curr_motion.mlib.get_frames_for_id(0)
            char_model = curr_motion.char.char_model
            body_constraints = settings.body_constraints

        body_points = geom_util.get_char_point_samples(char_model)

        opt_frames = None

        def setup_opt_vis_meshes():
            settings.clear_opt_vis_meshes()
            num_frames = src_frames.root_pos.shape[0]
            settings.opt_vis_frame_idxs = list(range(0, num_frames, 5))
            if len(settings.opt_vis_frame_idxs) == 0:
                settings.opt_vis_frame_idxs = [0]

            vis_color = np.array([1.0, 0.6, 0.2], dtype=np.float32)
            for frame_idx in settings.opt_vis_frame_idxs:
                mesh_name = f"opt_frame_{frame_idx}"
                settings.opt_vis_meshes.append(
                    ps_util.create_char_mesh(mesh_name, color=vis_color, transparency=0.5, char_model=char_model)
                )

        

        if settings.visualize_optimization:
            setup_opt_vis_meshes()
            g_opt_generator = moopt.motion_contact_optimization(
                src_motion_frames=src_frames,
                body_points=body_points,
                terrain=terrain,
                char_model=char_model,
                num_iters=settings.num_iters,
                step_size=settings.step_size,
                w_root_pos=settings.w_root_pos,
                w_root_rot=settings.w_root_rot,
                w_joint_rot=settings.w_joint_rot,
                w_smoothness=settings.w_smoothness,
                w_penetration=settings.w_penetration,
                w_contact=settings.w_contact,
                w_sliding=settings.w_sliding,
                w_body_constraints=settings.w_body_constraints,
                w_jerk=settings.w_jerk,
                body_constraints=body_constraints,
                max_jerk=settings.max_jerk,
                exp_name=None,
                use_wandb=settings.use_wandb,
                log_file="output/opt_log.txt",
                yield_intermediate=True)

            # for opt_step_frames in opt_generator:
            #     opt_frames = opt_step_frames
            #     update_opt_vis_meshes(opt_frames)
        else:
            settings.clear_opt_vis_meshes()
            opt_frames = moopt.motion_contact_optimization(
                src_motion_frames=src_frames,
                body_points=body_points,
                terrain=terrain,
                char_model=char_model,
                num_iters=settings.num_iters,
                step_size=settings.step_size,
                w_root_pos=settings.w_root_pos,
                w_root_rot=settings.w_root_rot,
                w_joint_rot=settings.w_joint_rot,
                w_smoothness=settings.w_smoothness,
                w_penetration=settings.w_penetration,
                w_contact=settings.w_contact,
                w_sliding=settings.w_sliding,
                w_body_constraints=settings.w_body_constraints,
                w_jerk=settings.w_jerk,
                body_constraints=body_constraints,
                max_jerk=settings.max_jerk,
                exp_name=None,
                use_wandb=settings.use_wandb,
                log_file="output/opt_log.txt")
        
            if opt_frames.root_pos.device != main_vars.device:
                opt_frames.set_device(device=main_vars.device)
            
            g.MotionManager().make_new_motion(
                motion_frames=opt_frames,
                new_motion_name="opt_motion",
                motion_fps=curr_motion.mlib._motion_fps[0].item(),
                vis_fps=5,
                loop_mode=motion_lib.LoopMode.CLAMP,
                view_seq=True
            )

    if g_opt_generator is not None:
        try:
            opt_frames = next(g_opt_generator)
            char_model = g.MotionManager().get_curr_motion().char.char_model
            opt_frames.set_device(main_vars.device)
            update_opt_vis_meshes(opt_frames, settings, char_model)
        except StopIteration as stop:
            opt_frames = stop.value
            opt_frames.set_device(main_vars.device)
            g.MotionManager().make_new_motion(
                motion_frames=opt_frames,
                new_motion_name="opt_motion",
                motion_fps=curr_motion.mlib._motion_fps[0].item(),
                vis_fps=5,
                loop_mode=motion_lib.LoopMode.CLAMP,
                view_seq=True
            )
            g_opt_generator = None
            settings.clear_opt_vis_meshes()


        

    if psim.Button("Compute contact point constraints (for optimization)"):
        mlib = curr_motion.mlib
        active_terrain = g.TerrainMeshManager().get_active_terrain(require=True)
        body_constraints = moopt.compute_approx_body_constraints(mlib._frame_root_pos,
                                                                mlib._frame_root_rot,
                                                                mlib._frame_joint_rot,
                                                                mlib._frame_contacts,
                                                                mlib._kin_char_model,
                                                                active_terrain)
        settings.body_constraints = body_constraints
        settings.create_body_constraint_ps_meshes()

    return
