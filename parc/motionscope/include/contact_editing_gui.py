import polyscope.imgui as psim

import parc.anim.motion_lib as motion_lib
import parc.motionscope.include.global_header as g
import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.motion_edit_lib as medit_lib
import parc.util.motion_util as motion_util


def _motion_lib_from_dof_frames(dof_frames, body_contacts, char_model, device, fps, loop_mode):
    # dof_frames: legacy [num_frames, 6 + dof] frames; body_contacts: [num_frames, num_bodies]
    frames = motion_util.motion_frames_from_mlib_format(dof_frames, char_model)
    frames.contacts = body_contacts[:, char_model.get_contact_body_ids()] # MotionLib keeps contact-body columns
    return motion_lib.MotionLib.from_frames(frames, char_model, device, loop_mode=loop_mode, fps=fps,
                                            contact_info=True)


def contact_editing_gui():
    settings = g.ContactEditingSettings()
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()
    terrain = g.TerrainMeshManager().get_active_terrain()
    selected_body_name = curr_motion.char.char_model.get_body_name(settings.selected_body_id)
    opened = psim.BeginCombo("Contact Body", selected_body_name)
    if opened:
        for body_name in curr_motion.char.char_model._body_names:
            clicked, hovering = psim.Selectable(body_name, selected_body_name==body_name)
            if clicked:
                settings.selected_body_id = curr_motion.char.char_model.get_body_id(body_name)
        psim.EndCombo()
    if psim.Button("Set contact label"):
        fps = int(curr_motion.mlib._motion_fps[0].item())
        curr_frame_idx = int(round(main_vars.motion_time * fps))
        curr_motion.mlib._frame_contacts[curr_frame_idx, settings.selected_body_id] = 1.0
    if psim.Button("Remove contact label"):
        fps = int(curr_motion.mlib._motion_fps[0].item())
        curr_frame_idx = int(round(main_vars.motion_time * fps))
        curr_motion.mlib._frame_contacts[curr_frame_idx, settings.selected_body_id] = 0.0

    changed, settings.start_frame_idx = psim.InputInt("Start frame idx", settings.start_frame_idx)
    changed, settings.end_frame_idx = psim.InputInt("End frame idx", settings.end_frame_idx)
    if settings.start_frame_idx < 0:
        settings.start_frame_idx = 0
    if settings.end_frame_idx >= curr_motion.mlib._motion_num_frames[0].item():
        settings.end_frame_idx = curr_motion.mlib._motion_num_frames[0].item() - 1
    if settings.end_frame_idx < settings.start_frame_idx:
        settings.end_frame_idx = settings.start_frame_idx
    
    if psim.Button("Set contact labels on slice of frames [inclusive]"):
        curr_motion.mlib._frame_contacts[settings.start_frame_idx:settings.end_frame_idx+1, settings.selected_body_id] = 1.0
    if psim.Button("Remove contact labels on slice of frames [inclusive]"):
        curr_motion.mlib._frame_contacts[settings.start_frame_idx:settings.end_frame_idx+1, settings.selected_body_id] = 0.0


    if psim.TreeNode("Automatic Contact Labelling"):

        changed, settings.contact_eps = psim.InputFloat("contact eps", settings.contact_eps)    
        if psim.Button("Correct and Label Foot Contacts"):
            try:
                corrected_frames, contact_frames = medit_lib.compute_hf_foot_contacts_and_correct_pen(
                    curr_motion.mlib._motion_frames,
                    terrain,
                    curr_motion.char.char_model,
                    settings.contact_eps)
            except ValueError as exc: # e.g. mesh feet (G1): only box feet are supported
                print("Cannot label foot contacts:", exc)
                corrected_frames = None

            if corrected_frames is not None:
                curr_motion.mlib = _motion_lib_from_dof_frames(corrected_frames, contact_frames, curr_motion.char.char_model,
                                                               main_vars.device, fps=curr_motion.mlib._motion_fps[0].item(),
                                                               loop_mode=curr_motion.mlib.get_motion_loop_mode_enum(0))
                main_vars.use_contact_info = True

                main_vars.motion_time = 0.0
                curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
                curr_motion.update_sequence(0.0, curr_motion.mlib._motion_lengths[0].item(), int(round(curr_motion.mlib._motion_lengths[0].item() * 15)))
                curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(terrain))

        if psim.Button("Correct and Label Foot Contacts, and Label Hand Contacts"):
            try:
                corrected_frames, contact_frames = medit_lib.compute_hf_foot_contacts_and_correct_pen(
                    curr_motion.mlib._motion_frames,
                    terrain,
                    curr_motion.char.char_model,
                    settings.contact_eps)

                hand_contacts = medit_lib.compute_motion_terrain_hand_contacts(curr_motion.mlib._motion_frames, 
                                                                               terrain,
                                                                               curr_motion.char.char_model, 
                                                                               settings.contact_eps)
            except ValueError as exc: # e.g. mesh feet/hands (G1): only box feet and sphere hands are supported
                print("Cannot label foot and hand contacts:", exc)
                contact_frames = None

            if contact_frames is not None:
                # generic names, resolved per character (G1: *_wrist_yaw_link)
                lh_id = curr_motion.char.char_model.get_body_id("left_hand")
                rh_id = curr_motion.char.char_model.get_body_id("right_hand")
                contact_frames[:, lh_id] = hand_contacts[:, lh_id]
                contact_frames[:, rh_id] = hand_contacts[:, rh_id]

                curr_motion.mlib = _motion_lib_from_dof_frames(curr_motion.mlib._motion_frames, contact_frames,
                                                               curr_motion.char.char_model, main_vars.device,
                                                               fps=curr_motion.mlib._motion_fps[0].item(),
                                                               loop_mode=motion_lib.LoopMode.CLAMP)
                main_vars.use_contact_info = True

        psim.TreePop()

    return