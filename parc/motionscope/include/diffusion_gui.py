import polyscope.imgui as psim

import parc.motionscope.include.global_header as g


########## DIFFUSION GUI ##########
def diffusion_gui():
    settings = g.MDMSettings()
    gen_settings = g.MDMSettings().gen_settings

    if g.g_mdm_model is None:
        psim.TextUnformatted("MDM model is not loaded")
        return
    
    opened = psim.BeginCombo("Selected MDM", settings.current_mdm_key)
    if opened:
        for key in settings.loaded_mdm_models:
            _, selected = psim.Selectable(key, key==settings.current_mdm_key)
            if selected:
                settings.select_mdm(key)
        psim.EndCombo()

    changed, gen_settings.use_ddim = psim.Checkbox("use ddim", gen_settings.use_ddim)
    if gen_settings.use_ddim:
        changed, gen_settings.ddim_stride = psim.InputInt("ddim stride", gen_settings.ddim_stride)

    if psim.TreeNode("Guidance Params"):
        changed, gen_settings.cfg_scale = psim.InputFloat("CFG scale", gen_settings.cfg_scale)
        changed, gen_settings.use_cfg = psim.Checkbox("Use CFG", gen_settings.use_cfg)
        psim.TreePop()


    changed, gen_settings.prev_state_ind_key = psim.Checkbox("Condition on prev state(s)", gen_settings.prev_state_ind_key)
    changed, gen_settings.target_condition_key = psim.Checkbox("Condition on target", gen_settings.target_condition_key)
    changed, settings.append_mdm_motion_to_prev_motion = psim.Checkbox("Append MDM motion to prev motion", settings.append_mdm_motion_to_prev_motion)
    changed, settings.mdm_batch_size = psim.InputInt("mdm batch size", settings.mdm_batch_size)
    settings.mdm_batch_size = max(settings.mdm_batch_size, 1)
    
    changed, settings.sample_prev_states_only = psim.Checkbox("sample prev states only", settings.sample_prev_states_only)
    changed, settings.hide_batch_motions = psim.Checkbox("hide motions after batch gen", settings.hide_batch_motions)
    
    return