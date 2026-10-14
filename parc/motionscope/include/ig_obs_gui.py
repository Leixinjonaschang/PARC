import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

import parc.motionscope.include.global_header as g
import parc.motionscope.polyscope_util as ps_util
import parc.util.torch_util as torch_util


def animate_obs(motion_time, overlay_obs_on_motion=False, ignore_tar_obs_idxs=[]):

    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()
    char_model = curr_motion.char.char_model
    ig_obs = g.IGObsSettings()

    mlib = curr_motion.mlib


    fps = mlib._motion_fps[0].item()
    num_frames = mlib._motion_num_frames[0].item()

    float_frame_idx = motion_time * fps
    floor_frame_idx = int(np.floor(float_frame_idx))
    ceil_frame_idx = int(np.ceil(float_frame_idx))
    round_frame_idx = int(np.round(float_frame_idx))

    ceil_frame_idx = min(ceil_frame_idx, num_frames - 1)
    round_frame_idx = min(round_frame_idx, num_frames - 1)

    if overlay_obs_on_motion:
        global_root_pos = mlib._frame_root_pos[round_frame_idx]
        global_root_rot = mlib._frame_root_rot[round_frame_idx]
    else:
        global_root_pos = torch.zeros([3], dtype=torch.float32, device=g.MainVars().device)
        global_root_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=g.MainVars().device)
        
    
    global_heading_quat = torch_util.calc_heading_quat(global_root_rot)
    hf_obs_points = torch_util.quat_rotate(global_heading_quat.unsqueeze(0), ig_obs.hf_obs_points[round_frame_idx]) + global_root_pos

    psim.TextUnformatted("floor frame idx: " + str(floor_frame_idx))
    psim.TextUnformatted("ceil frame idx: " + str(ceil_frame_idx))
    psim.TextUnformatted("round frame idx: " + str(round_frame_idx))
    ps.get_point_cloud("hf_obs_points").update_point_positions(hf_obs_points)

    root_pos = global_root_pos
    root_rot = global_root_rot
    joint_rot = ig_obs.proprio_char_obs["joint_rot"][round_frame_idx]
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    ps_util.update_char_motion_mesh(body_pos, body_rot, ig_obs.ps_proprio_char_mesh, char_model)

    contacts = ig_obs.char_contacts
    tar_contacts = ig_obs.tar_contacts

    # set color based on contact info
    red = np.array([1.0, 0.0, 0.0])

    for body_id in range(0, char_model.get_num_joints()):
        body_contact = contacts[round_frame_idx, body_id].item()
        ig_obs.ps_proprio_char_mesh[body_id].set_color(red * body_contact + ig_obs.proprio_char_color * (1.0 - body_contact))
    
    # proprio key pos obs are canoicalized differently than target key pos obs
    key_pos_obs = torch_util.quat_rotate(global_heading_quat.unsqueeze(0), ig_obs.proprio_key_pos_obs[round_frame_idx]) + global_root_pos
    ig_obs.ps_proprio_key_pos_pc.update_point_positions(key_pos_obs.numpy())


    for idx in range(len(ignore_tar_obs_idxs)):
        for ps_mesh in ig_obs.ps_tar_char_meshes[idx]:
            ps_mesh.set_enabled(False)

    for i in range(ig_obs.num_tar_obs):
        if i in ignore_tar_obs_idxs:
            continue
        root_pos = torch_util.quat_rotate(global_heading_quat, ig_obs.root_pos_tar_obs[round_frame_idx, i]) + global_root_pos
        root_rot = torch_util.quat_multiply(global_heading_quat, ig_obs.root_rot_tar_obs[round_frame_idx, i])
        joint_rot = ig_obs.joint_rot_tar_obs[round_frame_idx, i]
        body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
        ps_util.update_char_motion_mesh(body_pos, body_rot, ig_obs.ps_tar_char_meshes[i], char_model)

        for body_id in range(0, char_model.get_num_joints()):
            body_contact = tar_contacts[round_frame_idx, i, body_id].item()
            ig_obs.ps_tar_char_meshes[i][body_id].set_color(red * body_contact + ig_obs.tar_char_color * (1.0 - body_contact))


    key_pos_tar_obs = torch_util.quat_rotate(global_heading_quat.unsqueeze(0), ig_obs.key_pos_tar_obs[round_frame_idx]) + global_root_pos
    ig_obs.ps_tar_key_pos_pc.update_point_positions(key_pos_tar_obs.numpy())

    return

def ig_obs_gui():

    ig_obs_settings = g.IGObsSettings()
    if not ig_obs_settings.has_obs:
        return
    
    
    for key, value in ig_obs_settings.obs_shapes.items():
        psim.TextUnformatted(key)

    changed, ig_obs_settings.overlay_obs_on_motion = psim.Checkbox("Overlay obs on motion", ig_obs_settings.overlay_obs_on_motion)
    changed, ig_obs_settings.view_tar_obs = psim.Checkbox("View tar obs", ig_obs_settings.view_tar_obs)
    if changed:
        ig_obs_settings.SetViewTarObs(ig_obs_settings.view_tar_obs)
    changed, ig_obs_settings.record_obs = psim.Checkbox("Record obs", ig_obs_settings.record_obs)

    changed, ig_obs_settings.view_key_points = psim.Checkbox("View key points", ig_obs_settings.view_key_points)
    if changed:
        ig_obs_settings.ps_tar_key_pos_pc.set_enabled(False)
        ig_obs_settings.ps_proprio_key_pos_pc.set_enabled(False)

    animate_obs(motion_time=g.MainVars().motion_time, overlay_obs_on_motion=ig_obs_settings.overlay_obs_on_motion,
                ignore_tar_obs_idxs=[0,1,2])#global_root_pos=global_root_pos, global_root_rot=global_root_rot)
    return