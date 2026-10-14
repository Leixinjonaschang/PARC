import copy
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.motion_generator.diffusion_util as diffusion_util
import parc.motion_generator.gen_util as gen_util
import parc.motion_generator.mdm as mdm
import parc.util.geom_util as geom_util
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util
from parc.util.motion_util import MotionFrames


class MDMPathSettings:
    next_node_lookahead = 7
    rewind_num_frames = 5
    end_of_path_buffer = 2
    max_motion_length = 10.0
    path_batch_size = 16
    mdm_batch_size = 32
    top_k = 4
    w_target = 2.0
    w_contact = 0.1
    w_pen = 0.1

def compute_motion_loss(motion_frames: MotionFrames, 
                        path_nodes: torch.Tensor,
                        terrain: terrain_util.SubTerrain,
                        char_model: kin_char_model.KinCharModel,
                        body_points: list,
                        w_contact: float,
                        w_pen: float,
                        w_path: float,
                        verbose: bool = True):
    
    batch_size = motion_frames.root_pos.shape[0]
    total_losses = torch.zeros(size=[batch_size], dtype=torch.float32, device=motion_frames.root_pos.device)


    

    # TODO: compute losses with batch dimension
    #for i in range(batch_size):
    penetration_loss = 0.0

    contact_loss = 0.0

    jerk_loss = 0.0

    root_pos = motion_frames.root_pos#[i]
    root_rot = motion_frames.root_rot#[i]
    joint_rot = motion_frames.joint_rot#[i]
    contacts = motion_frames.contacts#[i]

    # TODO: compute a loss based on how well the motion follows the path nodes

    # gen_dxyz = root_pos[:, -1, 0:3] - root_pos[:, num_prev_states-1, 0:3]
    # #target_vector = torch.nn.functional.normalize(target_vector, eps=1e-2, dim=0)
    # target_dist = 0.3 # TODO: make this a parameter
    # #target_loss = torch.clamp(target_dist - torch.sum(gen_dxyz * target_vector, dim=-1), min=0.0) * w_target
    # target_rot = 30.0 * torch.pi / 180.0
    # position_loss = torch.clamp(target_dist - torch.norm(gen_dxyz, dim=-1), min = 0.0) / target_dist
    # rotation_loss = torch.clamp(target_rot - torch.abs(torch_util.quat_diff_angle(root_rot[:, -1, :], root_rot[:, num_prev_states-1, :])), min=0.0) / target_rot

    # not_moving_loss = (position_loss * rotation_loss) **2
    #target_loss = not_moving_loss

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    num_bodies = char_model.get_num_joints()

    min_z = torch.min(terrain.hf).item()

    # TODO: do this without the for loop? would it be better?
    for b in range(num_bodies):
        num_points = body_points[b].shape[0]
        curr_body_points = body_points[b].unsqueeze(0).unsqueeze(0) # shape: [1, 1, num_points(b), 3]
        curr_body_pos = body_pos[..., b, :].unsqueeze(-2) # shape: [batch_size, num_frames, 1, 3]
        curr_body_rot = body_rot[..., b, :].unsqueeze(-2) # shape: [batch_size, num_frames, 1, 4]
        

        curr_body_points = torch_util.quat_rotate(curr_body_rot, curr_body_points) + curr_body_pos

        negative_sdfs = terrain_util.points_hf_sdf(curr_body_points.view(batch_size, -1, 3), 
                                        terrain.hf.unsqueeze(0).expand(batch_size, -1, -1), 
                                        terrain.min_point.unsqueeze(0).expand(batch_size, -1), 
                                        terrain.dxdy, base_z = min_z-10.0, inverted=True)

        negative_sdfs = torch.clamp(negative_sdfs, max=0.0)

        #penetration_loss += torch.sum(torch.square(negative_sdfs))
        penetration_loss += torch.sum(-negative_sdfs, dim=-1) * w_pen

        positive_sdfs = terrain_util.points_hf_sdf(curr_body_points.view(batch_size, -1, 3), 
                                        terrain.hf.unsqueeze(0).expand(batch_size, -1, -1), 
                                        terrain.min_point.unsqueeze(0).expand(batch_size, -1), 
                                        terrain.dxdy, base_z = min_z-10.0, inverted=False).squeeze(0)

        positive_sdfs = torch.clamp(positive_sdfs, min=0.0)
        positive_sdfs = positive_sdfs.view(batch_size, -1, num_points)
        closest_distances, closest_point_ids = torch.min(positive_sdfs, dim=-1)
        
        contact_errs = closest_distances * contacts[..., b] # loss is closest distance of a point in contact
        #contact_loss += torch.sum(torch.square(contact_errs))
        contact_loss += torch.sum(contact_errs, dim=-1) * w_contact
    
    total_losses = contact_loss + penetration_loss# + not_moving_loss
    #total_losses[i] = total_loss
    # if verbose:
    #     print("idx:", i)
    #     print("penetration loss:", penetration_loss.item())
    #     print("contact loss:", contact_loss.item())
    #     print("target loss:", target_loss.item())
    #     print("total loss:", total_loss)

    losses = {
        "total_loss": total_losses,
        "contact_loss": contact_loss,
        "pen_loss": penetration_loss
    }

    return losses#total_losses, contact_loss, penetration_loss#, not_moving_loss

def gen_mdm_motion_at_path_start(path_nodes_xyz: torch.Tensor, 
                                 terrain: terrain_util.SubTerrain,
                                 char_model: kin_char_model.KinCharModel,
                                 mdm_model: mdm.MDM,
                                 mdm_settings: gen_util.MDMGenSettings,
                                 prev_frames: Optional[MotionFrames],
                                 batch_size: int,
                                 heading_mode = "auto") -> MotionFrames:
    start_xyz = path_nodes_xyz[0]
    next_xyz = path_nodes_xyz[1]
    next_xy = next_xyz[0:2]

    mdm_settings = copy.deepcopy(mdm_settings)

    start_xy = start_xyz[0:2]
    start_z = start_xyz[2].item()
    if mdm_model._relative_z_style == diffusion_util.RelativeZStyle.RELATIVE_TO_ROOT:
        new_rel_root_z = random.random() * (0.9 - 0.7) + 0.7
        start_z += new_rel_root_z


    if prev_frames is None:
        if heading_mode == "auto":
            next_diff = next_xy - start_xy
            heading = torch_util.heading_angle_from_xy(next_diff[0], next_diff[1])
        elif heading_mode == "random":
            heading = torch.rand(size=[1], dtype=torch.float32, device=start_xyz.device) * 2.0 * torch.pi
        else:
            assert False, "undefined heading mode"

        prev_frames = MotionFrames()
        prev_frames.init_blank_frames(char_model, mdm_model._num_prev_states, batch_size=batch_size)


        prev_frames.root_pos[..., -1, 0:2] = start_xy
        prev_frames.root_pos[..., -1, 2] = start_z
        prev_frames.root_rot[..., -1, :] = torch_util.heading_to_quat(heading)
        mdm_settings.use_prev_state = False
        mdm_settings.use_cfg = False
    else:
        mdm_settings.use_prev_state = True

    gen_motion_frames = gen_util.gen_mdm_motion(target_world_pos=next_xyz.unsqueeze(0), 
                                                prev_frames=prev_frames, 
                                                terrain=terrain,
                                                mdm_model=mdm_model,
                                                char_model=char_model,
                                                mdm_settings=mdm_settings)

    return gen_motion_frames

def gen_mdm_motion_at_xy(xy: torch.Tensor,
                         terrain: terrain_util.SubTerrain,
                         char_model: kin_char_model.KinCharModel,
                         mdm_model: mdm.MDM,
                         mdm_settings: gen_util.MDMGenSettings):
    
    # xy: [batch_size, 2]
    batch_size = xy.shape[0]
    device = char_model._device

    mdm_settings = copy.deepcopy(mdm_settings)

    start_z = terrain.get_hf_val_from_points(xy)

    new_rel_root_z = random.random() * (0.9 - 0.7) + 0.7
    start_z += new_rel_root_z


    initial_heading = torch.rand(size=[batch_size], dtype=torch.float32, device=device) * 2.0 * torch.pi

    prev_frames = MotionFrames()
    prev_frames.init_blank_frames(char_model, mdm_model._num_prev_states, batch_size=batch_size)


    prev_frames.root_pos[..., -1, 0:2] = xy
    prev_frames.root_pos[..., -1, 2] = start_z
    prev_frames.root_rot[..., -1, :] = torch_util.heading_to_quat(initial_heading)
    mdm_settings.use_prev_state = False
    mdm_settings.use_cfg = False

    target_heading = initial_heading + (torch.rand(size=[batch_size], dtype=torch.float32, device=device) - 0.5) * torch.pi
    vec = torch.zeros(size=[batch_size, 2], dtype=torch.float32, device=device)
    vec[..., 0] = 1.0
    target_pos = xy + torch_util.rotate_2d_vec(vec, target_heading)

    gen_motion_frames = gen_util.gen_mdm_motion(target_world_pos = target_pos, 
                                                prev_frames = prev_frames, 
                                                terrain = terrain,
                                                mdm_model = mdm_model,
                                                char_model = char_model,
                                                mdm_settings = mdm_settings)
    
    return gen_motion_frames
    
def get_closest_path_node_idx(root_pos, path_nodes_xyz):
    # root_pos: [batch_size, 3]
    # path_nodes_xyz: [num_nodes, 3]
    
    # find closest node in path
    closest_node_idx = 0
    xy_sq_dist = torch.sum(torch.square(root_pos[:, 0:2].unsqueeze(1) - path_nodes_xyz[:, 0:2].unsqueeze(0)), dim=-1)
    _, closest_node_idx = torch.min(xy_sq_dist, dim=-1)

    return closest_node_idx

def generate_frames_along_path(prev_frames: MotionFrames,
                               path_nodes_xyz: torch.Tensor,
                               terrain: terrain_util.SubTerrain,
                               char_model: kin_char_model.KinCharModel,
                               mdm_model: mdm.MDM,
                               mdm_settings: gen_util.MDMGenSettings,
                               path_settings: MDMPathSettings,
                               verbose: bool = True):
    #assert prev_frames.root_pos.shape[0] == 1
    root_pos = prev_frames.root_pos[:, -1]
    root_rot = prev_frames.root_rot[:, -1]
    joint_rot = prev_frames.joint_rot[:, -1]

    # find closest node in path
    #closest_node_idx = 0
    num_path_nodes = path_nodes_xyz.shape[0]

    # TODO: compute this outside of function to save compute?
    #node_pos = terrain.get_point(path_nodes)
    final_node_pos = path_nodes_xyz[-1]#terrain.get_xyz_point(path_nodes[-1])

    #xy_sq_dist = torch.sum(torch.square(root_pos[:, 0:2].unsqueeze(1) - path_nodes_xyz[:, 0:2].unsqueeze(0)), dim=-1)
    # shape: [batch_size, num_nodes]

    closest_node_idx = get_closest_path_node_idx(root_pos, path_nodes_xyz)

    # heuristic: average foot position is close enough to the final node position in 3D space
    done_radius = 0.5
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")
    avg_foot_pos = (body_pos[..., lf_id, :] + body_pos[..., rf_id, :]) / 2.0
    done_dist = torch.norm(avg_foot_pos - final_node_pos, dim=-1)
    done = done_dist < done_radius

    next_node_idx = torch.clamp(closest_node_idx + path_settings.next_node_lookahead, max=num_path_nodes-1)
    next_node_pos = path_nodes_xyz[next_node_idx]

    target_world_pos = next_node_pos#torch.where(done.unsqueeze(-1), root_pos, next_node_pos)

    #prev_frames = prev_frames.expand_first_dim(batch_size)

    gen_motion_frames = gen_util.gen_mdm_motion(target_world_pos=target_world_pos, 
                                                prev_frames=prev_frames,
                                                terrain=terrain,
                                                char_model=char_model, 
                                                mdm_model=mdm_model,
                                                mdm_settings=mdm_settings)
    
    return gen_motion_frames, done
    
def generate_frames_until_end_of_path(path_nodes: torch.Tensor,
                                      terrain: terrain_util.SubTerrain,
                                      char_model: kin_char_model.KinCharModel,
                                      mdm_model: mdm.MDM,
                                      prev_frames: Optional[MotionFrames],
                                      mdm_path_settings: MDMPathSettings,
                                      mdm_gen_settings: gen_util.MDMGenSettings,
                                      add_noise_to_loss=True,
                                      verbose: bool = True,
                                      slice_terrain = True,
                                      first_heading_mode = "auto") -> Tuple[List[motion_util.MotionFrames], List[terrain_util.SubTerrain], dict]:

    mdm_device = torch.device(mdm_model._device)
    batch_size = mdm_path_settings.mdm_batch_size

    path_nodes = path_nodes.to(device=mdm_device)
    terrain = terrain.torch_copy()
    terrain.set_device(mdm_device)
    char_model = char_model if char_model._device == mdm_device else char_model.get_copy(mdm_device)

    if prev_frames is not None:
        prev_frames = prev_frames.get_copy(mdm_device)

        if len(prev_frames.root_pos.shape) == 2:
            prev_frames = prev_frames.unsqueeze(0).expand_first_dim(batch_size)

    ## Step 1: go to start node, move around optionally with noise, then sample heightmap there
    gen_motion_frames = gen_mdm_motion_at_path_start(path_nodes_xyz = path_nodes,
                                                     terrain = terrain,
                                                     char_model = char_model,
                                                     mdm_model = mdm_model,
                                                     mdm_settings = mdm_gen_settings,
                                                     prev_frames = prev_frames,
                                                     batch_size = batch_size,
                                                     heading_mode = first_heading_mode)
    
    body_points = geom_util.get_char_point_samples(char_model)

    start_frame = mdm_model._seq_len - 1 - mdm_path_settings.rewind_num_frames
    full_motion_frames = [gen_motion_frames.get_slice(slice(0, start_frame))]
    total_num_frames = start_frame


    fps = mdm_model._sequence_fps
    dt = 1.0 / fps
    num_prev_states = mdm_model._num_prev_states
    w_pen = mdm_path_settings.w_pen
    w_contact = mdm_path_settings.w_contact
    w_path = 1.0

    final_frame = torch.zeros(size=[batch_size], dtype=torch.int64, device=mdm_device)
    final_frame[:] = -1

    final_frame_found = torch.zeros(size=[batch_size], dtype=torch.bool, device=mdm_device)
    while True:
        if verbose:
            print("Generating at time: ", total_num_frames * dt)

        prev_frames = gen_motion_frames.get_slice(slice(start_frame-num_prev_states, start_frame))

        # TODO: a separate function that repeats this function call autoregressively
        gen_motion_frames, done = generate_frames_along_path(prev_frames=prev_frames,
                                                             path_nodes_xyz=path_nodes,
                                                             terrain=terrain,
                                                             char_model=char_model,
                                                             mdm_model=mdm_model,
                                                             mdm_settings=mdm_gen_settings,
                                                             path_settings=mdm_path_settings,
                                                             verbose=verbose)

        total_num_frames += start_frame - num_prev_states

        if torch.all(done):
            full_motion_frames.append(gen_motion_frames.get_slice(slice(num_prev_states, mdm_model._seq_len)))#[:, num_prev_states:])
            if verbose:
                print("reached end of path for all motions")
            break

        # for each motion, check if the done flag is true. If yes, then we note that is where the motion ends
        # if final frame not found AND done, then we input the final frame
        new_done_indices = torch.nonzero(torch.logical_and(~final_frame_found, done))
        final_frame[new_done_indices] = total_num_frames

        final_frame_found = torch.logical_or(final_frame_found, done)

        if total_num_frames * dt > mdm_path_settings.max_motion_length:
            full_motion_frames.append(gen_motion_frames.get_slice(slice(num_prev_states, mdm_model._seq_len)))
            if verbose:
                print("reached motion time limit")
            break

        full_motion_frames.append(gen_motion_frames.get_slice(slice(num_prev_states, start_frame)))

    full_motion_frames = motion_util.cat_motion_frames(full_motion_frames)
    if final_frame_found.any():
        min_num_frames = torch.min(final_frame[final_frame_found]).item()
    else:
        min_num_frames = full_motion_frames.root_pos.shape[1]
    print("min num frames:", min_num_frames)
    
    sliced_motion_frames = []
    sliced_terrains = []
    all_losses = []
    contact_losses = []
    pen_losses = []
    not_finished_penalty = 100.0
    for i in range(batch_size):
        curr_motion_frames = full_motion_frames.get_idx(i).unsqueeze(0).get_slice(slice(0, final_frame[i]))
        if slice_terrain: # TODO: figure out why this is wrong sometimes
            curr_terrain = terrain_util.slice_terrain_around_motion(curr_motion_frames.root_pos, terrain, padding = 1.2, localize=False)
        else:
            curr_terrain = terrain.torch_copy()
        losses = compute_motion_loss(curr_motion_frames, path_nodes, curr_terrain, char_model, body_points, w_contact, w_pen, w_path)
        total_loss = losses["total_loss"]
        # Penalty for not reaching end node:
        if not final_frame_found[i]:
            total_loss = total_loss + not_finished_penalty
        print("motion", i, "loss:", total_loss)

        sliced_motion_frames.append(curr_motion_frames)
        sliced_terrains.append(curr_terrain)
        all_losses.append(total_loss)
        contact_losses.append(losses["contact_loss"])
        pen_losses.append(losses["pen_loss"])

    all_losses = torch.cat(all_losses)
    contact_losses = torch.cat(contact_losses)
    pen_losses = torch.cat(pen_losses)

    #true_all_losses = all_losses.clone()
    if add_noise_to_loss:
        all_losses = all_losses + torch.randn_like(all_losses)
    #best_losses, best_ids = torch.topk(all_losses, k=mdm_path_settings.top_k, largest=False)

    sorted_losses, sorted_ids = torch.sort(all_losses)

    sorted_motion_frames = []
    sorted_motion_terrains = []
    for id in sorted_ids:
        sorted_motion_frames.append(sliced_motion_frames[id])#full_motion_frames.get_idx(best_ids)
        sorted_motion_terrains.append(sliced_terrains[id])

    info = dict()
    info["losses"] = sorted_losses.squeeze()

    info["contact_losses"] = contact_losses[sorted_ids.squeeze()]
    info["pen_losses"] = pen_losses[sorted_ids.squeeze()]


    return sorted_motion_frames, sorted_motion_terrains, info