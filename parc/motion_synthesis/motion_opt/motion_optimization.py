import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import List
import enum

import parc.anim.kin_char_model as kin_char_model
import parc.util.torch_util as torch_util
import parc.util.terrain_util as terrain_util
import parc.util.geom_util as geom_util
import parc.util.wandb_logger as wandb_logger
import parc.util.motion_util as motion_util

class LossType(enum.Enum):
    ROOT_POS_LOSS = 0
    ROOT_ROT_LOSS = 1
    JOINT_ROT_LOSS = 2
    SMOOTHNESS_LOSS = 3
    PENETRATION_LOSS = 4
    CONTACT_LOSS = 5
    SLIDING_LOSS = 6
    BODY_CONSTRAINT_LOSS = 7
    JERK_LOSS = 8
    LOOPING_LOSS = 9

@dataclass
class BodyConstraint:
    start_frame_idx = 0
    end_frame_idx = 0
    constraint_point = None # 3D torch vector


def _align_contacts_to_bodies(
    contacts: torch.Tensor,
    char_model: kin_char_model.KinCharModel,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Expand per-contact-body contacts to per-body contacts."""

    if char_model.get_num_bodies() == char_model.get_num_contact_bodies():
        if contacts.shape[-1] != char_model.get_num_bodies():
            assert False, (
                "Contact tensor has incompatible shape: "
                f"expected last dim {char_model.get_num_bodies()}, got {contacts.shape[-1]}"
            )
        return contacts

    if contacts.shape[-1] != char_model.get_num_contact_bodies():
        assert False, (
            "Contact tensor has incompatible shape: "
            f"expected last dim {char_model.get_num_contact_bodies()}, got {contacts.shape[-1]}"
        )

    aligned_contacts = torch.zeros(
        size=[contacts.shape[0], char_model.get_num_bodies()],
        dtype=contacts.dtype,
        device=device,
    )

    for body_idx in range(char_model.get_num_bodies()):
        body_name = char_model.get_body_name(body_idx)
        if char_model.is_contact_body(body_name):
            contact_body_id = char_model.get_contact_body_id(body_name)
            aligned_contacts[:, body_idx] = contacts[:, contact_body_id]

    return aligned_contacts

def compute_approx_body_constraints(root_pos, root_rot, joint_rot, contacts, 
                                    char_model: kin_char_model.KinCharModel,
                                    terrain: terrain_util.SubTerrain):

    def extract_consecutive_trues(bool_tensor):
        # Ensure the tensor is 1D
        bool_tensor = bool_tensor.flatten()

        # Find the indices of `True` values
        indices = torch.nonzero(bool_tensor, as_tuple=True)[0]

        # If no `True` values are found, return an empty list
        if len(indices) == 0:
            return []

        # Find the breaks in consecutive `True` values
        diff = indices[1:] - indices[:-1]
        breaks = torch.nonzero(diff > 1, as_tuple=True)[0] + 1

        breaks = [0] + breaks.tolist()
        split_indices = []
        # Split indices into groups of consecutive values
        for i in range(len(breaks)-1):
            split_indices.append(indices[breaks[i]:breaks[i+1]].clone())
        if breaks[-1] < indices.shape[0] - 1:
            split_indices.append(indices[breaks[-1]:indices.shape[0]].clone())
        #split_indices = torch.split(indices, breaks.tolist())

        # # Create tensors for each group
        # tensors = [bool_tensor.clone() for _ in split_indices]
        # for t, group in zip(tensors, split_indices):
        #     t[:] = False  # Clear all entries
        #     t[group] = True  # Set only the group's indices to True

        return split_indices
    
    def estimate_approx_contact_points(contacts, body_pos, contact_threshold=0.9):
        contacts = contacts > contact_threshold
        contact_frame_ids = extract_consecutive_trues(contacts)

        contact_points = []
        for group in contact_frame_ids:
            #print(lf_pos[group])
            #if group.shape[0] <= 1: # don't make a contact constraint for potential outlier contacts
            #    continue
            avg_pos = torch.mean(body_pos[group], axis=0)
            #print(avg_pos)
            contact_points.append(avg_pos)
        if len(contact_points) > 0:
            contact_points = torch.stack(contact_points)
        else:
            contact_points = torch.tensor([], device=contacts.device)
        return contact_points, contact_frame_ids
    
    def optimize_contact_points(points):

        for i in range(points.shape[0]):
            base_z = terrain.hf.min().item() - 10.0
            point = points[i].clone()

            point.requires_grad = True
            optimizer = torch.optim.SGD([point], lr=0.01)


            for iter in range(1000):
                optimizer.zero_grad()
                sd = terrain_util.points_hf_sdf(point.unsqueeze(0).unsqueeze(0), terrain.hf.unsqueeze(0), 
                                                terrain.min_point.unsqueeze(0), terrain.dxdy, base_z=base_z,
                                                inverted=False)
                #print(sd.item())
                loss = torch.sum(torch.square(sd))
                loss.backward()

                optimizer.step()

            points[i] = point.detach().clone()
        return
    
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

    lf_id = char_model.get_body_id("left_foot")
    rf_id = char_model.get_body_id("right_foot")
    lh_id = char_model.get_body_id("left_hand")
    rh_id = char_model.get_body_id("right_hand")

    #ps.register_point_cloud("right hand positions", rh_pos, radius = 0.005)

    lf_pos = body_pos[:, lf_id]#.cpu().numpy()
    rf_pos = body_pos[:, rf_id]#.cpu().numpy()
    lf_rot = body_rot[:, lf_id]
    rf_rot = body_rot[:, rf_id]
    # get the center of the box
    lf_geom = char_model.get_geoms(lf_id)[0]
    lf_box_offset = lf_geom._offset
    lf_pos = lf_pos + torch_util.quat_rotate(lf_rot, lf_box_offset.unsqueeze(0))

    rf_geom = char_model.get_geoms(rf_id)[0]
    rf_box_offset = rf_geom._offset
    rf_pos = rf_pos + torch_util.quat_rotate(rf_rot, rf_box_offset.unsqueeze(0))

    lh_pos = body_pos[:, lh_id]
    rh_pos = body_pos[:, rh_id]

    lf_contacts = contacts[:, lf_id]
    rf_contacts = contacts[:, rf_id]
    lh_contacts = contacts[:, lh_id]
    rh_contacts = contacts[:, rh_id]

    lf_points, lf_contact_frame_ids = estimate_approx_contact_points(lf_contacts, lf_pos)
    rf_points, rf_contact_frame_ids = estimate_approx_contact_points(rf_contacts, rf_pos)
    lh_points, lh_contact_frame_ids = estimate_approx_contact_points(lh_contacts, lh_pos)
    rh_points, rh_contact_frame_ids = estimate_approx_contact_points(rh_contacts, rh_pos)

    optimize_contact_points(lf_points)
    optimize_contact_points(rf_points)
    optimize_contact_points(lh_points)
    optimize_contact_points(rh_points)

    def create_opt_constraints(points, contact_frame_ids):
        constraint_list = []
        for c_idx in range(points.shape[0]):
            constraint = BodyConstraint()
            constraint.start_frame_idx = contact_frame_ids[c_idx][0].item()
            constraint.end_frame_idx = contact_frame_ids[c_idx][-1].item()
            constraint.constraint_point = points[c_idx].clone()
            constraint_list.append(constraint)
        return constraint_list
    
    body_constraints = [[] for _ in range(char_model.get_num_joints())]
    
    body_constraints[lf_id] = create_opt_constraints(lf_points, lf_contact_frame_ids)
    body_constraints[rf_id] = create_opt_constraints(rf_points, rf_contact_frame_ids)
    body_constraints[lh_id] = create_opt_constraints(lh_points, lh_contact_frame_ids)
    body_constraints[rh_id] = create_opt_constraints(rh_points, rh_contact_frame_ids)

    return body_constraints

def motion_terrain_contact_loss(tgt_root_pos, tgt_root_rot, tgt_joint_dof,
                                src_root_pos, src_root_rot_quat, src_joint_rot, src_body_vels, src_body_rot_vels,
                                contacts, terrain: terrain_util.SubTerrain,
                                body_points: List[torch.Tensor],
                                char_model: kin_char_model.KinCharModel,
                                w_root_pos: float,
                                w_root_rot: float,
                                w_joint_rot: float,
                                w_smoothness: float,
                                w_penetration: float,
                                w_contact: float,
                                w_sliding: float,
                                w_body_constraints: float,
                                w_jerk: float,
                                body_constraints: list,
                                max_jerk: float):
    
    root_pos_err = tgt_root_pos - src_root_pos
    root_pos_loss = torch.sum(torch.square(root_pos_err))

    tgt_root_rot_quat = torch_util.exp_map_to_quat(tgt_root_rot)
    root_rot_err = torch_util.quat_diff_angle(tgt_root_rot_quat, src_root_rot_quat)
    root_rot_loss = torch.sum(torch.square(root_rot_err))


    tgt_joint_rot = char_model.dof_to_rot(tgt_joint_dof)
    joint_rot_err = torch_util.quat_diff_angle(tgt_joint_rot, src_joint_rot)
    joint_rot_loss = torch.sum(torch.square(joint_rot_err))

    
    tgt_body_pos, tgt_body_rot = char_model.forward_kinematics(tgt_root_pos, tgt_root_rot_quat, tgt_joint_rot)

    tgt_body_vels = tgt_body_pos[1:] - tgt_body_pos[:-1]
    body_vel_err = tgt_body_vels - src_body_vels
    body_vel_err_sq = torch.square(body_vel_err)

    tgt_body_rot_vels = torch_util.quat_diff_angle(tgt_body_rot[1:], tgt_body_rot[:-1])
    body_rot_vel_err = tgt_body_rot_vels - src_body_rot_vels
    body_rot_vel_err_sq = torch.square(body_rot_vel_err)


    smoothness_loss = torch.sum(body_vel_err_sq) + torch.sum(body_rot_vel_err_sq)

    #lf_id = char_model.get_body_id("left_hand")
    #tgt_lf_h = 1.0
    #custom_loss = torch.sum(torch.square((tgt_body_pos[164:174, lf_id, 2] - tgt_lf_h)))

    # first, do a general loss on body hf penetration
    # second, do a loss for contact bodies: at least one point on a contact body has to be touching the hf
    frame_change_in_contact = torch.min(torch.cat([contacts[1:].unsqueeze(-1), contacts[:-1].unsqueeze(-1)], dim=-1), dim=-1)[0]
    # get rid of negative values that might be there due to MDM issues
    frame_change_in_contact = torch.clamp(frame_change_in_contact, min=0.0)
    penetration_loss = 0.0
    contact_loss = 0.0
    sliding_loss = 0.0
    body_constraint_loss = 0.0
    num_frames = tgt_root_pos.shape[0]
    num_bodies = char_model.get_num_joints()
    for b in range(num_bodies):
        num_points = body_points[b].shape[0]
        curr_body_points = body_points[b].unsqueeze(0) # shape: [1, num_points(b), 3]
        curr_body_rot = tgt_body_rot[:, b].unsqueeze(1) # shape: [num_frames, 1, 4]
        curr_body_pos = tgt_body_pos[:, b].unsqueeze(1) # shape: [num_frames, 1, 3]

        curr_body_points = torch_util.quat_rotate(curr_body_rot, curr_body_points) + curr_body_pos

        negative_sdfs = terrain_util.points_hf_sdf(curr_body_points.view(-1, 3).unsqueeze(0), 
                                        terrain.hf.unsqueeze(0), 
                                        terrain.min_point.unsqueeze(0), 
                                        terrain.dxdy, base_z = -10.0, inverted=True).squeeze(0)

        negative_sdfs = torch.clamp(negative_sdfs, max=0.0)

        #penetration_loss += torch.sum(torch.square(negative_sdfs))
        penetration_loss += torch.sum(-negative_sdfs)


        if w_contact != 0.0:
            positive_sdfs = terrain_util.points_hf_sdf(curr_body_points.view(-1, 3).unsqueeze(0), 
                                            terrain.hf.unsqueeze(0), 
                                            terrain.min_point.unsqueeze(0), 
                                            terrain.dxdy, base_z = -10.0, inverted=False).squeeze(0)

            positive_sdfs = torch.clamp(positive_sdfs, min=0.0)
            positive_sdfs = positive_sdfs.view(num_frames, num_points)
            closest_distances, closest_point_ids = torch.min(positive_sdfs, dim=-1)
            

            body_name = char_model.get_body_name(b)
            if body_name in char_model._contact_body_names:
                contact_body_id = char_model.get_contact_body_id(body_name)
                contact_errs = closest_distances * contacts[:, contact_body_id] # loss is closest distance of a point in contact
                #contact_loss += torch.sum(torch.square(contact_errs))
                contact_loss += torch.sum(contact_errs)

        # gather from (N, M, 3) tensor with (N) tensor whose values or in [0, M-1]
        # closest_points = torch.gather(curr_body_points, dim=1, index=closest_point_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)).squeeze(1)
        # closest_points_speed_sq = torch.sum(torch.square(closest_points[1:] - closest_points[:-1]), dim=-1)

        # sliding_loss += torch.sum(closest_points_speed_sq * frame_change_in_contact[:, b])


        # TODO: sliding loss
        # If closest point id between two frames is the same, then the closest point speed should be 0



        if body_constraints is not None:
            curr_body_constraint_list = body_constraints[b] # list of constraints

            for curr_body_constraint in curr_body_constraint_list:
                start_frame_idx = curr_body_constraint.start_frame_idx
                end_frame_idx = curr_body_constraint.end_frame_idx
                constraint_point = curr_body_constraint.constraint_point

                # NOTE: assuming all bodies are spheres or boxes for now
                if char_model.get_geoms(b)[0]._shape_type == kin_char_model.GeomType.SPHERE:
                    radius = char_model.get_geoms(b)[0]._dims
                    offset = char_model.get_geoms(b)[0]._offset

                    # everything broadcastable to shape: [num_frames, 4 or 3]
                    body_center = torch_util.quat_rotate(curr_body_rot.squeeze(1), offset.unsqueeze(0)) + curr_body_pos.squeeze(1)
                    # final shape: [num_frames, 3]

                    body_constraint_diff = geom_util.sdSphere(constraint_point.unsqueeze(0), body_center[start_frame_idx:end_frame_idx+1], radius)
                    body_constraint_loss += torch.sum(torch.abs(body_constraint_diff))

                elif char_model.get_geoms(b)[0]._shape_type == kin_char_model.GeomType.BOX:
                    radius = torch.norm(char_model.get_geoms(b)[0]._dims) * 1.25
                    #box_offset = char_model.get_geoms(b)[0]._offset

                    # # everything broadcastable to shape: [num_frames, 4 or 3]
                    # body_center = torch_util.quat_rotate(curr_body_rot.squeeze(1), offset.unsqueeze(0)) + curr_body_pos.squeeze(1)
                    # # final shape: [num_frames, 3]

                    # box_points = geom_util.get_box_points_batch(curr_body_pos, curr_body_rot, box_dims, box_offset)
                    # TODO:
                    # - get box points batch
                    #curr_body_points
                    
                    # NOTE: hard coded the foot sole points indices
                    foot_sole_points = curr_body_points[start_frame_idx:end_frame_idx+1, 0:18].reshape(-1, 3)
                    body_constraint_diff = geom_util.sdSphere(constraint_point.unsqueeze(0), foot_sole_points, radius)
                    body_constraint_loss += torch.sum(torch.clamp(body_constraint_diff, min=0.0))
                else:
                    continue

                

                # Also, the body constraint should null the sliding loss, since these will be two competing things
                # and the body constraint should take priority
                body_vel_err_sq = body_vel_err_sq.clone()
                body_vel_err_sq[start_frame_idx:end_frame_idx+1, b] *= 0.0
                body_rot_vel_err_sq = body_rot_vel_err_sq.clone()
                body_rot_vel_err_sq[start_frame_idx:end_frame_idx+1, b] *= 0.0


            # the constraint will be the part of the surface the body is supposed to be in contact with
            # We will use a point to body shape SDF (e.g. point to sphere for sphere body parts)
            


    # TODO: 
    # min(contacts==True with positive_sdfs, dim=points)
    # so bodies in contact will be penalized for having distance between the closest points
    # and the hf

    # loss to prevent bodies in contact from sliding, relative to original motion
    # pseudo-huber losses
    if w_sliding != 0.0 and frame_change_in_contact.shape[-1] == num_bodies:
        c = 0.03
        c2 = 0.0009
        sliding_loss = torch.sum((torch.sqrt(torch.sum(body_vel_err_sq, dim=-1) + c2) - c) * frame_change_in_contact)\
                        + torch.sum((torch.sqrt(body_rot_vel_err_sq + c2) - c) * frame_change_in_contact)
    else:
        sliding_loss = 0.0
    
    # jerk loss
    tgt_body_acc = tgt_body_vels[1:] - tgt_body_vels[:-1]
    tgt_body_jerk = tgt_body_acc[1:] - tgt_body_acc[:-1]
    tgt_body_jerk_mag = torch.norm(tgt_body_jerk, dim=-1)
    dt = 1.0 / 30.0 # TODO don't hardcode
    max_jerk = max_jerk * (dt ** 3)
    jerk_loss = torch.sum(torch.clamp(tgt_body_jerk_mag - max_jerk, min=0.0))

    losses = dict()
    losses[LossType.ROOT_POS_LOSS] = root_pos_loss.item()
    losses[LossType.ROOT_ROT_LOSS] = root_rot_loss.item()
    losses[LossType.JOINT_ROT_LOSS] = joint_rot_loss.item()
    losses[LossType.SMOOTHNESS_LOSS] = smoothness_loss.item()
    losses[LossType.PENETRATION_LOSS] = penetration_loss.item()
    if isinstance(contact_loss, torch.Tensor):
        losses[LossType.CONTACT_LOSS] = contact_loss.item()
    else:
        losses[LossType.CONTACT_LOSS] = contact_loss
    if isinstance(sliding_loss, torch.Tensor):
        losses[LossType.SLIDING_LOSS] = sliding_loss.item()
    else:
        losses[LossType.SLIDING_LOSS] = sliding_loss
    losses[LossType.JERK_LOSS] = jerk_loss.item()
    if isinstance(body_constraint_loss, torch.Tensor):
        losses[LossType.BODY_CONSTRAINT_LOSS] = body_constraint_loss.item()
    else:
        losses[LossType.BODY_CONSTRAINT_LOSS] = body_constraint_loss

    loss = w_root_pos * root_pos_loss \
            + w_root_rot * root_rot_loss \
            + w_joint_rot * joint_rot_loss \
            + w_smoothness * smoothness_loss \
            + w_penetration * penetration_loss \
            + w_contact * contact_loss \
            + w_sliding * sliding_loss \
            + w_body_constraints * body_constraint_loss \
            + w_jerk * jerk_loss

    #loss += custom_loss * 1000.0
    return loss, losses


def motion_terrain_contact_loss_localized(tgt_root_pos, tgt_root_rot, tgt_joint_dof,
                                          src_root_pos, src_root_rot_quat, src_joint_rot, src_body_vels, src_body_rot_vels,
                                          contacts, terrain: terrain_util.SubTerrain,
                                          body_points: List[torch.Tensor],
                                          char_model: kin_char_model.KinCharModel,
                                          w_root_pos: float,
                                          w_root_rot: float,
                                          w_joint_rot: float,
                                          w_smoothness: float,
                                          w_penetration: float,
                                          w_contact: float,
                                          w_sliding: float,
                                          w_body_constraints: float,
                                          w_jerk: float,
                                          body_constraints: list,
                                          max_jerk: float):
    """Memory-optimized variant of :func:`motion_terrain_contact_loss`.

    The original implementation evaluates SDF queries against the full
    heightfield for every frame, which can become prohibitively memory
    expensive for large terrains.  This variant slices a per-frame
    heightfield patch centred around the character, reuses that patch
    across all SDF evaluations, and keeps the overall computation fully
    vectorised.
    """

    root_pos_err = tgt_root_pos - src_root_pos
    root_pos_loss = torch.sum(torch.square(root_pos_err))

    tgt_root_rot_quat = torch_util.exp_map_to_quat(tgt_root_rot)
    root_rot_err = torch_util.quat_diff_angle(tgt_root_rot_quat, src_root_rot_quat)
    root_rot_loss = torch.sum(torch.square(root_rot_err))

    tgt_joint_rot = char_model.dof_to_rot(tgt_joint_dof)
    joint_rot_err = torch_util.quat_diff_angle(tgt_joint_rot, src_joint_rot)
    joint_rot_loss = torch.sum(torch.square(joint_rot_err))

    tgt_body_pos, tgt_body_rot = char_model.forward_kinematics(tgt_root_pos, tgt_root_rot_quat, tgt_joint_rot)

    tgt_body_vels = tgt_body_pos[1:] - tgt_body_pos[:-1]
    body_vel_err = tgt_body_vels - src_body_vels
    body_vel_err_sq = torch.square(body_vel_err)

    tgt_body_rot_vels = torch_util.quat_diff_angle(tgt_body_rot[1:], tgt_body_rot[:-1])
    body_rot_vel_err = tgt_body_rot_vels - src_body_rot_vels
    body_rot_vel_err_sq = torch.square(body_rot_vel_err)

    smoothness_loss = torch.sum(body_vel_err_sq) + torch.sum(body_rot_vel_err_sq)

    frame_change_in_contact = torch.min(torch.cat([contacts[1:].unsqueeze(-1), contacts[:-1].unsqueeze(-1)], dim=-1), dim=-1)[0]
    frame_change_in_contact = torch.clamp(frame_change_in_contact, min=0.0)

    penetration_loss = 0.0
    contact_loss = 0.0
    sliding_loss = 0.0
    body_constraint_loss = 0.0

    num_frames = tgt_root_pos.shape[0]
    num_bodies = char_model.get_num_joints()

    body_world_points = []
    body_rot_list = []
    body_pos_list = []
    body_point_counts = []

    for b in range(num_bodies):
        local_body_points = body_points[b].unsqueeze(0)
        curr_body_rot = tgt_body_rot[:, b].unsqueeze(1)
        curr_body_pos = tgt_body_pos[:, b].unsqueeze(1)

        world_points = torch_util.quat_rotate(curr_body_rot, local_body_points) + curr_body_pos

        body_world_points.append(world_points)
        body_rot_list.append(curr_body_rot)
        body_pos_list.append(curr_body_pos)
        body_point_counts.append(world_points.shape[1])

    all_body_points = torch.cat(body_world_points, dim=1)

    terrain_dxdy = terrain.dxdy
    terrain_min_point = terrain.min_point
    terrain_dims = terrain.dims
    device = terrain.hf.device

    padding = terrain_dxdy * 2.0
    padding = padding.unsqueeze(0)

    body_xy = all_body_points[..., :2]
    min_xy = body_xy.min(dim=1)[0] - padding
    max_xy = body_xy.max(dim=1)[0] + padding

    inv_dxdy = 1.0 / terrain_dxdy
    inv_dxdy = inv_dxdy.unsqueeze(0)
    terrain_min_point_unsq = terrain_min_point.unsqueeze(0)

    grid_min = torch.floor((min_xy - terrain_min_point_unsq) * inv_dxdy).to(dtype=torch.int64)
    grid_max = torch.ceil((max_xy - terrain_min_point_unsq) * inv_dxdy).to(dtype=torch.int64)

    zero_long = torch.zeros(2, device=device, dtype=torch.int64)
    max_grid = (terrain_dims - 1).to(dtype=torch.int64)
    grid_min = torch.maximum(grid_min, zero_long.unsqueeze(0))
    grid_max = torch.minimum(grid_max, max_grid.unsqueeze(0))

    frame_sizes = grid_max - grid_min + 1
    patch_size = torch.maximum(frame_sizes.max(dim=0).values, torch.ones_like(terrain_dims))
    patch_size = torch.minimum(patch_size, terrain_dims)

    max_start = torch.maximum(terrain_dims - patch_size, torch.zeros_like(terrain_dims))
    start_idx = torch.minimum(grid_min, max_start.unsqueeze(0))
    start_idx = torch.maximum(start_idx, zero_long.unsqueeze(0))

    patch_size_x = int(patch_size[0].item())
    patch_size_y = int(patch_size[1].item())

    x_range = torch.arange(patch_size_x, device=device, dtype=torch.int64)
    y_range = torch.arange(patch_size_y, device=device, dtype=torch.int64)

    x_indices = start_idx[:, 0].unsqueeze(1) + x_range.unsqueeze(0)
    y_indices = start_idx[:, 1].unsqueeze(1) + y_range.unsqueeze(0)

    x_idx = x_indices.unsqueeze(2).expand(num_frames, patch_size_x, patch_size_y)
    y_idx = y_indices.unsqueeze(1).expand(num_frames, patch_size_x, patch_size_y)

    hf_patches = terrain.hf[x_idx, y_idx]

    patch_min_box_center = terrain_min_point_unsq + start_idx.to(dtype=torch.float32) * terrain_dxdy.unsqueeze(0)

    all_negative_sdfs = terrain_util.points_hf_sdf(all_body_points, hf_patches, patch_min_box_center,
                                                   terrain_dxdy, base_z=-10.0, inverted=True)
    all_negative_sdfs = torch.clamp(all_negative_sdfs, max=0.0)
    penetration_loss += torch.sum(-all_negative_sdfs)

    if w_contact != 0.0:
        all_positive_sdfs = terrain_util.points_hf_sdf(all_body_points, hf_patches, patch_min_box_center,
                                                       terrain_dxdy, base_z=-10.0, inverted=False)
        all_positive_sdfs = torch.clamp(all_positive_sdfs, min=0.0)
        positive_sdfs_split = torch.split(all_positive_sdfs, body_point_counts, dim=1)
    else:
        positive_sdfs_split = [None] * num_bodies

    for b in range(num_bodies):
        curr_body_points = body_world_points[b]
        curr_body_rot = body_rot_list[b]
        curr_body_pos = body_pos_list[b]

        if w_contact != 0.0:
            positive_sdfs = positive_sdfs_split[b]
            closest_distances, closest_point_ids = torch.min(positive_sdfs, dim=-1)

            body_name = char_model.get_body_name(b)
            if body_name in char_model._contact_body_names:
                contact_body_id = char_model.get_contact_body_id(body_name)
                contact_errs = closest_distances * contacts[:, contact_body_id]
                contact_loss += torch.sum(contact_errs)

        if body_constraints is not None:
            curr_body_constraint_list = body_constraints[b]

            for curr_body_constraint in curr_body_constraint_list:
                start_frame_idx = curr_body_constraint.start_frame_idx
                end_frame_idx = curr_body_constraint.end_frame_idx
                constraint_point = curr_body_constraint.constraint_point

                if char_model.get_geoms(b)[0]._shape_type == kin_char_model.GeomType.SPHERE:
                    radius = char_model.get_geoms(b)[0]._dims
                    offset = char_model.get_geoms(b)[0]._offset

                    body_center = torch_util.quat_rotate(curr_body_rot.squeeze(1), offset.unsqueeze(0)) + curr_body_pos.squeeze(1)
                    body_constraint_diff = geom_util.sdSphere(constraint_point.unsqueeze(0),
                                                              body_center[start_frame_idx:end_frame_idx+1], radius)
                    body_constraint_loss += torch.sum(torch.abs(body_constraint_diff))

                elif char_model.get_geoms(b)[0]._shape_type == kin_char_model.GeomType.BOX:
                    radius = torch.norm(char_model.get_geoms(b)[0]._dims) * 1.25
                    foot_sole_points = curr_body_points[start_frame_idx:end_frame_idx+1, 0:18].reshape(-1, 3)
                    body_constraint_diff = geom_util.sdSphere(constraint_point.unsqueeze(0), foot_sole_points, radius)
                    body_constraint_loss += torch.sum(torch.clamp(body_constraint_diff, min=0.0))
                else:
                    continue

                body_vel_err_sq = body_vel_err_sq.clone()
                body_vel_err_sq[start_frame_idx:end_frame_idx+1, b] *= 0.0
                body_rot_vel_err_sq = body_rot_vel_err_sq.clone()
                body_rot_vel_err_sq[start_frame_idx:end_frame_idx+1, b] *= 0.0

    if w_sliding != 0.0 and frame_change_in_contact.shape[-1] == num_bodies:
        c = 0.03
        c2 = 0.0009
        sliding_loss = torch.sum((torch.sqrt(torch.sum(body_vel_err_sq, dim=-1) + c2) - c) * frame_change_in_contact)\
                        + torch.sum((torch.sqrt(body_rot_vel_err_sq + c2) - c) * frame_change_in_contact)
    else:
        sliding_loss = 0.0

    tgt_body_acc = tgt_body_vels[1:] - tgt_body_vels[:-1]
    tgt_body_jerk = tgt_body_acc[1:] - tgt_body_acc[:-1]
    tgt_body_jerk_mag = torch.norm(tgt_body_jerk, dim=-1)
    dt = 1.0 / 30.0
    max_jerk = max_jerk * (dt ** 3)
    jerk_loss = torch.sum(torch.clamp(tgt_body_jerk_mag - max_jerk, min=0.0))

    losses = dict()
    losses[LossType.ROOT_POS_LOSS] = root_pos_loss.item()
    losses[LossType.ROOT_ROT_LOSS] = root_rot_loss.item()
    losses[LossType.JOINT_ROT_LOSS] = joint_rot_loss.item()
    losses[LossType.SMOOTHNESS_LOSS] = smoothness_loss.item()
    losses[LossType.PENETRATION_LOSS] = penetration_loss.item()
    if isinstance(contact_loss, torch.Tensor):
        losses[LossType.CONTACT_LOSS] = contact_loss.item()
    else:
        losses[LossType.CONTACT_LOSS] = contact_loss
    if isinstance(sliding_loss, torch.Tensor):
        losses[LossType.SLIDING_LOSS] = sliding_loss.item()
    else:
        losses[LossType.SLIDING_LOSS] = sliding_loss
    losses[LossType.JERK_LOSS] = jerk_loss.item()
    if isinstance(body_constraint_loss, torch.Tensor):
        losses[LossType.BODY_CONSTRAINT_LOSS] = body_constraint_loss.item()
    else:
        losses[LossType.BODY_CONSTRAINT_LOSS] = body_constraint_loss

    loss = w_root_pos * root_pos_loss \
            + w_root_rot * root_rot_loss \
            + w_joint_rot * joint_rot_loss \
            + w_smoothness * smoothness_loss \
            + w_penetration * penetration_loss \
            + w_contact * contact_loss \
            + w_sliding * sliding_loss \
            + w_body_constraints * body_constraint_loss \
            + w_jerk * jerk_loss

    return loss, losses

def simplified_motion_loss(tgt_root_pos: torch.Tensor,
                           tgt_root_rot: torch.Tensor,
                           tgt_joint_dof: torch.Tensor,
                           src_root_pos: torch.Tensor,
                           src_root_rot_quat: torch.Tensor,
                           src_joint_rot: torch.Tensor,
                           char_model: kin_char_model.KinCharModel,
                           w_root_pos: float,
                           w_root_rot: float,
                           w_joint_rot: float,
                           w_jerk: float):
    """Compute a simplified motion loss.

    The loss only regularizes differences between the source and target
    root position, root rotation, and joint rotations while also applying
    a jerk penalty on the target body positions.
    """

    root_pos_err = tgt_root_pos - src_root_pos
    root_pos_loss = torch.sum(torch.square(root_pos_err))

    tgt_root_rot_quat = torch_util.exp_map_to_quat(tgt_root_rot)
    root_rot_err = torch_util.quat_diff_angle(tgt_root_rot_quat, src_root_rot_quat)
    root_rot_loss = torch.sum(torch.square(root_rot_err))

    tgt_joint_rot = char_model.dof_to_rot(tgt_joint_dof)
    joint_rot_err = torch_util.quat_diff_angle(tgt_joint_rot, src_joint_rot)
    joint_rot_loss = torch.sum(torch.square(joint_rot_err))

    tgt_body_pos, _ = char_model.forward_kinematics(tgt_root_pos, tgt_root_rot_quat, tgt_joint_rot)

    if tgt_body_pos.shape[0] < 4:
        jerk_loss = torch.tensor(0.0, device=tgt_body_pos.device, dtype=tgt_body_pos.dtype)
    else:
        tgt_body_vels = tgt_body_pos[1:] - tgt_body_pos[:-1]
        tgt_body_acc = tgt_body_vels[1:] - tgt_body_vels[:-1]
        tgt_body_jerk = tgt_body_acc[1:] - tgt_body_acc[:-1]
        jerk_loss = torch.sum(torch.square(tgt_body_jerk))

    losses = {
        LossType.ROOT_POS_LOSS: root_pos_loss.item(),
        LossType.ROOT_ROT_LOSS: root_rot_loss.item(),
        LossType.JOINT_ROT_LOSS: joint_rot_loss.item(),
        LossType.JERK_LOSS: jerk_loss.item(),
    }

    loss = (
        w_root_pos * root_pos_loss
        + w_root_rot * root_rot_loss
        + w_joint_rot * joint_rot_loss
        + w_jerk * jerk_loss
    )

    return loss, losses


def build_logger(log_file, exp_name=None, use_wandb=True):
    log = wandb_logger.WandbLogger(project_name="motion optimization", exp_name=exp_name, connect_to_wandb=use_wandb)
    log.set_step_key("Iteration")
    if (log_file is not None):
        log.configure_output_file(log_file)
    return log

def motion_contact_optimization(src_motion_frames: motion_util.MotionFrames,
                                body_points: list,
                                terrain: terrain_util.SubTerrain,
                                char_model: kin_char_model.KinCharModel,
                                num_iters: int,
                                step_size: float,
                                w_root_pos: float, 
                                w_root_rot: float, 
                                w_joint_rot: float, 
                                w_smoothness: float, 
                                w_penetration: float, 
                                w_contact: float,
                                w_sliding: float,
                                w_body_constraints: float,
                                w_jerk: float,
                                body_constraints: list,
                                max_jerk: float,
                                exp_name: str,
                                use_wandb: bool,
                                log_file: str):
    # optimize a motion sequence to have non-penetrating contacts
    start_time = time.time()

    src_motion_frames.assert_num_dims(2)
    src_root_pos = src_motion_frames.root_pos
    src_root_rot_quat = src_motion_frames.root_rot
    src_root_rot = torch_util.quat_to_exp_map(src_root_rot_quat)
    src_joint_rot = src_motion_frames.joint_rot
    src_joint_dof = char_model.rot_to_dof(src_joint_rot)
    contacts = src_motion_frames.contacts

    src_body_pos, src_body_rot = char_model.forward_kinematics(src_root_pos, src_root_rot_quat, src_joint_rot)
    src_body_vels = src_body_pos[1:] - src_body_pos[:-1]
    src_body_rot_vels = torch_util.quat_diff_angle(src_body_rot[1:], src_body_rot[:-1])

    tgt_root_pos = src_root_pos.clone()
    tgt_root_rot = src_root_rot.clone()
    tgt_joint_dof = src_joint_dof.clone()

    tgt_root_pos.requires_grad = True
    tgt_root_rot.requires_grad = True
    tgt_joint_dof.requires_grad = True
    weights = [tgt_root_pos, tgt_root_rot, tgt_joint_dof]
    optimizer = torch.optim.Adam(weights, lr=step_size)

    log_iter_stride = 25

    logger = build_logger(log_file=log_file, exp_name=exp_name, use_wandb=use_wandb)

    for iter in range(num_iters):
        optimizer.zero_grad()
        loss, loss_dict = motion_terrain_contact_loss_localized(tgt_root_pos=tgt_root_pos,
                                                    tgt_root_rot=tgt_root_rot,
                                                    tgt_joint_dof=tgt_joint_dof,
                                                    src_root_pos=src_root_pos,
                                                    src_root_rot_quat=src_root_rot_quat,
                                                    src_joint_rot=src_joint_rot,
                                                    src_body_vels=src_body_vels,
                                                    src_body_rot_vels=src_body_rot_vels,
                                                    contacts = contacts,
                                                    terrain=terrain,
                                                    body_points=body_points,
                                                    char_model=char_model,
                                                    w_root_pos=w_root_pos,
                                                    w_root_rot=w_root_rot,
                                                    w_joint_rot=w_joint_rot,
                                                    w_smoothness=w_smoothness,
                                                    w_penetration=w_penetration,
                                                    w_contact=w_contact,
                                                    w_sliding=w_sliding,
                                                    w_jerk=w_jerk,
                                                    w_body_constraints=w_body_constraints,
                                                    body_constraints=body_constraints,
                                                    max_jerk=max_jerk)
        loss.backward()
        optimizer.step()
        
        if iter % log_iter_stride == 0:
            logger.log("Iteration", iter)
            logger.log("Time (min)", (time.time() - start_time) / 60.0)
            logger.log("TOTAL WEIGHTED LOSS", loss.item())
            for key, val in loss_dict.items():
                logger.log(key.name, val)

            logger.print_log()
            logger.write_log()


    if use_wandb:
        logger.end_wandb()
    if log_file is not None:
        logger.output_file.close()

    # ret_frames = [tgt_root_pos.detach(), 
    #               tgt_root_rot.detach(),
    #               tgt_joint_dof.detach()]
    # ret_frames = torch.cat(ret_frames, dim=-1)

    ret_frames = motion_util.MotionFrames(root_pos = tgt_root_pos.detach(),
                                          root_rot = torch_util.exp_map_to_quat(tgt_root_rot.detach()),
                                          joint_rot = char_model.dof_to_rot(tgt_joint_dof.detach()),
                                          contacts = src_motion_frames.contacts.clone())
    return ret_frames


def simplified_motion_optimization(src_motion_frames: motion_util.MotionFrames,
                                   char_model: kin_char_model.KinCharModel,
                                   num_iters: int,
                                   step_size: float,
                                   w_root_pos: float,
                                   w_root_rot: float,
                                   w_joint_rot: float,
                                   w_jerk: float,
                                   exp_name: str="",
                                   use_wandb: bool=False,
                                   log_file: str=None):
    """Optimize a motion sequence with only basic regularization and jerk loss."""

    start_time = time.time()

    src_motion_frames.assert_num_dims(2)
    src_root_pos = src_motion_frames.root_pos
    src_root_rot_quat = src_motion_frames.root_rot
    src_root_rot = torch_util.quat_to_exp_map(src_root_rot_quat)
    src_joint_rot = src_motion_frames.joint_rot
    src_joint_dof = char_model.rot_to_dof(src_joint_rot)

    tgt_root_pos = src_root_pos.clone()
    tgt_root_rot = src_root_rot.clone()
    tgt_joint_dof = src_joint_dof.clone()

    tgt_root_pos.requires_grad = True
    tgt_root_rot.requires_grad = True
    tgt_joint_dof.requires_grad = True

    optimizer = torch.optim.Adam([tgt_root_pos, tgt_root_rot, tgt_joint_dof], lr=step_size)

    log_iter_stride = 25
    logger = build_logger(log_file=log_file, exp_name=exp_name, use_wandb=use_wandb)

    for iter in range(num_iters):
        optimizer.zero_grad()
        loss, loss_dict = simplified_motion_loss(tgt_root_pos=tgt_root_pos,
                                                 tgt_root_rot=tgt_root_rot,
                                                 tgt_joint_dof=tgt_joint_dof,
                                                 src_root_pos=src_root_pos,
                                                 src_root_rot_quat=src_root_rot_quat,
                                                 src_joint_rot=src_joint_rot,
                                                 char_model=char_model,
                                                 w_root_pos=w_root_pos,
                                                 w_root_rot=w_root_rot,
                                                 w_joint_rot=w_joint_rot,
                                                 w_jerk=w_jerk)
        loss.backward()
        optimizer.step()

        if iter % log_iter_stride == 0:
            logger.log("Iteration", iter)
            logger.log("Time (min)", (time.time() - start_time) / 60.0)
            logger.log("TOTAL WEIGHTED LOSS", loss.item())
            for key, val in loss_dict.items():
                logger.log(key.name, val)

            logger.print_log()
            logger.write_log()

    if use_wandb:
        logger.end_wandb()
    if log_file is not None:
        logger.output_file.close()

    ret_frames = motion_util.MotionFrames(root_pos=tgt_root_pos.detach(),
                                          root_rot=torch_util.exp_map_to_quat(tgt_root_rot.detach()),
                                          joint_rot=char_model.dof_to_rot(tgt_joint_dof.detach()),
                                          contacts=src_motion_frames.contacts.clone())

    return ret_frames