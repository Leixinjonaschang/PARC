import torch

import parc.util.torch_util as torch_util


@torch.jit.script
def compute_box_obs(root_pos, root_rot, box_dims, box_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    # inputs must be in global coords
    # root_pos: (N, 3)
    # root_rot: (N, 4)
    # box_dims: (N, 3)
    # box_pos: (N, 3)

    rel_box_pos = box_pos - root_pos
    box_x_minus = rel_box_pos[:, 0] - box_dims[:, 0] / 2.0
    box_x_plus = rel_box_pos[:, 0] + box_dims[:, 0] / 2.0
    box_y_minus = rel_box_pos[:, 1] - box_dims[:, 1] / 2.0
    box_y_plus = rel_box_pos[:, 1] + box_dims[:, 1] / 2.0
    box_z_minus = torch.zeros_like(box_dims[:, 2])
    box_z_plus = box_dims[:, 2]
    corners = torch.stack([
        box_x_minus, box_y_minus, box_z_minus,
        box_x_minus, box_y_plus , box_z_minus,
        box_x_plus , box_y_plus , box_z_minus,
        box_x_plus , box_y_minus, box_z_minus,
        box_x_minus, box_y_minus, box_z_plus,
        box_x_minus, box_y_plus , box_z_plus,
        box_x_plus , box_y_plus , box_z_plus,
        box_x_plus , box_y_minus, box_z_plus
    ], dim=-1)

    # I think i did this rotation part very incorrectly?
    corners_vec_view = corners.view(corners.shape[0]*8, 3)
    root_rot = root_rot.unsqueeze(dim=1).repeat(1, 8, 1)
    root_rot = root_rot.view(root_rot.shape[0]*8, 4)
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)
    rotated_corners = torch_util.quat_rotate(heading_rot, corners_vec_view)
    #rotated_corners = corners_vec_view

    box_obs = rotated_corners.view(corners.shape[0], 8*3)

    return box_obs

@torch.jit.script
def get_box_corners(root_pos, box_dims, box_pos):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # inputs must be in global coords
    # root_pos: (N, 3)
    # box_dims: (N, 3)
    # box_pos: (N, 3)

    rel_box_pos = box_pos - root_pos
    box_x_minus = rel_box_pos[:, 0] - box_dims[:, 0] / 2.0
    box_x_plus = rel_box_pos[:, 0] + box_dims[:, 0] / 2.0
    box_y_minus = rel_box_pos[:, 1] - box_dims[:, 1] / 2.0
    box_y_plus = rel_box_pos[:, 1] + box_dims[:, 1] / 2.0
    box_z_minus = box_pos[:, 2] - box_dims[:, 2] / 2.0
    box_z_plus = box_pos[:, 2] + box_dims[:, 2] / 2.0
    corners = torch.stack([
        box_x_minus, box_y_minus, box_z_minus,
        box_x_minus, box_y_plus , box_z_minus,
        box_x_plus , box_y_plus , box_z_minus,
        box_x_plus , box_y_minus, box_z_minus,
        box_x_minus, box_y_minus, box_z_plus,
        box_x_minus, box_y_plus , box_z_plus,
        box_x_plus , box_y_plus , box_z_plus,
        box_x_plus , box_y_minus, box_z_plus
    ], dim=-1)

    box_obs = corners

    return box_obs

@torch.jit.script
def sample_point_in_rectangle(min_xy, max_xy):
    # type: (Tensor, Tensor) -> (Tensor)
    # border variables describe how close to edge we will let ourselves be
    
    # Get the number of rectangles
    num_rectangles = min_xy.shape[0]

    # Generate random samples for x and y coordinates within each rectangle
    point = min_xy + torch.rand_like(min_xy) * (min_xy - max_xy)

    return point

def sample_box(box_dims_mean, box_pos_mean, box_dims_std, box_pos_std, box_sampling, 
               num_envs, device,
               place_box_on_ground=True):
    ## Samples box geometry for the different envs

    # sample box dims and box positions
    if box_sampling == "none":
        box_dims = torch.clone(box_dims_mean).unsqueeze(dim=0).expand(num_envs, -1)

        if place_box_on_ground:
            box_pos_mean[2] = box_dims_mean[2] / 2.0

        box_pos = torch.clone(box_pos_mean).unsqueeze(dim=0).expand(num_envs, -1)
    elif box_sampling == "uniform" or box_sampling == "non-random-uniform":
        dims_lower_bound = box_dims_mean - box_dims_std
        dims_range = box_dims_std * 2

        pos_lower_bound = box_pos_mean - box_pos_std
        pos_range = box_pos_std * 2

        if box_sampling == "uniform":

            box_dims = dims_lower_bound + torch.rand(size=(num_envs, 3), device=device) * dims_range
            box_pos = pos_lower_bound + torch.rand(size=(num_envs, 3), device=device) * pos_range
        elif box_sampling == "non-random-uniform":
            lin = torch.linspace(0.0, 1.0, num_envs, device=device).unsqueeze(-1).expand(-1, 3)
            box_dims = dims_lower_bound + lin * dims_range
            box_pos = pos_lower_bound + lin * pos_range

        if place_box_on_ground:
            box_pos[..., 2] = box_dims[..., 2].clone() / 2.0

    return box_dims, box_pos


def inverse_char_obs(char_obs, root_height_obs):
    batch_size = char_obs.shape[0]
    device = char_obs.device
    off = int(root_height_obs)

    root_rot_slice = slice(off+0, off+6)
    root_vel_slice = slice(off+6, off+9)
    root_ang_vel_slice = slice(off+9, off+12)
    joint_rot_slice = slice(off+12, off+96)
    dof_vel_slice = slice(off+96, off+124)
    key_pos_slice = slice(off+124, off+124+12)

    root_rot = char_obs[..., root_rot_slice]
    root_vel = char_obs[..., root_vel_slice]
    root_ang_vel = char_obs[..., root_ang_vel_slice]
    joint_rot = char_obs[..., joint_rot_slice]
    dof_vel = char_obs[..., dof_vel_slice]
    key_pos = char_obs[..., key_pos_slice]

    if root_height_obs:
        root_height = char_obs[..., 0]
    
    # will return 0 for x y values of root pos
    root_pos = torch.zeros(size=(batch_size, 3), dtype=torch.float32, device=device)
    root_pos[..., 2] = root_height

    joint_rot = joint_rot.view(batch_size, joint_rot.shape[1] // 6, 6)
    joint_rot_quat = torch_util.tan_norm_to_quat(joint_rot.reshape(-1, 6)).reshape(batch_size, -1, 4)
    root_rot_quat = torch_util.tan_norm_to_quat(root_rot)

    return root_pos, root_rot_quat, joint_rot_quat

def extract_obs(obs, root_height_obs):
    off = int(root_height_obs)
    num_frames = obs.shape[0]

    root_rot_slice = slice(off+0, off+6)
    root_vel_slice = slice(off+6, off+9)
    root_ang_vel_slice = slice(off+9, off+12)
    joint_rot_slice = slice(off+12, off+96)
    dof_vel_slice = slice(off+96, off+124)
    key_pos_slice = slice(off+124, off+124+12)

    root_rot = torch_util.tan_norm_to_quat(obs[..., root_rot_slice])
    root_vel = obs[..., root_vel_slice]
    root_ang_vel = obs[..., root_ang_vel_slice]
    joint_rot = torch_util.tan_norm_to_quat(obs[..., joint_rot_slice].reshape(-1, 6)).reshape(num_frames, -1, 4)
    dof_vel = obs[..., dof_vel_slice]
    key_pos = obs[..., key_pos_slice]

    ret = dict()
    ret["root_rot"] = root_rot
    ret["root_vel"] = root_vel
    ret["root_ang_vel"] = root_ang_vel
    ret["joint_rot"] = joint_rot
    ret["dof_vel"] = dof_vel
    ret["key_pos"] = key_pos
    if root_height_obs:
        ret["root_height"] = obs[..., 0]
        
    return ret

def inverse_tar_obs(tar_obs, rots_as_quats = True):
    """
    Given observations of future states, get the motion frames in our usual format.
    Very hard-coded.
    """
    # expects tar obs in (batch_size, num_tar_obs, dofs) format
    if len(tar_obs.shape) == 2:
        input_shape_len_is_2 = True
        tar_obs = tar_obs.unsqueeze(0)
    else:
        input_shape_len_is_2 = False

    num_tar_obs = tar_obs.shape[1]

    total_root_pos = []
    total_root_rot = []
    total_joint_rot = []
    total_key_pos = []


    for frame_idx in range(num_tar_obs):
        batch_size = tar_obs.shape[0]

        curr_tar_obs = tar_obs[:, frame_idx, :]

        root_pos = curr_tar_obs[:, 0:3]
        root_rot_tan_norm = curr_tar_obs[:, 3:9]
        curr_joint_rot_tan_norm = curr_tar_obs[:, 9:9+84]
        curr_key_pos = curr_tar_obs[:, 9+84:].view(batch_size, -1, 3)


        if rots_as_quats:
            root_rot = torch_util.tan_norm_to_quat(root_rot_tan_norm)

            joint_rot = torch_util.tan_norm_to_quat(curr_joint_rot_tan_norm.reshape(-1, 6)).reshape(batch_size, -1, 4)

        else:
            root_rot = root_rot_tan_norm
            joint_rot = curr_joint_rot_tan_norm.reshape(-1, 6)

        total_root_pos.append(root_pos)
        total_root_rot.append(root_rot)
        total_joint_rot.append(joint_rot)
        total_key_pos.append(curr_key_pos)

    total_root_pos = torch.stack(total_root_pos, dim=1)
    total_root_rot = torch.stack(total_root_rot, dim=1)
    total_joint_rot = torch.stack(total_joint_rot, dim=1)
    total_key_pos = torch.stack(total_key_pos, dim=1)

    if input_shape_len_is_2:
        return total_root_pos.squeeze(0), total_root_rot.squeeze(0), total_joint_rot.squeeze(0), total_key_pos.squeeze(0)
    else:
        return total_root_pos, total_root_rot, total_joint_rot, total_key_pos