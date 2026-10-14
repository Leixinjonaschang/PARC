import numpy as np
import torch
import trimesh

import parc.anim.kin_char_model as kin_char_model
import parc.util.torch_util as torch_util


def point_plane_halfspace(point, plane_point, plane_normal, eps = 1e-5):
    
    temp = point - plane_point

    return torch.dot(temp, plane_normal) > eps

def box_plane_intersection(box_points, plane_point, plane_normal, eps = 1e-5):
    assert len(box_points) == 8

    above = 0
    below = 0
    for point in box_points:
        if point_plane_halfspace(point, plane_point, plane_normal, eps):
            above += 1
        else:
            below +=1

    if above == 8:
        return False
    elif below == 8:
        return False
    else:
        return True
    
def box_ground_plane_contact(box_points, plane_point, plane_normal, eps=1e-5):
    # This time, ground plane is treated as solid, so if all points are below, then it is considered in contact
    assert len(box_points) == 8

    above = 0
    below = 0
    for point in box_points:
        if point_plane_halfspace(point, plane_point, plane_normal, eps):
            above += 1
        else:
            below +=1

    if above == 8:
        return False
    elif below == 8:
        return True
    else:
        return True

@torch.no_grad()
def get_box_points(body_pos, body_rot, box_size, box_offset):
    # returns the 8 points of the box rigid body for parc/data/assets/humanoid.xml
    foot_points = []
    p1 = - box_size.clone().detach()
    p2 =  - box_size.clone().detach()
    p2[0] += 2*box_size[0]
    p3 = - box_size.clone().detach()
    p3[0] += 2*box_size[0]
    p3[1] += 2*box_size[1]
    p4 = - box_size.clone().detach()
    p4[1] += 2*box_size[1]
    foot_points = [
        p1, p2, p3, p4
    ]
    upper_foot_points = []
    
    for point in foot_points:
        p = point.clone().detach()
        p[2] += 2*box_size[2]
        upper_foot_points.append(p)
    foot_points.extend(upper_foot_points)

    for i in range(len(foot_points)):
        foot_points[i] += box_offset
        foot_points[i] = torch_util.quat_rotate(body_rot, foot_points[i])
        foot_points[i] += body_pos

    return foot_points

def get_box_points_batch(body_pos, body_rot, box_dims, box_offset):
    # body_pos: (N, 3) of box positions
    # body_rot: (N, 4) of box rotations around the box positions
    # box_dims: (N, 3) or (3) of box half dimensions
    # box_offset: (N, 3) or (3) of box center of mass offsets (from box positions)

    # returns: (N, 8, 3) of box corner positions
    if len(box_offset.shape) == 2:
        box_offset = box_offset.unsqueeze(dim=1) # (N, 1, 3)
    elif len(box_offset.shape) != 1:
        assert False

    foot_points = torch.zeros_like(body_pos).unsqueeze(dim=1).repeat(1, 8, 1)
    foot_points[:, 0, :] = -box_dims
    foot_points[:, 1, 1:3] = -box_dims[..., 1:3]
    foot_points[:, 1, 0] = box_dims[..., 0].clone()
    foot_points[:, 2, 2] = -box_dims[..., 2]
    foot_points[:, 2, 0:2] = box_dims[..., 0:2].clone()
    foot_points[:, 3, :] = -box_dims
    foot_points[:, 3, 1] = box_dims[..., 1].clone()

    # copy lower foot points xy to upper foot points xy
    temp = foot_points[:, 0:4, 0:2].clone()
    foot_points[:, 4:8, 0:2] = temp
    foot_points[:, 4:8, 2] = box_dims[..., 2].clone().unsqueeze(-1) # (N, 4) = (N, 1) broadcasting

    body_rot = body_rot.unsqueeze(dim=1).expand(-1, 8, -1)

    foot_points += box_offset
    foot_points = torch_util.quat_rotate(body_rot, foot_points)
    foot_points += body_pos.unsqueeze(dim=1)
    return foot_points

def sdRoundBox(point, box_dims, radius):
    # return signed distance from point to round box
    #https://iquilezles.org/articles/distfunctions/
    #vec3 q = abs(p) - b;
    #return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
    # box dims are half of length, width, height

    return sdBox(point, box_dims) - radius

def sdBox(point, box_halfdims):
    # point: (..., 3)
    # box_halfdims: (..., 3)
    # point.shape == box_dims.shape
    # Box is assumed to be cenetered at origin, points are transformed to box's coordinate frame

    # displacement of point to box when point is reflected to positive octant
    q = torch.abs(point) - box_halfdims # shape: (..., 3)
    
    # case where point is outside of the box
    positive_displacement = torch.clamp(q, min=0.0)
    positive_distance = torch.norm(positive_displacement, dim=-1)
    #z1 = torch.zeros_like(q[0])
    #temp2 = torch.min(torch.max(q[0], torch.max(q[1], q[2])), z1)

    # case where point is inside of the box
    closest_inside_surface = torch.max(q, dim=-1)[0]
    negative_distance = torch.clamp(closest_inside_surface, max=0.0)

    signed_distance = positive_distance + negative_distance

    return signed_distance

def sdPointLine(p, a, b):
    # Find the signed distance between a point 'p' and an infinite line described by points 'a' and 'b'.
    # One of the spaces split by the line is the negative space, and one is the positive space
    # p: (N, num_points, 2)
    # a: (N, 1, 2)
    # b: (N, 1, 2)
    
    pa = p - a
    ba = b - a
    # find the projection of pa onto ba
    dot_paba = torch.sum(pa*ba, dim=-1, keepdim=True)
    dot_baba = torch.sum(ba*ba, dim=-1, keepdim=True)
    proj_paba = dot_paba / dot_baba * ba

    ba_normalized = ba / torch.norm(ba, dim=-1, keepdim=True)


    orthogonal = pa - proj_paba

    sd = orthogonal[..., 0] *ba_normalized[..., 1] - orthogonal[..., 1] * ba_normalized[..., 0]
    return sd

def sdSphere(p, c, r):
    # p: point
    # c: center of sphere
    # r: radius of sphere
    return torch.norm(p - c, dim=-1) - r

def get_box_roof_points(box_pos, box_dims):
    # TODO: add z-axis rotation angle as a param

    # box_pos: (N, 3)
    # box_dims: (N, 3)
    box_dx = box_dims[..., 0:1]/2
    box_dy = box_dims[..., 1:2]/2
    box_x = box_pos[..., 0:1]
    box_y = box_pos[..., 1:2]

    # points: (N, 2)
    point_neg_neg = torch.cat([box_x - box_dx, box_y - box_dy], dim=-1)
    point_neg_pos = torch.cat([box_x - box_dx, box_y + box_dy], dim=-1)
    point_pos_pos = torch.cat([box_x + box_dx, box_y + box_dy], dim=-1)
    point_pos_neg = torch.cat([box_x + box_dx, box_y - box_dy], dim=-1)

    # box_points: (N, 4, 2)
    box_points = torch.stack([
        point_neg_neg,
        point_neg_pos,
        point_pos_pos,
        point_pos_neg
    ], dim=1)

    return box_points

def get_xy_grid_points_2(min_point, max_point, dxdy):
    device = min_point.device
    dims = ((max_point - min_point) / dxdy).to(dtype=torch.int64)
    x_points = torch.linspace(min_point[0], max_point[0], dims[0].item(), device=device)
    y_points = torch.linspace(min_point[1], max_point[1], dims[1].item(), device=device)

    x, y = torch.meshgrid(x_points, y_points, indexing='ij')

    xy_points = torch.stack([x, y], dim=-1)
    return xy_points

def get_xy_grid_points(center, dx, dy, num_x_neg, num_x_pos, num_y_neg, num_y_pos):

    device = center.device
    x_dim = num_x_neg + num_x_pos + 1
    y_dim = num_y_neg + num_y_pos + 1
    x_points = torch.linspace(center[0] - dx*num_x_neg, center[0] + dx*num_x_pos, x_dim, device=device)
    y_points = torch.linspace(center[1] - dy*num_y_neg, center[1] + dy*num_y_pos, y_dim, device=device)

    x, y = torch.meshgrid(x_points, y_points, indexing='ij')

    xy_points = torch.stack([x, y], dim=-1)
    return xy_points


def get_xy_grid_points_coarse2fine(center, dx, dy, num_x, num_y, root_x_offset, dx_incr_rate, dy_incr_rate):
    device = center.device
    x_dim = num_x * 2 + 1
    y_dim = num_y * 2 + 1
    
    x_points = torch.linspace(center[0] - dx * num_x, center[0] + dx * num_x, x_dim, device=device)
    y_points = torch.linspace(center[1] - dy * num_y, center[1] + dy * num_y, y_dim, device=device)
    
    x_inc = torch.cumsum(torch.ones_like(x_points[...,:num_x-1]) * dx_incr_rate, dim=-1)
    x_inc_reverse = torch.flip(x_inc, (-1,))
    y_inc = torch.cumsum(torch.ones_like(y_points[...,:num_y-1]) * dy_incr_rate, dim=-1)
    y_inc_reverse = torch.flip(y_inc, (-1,))
    
    x_points[...,num_x+2:] += x_inc
    x_points[...,:num_x-1] -= x_inc_reverse

    y_points[...,num_y+2:] -= y_inc
    y_points[...,:num_y-1] += y_inc_reverse

    x_points += root_x_offset
    x, y = torch.meshgrid(x_points, y_points, indexing='ij')
    
    xy_points = torch.stack([x, y], dim=-1)
    return xy_points

def get_xy_points_cone(center, dx, num_neg, num_pos, num_rays_neg, num_rays_pos, angle_between_rays):
    # get xy points for a cone
    device = center.device

    dim = num_neg + num_pos + 1
    y_points = torch.tensor([0.0], device=device, dtype=torch.float32)
    x_points = torch.linspace(-dx*num_neg, dx*num_pos, dim, device=device, dtype=torch.float32)

    x, y = torch.meshgrid(x_points, y_points, indexing='ij')

    xy_points = torch.stack([x, y], dim=-1).view(-1, 2)
    num_points = xy_points.shape[0]

    xy_points_rays = []
    num_rays = num_rays_neg + 1 + num_rays_pos
    for i in range(0, num_rays):
        angle = -angle_between_rays * (num_rays_neg-i)
        angle = torch.ones(size=(num_points,), device=device, dtype=torch.float32) * angle
        xy_points_rays.append(torch_util.rotate_2d_vec(xy_points, angle))

    xy_points = torch.cat(xy_points_rays, dim=0)
    return xy_points

def heightfield_from_box(xy_points,
                         box_4_points, box_height):
    # helper function for getting height fields from box (FOR TRAINING MDM)
    # 1) for each grid point, get a vertical line
    # 2) compute the z-intersection with the box plane and the vertical lines
    #       to get the (x, y, z) intersection point
    # 3) check if (x, y) are within the 4 box points. If yes, use z for the grid point
    # 4) otherwise, use 0

    # xy_points: (N, num_points, 2)
    # box_4_points: (N, 4, 2)
    # box_heights: (N, 2)
    # output: (N, num_points)
    

    check_1 = sdPointLine(xy_points, box_4_points[:, 0:1], box_4_points[:, 1:2]) > 0.0
    check_2 = sdPointLine(xy_points, box_4_points[:, 1:2], box_4_points[:, 2:3]) > 0.0
    check_3 = sdPointLine(xy_points, box_4_points[:, 2:3], box_4_points[:, 3:4]) > 0.0
    check_4 = sdPointLine(xy_points, box_4_points[:, 3:4], box_4_points[:, 0:1]) > 0.0
    in_box_checks = torch.stack([check_1, check_2, check_3, check_4], dim=-1)
    in_box = torch.any(torch.stack([torch.all(in_box_checks, dim=-1), torch.all(~in_box_checks, dim=-1)], dim=-1), dim=-1)

    hf = in_box.float() * box_height
    return hf




def area_triangle(A, B, C):
    # A, B, C: (N, 2)
    # returns: (N,)

    x1 = A[..., 0]
    x2 = B[..., 0]
    x3 = C[..., 0]
    y1 = A[..., 1]
    y2 = B[..., 1]
    y3 = C[..., 1]

    return 0.5 * torch.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def point_in_rectangle(P, A, B, C, D):
    # returns true if point P is inside the rectangle defined by ABCD
    # P: (N, 2)
    # A, B, C, D: (N, 2)
    # returns: (N,)

    PAB = area_triangle(P, A, B)
    PBC = area_triangle(P, B, C)
    PCD = area_triangle(P, C, D)
    PDA = area_triangle(P, D, A)

    ABCD = area_triangle(A, B, C) + area_triangle(A, C, D)

    # If P is inside the rectangle, then the sum of the areas of the 
    # triangles PAB, PBC, PCD, PDA will be equal to the area of the rectangle.
    # Otherwise that sum will be greater than the area of the rectangle

    return torch.abs(PAB + PBC + PCD + PDA - ABCD) < 1e-5

#@torch.jit.script
def obb_obb(posA, dimsA, quatA, posB, dimsB, quatB):
    ## type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # Pytorch parallelized implementation of separating axis test for oriented bounding boxes
    # adapted from pseudocode in real-time collision detection textbook, chapter 4.4

    # pos: center of box
    # dims: positive half-widths of box
    
    eps = 1e-5

    # Get the rotation of obb B in the coordinate frame of obb A
    quatAB = torch_util.quat_unit(torch_util.quat_multiply(torch_util.quat_conjugate(quatA), quatB))
    R = torch_util.quat_to_matrix(quatAB)
    absR = torch.abs(R) + eps # eps for parallel edges case

    # For OBB-OBB intersection, there are 15 axes to test
    # 3 from the coordinate axes of OBB A, 3 from the coordinate axes of OBB B
    # and 9 from the cross product of the coordinate axes of A and B

    # compute translation vector relative to A's coordinate frame
    t = posB - posA
    t = torch_util.quat_rotate(torch_util.quat_inv(quatA), t)

    sep_checks = []
    rs = []
    axis_lens = []

    # box A's face axes
    for i in range(3):
        ra = dimsA[..., i]
        rb = absR[..., i, 0] * dimsB[..., 0] + absR[..., i, 1] * dimsB[..., 1] + absR[..., i, 2] * dimsB[..., 2]
        r = ra+rb
        axis_len = torch.abs(t[..., i])
        rs.append(r)
        axis_lens.append(axis_len)
        
        sep_checks.append(r < axis_len)

    # box B's face axes
    for i in range(3):
        ra = dimsA[..., 0] * absR[..., 0, i] + dimsA[..., 1] * absR[..., 1, i] + dimsA[..., 2] * absR[..., 2, i]
        rb = dimsB[..., i]
        r = ra + rb
        axis_len = torch.abs(t[..., 0] * R[..., 0, i] + t[..., 1] * R[..., 1, i] + t[..., 2] * R[..., 2, i])

        rs.append(r)
        axis_lens.append(axis_len)

        sep_checks.append(r < axis_len)

    # Test axis L = A0 x B0
    ra = dimsA[..., 1] * absR[..., 2, 0] + dimsA[..., 2] * absR[..., 1, 0]
    rb = dimsB[..., 1] * absR[..., 0, 2] + dimsB[..., 2] * absR[..., 0, 1]
    r = ra + rb
    axis_len = torch.abs(t[..., 2] * R[..., 1, 0] - t[..., 1] * R[..., 2, 0])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    # Test axis L = A0 x B1
    ra = dimsA[..., 1] * absR[..., 2, 1] + dimsA[..., 2] * absR[..., 1, 1]
    rb = dimsB[..., 0] * absR[..., 0, 2] + dimsB[..., 2] * absR[..., 0, 0]
    r = ra + rb
    axis_len = torch.abs(t[..., 2] * R[..., 1, 1] - t[..., 1] * R[..., 2, 1])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    # Test axis L = A0 x B2
    ra = dimsA[..., 1] * absR[..., 2, 2] + dimsA[..., 2] * absR[..., 1, 2]
    rb = dimsB[..., 0] * absR[..., 0, 1] + dimsB[..., 1] * absR[..., 0, 0]
    r = ra + rb
    axis_len = torch.abs(t[..., 2] * R[..., 1, 2] - t[..., 1] * R[..., 2, 2])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    # Test axis L = A1 x B0
    ra = dimsA[..., 0] * absR[..., 2, 0] + dimsA[..., 2] * absR[..., 0, 0]
    rb = dimsB[..., 1] * absR[..., 1, 2] + dimsB[..., 2] * absR[..., 1, 1]
    r = ra + rb
    axis_len = torch.abs(t[..., 0] * R[..., 2, 0] - t[..., 2] * R[..., 0, 0])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    # Test axis L = A1 x B1
    ra = dimsA[..., 0] * absR[..., 2, 1] + dimsA[..., 2] * absR[..., 0, 1]
    rb = dimsB[..., 0] * absR[..., 1, 2] + dimsB[..., 2] * absR[..., 1, 0]
    r = ra + rb
    axis_len = torch.abs(t[..., 0] * R[..., 2, 1] - t[..., 2] * R[..., 0, 1])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    # Test axis L = A1 x B2
    ra = dimsA[..., 0] * absR[..., 2, 2] + dimsA[..., 2] * absR[..., 0, 2]
    rb = dimsB[..., 0] * absR[..., 1, 1] + dimsB[..., 1] * absR[..., 1, 0]
    r = ra + rb
    axis_len = torch.abs(t[..., 0] * R[..., 2, 2] - t[..., 2] * R[..., 0, 2])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    # Test axis L = A2 x B0
    ra = dimsA[..., 0] * absR[..., 1, 0] + dimsA[..., 1] * absR[..., 0, 0]
    rb = dimsB[..., 1] * absR[..., 2, 2] + dimsB[..., 2] * absR[..., 2, 1]
    r = ra + rb
    axis_len = torch.abs(t[..., 1] * R[..., 0, 0] - t[..., 0] * R[..., 1, 0])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    # Test axis L = A2 x B1
    ra = dimsA[..., 0] * absR[..., 1, 1] + dimsA[..., 1] * absR[..., 0, 1]
    rb = dimsB[..., 0] * absR[..., 2, 2] + dimsB[..., 2] * absR[..., 2, 0]
    r = ra + rb
    axis_len = torch.abs(t[..., 1] * R[..., 0, 1] - t[..., 0] * R[..., 1, 1])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    # Test axis L = A2 x B2
    ra = dimsA[..., 0] * absR[..., 1, 2] + dimsA[..., 1] * absR[..., 0, 2]
    rb = dimsB[..., 0] * absR[..., 2, 1] + dimsB[..., 1] * absR[..., 2, 0]
    r = ra + rb
    axis_len = torch.abs(t[..., 1] * R[..., 0, 2] - t[..., 0] * R[..., 1, 2])
    rs.append(r)
    axis_lens.append(axis_len)
    sep_checks.append(r < axis_len)

    rs = torch.stack(rs, dim=-1)
    axis_lens = torch.stack(axis_lens, dim=-1)
    sep_checks = torch.stack(sep_checks, dim=-1)
    #print(sep_checks)
    return sep_checks.any(dim=-1), rs, axis_lens

def obb_SAT(posA, quatA, dimsA, posB, quatB, dimsB,
            ret_debug_info=False):
    ## type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor

    # A less code-optimized version of the OBB separating axis test,
    # but more mathematically readable.
    # This function will also return the potential separating axis vectors.
    """
    posA: (N, 3)
    quatA: (N, 4)
    dimsA: (N, 3)
    posB: (M, 3)
    quatB: (M, 4)
    dimsB: (M, 3)
    """
    
    ## Assume first dimension is the batch dimension
    assert len(posA.shape) == 2
    batch_size_A = posA.shape[0]
    batch_size_B = posB.shape[0]


    device = posA.device

    t = posB.unsqueeze(0) - posA.unsqueeze(1)
    #t = torch_util.quat_rotate(torch_util.quat_inv(quatA), t)

    # Get the rotation of obb B in the coordinate frame of obb A
    #quatAB = torch_util.quat_unit(torch_util.quat_multiply(torch_util.quat_conjugate(quatA), quatB))
    #R = torch_util.quat_to_matrix(quatAB)

    ## Potential separating axes
    # The column vectors are the axes of the rotation matrix.
    # To make sure the axes are found in the last dimension of the tensor, we transpose
    uA = torch.transpose(torch_util.quat_to_matrix(quatA), -2, -1)
    uB = torch.transpose(torch_util.quat_to_matrix(quatB), -2, -1)

    #uA0_x_uB0 = torch.cross(uA[..., 0, :], uB[..., 0, :])

    uA_x_uB = torch.cross(uA.unsqueeze(-2).unsqueeze(1), uB.unsqueeze(-3).unsqueeze(0), dim=-1)
    uA_x_uB = uA_x_uB.view(batch_size_A, batch_size_B, -1, 3)
    #uA_x_uB = torch.cross(uA.unsqueeze(1), uB.unsqueeze(0), dim=-1)

    uA = uA.unsqueeze(1).expand(size=[-1, batch_size_B, -1, -1])
    uB = uB.unsqueeze(0).expand(size=[batch_size_A, -1, -1, -1])
    axes = torch.cat([uA, uB, uA_x_uB], dim=2)
    axis_lengths = torch.norm(axes, dim=-1).unsqueeze(-1)
    edge_case_axis = torch.zeros_like(axes)
    edge_case_axis[..., 0] = 1.0
    axes = torch.where(axis_lengths > 1e-5, axes / axis_lengths, edge_case_axis)

    
    ## PROJECTING ONTO AXIS
    # for each axis, rotate the points of the box with the axis
    # such that the axis is [1, 0, 0]
    # Then compute the max and min points of the boxes along the x-axis

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    inv_axes_rots = torch_util.quat_diff_vec(axes, x_axis)

    pointsA = get_box_points_batch(posA, quatA, dimsA, torch.zeros_like(posA)) # (N, 8, 3)
    pointsA = pointsA.unsqueeze(1).unsqueeze(3) # (N, 1, 8, 1, 3)
    pointsB = get_box_points_batch(posB, quatB, dimsB, torch.zeros_like(posB)) # (M, 8, 3)
    pointsB = pointsB.unsqueeze(0).unsqueeze(3) # (1, M, 8, 1, 3)
    #rA = torch.dot(dimsA.unsqueeze(-1) * L)

    rotated_pointsA = torch_util.quat_rotate(inv_axes_rots.unsqueeze(2), pointsA) # (N, M, 8, 15, 3)
    max_proj_A, _ = torch.max(rotated_pointsA[..., 0], dim=2) # (N, M, 15)
    min_proj_A, _ = torch.min(rotated_pointsA[..., 0], dim=2) # (N, M, 15)
    
    rA = (max_proj_A - min_proj_A) / 2.0

    rotated_pointsB = torch_util.quat_rotate(inv_axes_rots.unsqueeze(2), pointsB) # (N, M, 8, 15, 3)
    max_proj_B, _ = torch.max(rotated_pointsB[..., 0], dim=2) # (N, M, 15)
    min_proj_B, _ = torch.min(rotated_pointsB[..., 0], dim=2) # (N, M, 15)
    rB = (max_proj_B - min_proj_B) / 2.0

    abs_t_dot_L = torch.abs(torch.sum(t.unsqueeze(2) * axes, dim=-1))

    sep_val = abs_t_dot_L - (rA + rB)

    if ret_debug_info:
        info = dict()
        info["abs_t_dot_L"] = abs_t_dot_L
        info["max_proj_A"] = max_proj_A
        info["min_proj_A"] = min_proj_A
        info["max_proj_B"] = max_proj_B
        info["min_proj_B"] = min_proj_B
        info["sep_val"] = sep_val
        info["axes"] = axes
        return info
    else:
        return sep_val
    

def get_obbs_from_motion(root_pos: torch.Tensor, 
                         root_rot: torch.Tensor, 
                         joint_rot: torch.Tensor, 
                         char_model: kin_char_model.KinCharModel):
    
    device = root_pos.device

    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    char_obb_pos = []
    char_obb_rot = []
    char_obb_dims = []

    body_ids = []

    num_frames = root_pos.shape[0]

    # Get the obbs for the box geometries on the character
    # NOTE: assumes only 1 box per body
    for b in range(char_model.get_num_joints()):
        geoms = char_model.get_geoms(b)
        for geom in geoms:
            if geom._shape_type == kin_char_model.GeomType.SPHERE: # skip spheres for now
                continue
            if geom._shape_type == kin_char_model.GeomType.CAPSULE: # skip capsules for now
                continue
            offset = geom._offset
            dims = geom._dims.unsqueeze(0).repeat(num_frames, 1)

            pos = body_pos[:, b] + torch_util.quat_rotate(body_rot[:, b], offset.unsqueeze(0))
            rot = body_rot[:, b]

            char_obb_pos.append(pos)
            char_obb_rot.append(rot)
            char_obb_dims.append(dims)

            body_ids.append(b)

            # TODO: visualize the OBBs as transparent OBBs

    char_obb_pos = torch.cat(char_obb_pos, dim=0)
    char_obb_rot = torch.cat(char_obb_rot, dim=0)
    char_obb_dims = torch.cat(char_obb_dims, dim=0)

    body_ids = torch.tensor(body_ids, dtype=torch.int64, device=device)
    return char_obb_pos, char_obb_rot, char_obb_dims, body_ids


def np_euler2mat(x, y, z):
    # x, y, z: euler angles as degrees
    x = x * np.pi / 180.0
    y = y * np.pi / 180.0
    z = z * np.pi / 180.0


    cx = np.cos(x)
    cy = np.cos(y)
    cz = np.cos(z)
    sx = np.sin(x)
    sy = np.sin(y)
    sz = np.sin(z)

    mat = np.eye(3)
    mat[0, :] = [cy * cz, -cy * sz, sy]
    mat[1, :] = [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx]
    mat[2, :] = [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy]
    return mat

def is_point_in_capsule(x, y, z, h, r):
    # Capsule axis along the z-axis

    def sdVerticalCapsule(x, y, z, h, r):
        z = z - torch.clamp(z, -torch.ones_like(z) * h / 2, torch.ones_like(z) * h / 2)
        return x**2 + y**2 + z**2 - r*r
    
    return sdVerticalCapsule(x, y, z, h, r) <= 0.0
    #d = torch.sqrt(x**2 + y**2)
    
    # # Check if point is within the cylindrical part
    # inside_cylinder = (d <= R) & (z.abs() <= L / 2)
    
    # # Check if point is within the hemispherical caps
    # # Cap 1: center at (0, 0, L/2)
    # cap1 = (d <= R) & (z > L / 2) & ((x**2 + y**2 + (z - L / 2)**2) <= R**2)
    
    # # Cap 2: center at (0, 0, -L/2)
    # cap2 = (d <= R) & (z < -L / 2) & ((x**2 + y**2 + (z + L / 2)**2) <= R**2)
    
    #return inside_cylinder | cap1 | cap2

def is_point_in_box(x, y, z, halfdims):

    check_x = (-halfdims[0] <= x) & (x <= halfdims[0])
    check_y = (-halfdims[1] <= y) & (y <= halfdims[1])
    check_z = (-halfdims[2] <= z) & (z <= halfdims[2])

    return check_x & check_y & check_z

def is_point_in_sphere(x, y, z, r):

    return torch.square(x) + torch.square(y) + torch.square(z) <= r*r

def generate_centered_3D_grid_points(grid_bounds, dx, device="cpu"):

    x = torch.arange(-grid_bounds, grid_bounds + dx, dx, device=device)
    y = torch.arange(-grid_bounds, grid_bounds + dx, dx, device=device)
    z = torch.arange(-grid_bounds, grid_bounds + dx, dx, device=device)
    
    # Create meshgrid and flatten
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    grid_z = grid_z.flatten()
    
    return grid_x, grid_y, grid_z

def get_capsule_volume_samples(
    capsule_length,  # Length of the capsule
    capsule_radius,   # Radius of the capsule
    grid_bounds,  # Grid bounds
    dx,   # Grid spacing
):
    # Generate grid points
    grid_x, grid_y, grid_z = generate_centered_3D_grid_points(grid_bounds, dx)

    # Filter points inside the capsule
    inside_capsule = is_point_in_capsule(grid_x, grid_y, grid_z, capsule_length, capsule_radius)
    filtered_points = torch.stack([grid_x[inside_capsule], grid_y[inside_capsule], grid_z[inside_capsule]], dim=1)

    return filtered_points

def get_box_point_volume_samples(
    box_halfdims,
    grid_bounds,
    dx
):
    # Generate grid points
    grid_x, grid_y, grid_z = generate_centered_3D_grid_points(grid_bounds, dx)

    # Filter points inside the box
    inside_box = is_point_in_box(grid_x, grid_y, grid_z, box_halfdims)
    filtered_points = torch.stack([grid_x[inside_box], grid_y[inside_box], grid_z[inside_box]], dim=1)

    return filtered_points

def get_sphere_point_volume_samples(
    radius,
    grid_bounds,
    dx
):
    # Generate grid points
    grid_x, grid_y, grid_z = generate_centered_3D_grid_points(grid_bounds, dx)

    # Filter points inside the capsule
    inside_sphere = is_point_in_sphere(grid_x, grid_y, grid_z, radius)
    filtered_points = torch.stack([grid_x[inside_sphere], grid_y[inside_sphere], grid_z[inside_sphere]], dim=1)

    return filtered_points


def get_box_point_surface_samples(
    box_halfdims,
    device,
    num_slices = 2,
    dim_x = 6,
    dim_y = 3
):
    z = torch.linspace(0.0, 1.0, num_slices, device=device) * box_halfdims[2] * 2.0 - box_halfdims[2]
    x = torch.linspace(0.0, 1.0, dim_x, device=device) * box_halfdims[0] * 2.0 - box_halfdims[0]
    y = torch.linspace(0.0, 1.0, dim_y, device=device) * box_halfdims[1] * 2.0 - box_halfdims[1]
    x, y = torch.meshgrid(x, y, indexing='ij') # [dim_x, dim_y]

    box_points = torch.cat([x.reshape(-1).unsqueeze(0).unsqueeze(-1).expand(num_slices, dim_x * dim_y, 1), 
                            y.reshape(-1).unsqueeze(0).unsqueeze(-1).expand(num_slices, dim_x * dim_y, 1), 
                            z.unsqueeze(-1).unsqueeze(-1).expand(num_slices, dim_x * dim_y, 1)], dim = -1)
    return box_points.reshape(-1, 3)

def get_sphere_point_surface_samples(
    radius,
    device,
    num_subdivisions = 0,
):
    icosphere_trimesh = trimesh.creation.icosphere(subdivisions=num_subdivisions, radius=radius)
    icosphere_verts = icosphere_trimesh.vertices
    
    return torch.from_numpy(icosphere_verts).to(device=device, dtype=torch.float32)

def get_capsule_point_surface_samples(
    capsule_length,  # Length of the capsule
    capsule_radius,   # Radius of the capsule
    device,
    num_cylinder_slices = 3,
    num_circle_points = 4,
    num_sphere_subdivisons = 0,
    ignore_hemispheres = True
):
    capsule_points = []
    if not ignore_hemispheres:
        sphere_points = get_sphere_point_surface_samples(capsule_radius, device, num_subdivisions=num_sphere_subdivisons)
        
        upper_hemisphere_points = sphere_points[sphere_points[..., 2] > 1e-5].clone()
        upper_hemisphere_points[..., 2] += capsule_length / 2.0
        lower_hemisphere_points = sphere_points[sphere_points[..., 2] < -1e-5].clone()
        lower_hemisphere_points[..., 2] -= capsule_length / 2.0
        capsule_points.append(upper_hemisphere_points)
        capsule_points.append(lower_hemisphere_points)

    z = torch.linspace(0.0, 1.0, num_cylinder_slices, device=device) * capsule_length - capsule_length / 2.0
    # shape: [num_cylinder_slices]
    
    theta = torch.linspace(0, 2 * torch.pi, num_circle_points + 1, device=device)[:-1] # ignore last duplicate point
    
    # Compute x and y coordinates
    x = capsule_radius * torch.cos(theta) # shape: [num_circle_points]
    y = capsule_radius * torch.sin(theta) # shape: [num_circle_points]
    middle_points = torch.cat([x.unsqueeze(-1).unsqueeze(-1).expand(num_circle_points, num_cylinder_slices, 1), 
                               y.unsqueeze(-1).unsqueeze(-1).expand(num_circle_points, num_cylinder_slices, 1), 
                               z.unsqueeze(0).unsqueeze(-1).expand(num_circle_points, num_cylinder_slices, 1)], dim=-1)
    middle_points = middle_points.view(-1, 3)
    capsule_points.append(middle_points)
    capsule_points = torch.cat(capsule_points, dim=0)
    return capsule_points

def get_char_point_samples(char_model: kin_char_model.KinCharModel,
                           sphere_num_subdivisions = 0,
                           box_num_slices = 2,
                           box_dim_x = 3,
                           box_dim_y = 6,
                           capsule_num_circle_points = 4,
                           capsule_num_sphere_subdivisons = 0,
                           capsule_num_cylinder_slices = 4):
    all_points = []
    device = char_model._device
    total_num_points = 0
    for b in range(0, char_model.get_num_joints()):
        geoms = char_model.get_geoms(b)

        curr_body_points = []
        for geom in geoms:
            #trimesh.primitives.Capsule()
            if geom._shape_type == kin_char_model.GeomType.SPHERE:
                r = geom._dims.item()
                offset = geom._offset
                
                #sphere_points = get_sphere_point_volume_samples(r, r * 1.5, dx)
                sphere_points = get_sphere_point_surface_samples(r, device=device,
                                                                 num_subdivisions=sphere_num_subdivisions)
                sphere_points = sphere_points + offset
                curr_body_points.append(sphere_points)
            elif geom._shape_type == kin_char_model.GeomType.BOX:
                halfdims = geom._dims
                offset = geom._offset
                #largest_halfdim = torch.max(halfdims).item()
                #box_points = get_box_point_volume_samples(halfdims, largest_halfdim * 1.5, dx)
                box_points = get_box_point_surface_samples(halfdims, device,
                                                           num_slices=box_num_slices,
                                                           dim_x=box_dim_x,
                                                           dim_y=box_dim_y)
                box_points = box_points + offset
                curr_body_points.append(box_points)
            elif geom._shape_type == kin_char_model.GeomType.CAPSULE:
                fromto = geom._dims
                offset = geom._offset + fromto/2.0 # because capsules start at their centers

                z_axis = torch.tensor([0.0, 0.0, 1.0], device=char_model._device)
                axis = torch.cross(z_axis, geom._dims)
                if (torch.norm(axis) < 1e-5):
                    axis = z_axis
                else:
                    axis = axis / torch.norm(axis)

                # Compute rotation angle using dot product
                angle = torch.acos(torch.dot(axis, geom._dims))

                rotation = torch_util.axis_angle_to_quat(axis, angle)
                #transform[:3, :3] = rotation.cpu().numpy()


                h = torch.norm(fromto).item()
                radius = geom._radius
                #capsule_points = get_capsule_volume_samples(h, radius, h * 1.5, dx)
                capsule_points = get_capsule_point_surface_samples(h, radius, 
                                                                   num_circle_points=capsule_num_circle_points, 
                                                                   num_sphere_subdivisons=capsule_num_sphere_subdivisons,
                                                                   num_cylinder_slices=capsule_num_cylinder_slices,
                                                                   device=device)
                capsule_points = torch_util.quat_rotate(rotation.unsqueeze(0), capsule_points)
                capsule_points = capsule_points + offset

                curr_body_points.append(capsule_points)
            elif geom._shape_type == kin_char_model.GeomType.CYLINDER:
                # TODO
                curr_body_points.append(torch.zeros(size=[1, 3], dtype=torch.float32, device=device))
            elif geom._shape_type == kin_char_model.GeomType.MESH:
                # TODO
                curr_body_points.append(torch.zeros(size=[1, 3], dtype=torch.float32, device=device))

        if len(geoms) == 0:
            zero_point = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
            curr_body_points.append(zero_point.unsqueeze(0))

        curr_body_points = torch.cat(curr_body_points, dim=0)
        total_num_points += curr_body_points.shape[0]
        all_points.append(curr_body_points)
    #print("total num sampled char points =", total_num_points)
    return all_points


def get_minimal_char_point_samples(char_model: kin_char_model.KinCharModel):

    all_points = []
    device = char_model._device
    total_num_points = 0
    for b in range(0, char_model.get_num_joints()):
        geoms = char_model.get_geoms(b)

        curr_body_points = []
        for geom in geoms:
            #trimesh.primitives.Capsule()
            if geom._shape_type == kin_char_model.GeomType.SPHERE:
                r = geom._dims.item()
                offset = geom._offset
                
                sphere_points = offset.clone().unsqueeze(0)
                curr_body_points.append(sphere_points)
            elif geom._shape_type == kin_char_model.GeomType.CAPSULE:
                fromto = geom._dims
                offset = geom._offset + fromto/2.0 # because capsules start at their centers

                z_axis = torch.tensor([0.0, 0.0, 1.0], device=char_model._device)
                axis = torch.cross(z_axis, geom._dims)
                if (torch.norm(axis) < 1e-5):
                    axis = z_axis
                else:
                    axis = axis / torch.norm(axis)

                # Compute rotation angle using dot product
                angle = torch.acos(torch.dot(axis, geom._dims))

                rotation = torch_util.axis_angle_to_quat(axis, angle)
                #transform[:3, :3] = rotation.cpu().numpy()


                h = torch.norm(fromto).item()
                #radius = geom._radius
                #capsule_points = get_capsule_volume_samples(h, radius, h * 1.5, dx)
                capsule_points = torch.tensor([[0.0, 0.0, h/3.0], [0.0, 0.0, -h/3.0]], dtype=torch.float32, device=device)
                capsule_points = torch_util.quat_rotate(rotation.unsqueeze(0), capsule_points)
                capsule_points = capsule_points + offset

                curr_body_points.append(capsule_points)
            elif geom._shape_type == kin_char_model.GeomType.BOX:
                halfdims = geom._dims
                offset = geom._offset
                box_points = get_box_point_surface_samples(halfdims, device,
                                                           num_slices=2,
                                                           dim_x=2,
                                                           dim_y=2)
                box_points = box_points + offset
                curr_body_points.append(box_points)
            else:
                assert False

        curr_body_points = torch.cat(curr_body_points, dim=0)
        total_num_points += curr_body_points.shape[0]
        all_points.append(curr_body_points)
    #print("total num sampled char points =", total_num_points)
    return all_points