import torch
import parc.util.torch_util as torch_util

def fast_negexpf(x):
    return 1.0 / (1.0 + x + 0.48*x*x + 0.235*x*x*x)

def halflife_to_damping(halflife, eps = 1e-5):
    return 4.0 * 0.69314718056 / (halflife + eps)

def damping_to_halflife(damping, eps = 1e-5):
    return 4.0 * 0.69314718056 / (damping + eps)

def decayed_offset(x, v, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    j1 = v + x*y
    eydt = fast_negexpf(y * dt)
    return eydt * (x + j1 * dt)

def decayed_offset_cubic(
    x, # Initial Position
    v, # Initial Velocity
    blendtime, dt, eps=1e-8
):
    t = np.clip(dt / (blendtime + eps), a_min=0.0, a_max=1.0)

    d = x
    c = v * blendtime
    b = -3*d - 2*c
    a = 2*d + c
    
    return a*t*t*t + b*t*t + c*t + d

def intertialize_frames(motion_frames: torch.Tensor, ratio: float, 
                        halflife_start: float, halflife_end: float, dt: float):
    ## First compute difference between first and last frame

    dofs = motion_frames[:, 3:]

    diff_pos = dofs[-1] - dofs[0]
    diff_vel = (dofs[-1] - dofs[-2]) / dt - (dofs[1] - dofs[0]) / dt

    # loop over every frame
    num_frames = dofs.shape[0]
    offsets = []
    for i in range(num_frames):
        curr_offset = decayed_offset(ratio * diff_pos, ratio * diff_vel, halflife_start, i * dt) +\
                        decayed_offset((1.0-ratio) * -diff_pos, (1.0-ratio) * diff_vel, halflife_end, (num_frames - 1 - i) * dt)
        offsets.append(curr_offset)

    offsets = torch.stack(offsets)

    # apply offsets
    ret = torch.cat([motion_frames[:, 0:3], dofs + offsets], dim=-1)

    return ret

def calc_inertialize_both_offsets(diff_pos, diff_vel, num_frames: int,
                                  ratio: float, halflife_start: float, halflife_end: float, blendtime: float, dt: float):
    offsets = []
    for i in range(num_frames):
        # curr_offset = decayed_offset(ratio * diff_pos, ratio * diff_vel, halflife_start, i * dt) +\
        #                 decayed_offset((1.0-ratio) * -diff_pos, (1.0-ratio) * diff_vel, halflife_end, (num_frames - 1 - i) * dt)

        curr_offset = decayed_offset_cubic(ratio * diff_pos, ratio * diff_vel, blendtime, i * dt) +\
                         decayed_offset_cubic((1.0-ratio) * -diff_pos, (1.0-ratio) * diff_vel, blendtime, (num_frames - 1 - i) * dt)
        offsets.append(curr_offset)
    offsets = torch.stack(offsets)

    return offsets

def calc_loop_inertialization(body_pos, body_rot, 
                              ratio: float, halflife_start: float, halflife_end: float, 
                              blendtime: float, dt: float):
    
    first_body_pos = body_pos[0:1].clone()
    first_body_rot = body_rot[0:1].clone()

    #body_pos = body_pos[:-1]
    #body_rot = body_rot[:-1]

    diff_pos = body_pos[-1] - body_pos[0]
    diff_pos[:, 0] = 0.0 # x is the axis of symmetry
    diff_vel = (body_pos[-1] - body_pos[-2]) / dt - (body_pos[1] - body_pos[0]) / dt

    # quat_diff ordering is opposite
    diff_rot = torch_util.quat_to_exp_map(torch_util.quat_diff(body_rot[0], body_rot[-1]))

    diff_ang_vel = torch_util.quat_differentiate_angular_velocity(body_rot[-1], body_rot[-2], dt) -\
                torch_util.quat_differentiate_angular_velocity(body_rot[1], body_rot[0], dt)
    
    num_frames = body_pos.shape[0]
    
    pos_offsets = calc_inertialize_both_offsets(diff_pos, diff_vel, num_frames, ratio, halflife_start, halflife_end, blendtime, dt)
    rot_offsets = calc_inertialize_both_offsets(diff_rot, diff_ang_vel, num_frames, ratio, halflife_start, halflife_end, blendtime, dt)

    new_body_pos = body_pos + pos_offsets
    new_body_rot = torch_util.quat_mul(torch_util.exp_map_to_quat(rot_offsets), body_rot)

    # ensure first and last frame are the same # TODO add little x-positional offset here
    #first_body_pos[0, :, 0] += body_pos[-1, :, 0]
    #new_body_pos = torch.cat([new_body_pos, first_body_pos], dim=0)
    #new_body_rot = torch.cat([new_body_rot, first_body_rot], dim=0)

    return new_body_pos, new_body_rot