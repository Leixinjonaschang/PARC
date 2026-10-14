import numpy as np
import torch

import parc.motion_tracker.envs.base_env as base_env


def compute_td_lambda_return(r, next_vals, done, discount, td_lambda):
    assert(r.shape == next_vals.shape)

    return_t = torch.zeros_like(r)
    reset_mask = done != base_env.DoneFlags.NULL.value
    reset_mask = reset_mask.type(torch.float)

    last_val = r[-1] + discount * next_vals[-1]
    return_t[-1] = last_val

    timesteps = r.shape[0]
    for i in reversed(range(0, timesteps - 1)):
        curr_r = r[i]
        curr_reset = reset_mask[i]
        next_v = next_vals[i]
        next_ret = return_t[i + 1]

        curr_lambda = td_lambda * (1.0 - curr_reset)
        curr_val = curr_r + discount * ((1.0 - curr_lambda) * next_v + curr_lambda * next_ret)
        return_t[i] = curr_val

    #_debug_td_lambda(r, next_vals, done, discount, td_lambda, return_t)

    return return_t

def _debug_td_lambda(r, next_vals, done, discount, td_lambda, ref_ret):
    # brute force td lambda calculation to check that the dynamic programming implementation is correct
    r = r.cpu().numpy()
    next_vals = next_vals.cpu().numpy()
    done = done.cpu().numpy()
    ref_ret = ref_ret.cpu().numpy()
    return_t = np.zeros_like(r)

    reset_mask = done != base_env.DoneFlags.NULL.value
    reset_mask = reset_mask.astype(np.float32)

    timesteps = r.shape[0]
    batch_size = r.shape[1]
    for i in range(batch_size):
        for t0 in range(timesteps):
            new_val = 0.0
            sum_r = 0.0
            curr_discount = 1.0
            curr_lambda = 1.0

            for t in range(t0, timesteps):
                curr_r = r[t, i]
                curr_reset = reset_mask[t, i]
                next_v = next_vals[t, i]

                sum_r += curr_discount * curr_r
                curr_val = sum_r + curr_discount * discount * next_v

                if (curr_reset == 0.0 and t < timesteps - 1):
                    new_val += (1 - td_lambda) * curr_lambda * curr_val
                else:
                    new_val += curr_lambda * curr_val
                    break
                
                curr_discount *= discount
                curr_lambda *= td_lambda

            return_t[t0, i] = new_val

    ret_diff = return_t - ref_ret
    max_ret_err = np.max(np.abs(ret_diff))

    return