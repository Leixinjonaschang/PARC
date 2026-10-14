import gym
import numpy as np
import torch

import parc.motion_tracker.learning.distribution_gaussian_diag as distribution_gaussian_diag
import parc.util.torch_util as torch_util


class BaseModel(torch.nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self._activation = torch.nn.ReLU
        return

    def _build_action_distribution(self, config, env, input):
        a_space = env.get_action_space()
        in_size = torch_util.calc_layers_out_size(input)
        
        if (isinstance(a_space, gym.spaces.Box)):
            a_size = np.prod(a_space.shape)
            a_init_output_scale = config["actor_init_output_scale"]
            a_std_type = distribution_gaussian_diag.StdType[config["actor_std_type"]]
            a_std = config["action_std"]
            a_dist = distribution_gaussian_diag.DistributionGaussianDiagBuilder(in_size, a_size, std_type=a_std_type,
                                                                            init_std=a_std, init_output_scale=a_init_output_scale)
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)

        return a_dist