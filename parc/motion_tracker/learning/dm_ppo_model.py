import gym
import numpy as np
import torch

import parc.motion_tracker.learning.base_model as base_model
import parc.motion_tracker.learning.distribution_gaussian_diag as distribution_gaussian_diag
import parc.motion_tracker.learning.ppo_model as ppo_model
import parc.util.nets.net_builder as net_builder
import parc.util.torch_util as torch_util


class DMPPOModel(ppo_model.PPOModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        return
    
    def _build_action_distribution_dm_ViT(self, config, env, in_size):
        a_space = env.get_action_space()
        
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

    def _build_nets(self, config, env):

        if config["actor_net"][0:6] == "dm_vit":
            assert config["actor_net"] == config["critic_net"]

            input_dict = {
                "obs_shapes": env._compute_obs(ret_obs_shapes=True),
                "device": env._device
            }
            for key in config:
                input_dict[key] = config[key]
            
            self._dm_vit, info = net_builder.build_net(config["actor_net"],
                                                 input_dict,
                                                 self._activation)

            self._actor_layers = self._dm_vit.forward_actor
            self._critic_layers = self._dm_vit.forward_critic


            self._action_dist = self._build_action_distribution_dm_ViT(config, env, self._dm_vit.get_out_token_dim())

            self._critic_out = torch.nn.Linear(self._dm_vit.get_out_token_dim(), 1)
            torch.nn.init.zeros_(self._critic_out.bias)
        elif config["actor_net"] == "dm_cnn_mlp":

            assert config["actor_net"] == config["critic_net"]

            input_dict = {
                "obs_shapes": env._compute_obs(ret_obs_shapes=True),
                "device": env._device
            }
            for key in config:
                input_dict[key] = config[key]

            self._cnn_mlp, info = net_builder.build_net(config["actor_net"],
                                                 input_dict,
                                                 self._activation)

            self._actor_layers = self._cnn_mlp.forward_actor
            self._critic_layers = self._cnn_mlp.forward_critic


            self._action_dist = self._build_action_distribution_dm_ViT(config, env, self._cnn_mlp._actor_out_dim)

            self._critic_out = torch.nn.Linear(self._cnn_mlp._critic_out_dim, 1)
            torch.nn.init.zeros_(self._critic_out.bias)

        else:
            super()._build_nets(config, env)

        return