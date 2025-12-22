import torch
import torch.nn as nn

import gym
import numpy as np

import learning.nets.net_builder as net_builder
import learning.distribution_gaussian_diag as distribution_gaussian_diag
import learning.distribution_categorical as distribution_categorical

import learning.ppo_model as ppo_model

import learning.obs_encoder as obs_encoder

class DMPPOModel(ppo_model.PPOModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        self._predictor_test_mode = False
        self._build_future_pose_predictor(config)
        return

    def set_predictor_test_mode(self, enabled: bool):
        self._predictor_test_mode = enabled
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
        elif (isinstance(a_space, gym.spaces.Discrete)):
            num_actions = a_space.n
            a_init_output_scale = config["actor_init_output_scale"]
            a_dist = distribution_categorical.DistributionCategoricalBuilder(in_size, num_actions, 
                                                                             init_output_scale=a_init_output_scale)
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)

        return a_dist

    def _build_nets(self, config, env):
        self.obs_encoder = obs_encoder.ObservationEncoder(config, env, env._device)

        if config["actor_net"][0:6] == "dm_vit":
            assert config["actor_net"] == config["critic_net"]

            input_dict = self.obs_encoder.get_net_builder_input_dict()
            for key in config:
                if key not in ["obs_shapes", "obs_dim"]:
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

            input_dict = self.obs_encoder.get_net_builder_input_dict()
            for key in config:
                if key not in ["obs_shapes", "obs_dim"]:
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

    def _build_actor_input_dict(self, env):
        if hasattr(self, "obs_encoder"):
            obs_space = self.obs_encoder.get_mock_obs_space()
            return {"obs": obs_space}
        return super()._build_actor_input_dict(env)

    def _build_critic_input_dict(self, env):
        if hasattr(self, "obs_encoder"):
            obs_space = self.obs_encoder.get_mock_obs_space()
            return {"obs": obs_space}
        return super()._build_critic_input_dict(env)

    def eval_actor(self, obs):
        if self._predictor_test_mode:
            predictor_input = self._get_predictor_input(obs)
            predicted_latent = self._predictor(predictor_input)
            processed_obs = self.obs_encoder(obs, overrides={"tar_obs": predicted_latent})
        else:
            processed_obs = self.obs_encoder(obs)
            
        return super(DMPPOModel, self).eval_actor(processed_obs)

    def eval_critic(self, obs):
        processed_obs = self.obs_encoder(obs)
        return super().eval_critic(processed_obs)

    def _build_future_pose_predictor(self, config):
        # Predictor Input size: Sum of raw dims of all obs EXCEPT "tar_obs"
        raw_obs_shapes = self.obs_encoder.get_obs_shapes()
        input_dim = 0
        target_dim = 0
        predictor_config = config.get("future_pose_predictor", {})
        predictor_hidden_dims = predictor_config.get("hidden_dims", [512, 512, 256])

        # Compute the predictor input and target dimensions
        for key, info in raw_obs_shapes.items():
            if key == "tar_obs":
                target_latent_info = self.obs_encoder.get_output_obs_shapes()[key]
                target_dim = int(np.prod(target_latent_info["shape"]))
            else:
                input_dim += int(np.prod(info["shape"]))
        assert target_dim != 0, "tar_obs not found in observation shapes. Predictor cannot be built."

        
        self._predictor = obs_encoder.MLPEncoder(
            input_dim=input_dim,
            output_dim=target_dim,
            hidden_dims=predictor_hidden_dims,
            activation=self._activation
        )
        self._predictor.to(self.obs_encoder._device)
        
        # lr = predictor_config.get("lr", 1e-4) 
        # print(f"debug: Future pose predictor learning rate: {lr}")
        self.predictor_optimizer = torch.optim.Adam(self._predictor.parameters(), lr=1e-4)
        return

    def _get_predictor_input(self, obs):
        raw_obs_shapes = self.obs_encoder.get_obs_shapes()
        predictor_input_parts = []
        
        for key in raw_obs_shapes:
            if key != "tar_obs":
                raw_part = self.obs_encoder.get_raw_part(obs, key)
                predictor_input_parts.append(raw_part)
                
        return torch.cat(predictor_input_parts, dim=-1)

    def train_future_pose_predictor(self, obs):
        if not hasattr(self, "_predictor") or self._predictor is None:
            assert False, "Future pose predictor not built."
        
        with torch.no_grad():
            target_latent = self.obs_encoder.get_latent(obs, "tar_obs").detach()
            predictor_input = self._get_predictor_input(obs)
        
        prediction = self._predictor(predictor_input)
        loss = nn.functional.mse_loss(prediction, target_latent)
        
        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()
        
        return {"predictor_loss": loss.detach()}
