import torch
import torch.nn as nn
import numpy as np
import copy
from collections import OrderedDict
import gym
from learning.modules.mlp_encoder import MLPEncoder

class ObservationEncoder(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self._device = device
        self._config = config
        self._env = env
        
        # 1. Compute Obs Shapes and Slices
        self._obs_shapes = env._compute_obs(ret_obs_shapes=True)
        self._obs_slices = {}
        self._encoders = nn.ModuleDict()
        
        self._output_obs_shapes = copy.deepcopy(self._obs_shapes)
        
        curr_offset = 0
        
        # 1.5 Get Encoder Config
        encoders_cfg = config.get("obs_encoders", {})
        
        # Iterate to build slices and encoders
        for key, info in self._obs_shapes.items():
            shape = info["shape"]
            size = int(np.prod(shape))
            self._obs_slices[key] = (curr_offset, curr_offset + size)
            curr_offset += size
            
            # Check for Encoders
            # Legacy support for old config style
            if key == "tar_obs" and "tar_obs_latent_dim" in config and key not in encoders_cfg:
                 encoders_cfg[key] = {
                     "type": "mlp",
                     "latent_dim": config.get("tar_obs_latent_dim", 32),
                     "hidden_dims": config.get("tar_obs_hidden_dim", [512, 512, 256])
                 }

            if key in encoders_cfg:
                cfg = encoders_cfg[key]
                enc_type = cfg.get("type", "mlp")
                latent_dim = cfg["latent_dim"]
                
                if enc_type == "mlp":
                    hidden_dims = cfg.get("hidden_dims", [256, 128])
                    self._encoders[key] = MLPEncoder(size, latent_dim, hidden_dims=hidden_dims)
                
                # Update output shape
                self._output_obs_shapes[key]["shape"] = (latent_dim,)

        # Compute total output dimension
        self._total_output_dim = 0
        for key, info in self._output_obs_shapes.items():
            self._total_output_dim += int(np.prod(info["shape"]))
            
        return

    def get_output_obs_shapes(self):
        return self._output_obs_shapes
    
    def get_obs_shapes(self):
        return self._obs_shapes

    def get_output_dim(self):
        return self._total_output_dim

    def get_net_builder_input_dict(self):
        """Standardized input dict for net_builder"""
        return {
            "device": self._device,
            "obs_shapes": self._output_obs_shapes,
            "obs_dim": self._total_output_dim
        }

    def get_mock_obs_space(self):
        """Returns a gym Box space matching the output dimension"""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._total_output_dim,), dtype=np.float32)

    def get_latent(self, obs, key):
        """
        Get the latent vector encoded by the encoder for a given key.
        """
        if key not in self._obs_slices:
            raise KeyError(f"Key '{key}' not found in observation shapes. Available keys: {list(self._obs_slices.keys())}")

        if key not in self._encoders:
            raise KeyError(f"Key '{key}' does not have a corresponding encoder. Available encoder keys: {list(self._encoders.keys())}")

        start, end = self._obs_slices[key]
        part = obs[..., start:end]
        return self._encoders[key](part)

    def get_raw_part(self, obs, key):
        """
        Get the raw observation part for a given key without any encoding.
        """
        if key not in self._obs_slices:
            raise KeyError(f"Key '{key}' not found in observation shapes. Available keys: {list(self._obs_slices.keys())}")
            
        start, end = self._obs_slices[key]
        return obs[..., start:end]

    def forward(self, obs):
        processed_parts = []
        
        # We need to iterate in the same order as obs_shapes
        for key in self._obs_shapes:
            start, end = self._obs_slices[key]
            part = obs[..., start:end]
            
            if key in self._encoders:
                part = self._encoders[key](part)
            
            processed_parts.append(part)
            
        return torch.cat(processed_parts, dim=-1)


