import torch
import torch.nn as nn
import numpy as np
import copy
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
        
        self._output_obs_shapes = {}
        
        curr_offset = 0
        
        # Build slices for raw inputs
        for key, info in self._obs_shapes.items():
            shape = info["shape"]
            size = int(np.prod(shape))
            self._obs_slices[key] = (curr_offset, curr_offset + size)
            curr_offset += size

        # 1.5 Get Encoder Config
        encoders_cfg = config.get("obs_encoders", {})

        # Helper mappings
        self._key_to_encoder = {} # input_key -> encoder_name
        self._encoder_input_keys = {} # encoder_name -> [input_keys]

        # Build Encoders
        for enc_name, cfg in encoders_cfg.items():
            input_keys = cfg.get("input_keys", [enc_name])
            self._encoder_input_keys[enc_name] = input_keys
            
            # Calculate total input dimension for this encoder
            total_input_dim = 0
            for k in input_keys:
                if k not in self._obs_shapes:
                    raise ValueError(f"Input key '{k}' for encoder '{enc_name}' not found in observation shapes.")
                
                # Map input key to this encoder
                if k in self._key_to_encoder:
                    raise ValueError(f"Key '{k}' is already assigned to encoder '{self._key_to_encoder[k]}'. Cannot assign to '{enc_name}'.")
                self._key_to_encoder[k] = enc_name
                
                total_input_dim += int(np.prod(self._obs_shapes[k]["shape"]))

            # Build the encoder module
            enc_type = cfg.get("type", "mlp")
            latent_dim = cfg["latent_dim"]
            
            if enc_type == "mlp":
                hidden_dims = cfg.get("hidden_dims", [256, 128])
                self._encoders[enc_name] = MLPEncoder(total_input_dim, latent_dim, hidden_dims=hidden_dims)

        # Build Output Shapes
        # We iterate through original keys to maintain a deterministic order, 
        # but replace encoded keys with the encoder output (once per encoder)
        processed_encoders = set()
        
        for key, info in self._obs_shapes.items():
            if key in self._key_to_encoder:
                enc_name = self._key_to_encoder[key]
                if enc_name not in processed_encoders:
                    # Add encoder output shape
                    latent_dim = encoders_cfg[enc_name]["latent_dim"]
                    self._output_obs_shapes[enc_name] = {"shape": (latent_dim,)}
                    processed_encoders.add(enc_name)
                # Else: this key is part of an already processed encoder, skip
            else:
                # Pass-through raw key
                self._output_obs_shapes[key] = info

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

    def get_encoder_input_keys(self, enc_name):
        return self._encoder_input_keys.get(enc_name, [])

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

    def get_latent(self, obs, enc_name):
        """
        Get the latent vector from a specific encoder.
        """
        if enc_name not in self._encoders:
             raise KeyError(f"Encoder '{enc_name}' not found.")
             
        input_keys = self._encoder_input_keys[enc_name]
        parts = []
        for k in input_keys:
            parts.append(self.get_raw_part(obs, k))
            
        inp = torch.cat(parts, dim=-1)
        return self._encoders[enc_name](inp)

    def get_raw_part(self, obs, key):
        """
        Get the raw observation part for a given key without any encoding.
        """
        if key not in self._obs_slices:
            raise KeyError(f"Key '{key}' not found in observation shapes.")
            
        start, end = self._obs_slices[key]
        return obs[..., start:end]

    def forward(self, obs, overrides=None):
        processed_parts = []
        processed_encoders = set()
        
        # We iterate in the same order as obs_shapes to determine output order
        for key in self._obs_shapes:
            if key in self._key_to_encoder:
                enc_name = self._key_to_encoder[key]
                
                # Check overrides first
                if overrides is not None and enc_name in overrides:
                    if enc_name not in processed_encoders:
                        processed_parts.append(overrides[enc_name])
                        processed_encoders.add(enc_name)
                    continue

                if enc_name not in processed_encoders:
                    # Run Encoder
                    input_keys = self._encoder_input_keys[enc_name]
                    parts = []
                    for k in input_keys:
                        # For inputs to the encoder, we need raw data from obs
                        start, end = self._obs_slices[k]
                        parts.append(obs[..., start:end])
                    
                    inp = torch.cat(parts, dim=-1)
                    latent = self._encoders[enc_name](inp)
                    
                    processed_parts.append(latent)
                    processed_encoders.add(enc_name)
                # Else: skipped (part of group already processed)
            else:
                # Raw pass-through
                if overrides is not None and key in overrides:
                    processed_parts.append(overrides[key])
                else:
                    start, end = self._obs_slices[key]
                    processed_parts.append(obs[..., start:end])
            
        return torch.cat(processed_parts, dim=-1)
