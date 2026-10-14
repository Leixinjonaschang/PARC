import collections

import gym
import numpy as np
import torch
import torch.nn as nn

from parc.motion_tracker.learning.modules.mlp_encoder import MLPEncoder


class ObservationEncoder(nn.Module):
    """Replaces groups of observation keys with learned latents and passes the other keys through.

    The flat observation is the concatenation, in key order, of the parts reported by
    env._compute_obs(ret_obs_shapes=True). config["obs_encoders"] maps an encoder name to
    {type: mlp, latent_dim, hidden_dims, input_keys}; input_keys defaults to [name]. In the output,
    each encoder's latent takes the place of its first input key, its other input keys are dropped,
    and every other key passes through unchanged. For example, with the G1 observation and a
    future_pose encoder on [tar_obs, tar_contacts]:

        input  (2363): char_obs 266 | tar_obs 1404 | tar_contacts 216 | char_contacts 36 | hf 441
        output  (775): char_obs 266 | future_pose 32 | char_contacts 36 | hf 441

    Only observation shapes are taken from the env (not the env itself), so the module can be
    deep-copied.
    """

    def __init__(self, config, env):
        super().__init__()

        # Raw observation layout
        self._obs_shapes = collections.OrderedDict()
        self._obs_slices = collections.OrderedDict()
        curr_offset = 0
        for key, info in env._compute_obs(ret_obs_shapes=True).items():
            info = dict(info)
            info["shape"] = tuple(int(dim) for dim in info["shape"])
            size = int(np.prod(info["shape"]))
            self._obs_shapes[key] = info
            self._obs_slices[key] = (curr_offset, curr_offset + size)
            curr_offset += size

        obs_dim = int(env.get_obs_space().shape[0])
        if (curr_offset != obs_dim):
            raise ValueError("The observation parts {} sum to {:d} dims, but the observation space has {:d} dims."
                             .format(dict((key, info["shape"]) for key, info in self._obs_shapes.items()),
                                     curr_offset, obs_dim))

        # Encoders
        self._encoders = nn.ModuleDict()
        self._key_to_encoder = dict() # input key -> encoder name
        self._encoder_input_keys = collections.OrderedDict() # encoder name -> input keys
        self._encoder_latent_dims = dict()

        encoders_config = config.get("obs_encoders", None) or dict()
        for enc_name, enc_config in encoders_config.items():
            input_keys = list(enc_config.get("input_keys", [enc_name]))
            if (len(input_keys) == 0):
                raise ValueError("Obs encoder '{}' has no input_keys.".format(enc_name))
            if (enc_name in self._obs_shapes and enc_name not in input_keys):
                raise ValueError("Obs encoder '{}' has the name of an observation key that it does not encode."
                                 .format(enc_name))

            input_dim = 0
            for key in input_keys:
                if (key not in self._obs_shapes):
                    raise ValueError("Input key '{}' of obs encoder '{}' is not an observation key: {}"
                                     .format(key, enc_name, list(self._obs_shapes.keys())))
                if (key in self._key_to_encoder):
                    raise ValueError("Observation key '{}' is already encoded by '{}', cannot also encode it with '{}'."
                                     .format(key, self._key_to_encoder[key], enc_name))
                self._key_to_encoder[key] = enc_name
                input_dim += self._get_raw_size(key)

            enc_type = enc_config.get("type", "mlp")
            latent_dim = int(enc_config["latent_dim"])
            if (enc_type == "mlp"):
                hidden_dims = enc_config.get("hidden_dims", [256, 128])
                self._encoders[enc_name] = MLPEncoder(input_dim, latent_dim, hidden_dims=hidden_dims)
            else:
                raise ValueError("Unsupported type '{}' for obs encoder '{}'.".format(enc_type, enc_name))

            self._encoder_input_keys[enc_name] = input_keys
            self._encoder_latent_dims[enc_name] = latent_dim

        # Output layout: raw key order, with each encoder in place of its first input key
        self._output_obs_shapes = collections.OrderedDict()
        for key, info in self._obs_shapes.items():
            enc_name = self._key_to_encoder.get(key, None)
            if (enc_name is None):
                self._output_obs_shapes[key] = info
            elif (enc_name not in self._output_obs_shapes):
                self._output_obs_shapes[enc_name] = {"shape": (self._encoder_latent_dims[enc_name],)}

        self._total_output_dim = sum(int(np.prod(info["shape"])) for info in self._output_obs_shapes.values())
        return

    def get_obs_shapes(self):
        return self._obs_shapes

    def get_output_obs_shapes(self):
        return self._output_obs_shapes

    def get_output_dim(self):
        return self._total_output_dim

    def has_encoder(self, enc_name):
        return enc_name in self._encoders

    def get_encoder_input_keys(self, enc_name):
        return list(self._encoder_input_keys[enc_name])

    def get_mock_obs_space(self):
        """A Box space with the size of the encoded observation, for building the actor and critic nets."""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._total_output_dim,), dtype=np.float32)

    def get_raw_part(self, obs, key):
        """The slice of the flat observation that holds `key`, without encoding."""
        start, end = self._obs_slices[key]
        return obs[..., start:end]

    def get_latent(self, obs, enc_name):
        """The latent of encoder `enc_name` for the flat observation `obs`."""
        parts = [self.get_raw_part(obs, key) for key in self._encoder_input_keys[enc_name]]
        enc_input = torch.cat(parts, dim=-1)
        return self._encoders[enc_name](enc_input)

    def forward(self, obs, overrides=None):
        """Encodes the flat observation. `overrides` maps an output name (an encoder name or a
        pass-through key) to a tensor that is used in place of that part."""
        parts = []
        for name in self._output_obs_shapes:
            if (overrides is not None and name in overrides):
                parts.append(overrides[name])
            elif (name in self._encoders):
                parts.append(self.get_latent(obs, name))
            else:
                parts.append(self.get_raw_part(obs, name))
        return torch.cat(parts, dim=-1)

    def _get_raw_size(self, key):
        start, end = self._obs_slices[key]
        return end - start
