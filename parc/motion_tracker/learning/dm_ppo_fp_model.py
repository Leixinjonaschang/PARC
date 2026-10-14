import numpy as np
import torch

import parc.motion_tracker.learning.dm_ppo_model as dm_ppo_model
import parc.motion_tracker.learning.obs_encoder as obs_encoder
from parc.motion_tracker.learning.modules.mlp_encoder import MLPEncoder


class FuturePoseDMPPOModel(dm_ppo_model.DMPPOModel):
    """DMPPOModel with observation encoders and a future-pose predictor (teacher-student distillation).

    The actor and critic read the observation through an ObservationEncoder (config["obs_encoders"]).
    The predictor (student) is an MLPEncoder that predicts the latent of one encoder (the teacher,
    config["future_pose_predictor"]["target_encoder"], default "future_pose") from all the raw
    observation keys that this encoder does not consume. For G1 it maps char_obs, char_contacts and hf
    (743 dims) to the 32-dim latent of [tar_obs, tar_contacts]. In predictor test mode, the actor uses
    the predicted latent instead of the encoder's; the critic always uses the encoder.

    The attribute names (obs_encoder, _predictor) are develop's, so that develop's generation-C
    checkpoints load unchanged.
    """

    DEFAULT_PREDICTOR_TARGET = "future_pose"
    DEFAULT_PREDICTOR_HIDDEN_DIMS = [512, 512, 256]

    def __init__(self, config, env):
        super().__init__(config, env)

        self._predictor_test_mode = False
        self._build_predictor(config)
        return

    def eval_actor(self, obs):
        if (self._predictor_test_mode):
            pred_latent = self._predictor(self._get_predictor_input(obs))
            encoded_obs = self.obs_encoder(obs, overrides={self._predictor_target: pred_latent})
        else:
            encoded_obs = self.obs_encoder(obs)
        return super().eval_actor(encoded_obs)

    def eval_critic(self, obs):
        encoded_obs = self.obs_encoder(obs)
        return super().eval_critic(encoded_obs)

    def has_predictor(self):
        return self._predictor is not None

    def get_predictor_params(self):
        if (self.has_predictor()):
            return list(self._predictor.parameters())
        return []

    def set_predictor_test_mode(self, enabled):
        if (enabled and not self.has_predictor()):
            raise RuntimeError("Predictor test mode needs a future-pose predictor, but the model has no '{:s}' obs encoder."
                               .format(self._predictor_target))
        self._predictor_test_mode = enabled
        return

    def get_predictor_test_mode(self):
        return self._predictor_test_mode

    def compute_predictor_loss(self, norm_obs):
        """MSE between the predicted latent and the target encoder's latent of the normalized observation.

        The target is computed without gradients, so the loss only trains the predictor. The caller steps
        the optimizer.
        """
        if (not self.has_predictor()):
            raise RuntimeError("The model has no future-pose predictor.")

        with torch.no_grad():
            tar_latent = self.obs_encoder.get_latent(norm_obs, self._predictor_target)
        pred_latent = self._predictor(self._get_predictor_input(norm_obs))
        loss = torch.nn.functional.mse_loss(pred_latent, tar_latent)
        return loss

    def _build_nets(self, config, env):
        for net_key in ["actor_net", "critic_net"]:
            net_name = config[net_key]
            if (net_name.startswith("dm_vit") or net_name == "dm_cnn_mlp"):
                raise ValueError("{:s} '{:s}' is not supported with observation encoders.".format(net_key, net_name))

        # Built first: the actor and critic input size is the encoded observation size.
        self.obs_encoder = obs_encoder.ObservationEncoder(config, env)
        super()._build_nets(config, env)
        return

    def _build_actor_input_dict(self, env):
        return {"obs": self.obs_encoder.get_mock_obs_space()}

    def _build_critic_input_dict(self, env):
        return {"obs": self.obs_encoder.get_mock_obs_space()}

    def _build_predictor(self, config):
        predictor_config = config.get("future_pose_predictor", None) or dict()
        self._predictor_target = predictor_config.get("target_encoder", self.DEFAULT_PREDICTOR_TARGET)
        self._predictor_input_keys = []
        self._predictor = None

        # The predictor is built whenever its target encoder exists, independently of whether it is
        # trained, so that the checkpoint layout does not depend on the training flag.
        if (not self.obs_encoder.has_encoder(self._predictor_target)):
            if ("target_encoder" in predictor_config):
                raise ValueError("future_pose_predictor.target_encoder '{:s}' is not an obs encoder."
                                 .format(self._predictor_target))
            return

        target_input_keys = self.obs_encoder.get_encoder_input_keys(self._predictor_target)
        obs_shapes = self.obs_encoder.get_obs_shapes()
        self._predictor_input_keys = [key for key in obs_shapes if key not in target_input_keys]
        if (len(self._predictor_input_keys) == 0):
            raise ValueError("Obs encoder '{:s}' consumes every observation key, so there is no predictor input."
                             .format(self._predictor_target))

        input_dim = sum(int(np.prod(obs_shapes[key]["shape"])) for key in self._predictor_input_keys)
        output_shapes = self.obs_encoder.get_output_obs_shapes()
        latent_dim = int(np.prod(output_shapes[self._predictor_target]["shape"]))
        hidden_dims = predictor_config.get("hidden_dims", self.DEFAULT_PREDICTOR_HIDDEN_DIMS)

        self._predictor = MLPEncoder(input_dim, latent_dim, hidden_dims=hidden_dims, activation=self._activation)
        return

    def _get_predictor_input(self, obs):
        parts = [self.obs_encoder.get_raw_part(obs, key) for key in self._predictor_input_keys]
        return torch.cat(parts, dim=-1)
