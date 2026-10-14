import torch

import parc.motion_tracker.learning.dm_ppo_agent as dm_ppo_agent
import parc.motion_tracker.learning.dm_ppo_fp_model as dm_ppo_fp_model
import parc.motion_tracker.learning.optimizer as optimizer
from parc.util.logger import Logger


class FuturePoseDMPPOAgent(dm_ppo_agent.DMPPOAgent):
    """DMPPOAgent with observation encoders and a future-pose predictor (FuturePoseDMPPOModel).

    The encoders, actor and critic are trained by the PPO loss. The predictor has its own optimizer and
    takes one step per PPO minibatch on the MSE to the target encoder's latent. It is configured by
    config["model"]["future_pose_predictor"]:
        enabled:        train the predictor (default true). It is built either way, so checkpoints
                        always have the same layout.
        learning_rate:  learning rate of the predictor's Adam optimizer (default 1e-4).
        hidden_dims, target_encoder: see FuturePoseDMPPOModel.
    """

    NAME = "DM_PPO_FP"
    PREDICTOR_KEY_PREFIX = "_model._predictor."
    DEFAULT_PREDICTOR_LR = 1e-4

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._predictor_weights_missing = False
        return

    def set_predictor_test_mode(self, enabled):
        if (enabled and self._predictor_weights_missing):
            raise RuntimeError("The loaded checkpoint has no future-pose predictor weights, "
                               "so predictor test mode would run an untrained predictor.")
        self._model.set_predictor_test_mode(enabled)
        return

    def load(self, in_file):
        # Strict, except that the predictor may be missing from the checkpoint (e.g. generation-B1
        # checkpoints, trained before the predictor existed) or from the model.
        state_dict = torch.load(in_file, map_location=self._device)

        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing_keys = sorted(model_keys - ckpt_keys)
        unexpected_keys = sorted(ckpt_keys - model_keys)
        mismatched_keys = [key for key in missing_keys + unexpected_keys
                           if not key.startswith(self.PREDICTOR_KEY_PREFIX)]
        if (len(mismatched_keys) > 0):
            raise RuntimeError("Checkpoint {:s} does not match the model. Missing keys: {}. Unexpected keys: {}."
                               .format(str(in_file), [key for key in missing_keys if key in mismatched_keys],
                                       [key for key in unexpected_keys if key in mismatched_keys]))

        if (len(missing_keys) > 0):
            Logger.print("Warning: {:s} has no future-pose predictor weights; the predictor keeps its "
                         "initial weights.".format(str(in_file)))
        if (len(unexpected_keys) > 0):
            Logger.print("Warning: {:s} has future-pose predictor weights, but the model has no predictor; "
                         "they are ignored.".format(str(in_file)))

        self.load_state_dict(state_dict, strict=False)
        self._predictor_weights_missing = len(missing_keys) > 0
        self._sync_optimizer()
        Logger.print("Loaded model parameters from {:s}".format(str(in_file)))
        return

    def _load_params(self, config):
        super()._load_params(config)

        predictor_config = config["model"].get("future_pose_predictor", None) or dict()
        self._has_predictor_config = len(predictor_config) > 0

        enabled = predictor_config.get("enabled", True)
        if (not isinstance(enabled, bool)):
            raise ValueError("future_pose_predictor.enabled must be true or false, got {}".format(repr(enabled)))
        self._enable_predictor_training = enabled

        # float(): yaml reads a number like 1e-4 (without a decimal point) as a string.
        self._predictor_lr = float(predictor_config.get("learning_rate", self.DEFAULT_PREDICTOR_LR))
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = dm_ppo_fp_model.FuturePoseDMPPOModel(model_config, self._env)
        return

    def _build_optimizer(self, config):
        predictor_params = self._model.get_predictor_params()
        predictor_param_ids = set(id(param) for param in predictor_params)
        params = [param for param in self.parameters()
                  if param.requires_grad and id(param) not in predictor_param_ids]
        self._optimizer = optimizer.Optimizer(config["optimizer"], params)

        self._predictor_optimizer = None
        if (self._enable_predictor_training):
            if (self._model.has_predictor()):
                # "Adam" is AdamW, which is Adam when weight_decay is 0.
                predictor_opt_config = {
                    "type": "Adam",
                    "learning_rate": self._predictor_lr,
                    "weight_decay": 0.0
                }
                self._predictor_optimizer = optimizer.Optimizer(predictor_opt_config, predictor_params)
            elif (self._has_predictor_config):
                Logger.print("Warning: future_pose_predictor is enabled, but there is no predictor because the "
                             "model has no target obs encoder.")
        return

    def _sync_optimizer(self):
        super()._sync_optimizer()
        if (self._predictor_optimizer is not None):
            self._predictor_optimizer.sync()
        return

    def _compute_loss(self, batch):
        info = super()._compute_loss(batch)

        if (self._predictor_optimizer is not None):
            info["predictor_loss"] = self._update_predictor(batch["norm_obs"])
        return info

    def _update_predictor(self, norm_obs):
        loss = self._model.compute_predictor_loss(norm_obs)
        self._check_nan(loss, "predictor_loss")
        self._predictor_optimizer.step(loss)

        # The main optimizer does not hold the predictor, so it never clears these gradients, and its
        # gradient clipping over model.parameters() would include them.
        for param in self._model.get_predictor_params():
            param.grad = None
        return loss.detach()
