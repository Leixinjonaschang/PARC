import torch
import torch.nn as nn
import numpy as np
import gym


class CVAESkillModule(nn.Module):
    """
    CVAE module for learning latent skill space from teacher policy rollouts.

    Architecture:
    - Encoder: q(z | condition, action) -> mu, logvar
    - Decoder: p(action | condition, z) -> action
    - Prior: p(z | char_obs, hf) -> mu_p, logvar_p
    """

    def __init__(self, config, env, device):
        super().__init__()
        self._device = device
        self._config = config
        self._env = env

        # Get dimensions from environment
        self._action_dim = self._get_action_dim()
        self._char_obs_dim, self._tar_obs_dim, self._tar_contacts_dim, self._hf_dim, self._char_contacts_dim, self._heading_dim = self._get_obs_dims()

        # CVAE config
        cvae_cfg = config.get("cvae", {})
        self._latent_dim = cvae_cfg.get("latent_dim", 128)
        self._beta = cvae_cfg.get("beta", 0.1)

        # Encoder hidden dims: condition + action -> ... -> mu, logvar
        encoder_hidden_dims = cvae_cfg.get("encoder_hidden_dims", [1024, 1024, 512])

        # Decoder hidden dims: z + condition -> ... -> action
        decoder_hidden_dims = cvae_cfg.get("decoder_hidden_dims", [512, 2048, 1024, 512])

        # Prior hidden dims: char_obs + hf -> ... -> mu_p, logvar_p
        prior_hidden_dims = cvae_cfg.get("prior_hidden_dims", [512, 256])

        # Encoder input dimension: char_obs + tar_obs + tar_contacts + char_contacts
        self._encoder_condition_dim = self._char_obs_dim + self._tar_obs_dim + self._tar_contacts_dim + self._char_contacts_dim
        # Decoder condition dimension: char_obs + char_contacts + hf + target_xy(heading)
        self._decoder_condition_dim = self._char_obs_dim + self._char_contacts_dim + self._hf_dim + self._heading_dim

        # Build networks
        self._build_encoder(encoder_hidden_dims)
        self._build_decoder(decoder_hidden_dims)
        self._build_prior(prior_hidden_dims)

        # Move module to device
        self.to(self._device)

        # Optimizer
        lr = float(cvae_cfg.get("learning_rate", 1e-4))
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return

    def _get_action_dim(self):
        a_space = self._env.get_action_space()
        if isinstance(a_space, gym.spaces.Box):
            return int(np.prod(a_space.shape))
        elif isinstance(a_space, gym.spaces.Discrete):
            return 1
        else:
            raise ValueError(f"Unsupported action space: {a_space}")

    def _get_obs_dims(self):
        """Get observation dimensions from the environment."""
        obs_shapes = self._env._compute_obs(ret_obs_shapes=True)

        # char_obs: all non-tar observations (pass-through)
        char_obs_dim = 0
        tar_obs_dim = 0
        tar_contacts_dim = 0
        char_contacts_dim = 0
        hf_dim = 0
        heading_dim = 0

        for key, info in obs_shapes.items():
            dim = int(np.prod(info["shape"]))
            if key == "tar_obs":
                tar_obs_dim = dim
            elif key == "tar_contacts":
                tar_contacts_dim = dim
            elif key == "char_contacts":
                char_contacts_dim = dim
            elif key == "hf":
                hf_dim = dim
            elif key == "target_xy":
                heading_dim = dim
            else:
                # All other obs go to char_obs
                char_obs_dim += dim

        return char_obs_dim, tar_obs_dim, tar_contacts_dim, hf_dim, char_contacts_dim, heading_dim

    def _build_encoder(self, hidden_dims):
        """Build encoder: q(z | condition, action) -> mu, logvar"""
        input_dim = self._encoder_condition_dim + self._action_dim # weird here, why the encoder input includes the action dimensions?

        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h

        self._encoder = nn.Sequential(*layers)
        self._encoder_mu = nn.Linear(curr_dim, self._latent_dim)
        self._encoder_logvar = nn.Linear(curr_dim, self._latent_dim)

        return

    def _build_decoder(self, hidden_dims):
        """Build decoder: p(action | condition, z) -> action"""
        input_dim = self._latent_dim + self._decoder_condition_dim

        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h

        layers.append(nn.Linear(curr_dim, self._action_dim))
        # Use tanh to bound actions (normalized action space [-1, 1])
        self._decoder = nn.Sequential(*layers)

        return

    def _build_prior(self, hidden_dims):
        """Build prior: p(z | char_obs, hf) -> mu_p, logvar_p"""
        input_dim = self._char_obs_dim + self._hf_dim

        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h

        self._prior = nn.Sequential(*layers)
        self._prior_mu = nn.Linear(curr_dim, self._latent_dim)
        self._prior_logvar = nn.Linear(curr_dim, self._latent_dim)

        return

    def encode(self, condition, action):
        """
        Encode condition and action to latent distribution.

        Args:
            condition: (B, encoder_condition_dim) - encoder condition
            action: (B, action_dim) - action to encode

        Returns:
            mu: (B, latent_dim) - mean of latent distribution
            logvar: (B, latent_dim) - log variance of latent distribution
        """
        x = torch.cat([condition, action], dim=-1)
        h = self._encoder(x)
        mu = self._encoder_mu(h)
        logvar = self._encoder_logvar(h)
        return mu, logvar

    def decode(self, condition, z):
        """
        Decode latent z and condition to action.

        Args:
            condition: (B, decoder_condition_dim) - decoder condition
            z: (B, latent_dim) - latent sample

        Returns:
            action_pred: (B, action_dim) - predicted action
        """
        x = torch.cat([z, condition], dim=-1)
        action_pred = self._decoder(x)
        return action_pred

    def prior(self, char_obs, hf):
        """
        Compute prior distribution from character observation and heightfield.

        Args:
            char_obs: (B, char_obs_dim) - character observation
            hf: (B, hf_dim) - heightfield observation

        Returns:
            mu_p: (B, latent_dim) - mean of prior distribution
            logvar_p: (B, latent_dim) - log variance of prior distribution
        """
        x = torch.cat([char_obs, hf], dim=-1)
        h = self._prior(x)
        mu_p = self._prior_mu(h)
        logvar_p = self._prior_logvar(h)
        return mu_p, logvar_p

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from N(mu, sigma).

        Args:
            mu: (B, latent_dim) - mean
            logvar: (B, latent_dim) - log variance

        Returns:
            z: (B, latent_dim) - sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, batch, use_prior=False):
        """
        Forward pass through CVAE.

        Args:
            batch: dict with keys:
                - char_obs: (B, char_obs_dim)
                - tar_obs: (B, tar_obs_dim)
                - tar_contacts: (B, tar_contacts_dim)
                - char_contacts: (B, char_contacts_dim)
                - hf: (B, hf_dim)
                - heading: (B, 2) - target_xy
                - action: (B, action_dim) - ground truth action
            use_prior: if True, use prior instead of encoder (for inference)

        Returns:
            dict with:
                - action_pred: reconstructed action
                - z: sampled latent
                - mu: encoder mean
                - logvar: encoder logvar
                - mu_p: prior mean
                - logvar_p: prior logvar
        """
        char_obs = batch["char_obs"]
        tar_obs = batch["tar_obs"]
        tar_contacts = batch["tar_contacts"]
        char_contacts = batch["char_contacts"]
        hf = batch["hf"]
        heading = batch["heading"]
        action = batch["action"]

        # Encoder condition: char_obs + tar_obs + tar_contacts + char_contacts
        encoder_condition = torch.cat([char_obs, tar_obs, tar_contacts, char_contacts], dim=-1)

        # Decoder condition: char_obs + char_contacts + hf + heading
        decoder_parts = [char_obs, char_contacts, hf]
        if heading is not None and self._heading_dim > 0:
            decoder_parts.append(heading)
        decoder_condition = torch.cat(decoder_parts, dim=-1)

        # Encode or use prior
        if use_prior:
            mu_p, logvar_p = self.prior(char_obs, hf)
            z = self.reparameterize(mu_p, logvar_p)
            mu = None
            logvar = None
        else:
            mu, logvar = self.encode(encoder_condition, action)
            z = self.reparameterize(mu, logvar)
            mu_p, logvar_p = self.prior(char_obs, hf)

        # Decode
        action_pred = self.decode(decoder_condition, z)

        return {
            "action_pred": action_pred,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "mu_p": mu_p,
            "logvar_p": logvar_p
        }

    def compute_loss(self, batch):
        """
        Compute CVAE loss: L_recon + beta * L_KL

        Args:
            batch: dict with observations and actions

        Returns:
            dict with losses and statistics
        """
        char_obs = batch["char_obs"]
        tar_obs = batch["tar_obs"]
        tar_contacts = batch["tar_contacts"]
        char_contacts = batch["char_contacts"]
        hf = batch["hf"]
        heading = batch["heading"]
        action = batch["action"]

        # Encoder condition and prior
        encoder_condition = torch.cat([char_obs, tar_obs, tar_contacts, char_contacts], dim=-1)
        mu, logvar = self.encode(encoder_condition, action)
        mu_p, logvar_p = self.prior(char_obs, hf)

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        logvar_p = torch.clamp(logvar_p, min=-10, max=10)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decoder condition
        decoder_parts = [char_obs, char_contacts, hf]
        if heading is not None and self._heading_dim > 0:
            decoder_parts.append(heading)
        decoder_condition = torch.cat(decoder_parts, dim=-1)

        # Decode
        action_pred = self.decode(decoder_condition, z)

        # Reconstruction loss (MSE)
        recon_loss = torch.mean(torch.square(action_pred - action))

        # KL divergence: KL(q(z|c,a) || p(z|c,hf))
        # KL = 0.5 * (logvar_p - logvar - 1 + exp(logvar - logvar_p) + (mu - mu_p)^2 / exp(logvar_p))
        kl_loss = 0.5 * (
            logvar_p - logvar - 1 +
            torch.exp(logvar - logvar_p) +
            torch.square(mu - mu_p) / torch.exp(logvar_p)
        )
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        # Total loss
        total_loss = recon_loss + self._beta * kl_loss

        # Statistics
        z_mean = torch.mean(z).item()
        z_std = torch.std(z).item()

        return {
            "cvae_loss": total_loss,
            "cvae_recon_loss": recon_loss.detach(),
            "cvae_kl_loss": kl_loss.detach(),
            "cvae_z_mean": z_mean,
            "cvae_z_std": z_std
        }

    def update(self, batch):
        """
        Update CVAE with a batch.

        Args:
            batch: dict with observations and actions

        Returns:
            dict with loss information
        """
        self._optimizer.zero_grad()
        loss_info = self.compute_loss(batch)

        # Check for NaN loss
        if torch.isnan(loss_info["cvae_loss"]):
            print("Warning: NaN loss detected, skipping update")
            loss_info["cvae_loss"] = torch.tensor(0.0, device=self._device)
            loss_info["cvae_recon_loss"] = torch.tensor(0.0, device=self._device)
            loss_info["cvae_kl_loss"] = torch.tensor(0.0, device=self._device)
            return loss_info

        loss_info["cvae_loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self._optimizer.step()
        return loss_info

    def get_action(self, batch, deterministic=False):
        """
        Use CVAE as a policy to generate actions.
        Samples z from prior during rollout, decodes to action.

        Args:
            batch: dict with keys:
                - char_obs: (B, char_obs_dim)
                - char_contacts: (B, char_contacts_dim)
                - hf: (B, hf_dim)
                - heading: (B, heading_dim) - optional
            deterministic: if True, use mean instead of sampling

        Returns:
            action: (B, action_dim) - generated action
        """
        char_obs = batch["char_obs"]
        char_contacts = batch["char_contacts"]
        hf = batch["hf"]
        heading = batch.get("heading")

        # Sample z from prior
        mu_p, logvar_p = self.prior(char_obs, hf)
        if deterministic:
            z = mu_p
        else:
            z = self.reparameterize(mu_p, logvar_p)

        # Build decoder condition
        decoder_parts = [char_obs, char_contacts, hf]
        if heading is not None and self._heading_dim > 0:
            decoder_parts.append(heading)
        decoder_condition = torch.cat(decoder_parts, dim=-1)

        # Decode to action
        action = self.decode(decoder_condition, z)
        return action