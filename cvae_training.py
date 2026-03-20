"""
Standalone CVAE Training Script

This script trains the CVAE skill module independently from tracker training.
It can use DAgger (iterative student-expert alignment) or standard training.

Usage:
    python cvae_training.py --config PARC/cvae_config/cvae_g1.yaml
"""
import os
import sys
import yaml
from pathlib import Path
import random

# Add PARC to path
PARC_PATH = Path(__file__).parent
sys.path.insert(0, str(PARC_PATH))

# IsaacGym must be imported before torch
import envs.env_builder as env_builder
import learning.dm_ppo_model as dm_ppo_model
import learning.modules.cvae_skill as cvae_skill
from learning.base_agent import AgentMode
import util.wandb_logger as wandb_logger
import util.tb_logger as tb_logger
import util.mp_util as mp_util

# torch must be imported after isaacgym modules
import torch
import numpy as np


def load_config(config_path):
    """Load CVAE training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_env(env_config_path, num_envs, device, visualize=False):
    """Build environment for CVAE training."""
    env = env_builder.build_env(env_config_path, num_envs, device, visualize)
    return env


def load_teacher_model(checkpoint_path, config, env, device):
    """Load teacher model from checkpoint for CVAE distillation."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    teacher_model = dm_ppo_model.DMPPOModel(config["model"], env)
    actor_state_dict = {k: v for k, v in checkpoint.items()
                        if 'actor' in k.lower() or 'obs_encoder' in k.lower()}
    teacher_model.load_state_dict(actor_state_dict, strict=False)
    teacher_model.to(device)
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.requires_grad = False

    print(f"Loaded teacher model from {checkpoint_path}")
    return teacher_model


class CVAETrainer:
    """Standalone CVAE trainer with DAgger support."""

    def __init__(self, config, env, teacher_model, device):
        self._config = config
        self._env = env
        self._teacher_model = teacher_model
        self._device = device

        # Logger Setup
        cvae_cfg = config.get("cvae", {})
        logging_cfg = cvae_cfg.get("logging", {})
        
        project_name = logging_cfg.get("project_name", "parkour_cvae")
        exp_name = logging_cfg.get("exp_name", "cvae_training")
        from datetime import datetime
        exp_name = f"{exp_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
        use_wandb = logging_cfg.get("use_wandb", False)
        
        if use_wandb:
            self._logger = wandb_logger.WandbLogger(project_name=project_name, 
                                                    exp_name=exp_name, 
                                                    connect_to_wandb=True, 
                                                    param_config=config)
        else:
            self._logger = tb_logger.TBLogger()
            
        self._logger.set_step_key("Iteration")

        # CVAE module
        self._cvae_module = cvae_skill.CVAESkillModule(config, env, device)

        # Training config
        cvae_cfg = config.get("cvae", {})
        self._batch_size = cvae_cfg.get("batch_size", 64)
        self._num_batches = cvae_cfg.get("num_batches", 4)
        self._update_interval = cvae_cfg.get("update_interval", 10)

        # Buffer config
        buffer_size = cvae_cfg.get("buffer_size", 10000)
        self._num_envs = env._num_envs
        self._buffer_length = max(1, buffer_size // self._num_envs)  # number of timesteps
        self._total_buffer_size = self._buffer_length * self._num_envs  # T * N

        # Get observation dimensions from env
        obs_shapes = self._env._compute_obs(ret_obs_shapes=True)
        self._obs_dims = {}
        separated_keys = {"tar_obs", "tar_contacts", "char_contacts", "hf", "target_xy"}

        # char_obs dimension: all keys except separated ones
        char_obs_dim = 0
        for key in obs_shapes:
            if key not in separated_keys:
                char_obs_dim += int(np.prod(obs_shapes[key]["shape"]))
        self._obs_dims["char_obs"] = char_obs_dim
        self._obs_dims["tar_obs"] = int(np.prod(obs_shapes["tar_obs"]["shape"])) if "tar_obs" in obs_shapes else 0
        self._obs_dims["tar_contacts"] = int(np.prod(obs_shapes["tar_contacts"]["shape"])) if "tar_contacts" in obs_shapes else 0
        self._obs_dims["char_contacts"] = int(np.prod(obs_shapes["char_contacts"]["shape"])) if "char_contacts" in obs_shapes else 0
        self._obs_dims["hf"] = int(np.prod(obs_shapes["hf"]["shape"])) if "hf" in obs_shapes else 0
        self._obs_dims["heading"] = int(np.prod(obs_shapes["target_xy"]["shape"])) if "target_xy" in obs_shapes else 0

        # Action dimension (use first action shape)
        action_dim = env._action_space.shape[0] if hasattr(env._action_space, 'shape') else env.action_space.shape[0]
        self._obs_dims["action"] = action_dim

        # Teacher buffer: pre-allocated tensors (T, N, dim)
        self._teacher_buffer = {
            "char_obs": torch.zeros(self._buffer_length, self._num_envs, self._obs_dims["char_obs"]),
            "tar_obs": torch.zeros(self._buffer_length, self._num_envs, self._obs_dims["tar_obs"]) if self._obs_dims["tar_obs"] > 0 else None,
            "tar_contacts": torch.zeros(self._buffer_length, self._num_envs, self._obs_dims["tar_contacts"]) if self._obs_dims["tar_contacts"] > 0 else None,
            "char_contacts": torch.zeros(self._buffer_length, self._num_envs, self._obs_dims["char_contacts"]) if self._obs_dims["char_contacts"] > 0 else None,
            "hf": torch.zeros(self._buffer_length, self._num_envs, self._obs_dims["hf"]) if self._obs_dims["hf"] > 0 else None,
            "heading": torch.zeros(self._buffer_length, self._num_envs, self._obs_dims["heading"]) if self._obs_dims["heading"] > 0 else None,
            "action": torch.zeros(self._buffer_length, self._num_envs, self._obs_dims["action"]),
        }
        # Flat views for efficient sampling: (T*N, dim)
        self._teacher_buffer_flat = {}
        for key in self._teacher_buffer:
            if self._teacher_buffer[key] is not None:
                self._teacher_buffer_flat[key] = self._teacher_buffer[key].view(-1, self._obs_dims[key])
        self._buffer_head = 0
        self._teacher_buffer_samples = 0

        # DAgger config
        dagger_cfg = cvae_cfg.get("dagger", {})
        self._dagger_enabled = dagger_cfg.get("enabled", False)
        self._dagger_iterations = dagger_cfg.get("num_iterations", 10)
        self._dagger_student_steps = dagger_cfg.get("student_rollouts_per_iter", 1000)
        self._dagger_expert_steps = dagger_cfg.get("expert_rollouts_per_iter", 100)
        self._dagger_mix_ratio = dagger_cfg.get("mix_ratio", 0.5)
        dagger_buffer_size = dagger_cfg.get("buffer_size", 50000)
        self._dagger_buffer_length = max(1, dagger_buffer_size // self._num_envs)
        self._dagger_total_buffer_size = self._dagger_buffer_length * self._num_envs

        # DAgger buffer: pre-allocated tensors (T, N, dim)
        self._dagger_buffer = {
            "char_obs": torch.zeros(self._dagger_buffer_length, self._num_envs, self._obs_dims["char_obs"]),
            "tar_obs": torch.zeros(self._dagger_buffer_length, self._num_envs, self._obs_dims["tar_obs"]) if self._obs_dims["tar_obs"] > 0 else None,
            "tar_contacts": torch.zeros(self._dagger_buffer_length, self._num_envs, self._obs_dims["tar_contacts"]) if self._obs_dims["tar_contacts"] > 0 else None,
            "char_contacts": torch.zeros(self._dagger_buffer_length, self._num_envs, self._obs_dims["char_contacts"]) if self._obs_dims["char_contacts"] > 0 else None,
            "hf": torch.zeros(self._dagger_buffer_length, self._num_envs, self._obs_dims["hf"]) if self._obs_dims["hf"] > 0 else None,
            "heading": torch.zeros(self._dagger_buffer_length, self._num_envs, self._obs_dims["heading"]) if self._obs_dims["heading"] > 0 else None,
            "action": torch.zeros(self._dagger_buffer_length, self._num_envs, self._obs_dims["action"]),
        }
        # Flat views for efficient sampling: (T*N, dim)
        self._dagger_buffer_flat = {}
        for key in self._dagger_buffer:
            if self._dagger_buffer[key] is not None:
                self._dagger_buffer_flat[key] = self._dagger_buffer[key].view(-1, self._obs_dims[key])
        self._dagger_buffer_head = 0
        self._dagger_buffer_samples = 0

        # State
        self._curr_obs = None
        self._curr_info = None
        self._iter = 0

        return

    def _reset_env(self):
        """Reset environment and get initial observation."""
        self._curr_obs, self._curr_info = self._env.reset()
        return

    def _step_env(self, action):
        """Step environment with given action."""
        next_obs, reward, done, info = self._env.step(action)
        return next_obs, reward, done, info

    def _reset_done_envs(self, done):
        """Reset environments that are done."""
        reset_mask = done.any(dim=0) if done.dim() > 1 else done
        if reset_mask.any():
            new_obs, new_info = self._env.reset()
            for i in range(self._num_envs):
                if reset_mask[i]:
                    self._curr_obs = new_obs
                    self._curr_info = new_info
        return self._curr_obs, self._curr_info

    def _extract_obs_part(self, obs, key, obs_shapes=None):
        """Extract a specific observation part from the full observation tensor."""
        if obs_shapes is None:
            obs_shapes = self._env._compute_obs(ret_obs_shapes=True)

        if key not in obs_shapes:
            return None

        shape = obs_shapes[key]["shape"]
        size = int(np.prod(shape))

        curr_offset = 0
        for k, info in obs_shapes.items():
            if k == key:
                start = curr_offset
                end = curr_offset + size
                return obs[..., start:end]
            curr_offset += int(np.prod(info["shape"]))
        return None

    def _build_cvae_batch_from_obs(self, obs):
        """Extract observation parts into CVAE batch format."""
        obs_shapes = self._env._compute_obs(ret_obs_shapes=True)
        separated_keys = {"tar_obs", "tar_contacts", "char_contacts", "hf", "target_xy"}

        # Extract char_obs (all keys except target keys)
        char_obs_parts = []
        for key in obs_shapes:
            if key not in separated_keys:
                part = self._extract_obs_part(obs, key, obs_shapes)
                if part is not None:
                    char_obs_parts.append(part)
        char_obs = torch.cat(char_obs_parts, dim=-1)

        # Extract other observation parts
        tar_obs = self._extract_obs_part(obs, "tar_obs", obs_shapes)
        tar_contacts = self._extract_obs_part(obs, "tar_contacts", obs_shapes)
        char_contacts = self._extract_obs_part(obs, "char_contacts", obs_shapes)
        hf = self._extract_obs_part(obs, "hf", obs_shapes)
        heading = self._extract_obs_part(obs, "target_xy", obs_shapes)

        return {
            "char_obs": char_obs,
            "tar_obs": tar_obs,
            "tar_contacts": tar_contacts,
            "char_contacts": char_contacts,
            "hf": hf,
            "heading": heading
        }

    def _decide_teacher_action(self, obs, info):
        """Get action from frozen teacher model."""
        with torch.no_grad():
            action_dist = self._teacher_model.eval_actor(obs)
            action = action_dist.mode
        return action, {"a_logp": None, "rand_action_mask": None}

    def _store_teacher_sample(self, obs, action):
        """Store sample in teacher buffer."""
        batch = self._build_cvae_batch_from_obs(obs)

        # Store at buffer head position: (T, N, dim) at [head] = (N, dim)
        self._teacher_buffer["char_obs"][self._buffer_head] = batch["char_obs"].cpu()
        if self._teacher_buffer["tar_obs"] is not None:
            self._teacher_buffer["tar_obs"][self._buffer_head] = batch["tar_obs"].cpu()
        if self._teacher_buffer["tar_contacts"] is not None:
            self._teacher_buffer["tar_contacts"][self._buffer_head] = batch["tar_contacts"].cpu()
        if self._teacher_buffer["char_contacts"] is not None:
            self._teacher_buffer["char_contacts"][self._buffer_head] = batch["char_contacts"].cpu()
        if self._teacher_buffer["hf"] is not None:
            self._teacher_buffer["hf"][self._buffer_head] = batch["hf"].cpu()
        if self._teacher_buffer["heading"] is not None:
            self._teacher_buffer["heading"][self._buffer_head] = batch["heading"].cpu()
        self._teacher_buffer["action"][self._buffer_head] = action.cpu()

        # Update buffer head and sample count
        self._buffer_head = (self._buffer_head + 1) % self._buffer_length
        self._teacher_buffer_samples = min(self._teacher_buffer_samples + self._num_envs, self._total_buffer_size)

    def _sample_teacher_batch(self, batch_size):
        """Sample a batch from the teacher buffer."""
        if self._teacher_buffer_samples == 0:
            return None

        # Fix Bug 1: indices must be on CPU, then move sliced data to GPU
        rand_idx = torch.randint(0, self._teacher_buffer_samples, (batch_size,))

        batch = {}
        # Fix Bug 2: iterate over all keys in original buffer to preserve None entries
        for key in self._teacher_buffer:
            if self._teacher_buffer[key] is not None:
                batch[key] = self._teacher_buffer_flat[key][rand_idx].to(self._device)
            else:
                batch[key] = None

        return batch

    def _collect_teacher_rollouts(self, num_steps):
        """Collect teacher rollouts and store in buffer."""
        for _ in range(num_steps):
            action, _ = self._decide_teacher_action(self._curr_obs, self._curr_info)
            self._store_teacher_sample(self._curr_obs, action)
            next_obs, r, done, next_info = self._step_env(action)
            self._curr_obs, self._curr_info = self._reset_done_envs(done)
        return

    def _train_cvae(self):
        """Train CVAE module using teacher buffer."""
        if self._teacher_buffer_samples == 0:
            return {
                "cvae_loss": torch.tensor(0.0, device=self._device),
                "cvae_recon_loss": torch.tensor(0.0, device=self._device),
                "cvae_kl_loss": torch.tensor(0.0, device=self._device)
            }

        self._cvae_module.train()
        total_info = {
            "cvae_loss": torch.tensor(0.0, device=self._device),
            "cvae_recon_loss": torch.tensor(0.0, device=self._device),
            "cvae_kl_loss": torch.tensor(0.0, device=self._device)
        }

        for _ in range(self._num_batches):
            batch = self._sample_teacher_batch(self._batch_size)
            if batch is None:
                break
            loss_info = self._cvae_module.update(batch)
            for k in total_info:
                total_info[k] += loss_info[k]

        for k in total_info:
            total_info[k] = total_info[k] / self._num_batches

        return total_info

    def _store_dagger_sample(self, obs, expert_action):
        """Store sample in dagger buffer."""
        batch = self._build_cvae_batch_from_obs(obs)

        # Store at buffer head position: (T, N, dim) at [head] = (N, dim)
        self._dagger_buffer["char_obs"][self._dagger_buffer_head] = batch["char_obs"].cpu()
        if self._dagger_buffer["tar_obs"] is not None:
            self._dagger_buffer["tar_obs"][self._dagger_buffer_head] = batch["tar_obs"].cpu()
        if self._dagger_buffer["tar_contacts"] is not None:
            self._dagger_buffer["tar_contacts"][self._dagger_buffer_head] = batch["tar_contacts"].cpu()
        if self._dagger_buffer["char_contacts"] is not None:
            self._dagger_buffer["char_contacts"][self._dagger_buffer_head] = batch["char_contacts"].cpu()
        if self._dagger_buffer["hf"] is not None:
            self._dagger_buffer["hf"][self._dagger_buffer_head] = batch["hf"].cpu()
        if self._dagger_buffer["heading"] is not None:
            self._dagger_buffer["heading"][self._dagger_buffer_head] = batch["heading"].cpu()
        self._dagger_buffer["action"][self._dagger_buffer_head] = expert_action.cpu()

        # Update buffer head and sample count
        self._dagger_buffer_head = (self._dagger_buffer_head + 1) % self._dagger_buffer_length
        self._dagger_buffer_samples = min(self._dagger_buffer_samples + self._num_envs, self._dagger_total_buffer_size)

    def _sample_dagger_batch(self, batch_size):
        """Sample a batch from the DAgger buffer."""
        if self._dagger_buffer_samples == 0:
            return None

        # Fix Bug 1: indices must be on CPU, then move sliced data to GPU
        rand_idx = torch.randint(0, self._dagger_buffer_samples, (batch_size,))

        batch = {}
        # Fix Bug 2: iterate over all keys in original buffer to preserve None entries
        for key in self._dagger_buffer:
            if self._dagger_buffer[key] is not None:
                batch[key] = self._dagger_buffer_flat[key][rand_idx].to(self._device)
            else:
                batch[key] = None

        return batch

    def _cvae_rollout_student(self, num_steps):
        """Rollout using CVAE decoder as policy, label with expert."""
        for _ in range(num_steps):
            batch = self._build_cvae_batch_from_obs(self._curr_obs)

            with torch.no_grad():
                cvae_action = self._cvae_module.get_action(batch)

            expert_action, _ = self._decide_teacher_action(self._curr_obs, self._curr_info)

            next_obs, r, done, next_info = self._step_env(cvae_action)
            self._store_dagger_sample(self._curr_obs, expert_action)

            self._curr_obs, self._curr_info = self._reset_done_envs(done)
        return

    def _dagger_expert_rollout(self, num_steps):
        """Collect expert rollouts to expand state coverage."""
        for _ in range(num_steps):
            expert_action, _ = self._decide_teacher_action(self._curr_obs, self._curr_info)
            next_obs, r, done, next_info = self._step_env(expert_action)
            self._store_dagger_sample(self._curr_obs, expert_action)
            self._curr_obs, self._curr_info = self._reset_done_envs(done)
        return

    def _train_cvae_dagger(self):
        """Train CVAE on mixed buffer (dagger + teacher)."""
        dagger_size = self._dagger_buffer_samples
        teacher_size = self._teacher_buffer_samples

        if dagger_size == 0 and teacher_size == 0:
            return {
                "cvae_loss": torch.tensor(0.0, device=self._device),
                "cvae_recon_loss": torch.tensor(0.0, device=self._device),
                "cvae_kl_loss": torch.tensor(0.0, device=self._device)
            }

        self._cvae_module.train()
        total_info = {
            "cvae_loss": torch.tensor(0.0, device=self._device),
            "cvae_recon_loss": torch.tensor(0.0, device=self._device),
            "cvae_kl_loss": torch.tensor(0.0, device=self._device)
        }

        for _ in range(self._num_batches):
            total_size = dagger_size + teacher_size
            if total_size == 0:
                break

            target_dagger_ratio = min(self._dagger_mix_ratio, dagger_size / max(1, total_size))

            if dagger_size > 0 and random.random() < target_dagger_ratio:
                n_dagger = self._batch_size
                n_teacher = 0
            else:
                n_dagger = 0
                n_teacher = self._batch_size

            if n_dagger > 0:
                batch = self._sample_dagger_batch(n_dagger)
            elif n_teacher > 0:
                batch = self._sample_teacher_batch(n_teacher)
            else:
                continue

            if batch is None:
                continue

            loss_info = self._cvae_module.update(batch)
            for k in total_info:
                total_info[k] += loss_info[k]

        num_updates = sum(1 for _ in range(self._num_batches) if (
            (self._dagger_buffer_samples > 0 and random.random() < min(self._dagger_mix_ratio, self._dagger_buffer_samples / max(1, self._dagger_buffer_samples + self._teacher_buffer_samples)))
            or (self._dagger_buffer_samples == 0 and self._teacher_buffer_samples > 0)
        ))
        if num_updates > 0:
            for k in total_info:
                total_info[k] = total_info[k] / num_updates

        return total_info

    def _run_dagger_iteration(self, iteration):
        """Run one DAgger iteration: student rollout + expert rollout + train."""
        print(f"DAgger iteration {iteration + 1}/{self._dagger_iterations}")

        print(f"  Student rollout ({self._dagger_student_steps} steps)...")
        self._cvae_rollout_student(self._dagger_student_steps)

        print(f"  Expert rollout ({self._dagger_expert_steps} steps)...")
        self._dagger_expert_rollout(self._dagger_expert_steps)

        print("  Training CVAE...")
        cvae_info = self._train_cvae_dagger()

        dagger_size = self._dagger_buffer_samples
        teacher_size = self._teacher_buffer_samples
        print(f"  DAgger buffer size: {dagger_size}, Teacher buffer size: {teacher_size}")
        print(f"  CVAE loss: {cvae_info['cvae_loss'].item():.4f}, "
              f"Recon: {cvae_info['cvae_recon_loss'].item():.4f}, "
              f"KL: {cvae_info['cvae_kl_loss'].item():.4f}")

        return cvae_info

    def _log_metrics(self, step, cvae_info):
        """Log training metrics to WandB/TB."""
        self._logger.log("Iteration", step)
        self._logger.log("CVAE_Loss", cvae_info['cvae_loss'].item())
        self._logger.log("Recon_Loss", cvae_info['cvae_recon_loss'].item())
        self._logger.log("KL_Loss", cvae_info['cvae_kl_loss'].item())
        if 'cvae_z_mean' in cvae_info:
            self._logger.log("Z_Mean", cvae_info['cvae_z_mean'])
            self._logger.log("Z_Std", cvae_info['cvae_z_std'])
        self._logger.print_log()
        self._logger.write_log()

    def train(self, max_iterations, output_dir, save_interval=10):
        """Run CVAE training loop."""
        os.makedirs(output_dir, exist_ok=True)
        self._reset_env()

        log_file = os.path.join(output_dir, "cvae_log.txt")
        self._logger.configure_output_file(log_file)

        print(f"Starting CVAE training for {max_iterations} iterations")
        print(f"DAgger enabled: {self._dagger_enabled}")

        global_step = 0
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")

            if self._dagger_enabled:
                # DAgger training
                for i in range(self._dagger_iterations):
                    cvae_info = self._run_dagger_iteration(i)
                    global_step += 1
                    self._log_metrics(global_step, cvae_info)
            else:
                # Standard training
                print("Collecting teacher rollouts...")
                self._collect_teacher_rollouts(self._dagger_student_steps)
                cvae_info = self._train_cvae()
                global_step += 1
                self._log_metrics(global_step, cvae_info)

            # Save checkpoint

            # Save checkpoint
            if (iteration + 1) % save_interval == 0:
                save_path = os.path.join(output_dir, f"cvae_iter_{iteration + 1}.pt")
                torch.save(self._cvae_module.state_dict(), save_path)
                print(f"Saved CVAE checkpoint to {save_path}")

        # Final save
        final_path = os.path.join(output_dir, "cvae_final.pt")
        torch.save(self._cvae_module.state_dict(), final_path)
        print(f"Saved final CVAE to {final_path}")

        return

    def save(self, path):
        """Save CVAE weights."""
        torch.save(self._cvae_module.state_dict(), path)
        return


def train_cvae(config_path):
    """Main entry point for CVAE training."""
    config = load_config(config_path)

    # Get paths
    cvae_cfg = config.get("cvae", {})
    teacher_cfg = cvae_cfg.get("teacher_checkpoint", {})
    output_dir = config.get("output_dir", "/home/phi/CLX/projects/humanoid_loco/tests/parc/cvae_training/")

    # Determine env and agent configs from teacher checkpoint location
    # Configs are in the same folder as the checkpoint
    teacher_checkpoint_path = teacher_cfg.get("checkpoint_path")
    if teacher_checkpoint_path and os.path.exists(teacher_checkpoint_path):
        checkpoint_dir = os.path.dirname(teacher_checkpoint_path)
        # Go up from checkpoints/ to tracker folder
        tracker_dir = os.path.dirname(checkpoint_dir)
        env_config_path = os.path.join(tracker_dir, "dm_env.yaml")
        agent_config_path = os.path.join(tracker_dir, "agent_config.yaml")
    else:
        # Fallback to default location
        env_config_path = PARC_PATH / "PARC" / "tracker_config" / "dm_env_g1.yaml"
        agent_config_path = PARC_PATH / "PARC" / "tracker_config" / "dm_agent_g1.yaml"

    # Load agent config to get model config for teacher
    with open(agent_config_path, 'r') as f:
        agent_config = yaml.safe_load(f)

    # Override agent config with cvae config (since we need the model for teacher)
    agent_config["cvae"] = cvae_cfg

    # Setup
    device = config.get("device", "cuda:0")
    num_envs = config.get("num_envs", 512)

    print("Building environment...")
    env = build_env(str(env_config_path), num_envs, device)

    print("Loading teacher model...")
    teacher_checkpoint_path = teacher_cfg.get("checkpoint_path")
    if teacher_checkpoint_path is None or not os.path.exists(teacher_checkpoint_path):
        print(f"Warning: Teacher checkpoint not found at {teacher_checkpoint_path}")
        print("Using random teacher model (CVAE will still train but no meaningful distillation)")
        teacher_model = None
    else:
        teacher_model = load_teacher_model(teacher_checkpoint_path, agent_config, env, device)

    # Create trainer
    trainer = CVAETrainer(agent_config, env, teacher_model, device)

    # Run training
    max_iterations = config.get("max_iterations", 100)
    save_interval = config.get("save_interval", 10)

    trainer.train(max_iterations, output_dir, save_interval)

    print("CVAE training complete!")
    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone CVAE Training")
    parser.add_argument("--config", type=str, default="PARC/cvae_config/cvae_g1.yaml",
                        help="Path to CVAE config file")
    args = parser.parse_args()

    train_cvae(args.config)
