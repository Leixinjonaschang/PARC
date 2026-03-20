# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a humanoid robot locomotion research project with two main components:
- **PARC**: Core project - Physics-based Augmentation with Reinforcement Learning for Character Controllers (SIGGRAPH 2025)
- **GMR**: General Motion Retargeting library - converts human motion to robot motions, which serves for the training in PARC

## Directory Structure

```text
humanoid_loco/
├── GMR/                            # Motion retargeting to humanoid robots.
├── Data/                           # Motion datasets, retargeted bounds, and SMPL-X exports.
├── terrain-generator/              # Terrain generation utilities.
├── tests/                          # Test files and configuration iterations.
├── PARC/                           # Core project: Physics-based Augmentation with RL
│   ├── CLAUDE.md                   # AI AGENT ROOT PROMPT: Contains project architecture... READ FIRST.
│   ├── README.md                   # General project setup, IsaacGym installation instructions.
│   ├── parc_0_setup_iter.py        # Utility to scaffold config files for a new PARC iteration.
│   ├── parc_1_train_gen.py         # Stage 1: Train Motion Diffusion Model (MDM).
│   ├── parc_2_kin_gen.py           # Stage 2: Generate pure kinematic reference motions with MDM.
│   ├── parc_3_tracker.py           # Stage 3: Train Low-Level Physics Tracker in IsaacGym (PPO).
│   ├── parc_4_phys_record.py       # Stage 4: Record successful physics-based rollouts.
│   ├── cvae_training.py            # Standalone CVAE training script (DAgger or standard).
│   ├── cvae_config/                # CVAE configuration files.
│   │   └── cvae_g1.yaml            # CVAE hyperparameters for G1 robot.
│   ├── run.py                      # Underlying RL training execution engine.
│   │
│   ├── learning/                   # Core Reinforcement Learning, PPO, and Architectures
│   │   ├── base_agent.py           # Abstract RL agent base (buffers, reward tracking).
│   │   ├── ppo_agent.py            # Standard PPO math (Actor/Critic loss, TD-lambda).
│   │   ├── dm_ppo_agent.py         # Custom DeepMimic-style PPO (with optional CVAE loading).
│   │   ├── dm_ppo_model.py         # Actor/Critic nets & Future Pose Predictor MLPs.
│   │   ├── obs_encoder.py          # Modular observation encoders.
│   │   └── modules/                
│   │       ├── cvae_skill.py       # Core CVAE module: Encoder, Decoder, Reparameterization.
│   │       └── mlp_encoder.py      # Utility MLP blocks used by obs_encoder.
│   │
│   ├── envs/                       # IsaacGym Simulators, Physics, and Reward Calculation
│   │   ├── base_env.py             # Base IsaacGym API wrapper.
│   │   ├── ig_char_env.py          # Computes physical character states (char_obs, forces).
│   │   └── ig_parkour/                 
│   │       ├── ig_parkour_env.py   # Stage 3 Env: Computes tracking rewards, stitches states.
│   │       └── mgdm_dm_util.py     # Heightfield calculations, relative Heightfield raycasting.
│   │
│   ├── anim/                       # Kinematics and Reference Data Processors
│   │   ├── motion_lib.py           # Loads and interpolates raw motion dataset files.
│   │   └── kin_char_model.py       # Pure kinematic version of the robot.
│   │
│   └── diffusion/                  # Motion Diffusion Model (MDM) Implementation
```

## Quick Navigation for AI Tasks
- **CVAE Training**: See `cvae_training.py` and `cvae_config/cvae_g1.yaml`.
- **CVAE Module**: See `learning/modules/cvae_skill.py`.
- **Observation Space or Heightfield Issues**: See `envs/ig_parkour/ig_parkour_env.py` and `envs/ig_parkour/mgdm_dm_util.py`.
- **Policy/Network Architecture Changes**: See `learning/dm_ppo_model.py` and Config `.yaml` files.
- **Motion Data/Interpolation Bugs**: See `anim/motion_lib.py` and `anim/kin_char_model.py`.

## PARC (Core Project)

PARC is a physics-based character controller using RL. It consists of 4 main training stages:

### Training Pipeline Commands

```bash
# Stage 1: Train motion generator (diffusion model)
python parc_1_train_gen.py --config path/to/config

# Stage 2: Generate kinematic motions
python parc_2_kin_gen.py --config path/to/config

# Stage 3: Train tracking policy
python parc_3_tracker.py --config path/to/config

# Stage 4: Record physics data
python parc_4_phys_record.py --config path/to/config
```

### Setup for New Iteration
```bash
python parc_0_setup_iter.py
```

### Environment Setup

PARC requires IsaacGym and Python 3.8.19:
```bash
conda create -n parc python=3.8.19 -y
conda activate parc
pip install -r PARC/requirements.txt
# Install IsaacGym (download from NVIDIA developer website)
```

### Key PARC Modules

- **`anim/motion_lib.py`**: Motion data loading and processing
- **`anim/kin_char_model.py`**: Kinematic character model
- **`learning/`**: RL training (PPO agent, networks, normalizers)
- **`envs/`**: IsaacGym environments (base, deepmimic, parkour)
- **`diffusion/`**: Motion diffusion model (MDM) implementation

### Motion Visualization

```bash
# Use Motion Forge (built on Polyscope)
python PARC/motion_forge.py
# Edit motionscope/motion_forge_config.yaml with motion_filepath
```

## Architecture Summary

**PARC Training Flow:**
1. Motion diffusion model (MDM) generates motion embeddings
2. Kinematic generation produces joint trajectories
3. RL policy (PPO) learns to track motions in physics simulation
4. Final model can be deployed on real robots

**Data Pipeline:**
- Raw motion (AMASS/LAFAN1) → GMR retargeting → Robot motion → PARC training
- Motion data loaded via `anim/motion_lib.py` and `anim/kin_char_model.py`

## Dependencies

- **IsaacGym**: Physics simulation (PARC)
- **MuJoCo**: Physics + visualization (GMR)
- **PyTorch**: 1.13.1 (PARC)
- **SMPL-X**: Human body model
- **mink**: IK solver for motion retargeting

## Branch: feature/encode_contact

This branch implements **future pose prediction** via teacher-student distillation for the PARC tracking policy.

### Teacher-Student Distillation (Current Branch)

The distillation allows the tracker to infer future poses without having actual future target observations - useful for real-time control with latency.

**Teacher** (`future_pose` encoder): Encodes target observations into a latent space.
- **Input**: `tar_obs` (1098-dim), `tar_contacts` (216-dim) → concatenated 1314-dim
- **Output**: 32-dim latent vector

```yaml
obs_encoders:
  future_pose:
    type: mlp
    latent_dim: 32
    hidden_dims: [512, 512, 256]
    input_keys: ["tar_obs", "tar_contacts"]
```

**Student** (predictor MLP): Predicts the teacher's latent from partial observations.
- **Input**: All raw obs EXCEPT `tar_obs` and `tar_contacts` — i.e., `root_rot`(6), `root_vel`(3), `root_ang_vel`(3), `joint_rot`(174), `dof_vel`(29), `key_pos`(15), `char_contacts`(36), `hf`(441), `target_xy`(2), `replan_t`(1)
- **Output**: 32-dim latent (same space as teacher encoder)

**Training** (`learning/dm_ppo_model.py`):
- Teacher provides target latent (detached, no gradient)
- Student predicts latent from current observations
- MSE loss aligns student output with teacher

**Inference**: Can use predicted latent to replace actual future pose when `--test_predictor True` is passed.

### Observation Structure

The actor network receives processed observations from the `ObservationEncoder`. Based on G1 config (29 DOF, 36 bodies, 6 future steps):

**Encoded (via `future_pose` encoder, 32-dim latent)**:
| Component | Shape | Description |
|-----------|-------|-------------|
| `tar_obs` | 1098 | Future target pose (6 steps × [root_pos(3) + root_rot(6) + joint_rot(174)]) |
| `tar_contacts` | 216 | Future contacts (36 bodies × 6 steps) |

**Pass-through (unchanged)**:
| Component | Shape | Description |
|-----------|-------|-------------|
| `root_rot` | 6 | Root rotation (quaternion → tangent-normalized) |
| `root_vel` | 3 | Root linear velocity (local frame) |
| `root_ang_vel` | 3 | Root angular velocity (local frame) |
| `joint_rot` | 174 | Joint rotations (29 DOF × 6) |
| `dof_vel` | 29 | Joint velocities |
| `key_pos` | 15 | Key body positions (5 bodies × 3, local to root) |
| `char_contacts` | 36 | Current character contacts (36 bodies) |
| `hf` | 441 | Heightfield (63 points × 7 rays) |
| `target_xy` | 2 | Localized target position (if enabled) |
| `replan_t` | 1 | Replan timer (MGDM only) |

**Total**: ~2024 raw dims → ~774 encoded dims (varies with config)

### Terrain Height Observation (`hf`)

**Shape**: 441 dimensions (63 × 7 rays)

**Configuration** (from `dm_env_g1.yaml`):
| Parameter | Value |
|-----------|-------|
| `ray_points_ahead` | 60 |
| `ray_points_behind` | 2 |
| `ray_num_left` | 3 |
| `ray_num_right` | 3 |
| `ray_angle` | 0.262 rad (~15°) |
| `ray_dx` | 0.05 |
| `min_obs_h` | -3.0 |
| `max_obs_h` | 3.0 |

**How it works** (`envs/ig_parkour/mgdm_dm_util.py:_refresh_ray_obs_hfs`):
- Cone-shaped sampling grid centered on character
- Points sampled from terrain heightfield via `get_local_hf_from_terrain()`
- **Relative to character ROOT position** (not ground): `ray_hfs - char_root_pos_xyz[..., 2]`
- Clamped to `[min_obs_h, max_obs_h]`

**Out-of-bounds behavior** (`util/terrain_util.py`):
- Indices clamped to terrain boundaries via `get_inbounds_grid_index()`
- Returns boundary height value (typically 0.0) for points outside terrain
- No NaN values possible

**Visualization** (`envs/ig_parkour/ig_parkour_env.py:_draw_local_hf`):
- Enable with `--debug_visuals True` or `debug_visuals: true` in config
- Red points = ray samples, Blue points = MGDM reference trajectory
- Draws in env 0 using global/world coordinates

### Key Files (Current Branch)
- `learning/obs_encoder.py`: Modular observation encoder with MLP encoders
- `learning/modules/mlp_encoder.py`: MLP encoder module
- `learning/dm_ppo_model.py`: Predictor training and inference
- `PARC/tracker_config/dm_agent_g1.yaml`: G1 agent config with predictor settings


## Branch: test/cvae_latent

### Implementation Plan (Skill Latent Plan)
This plan outlines the process of implementing a Conditional Variational Autoencoder (CVAE) for latent skill space construction.

1. Encoder Architecture

The encoder is responsible for mapping the input (robot state, target observations, target contacts, current contacts, and character observations) to a latent skill representation.

Encoder Inputs:

tar_obs: Target observations (future target pose, root position, root rotation, joint rotations for 6 steps).

tar_contacts: Target contact information (contact points of the robot with the environment).

char_obs: Character observations (robot's current state, joint positions, velocities, etc.).

char_contacts: Current contact points or observations regarding interactions with the environment.

Encoder Output:

mu: Mean of the latent distribution.

sigma: Standard deviation of the latent distribution.

Encoder Network Architecture:

Input: Concatenate tar_obs, tar_contacts, char_obs, and char_contacts to form a single input vector.

Hidden Layers:

Layer 1: 1024 units

Layer 2: 1024 units

Layer 3: 512 units

Layer 4: 128 units

Output:

Mean 

μ and Standard Deviation 
σ of latent distribution.

Latent dimension: 128 

- Input of the decoder:
 - latent sample from the distribution constructed by mu and sigma (reparameterization trick)
 - condition:
  - heading (target_xy)
  - char_obs
  - char_contacts
  - hf

2. Decoder Architecture

The decoder takes the latent variable and reconstructs the output (robot actions). The decoder is conditioned on latent skill and various robot states and environmental data.

Decoder Inputs:

Latent sample: A sample from the latent distribution , generated using the reparameterization trick.

Conditioning information:

heading (target_xy): Target position.

char_obs: Character observations (robot's state).

char_contacts: Character contact information.

hf: Heightfield (terrain information).

Decoder Output:

a_t: The action generated by the robot (e.g., joint angles, velocities).

Decoder Network Architecture:

Input: Concatenate the latent sample z_t

 with the conditioning information: heading (target_xy), char_obs, char_contacts, and hf.

Hidden Layers:

Layer 1: 128 units (for latent skill)

Layer 2: 512 units

Layer 3: 2048 units

Layer 4: 1024 units

Layer 5: 512 units

Output: Action dimension

3. Learnable Prior Network

The learnable prior is modeled as a conditional distribution that depends on the robot's current state (char_obs,hf)

Prior Network Inputs: s_t: Robot's current state (char_obs,hf)
rior Network Output:
mu_p(char_obs,hf): Mean of the prior distribution.
sigma_p(char_obs,hf): Standard deviation of the prior distribution.

Prior Network Architecture:
Input: Robot state (char_obs,hf)
Hidden Layers:

Layer 1: 512 units

Layer 2: 256 units

Output: mu_p(char_obs,hf) and sigma_p(char_obs,hf)

4. Reparameterization Trick

To sample from the latent variable distribution, the reparameterization trick is applied to the output of the encoder.
z_t = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
This allows for backpropagation through the sampling process, enabling end-to-end training.

The loss function for training the CVAE consists of two terms:

Reconstruction Loss:
This measures the difference between the decoded actions and the real actions:
L_{\text{recon}} = \mathbb{E}\left[\|a_t - a_t^{\text{real}}\|^2\right]

KL Divergence Loss:
This regularizes the latent space to ensure the posterior q(z_t | s_t, \tilde s_{t+1}) stays close to the prior p(z_t^p | s_t)
L_{\text{KL}} = D_{\text{KL}}\left(q(z_t | s_t, \tilde s_{t+1}) \| p(z_t^p | s_t)\right)
L_{\text{total}} = L_{\text{recon}} + \lambda \cdot L_{\text{KL}}

the original tracker policy is treated as teacher. 

### CVAE Training (Standalone)

CVAE training has been decoupled from tracker training. Use the standalone `cvae_training.py` script to train the CVAE.

**Configuration**: All CVAE settings use the unified `cvae:` namespace.

**CVAE Config** (`PARC/cvae_config/cvae_g1.yaml`):
```yaml
cvae:
  enabled: True
  latent_dim: 128
  encoder_hidden_dims: [1024, 1024, 512]
  decoder_hidden_dims: [512, 2048, 1024, 512]
  prior_hidden_dims: [512, 256]
  beta: 0.1
  learning_rate: 1e-4
  dagger:
    enabled: True
    num_iterations: 10
    student_rollouts_per_iter: 1000
    expert_rollouts_per_iter: 100
    mix_ratio: 0.5
  teacher_checkpoint:
    checkpoint_path: "/path/to/tracker.pt"

output_dir: "../tests/parc/cvae_training/"
save_interval: 10
```

**Tracker Config** (`PARC/PARC/tracker_config/dm_agent_g1.yaml`):
```yaml
# CVAE loads pre-trained weights (no training during tracker training)
cvae:
  enabled: True
  checkpoint: null  # Set path after training CVAE
  # Architecture params (must match trained model)
  latent_dim: 128
  encoder_hidden_dims: [1024, 1024, 512]
  decoder_hidden_dims: [512, 2048, 1024, 512]
  prior_hidden_dims: [512, 256]
  beta: 0.1
  learning_rate: 1e-4
```

**Training CVAE:**

```bash
# 1. Train CVAE (DAgger or standard)
python PARC/cvae_training.py --config PARC/cvae_config/cvae_g1.yaml

# 2. Set the checkpoint path in dm_agent_g1.yaml, then run tracker training
# CVAE loads from checkpoint, no training occurs
python parc_3_tracker.py --config PARC/PARC/tracker_config/dm_agent_g1.yaml
```

**Key Files:**
- `PARC/cvae_training.py`: Standalone CVAE training script (DAgger + standard)
- `PARC/cvae_config/cvae_g1.yaml`: CVAE hyperparameters and training config
- `learning/modules/cvae_skill.py`: CVAESkillModule class (Encoder, Decoder, Prior) - reads config from `cvae:`
- `learning/dm_ppo_agent.py`: CVAE loading (no training) during tracker training