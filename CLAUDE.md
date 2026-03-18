# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a humanoid robot locomotion research project with two main components:
- **PARC**: Core project - Physics-based Augmentation with Reinforcement Learning for Character Controllers (SIGGRAPH 2025)
- **GMR**: General Motion Retargeting library - converts human motion to robot motions, which serves for the training in PARC

## Directory Structure

```
humanoid_loco/
├── PARC/           # Core RL training framework (main project)
├── GMR/            # Motion retargeting to humanoid robots
├── Data/           # Motion datasets and retargeted data
├── terrain-generator/  # Terrain generation utilities
└── tests/          # Test files
```

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
