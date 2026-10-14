# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PARC (Physics-based Augmentation with Reinforcement Learning for Character Controllers, SIGGRAPH 2025), built on MimicKit, adapted here for the Unitree G1 humanoid (29 DOF). This repo is PARC only. Training data comes from **GMR** (General Motion Retargeting: AMASS/LAFAN1 → robot motion), which lives outside this repo. Configs reference sibling directories by relative path: `../Data/` (datasets) and `../tests/parc/` (experiment outputs).

## Environment

Managed with [uv](https://docs.astral.sh/uv/): Python 3.8.20 (`.python-version`), torch 1.13.1 (CUDA 11.7). `pyproject.toml` + `uv.lock` are the source of truth; `requirements.txt` and the conda steps in `README.md` are upstream's.
```bash
uv python install     # pinned CPython 3.8.20
uv sync               # create .venv from uv.lock
# IsaacGym Preview 4 (not on PyPI): download from NVIDIA, extract anywhere, link it into .venv
realpath /path/to/isaacgym/python > .venv/lib/python3.8/site-packages/isaacgym.pth
uv run python -c "from isaacgym import gymapi, gymtorch; import torch; print('ok')"
uv add <package>      # add a dependency (updates pyproject.toml and uv.lock)
```
- Run scripts with `uv run python ...`, which puts `.venv/bin` (including `ninja`, needed by the `gymtorch` JIT build) on `PATH`.
- The `isaacgym.pth` link survives `uv sync`/`uv run`; redo it only if `.venv` is recreated. Don't `uv pip install` IsaacGym: the next `uv sync` uninstalls it.
- No `LD_LIBRARY_PATH` is needed (uv's Python links `libpython3.8.so.1.0` dynamically). Don't put CUDA 11/cuDNN 8 dirs on it: a stray `libcublasLt.so.11` breaks torch.
- Import `isaacgym` before `torch` (the entry scripts already do).

**Run everything from the repo root.** Paths in scripts and configs are relative to it (`PARC/tracker_default.yaml`, `data/assets/unitree_g1/...`).

There is no unit test suite or linter. `tools/motion_tests/` holds metric scripts (MDM validation, loss/stat computation), not tests.

## Commands

### PARC iteration pipeline
```bash
uv run python parc_0_setup_iter.py                       # writes per-stage configs into <output_dir>/p1_train_gen, p2_kin_gen, p3_tracker, p4_phys_record
uv run python parc_1_train_gen.py --config path/to/cfg   # train motion diffusion model (MDM)
uv run python parc_2_kin_gen.py   --config path/to/cfg   # generate kinematic motions on terrain
uv run python parc_3_tracker.py   --config path/to/cfg   # train tracking policy (PPO)
uv run python parc_4_phys_record.py --config path/to/cfg # record physically-corrected motions
uv run python parc_5_create_dataset.py                   # build a dataset YAML (class-weighted) from motion folders; reads PARC/create_dataset_config.yaml
```
`parc_0_setup_iter.py` takes no CLI args. Edit the paths and flags at the top of the file (`output_dir`, `input_model_path`, `input_dataset_folder_paths`, `write_*`). Without `--config`, each stage script falls back to its `PARC/*_default.yaml`.

### Running the RL agent directly (`run.py`)
`parc_3_tracker.py` and `parc_4_phys_record.py` rewrite the env/agent YAMLs into the output dir, then call `run.main()` with a generated argv (saved to `<output_dir>/train_args.txt`). To test a trained tracker:
```bash
uv run python run.py --env_config <dir>/dm_env.yaml --agent_config <dir>/agent_config.yaml \
  --model_file <dir>/checkpoints/model_XXXX.pt --mode test --visualize True --num_envs 128 --device cuda:0
# or put the args in a file:  uv run python run.py --arg_file test_tracker_args.txt
```
Key `run.py` args: `--mode {train,test,test2,record}`, `--num_workers N` (multi-process, spawn), `--debug_visuals True` (draw heightfield rays), `--test_predictor True` (swap the teacher latent for the student prediction), `--logger tb`, `--rand_seed`.

### Visualization
```bash
uv run python motion_forge.py   # Motionscope/Motion Forge (Polyscope); configure MOTION_FORGE/motion_forge_config.yaml (motion_filepath, MDM model, char model)
```
Other utilities are in `tools/`: `terrain_tool/view_terrain.py`, `motion_file_proc/pkl_preview.py`, `char_model/print_model_info.py`, and `motion_opt/`.

## Architecture

### Config-driven builders
- `envs/env_builder.py` dispatches on `env_name` in the env YAML. Only `ig_parkour` (`IGParkourEnv`) is registered.
- `learning/agent_builder.py` dispatches on `agent_name`: `PPO` or `DM_PPO` (`DMPPOAgent`, used for tracking).
- Tracker configs: `PARC/tracker_config/dm_env_g1.yaml` and `dm_agent_g1.yaml` (G1); `*_default.yaml` is the original humanoid. Stage configs live in `PARC/*.yaml`.

### Environment stack
`BaseEnv` → `IGEnv` → `IGCharEnv` → `IGParkourEnv` (`envs/ig_parkour/ig_parkour_env.py`). `IGParkourEnv` **composes** two sub-envs that share the IsaacGym sim and split `num_envs` between them. Both derive from `RefCharEnv` (`mgdm_dm_util.py`):
- `DeepMimicEnv` (`dm_env.py`, config key `env.dm`): tracks a fixed motion dataset (`motion_file`). Stage 3 uses this. The G1 config is DM-only (`fraction_dm_envs: 1.0`).
- `MotionGenDeepMimicEnv` (`mgdm_env.py`, config key `env.mgdm`): tracks motions the MDM generates online, with replanning (`replan_t` obs).

Shared observation code, including heightfield ray sampling, is in `mgdm_dm_util.py`.

### Learning stack
`BaseAgent` → `PPOAgent` → `DMPPOAgent`. Models: `BaseModel` → `PPOModel` → `DMPPOModel`. The env returns a flat obs tensor; `env._compute_obs(ret_obs_shapes=True)` gives the per-key shapes in concatenation order. `learning/obs_encoder.py` (`ObservationEncoder`) slices the flat obs by those offsets, routes configured keys through encoders (`learning/modules/mlp_encoder.py`) and passes the rest through unchanged before the actor/critic MLPs (`learning/nets/`).

### Motion generation
`diffusion/`: the MDM transformer (`mdm.py`, `mdm_transformer.py`) conditions on terrain heightfield and contacts (`mdm_heightfield_contact_motion_sampler.py`). Motion data is loaded through `anim/motion_lib.py` and `anim/kin_char_model.py`.

## Teacher-Student Future-Pose Distillation

This work came from `feature/encode_contact` and is now merged. It lets the tracker run without future target observations.

- **Teacher**: the `future_pose` encoder (`model.obs_encoders` in `dm_agent_g1.yaml`) maps `tar_obs` + `tar_contacts` (1620 dims for G1) to a 32-dim, L2-normalized latent.
- **Student**: the predictor MLP in `learning/dm_ppo_model.py` predicts that latent from all other obs keys (`char_obs`, `char_contacts`, `hf`: 743 dims). It is trained with MSE against the detached teacher latent, using its own Adam optimizer (lr hardcoded to 1e-4), and logged as `Predictor_Loss`.
- **Known bug**: `DMPPOAgent` reads `future_pose_predictor` from the top level of the agent config, but `dm_agent_g1.yaml` nests it under `model:`. So `enabled` (and `learning_rate`) are ignored and the predictor always trains.
- At inference, `--test_predictor True` calls `set_predictor_test_mode()` to swap in the predicted latent.

### Observation keys (G1 config: 29 DOF, 36 bodies, 6 future steps)
Flat obs order = key order below. Joint rotations cover all 35 non-root joints, including the 6 fixed ones.

| Key | Dim | Contents |
|-----|-----|----------|
| `char_obs` | 266 | root_rot 6 + root_vel 3 + root_ang_vel 3 + joint_rot 35×6 + dof_vel 29 + key_pos 5×3 |
| `tar_obs` | 6×234 = 1404 | per future step: root_pos 3 + root_rot 6 + joint_rot 35×6 + key_pos 5×3 |
| `tar_contacts` | 6×36 = 216 | reference contacts, not normalized |
| `char_contacts` | 36 | not normalized |
| `hf` | 441 | 7 rays × 63 points, not normalized |

Total 2363 raw dims → 775 after encoding (`char_obs` 266 + latent 32 + `char_contacts` 36 + `hf` 441). `target_xy` (2) and `replan_t` (1) are appended only when target obs or MGDM envs are enabled, which the G1 config doesn't do.

### Heightfield obs (`hf`)
Built by `_refresh_ray_obs_hfs` in `mgdm_dm_util.py`. Samples a cone-shaped ray grid (from `dm_env_g1.yaml`: 60 points ahead, 2 behind, 3 rays left, 3 right, 15° spacing, `ray_dx=0.05`). Heights are **relative to the character root height**, not the ground, and clamped to `[-3, 3]`. Out-of-bounds samples are clamped to the terrain edge by `get_inbounds_grid_index()` in `util/terrain_util.py`, so values are never NaN. Pass `--debug_visuals True` to draw them (red = ray samples of every env, blue = MGDM reference path).
