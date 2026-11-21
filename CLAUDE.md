# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A fork of upstream PARC (Physics-based Augmentation with Reinforcement Learning for Character Controllers, SIGGRAPH 2025; https://github.com/mshoe/PARC), on upstream's rewritten `parc/` layout. It adds the Unitree G1 humanoid (29 DOF) next to upstream's humanoid, and teacher-student future-pose distillation for the tracking policy. Motion retargeting (GMR: AMASS/LAFAN1 → robot motion) and all datasets live outside the repo. The repo ships humanoid example motions (`data/motion_terrains/`) but no G1 data.

- `doc/porting_plan.md` (Chinese): history of the migration from the old flat-layout `develop` branch, bug lists, and decisions D1–D22 (D13–D22 are still open).
- `doc/parc_guide.md`: upstream's walkthrough of the PARC loop.
- `README.md` is upstream's and describes a conda setup. This fork uses uv.

## Environment

uv project: Python 3.8.20 (`.python-version`), torch 2.2.0 (PyPI's CUDA 12.1 build), `pyproject.toml` + `uv.lock`.

```bash
uv python install     # CPython 3.8.20
uv sync               # .venv from uv.lock, with pytest and editable parc/ + scripts/
# IsaacGym Preview 4 is not on PyPI: extract it anywhere and link its python/ dir into .venv
realpath /path/to/isaacgym/python > .venv/lib/python3.8/site-packages/isaacgym.pth
uv run python -c "from isaacgym import gymapi, gymtorch; import torch; print('ok')"
```

- Don't `uv pip install` IsaacGym: `uv sync` removes packages that aren't in the lock. The `.pth` survives `uv sync`; redo it only when `.venv` is recreated.
- Use `uv run`. It puts `.venv/bin` on `PATH`, and the first `gymtorch` import JIT-builds the extension with `.venv/bin/ninja`; without it on `PATH`, the build fails with `RuntimeError: Ninja is required ...`. The build goes to `TORCH_EXTENSIONS_DIR` (default `~/.cache/torch_extensions/`); give each venv or worktree its own when several run IsaacGym at once. No `LD_LIBRARY_PATH` setup is needed.
- **`PARC_DATA_DIR`**: configs reference datasets, checkpoints and outputs as `$DATA_DIR/...`. `parc/util/path_loader.py` substitutes it in every string of a config loaded with `load_config`, and in `run_tracker` paths. The value comes from the `PARC_DATA_DIR` env var, else `DATA_DIR` in `data/configs/user_config.yaml` (tracked, holding upstream's macOS path, so set the env var instead). It must be an existing absolute directory; every config load asserts that. On a command line, quote it (`'$DATA_DIR/...'`).
- **Run everything from the repo root**, as `uv run python -m scripts.<name>`. Other relative paths in configs (`data/assets/...`, `data/configs/...`) resolve against the working directory.
- **wandb** is the only logger (`run_tracker --logger` is ignored). `use_wandb: false` calls `wandb.init(mode="offline")`, which overrides `WANDB_MODE` and writes a local run to `./wandb/` (or `$WANDB_DIR/wandb/`). Nothing is uploaded. `use_wandb: true` (the value in all shipped configs) logs online unless `WANDB_MODE` says otherwise. Run tests and smoke runs with `WANDB_MODE=disabled`.

## Commands

### Tests

```bash
export WANDB_MODE=disabled
uv run pytest -q                                   # ~350 CPU tests, ~2 min; CUDA-only tests skip without a GPU
uv run pytest tests/test_charmodel_g1.py -q        # one file
uv run pytest tests/test_learning_future_pose.py::test_encoder_maps_g1_obs_to_775_dims   # one test (or -k <expr>)
PARC_MOTIONSCOPE_HEADLESS=1 uv run pytest tests/test_motiontools_motionscope_headless.py  # opt-in GUI test (polyscope EGL, NVIDIA driver)
```

Tests set their own `PARC_DATA_DIR`. They use synthetic data and develop reference values from `tests/fixtures/`. `tests/test_import_sweep.py` imports the modules of `parc/` and `scripts/` in a subprocess, IsaacGym first. The GPU + IsaacGym smoke runs, with commands and reference numbers, are in three READMEs:
- `tests/smoke/README.md`: humanoid stages.
- `tests/smoke/g1_env/README.md`: G1 tracker env (build, train, test with and without the predictor, record, debug visuals).
- `tests/smoke/g1_pipeline/README.md`: G1 stages 0–2 on synthetic motions.

### PARC iteration (G1)

```bash
uv run python -m scripts.parc_0_setup_iter --config data/configs/parc_0_setup_iter_config_g1.yaml
OUT=$PARC_DATA_DIR/tests/parc/experiment_g1/iter_1     # output_dir of that config
uv run python -m scripts.parc_1_train_gen   --config $OUT/p1_train_gen/mdm_config.yaml
uv run python -m scripts.parc_2_kin_gen     --config $OUT/p2_kin_gen/boxes_0_49/kin_gen_config.yaml   # one job per batch: boxes_0_49 ... boxes_450_499
uv run python -m scripts.parc_3_tracker     --config $OUT/p3_tracker/tracker.yaml
uv run python -m scripts.parc_4_phys_record --config $OUT/p4_phys_record/phys_record.yaml
```

Each stage script takes only `--config`. Without it, a script uses upstream's humanoid config (`data/configs/*_default.yaml`, or `parc_0_setup_iter_config.yaml` for stage 0). parc_0 writes the stage configs with `$DATA_DIR` already resolved. Its `char_file` key sets the G1 MJCF in every stage: the MDM config, kin gen's `opt.char_model`, the dataset configs, and `p3_tracker/input_dm_env.yaml` (a copy of `dm_env_g1.yaml`). From the second iteration on, set `input_mdm_model_path`, `input_sampler_stats_path` and `input_tracker_model_path`.

### Running the tracker (`scripts/run_tracker.py`)

parc_3 writes `dm_env.yaml`, `agent_config.yaml` and `train_args.txt` to its output dir and calls `run_tracker.main()`. parc_4 writes `record_env.yaml` and `record_args.txt` and does the same. Test a trained tracker:

```bash
uv run python -m scripts.run_tracker --mode test --visualize False --num_envs 16 --test_episodes 32 \
  --env_config $OUT/p3_tracker/dm_env.yaml --agent_config $OUT/p3_tracker/agent_config.yaml \
  --model_file $OUT/p3_tracker/model.pt --test_predictor True
```

It has a custom parser: `--key value`, booleans as `True`/`False`. Command-line values override those in `--arg_file <file>` (e.g. `train_args.txt`).
- `--mode train|test|record` (default `train`). `record` is what parc_4 runs: it saves the successfully tracked motions as MS files to `env.output_motion_dir`.
- `--test_episodes N`: the default is unbounded, so test mode never ends without it.
- `--visualize` defaults to `True` (IsaacGym viewer, needs a display). Pass `False` for headless runs.
- `--test_predictor True`: the actor uses the student's predicted latent instead of the teacher's (test and record only).
- `--debug_visuals True`: draws the heightfield samples of every env and colours contact bodies. Needs `--visualize True`.
- Others: `--num_envs` (default 1), `--device` (default `cuda:0`), `--max_samples`, `--out_model_file`, `--int_output_dir`, `--log_file`, `--rand_seed`.

### Data tools

```bash
uv run python -m scripts.convert_legacy_motions --inputs <legacy dir> --outputs <new dir> --char-file data/assets/unitree_g1/g1_mocap_29dof.xml
uv run python -m scripts.robot_motion_preprocess --input <GMR output dir> --output <dir> --source_dir <source motions dir>
uv run python -m scripts.create_dataset_yaml --config data/configs/create_dataset_config_g1.yaml
```

- `convert_legacy_motions`: converts develop-era pickles to MS files, recursively, into a different directory. The default `--char-file` is the humanoid; `--dry-run` lists the files.
- `robot_motion_preprocess`: turns retargeted `{frames, fps}` pickles (GMR output; G1 by default) into MS files. It copies terrain and contacts from the source motion of the same name, fixes the height, and adds body constraints and `hf_mask_inds`. With `--allow_missing_source`, unmatched files get flat ground and geometric foot contacts.
- `create_dataset_yaml`: builds a weighted dataset yaml. parc_1, parc_3 and parc_4 do the same from their `create_dataset_config`.
- `scripts/tools/` (all take `--help`):
  - `pkl_preview` and `pkl_preview_example`: inspect an MS or legacy `.pkl`.
  - `view_terrain`: 2d or 3d terrain view.
  - `print_model_info`: body and DOF order of an MJCF.
  - `obj_to_heightfield`: OBJ terrain → `{terrain, path_nodes}` pickle for parc_2's FILE mode.
  - `pkl_extract_frames_fps` and `extract_motion.sh`: export `{frames, fps}`.
  - `rename_files`.
- `scripts/run_optimize_motions.py` and `run_simple_augment_motions.py` run the kinematic optimization or augmentation on MS files (configs in `parc/motion_synthesis/motion_opt/config/`). `scripts/motion_tests/` holds MDM validation and metric scripts, not tests.

### MotionScope

A Polyscope motion and terrain editor and MDM visualizer. Edit `parc/motionscope/motionscope_config.yaml` (`motions: [[<MS file>, humanoid | g1 | <xml path>]]`, `load_mdm`, `mdm_filepaths`), then run `uv run python -m scripts.run_motionscope`. It needs a display and imports IsaacGym on Linux.

## Architecture

### Config-driven builders
- `parc/motion_tracker/envs/env_builder.py`: `build_env(env_file, num_envs, device, visualize, debug_visuals=False)` dispatches on `env_name`. `ig_parkour` (`IGParkourEnv`) is the only env.
- `parc/motion_tracker/learning/agent_builder.py` dispatches on `agent_name`:
  - `PPO` → `PPOAgent`.
  - `DM_PPO` → `DMPPOAgent`, or `FuturePoseDMPPOAgent` when `model.obs_encoders` is non-empty (as in `dm_agent_g1.yaml`).
  - `DM_PPO_FP` → `FuturePoseDMPPOAgent`.
- `data/configs/` holds the stage configs `parc_{0..4}_*.yaml`, `create_dataset_config*.yaml` and `tracker_config/dm_{env,agent}_{default,g1}.yaml`. `*_default` and unsuffixed files are upstream's humanoid; `*_g1` are the G1 variants. Stage 4 has no G1 variant: parc_0 fills in `parc_4_phys_record_default.yaml`.

### Tracker env (`parc/motion_tracker/envs/`)
- The hierarchy is `BaseEnv` → `IGEnv` → `IGCharEnv` → `IGParkourEnv` (`ig_parkour/ig_parkour_env.py`). `IGParkourEnv` composes one `DeepMimicEnv` (`ig_parkour/dm_env.py`, config `env.dm`). It lays out the terrains of all dataset motions as one grid and tracks which motion each env follows.
- Upstream removed the MGDM sub-env (online MDM motions), with its `target_xy`/`replan_t` observations and multiprocess workers. `fraction_dm_envs` must be 1.0 (asserted).
- Observation and reward code is in `ig_parkour/mgdm_dm_util.py` (the name was kept).
- `IGCharEnv` takes joint limits from the kinematic model and PD gains from `env.pd_params` (`char_dof_props.py`). It checks that body and DOF order match IsaacGym's, and prints a per-DOF table (limits, hasLimits, effort, kp, kd) at startup.

### Learning stack (`parc/motion_tracker/learning/`)
Agents: `BaseAgent` → `PPOAgent` → `DMPPOAgent` → `FuturePoseDMPPOAgent` (`dm_ppo_fp_agent.py`). Models: `PPOModel` → `DMPPOModel` → `FuturePoseDMPPOModel` (`dm_ppo_fp_model.py`).

The env returns one flat observation, and `env._compute_obs(ret_obs_shapes=True)` gives the per-key shapes in concatenation order. `ObservationEncoder` (`obs_encoder.py`) slices it. Each encoder's latent (`modules/mlp_encoder.py`, L2-normalized output) replaces its input keys, at the position of the first one. The other keys pass through to the actor and critic MLPs (`parc/util/nets/`).

Future-pose distillation is configured under `model:` in `dm_agent_g1.yaml`:
- Teacher: the obs encoder `future_pose` maps `[tar_obs, tar_contacts]` to a 32-dim latent and is trained by the PPO loss.
- Student: an MLP that maps every raw key the teacher doesn't consume to the teacher latent. It is configured by `model.future_pose_predictor`: `hidden_dims`, `learning_rate` (default 1e-4), `target_encoder` (default `future_pose`).
- It trains on the MSE to the detached teacher latent, with one step per PPO minibatch on its own Adam optimizer. Its parameters are excluded from the main optimizer. It is logged as `Predictor_Loss`.
- `enabled` (bool, default `true`) only switches the predictor's training. The predictor is built whenever the target encoder exists, so checkpoint keys don't depend on `enabled`.
- `--test_predictor True` calls `set_predictor_test_mode(True)`: the actor gets the predicted latent, the critic always the teacher's. It fails if the loaded checkpoint has no predictor weights.
- `load()` is strict except for predictor keys, so develop's generation-C checkpoints load unchanged.
- A NaN in a loss or in the training data raises through `PPOAgent._check_nan`, so the process exits non-zero.

### Motion generator and pipeline stages
- `parc/motion_generator/mdm.py` (`MDM`, transformer in `mdm_transformer.py`) is a diffusion model that generates short windows (`sequence_duration` 0.5 s at 30 fps) autoregressively.
  - Conditions: previous frames, a local heightmap (CNN from `parc/util/nets/`) and a target direction.
  - It trains on `MDMHeightfieldContactMotionSampler` (weighted sampling with heightfield augmentation).
  - Checkpoints are `.ckpt` dicts (`save_checkpoint` / `MDM.load_checkpoint`), including EMA weights when `EMA` is configured.
- Stage 1 (`parc_1_train_gen`): dataset yaml → `p1_train_gen/sampler.pkl`, `sampler_stats.txt`, `checkpoints/model_<epoch>.ckpt` and `final_model.ckpt`.
- Stage 2 (`parc_2_kin_gen`) runs the MDM through these steps:
  1. Builds a procedural terrain (`procgen_mode` BOXES/PATHS/STAIRS/FILE) and plans an A* path on it (`parc/motion_synthesis/procgen/astar.py`).
  2. Generates candidates along the path (`procgen/mdm_path.py`) and keeps the `top_k` under the loss thresholds.
  3. Runs the kinematic optimization (`motion_synthesis/motion_opt/motion_optimization.py`), then removes hesitation frames.
  4. Writes `p2_kin_gen/<batch>/<name>_opt.pkl` plus mirrored `flipped/` copies. Raw generations go to `p2_kin_gen/ignore/raw/`.
- Stage 3 (`parc_3_tracker`): initial + generated motions → `p3_tracker/model.pt` and `checkpoints/model_<iter>.pt`.
- Stage 4 (`parc_4_phys_record`): one env per generated motion. A failed motion is retried from later start times. The successful ones go to `p4_phys_record/recorded_motions/` (MS files) and feed the next iteration's dataset.
- Datasets are yaml lists `motions: [{file, weight}]` built by `parc/util/create_dataset.py`:
  - A class is a first-level subfolder of a `folder_paths` entry.
  - Classes get equal weight; within a class, weight ∝ motion length.
  - Directories with `ignore` in their path, and `log/` dirs, are skipped.

### Motion files and character model
- MS format (`parc/util/file_io.py`): a pickled `{motion_data, terrain_data, misc_data}` dict whose sections are pickled separately.
  - `MSMotionData`: `root_pos [N,3]`; `root_rot [N,4]` and `joint_rot [N, num_bodies-1, 4]` (quaternions, xyzw); `body_contacts [N, num_contact_bodies]`; `fps`; `loop_mode`.
  - `MSTerrainData`: `hf`, `hf_maxmin`, `min_point`, `dx`.
  - `misc_data`: `hf_mask_inds`, `opt:body_constraints`, `path_nodes`, ...
  - Load with `file_io_helper.load_ms_file` (torch tensors) or `MotionLib.from_file` (`parc/anim/motion_lib.py`; takes a motion file or a dataset yaml).
- `parc/util/pickle_compat.py` (`load`, `loads`, `CompatUnpickler`) does all MS unpickling:
  - It remaps develop-era modules (`util.*` → `parc.util.*`, `tools.motion_opt` → `parc.motion_synthesis.motion_opt`, ...).
  - It maps `numpy._core` for NumPy-2 pickles such as GMR output (only under NumPy < 2).
  - It loads GPU-pickled tensors to `map_location` (default CPU).
  - Legacy flat motion dicts (`frames`, `terrain`, ...) raise `MSFileFormatError`, whose message gives the conversion command.
- `parc/anim/kin_char_model.py` (`KinCharModel`) parses MJCF:
  - It honours `<compiler angle="radian">` and stores joint limits in radians.
  - Joints default to hinge. Each body has 0, 1 or 3 hinges (3 become one spherical joint), and every hinge needs a `range`.
  - Contact bodies come from `<custom_contact_info>`, else all bodies.
  - `BODY_NAME_ALIASES` maps PARC's canonical names to other robots' links (`left_foot` → `left_ankle_roll_link`; `left_hand` → `left_wrist_yaw_link` or `left_rubber_hand`). `resolve_body_name`, `has_body` and `get_body_id` use it, so configs and code keep the canonical names.

## G1 specifics

- `data/assets/unitree_g1/g1_mocap_29dof.xml` has 36 bodies: the root, 29 bodies on hinge joints, and 6 on fixed joints (`left_toe_link`, `right_toe_link`, `pelvis_contour_link`, `head_link`, `head_mocap`, `imu_in_torso`). It has 29 DOFs. All 36 bodies are contact bodies for the contact observations and the contact reward. This is separate from `env.contact_bodies`, which only affects fall termination.
- In `dm_env_g1.yaml`, the per-joint lists (`joint_err_w`, `pose_termination_dist`) have 35 entries, and `contact_weights` has 36.
- PD gains come from `env.pd_params` in `dm_env_g1.yaml` (the MJCF has none). Each joint takes the first key that is a case-insensitive substring of its name. parc_3 and parc_4 re-save the env config with sorted keys, so keys must not overlap.

Observation keys, in flat order (checked at runtime by `tests/smoke/g1_env/check_g1_env.py`):

| Key | Shape | Normalized | Contents |
|---|---|---|---|
| `char_obs` | 266 | yes | root_rot 6 + root_vel 3 + root_ang_vel 3 + joint_rot 35×6 + dof_vel 29 + key_pos 5×3 |
| `tar_obs` | (6, 234) = 1404 | yes | per future step (`tar_obs_steps` 1, 2, 3, 10, 20, 30 at 30 Hz): root_pos 3 + root_rot 6 + joint_rot 35×6 + key_pos 5×3 |
| `tar_contacts` | (6, 36) = 216 | no | reference contacts |
| `char_contacts` | 36 | no | 1 where the contact force exceeds 1e-5 |
| `hf` | 441 | no | 7 rays (−45° to +45°, 15° apart) × 63 points (−0.10 to +3.00 m, step 0.05); terrain height minus root height, clamped to ±3 |

- Rotations are 6-D tangent-normal vectors. `joint_rot` is `dof_to_rot` over all 35 non-root joints, so it includes the 6 fixed ones (constant).
- The raw observation has 2363 dims. Encoded, it has 775: `char_obs` 266, `future_pose` 32, `char_contacts` 36, `hf` 441. This is the actor and critic input.
- The teacher's input has 1620 dims. The predictor's input has 743 (`char_obs` + `char_contacts` + `hf`).

Caveats (current behaviour; see `tests/smoke/g1_env/README.md` and the open decisions in `doc/porting_plan.md` §8):
- **Joint limits are not enforced.** No joint in the G1 MJCF is `limited="true"`, so IsaacGym reports `hasLimits` false on all 29 DOFs, and PhysX doesn't stop joints at their limits. The kinematic limits only set the PD action bounds (range midpoint ± 0.7 × range).
- **Effort is unlimited.** Every DOF has effort FLT_MAX, because IsaacGym ignores the MJCF's `actuatorfrcrange` (D13).
- **motion_opt body constraints do nothing on the G1.** They only support bodies whose first geom is a sphere or a box, and the G1's feet and hands start with a mesh (D16). A warning names the skipped bodies.
- **The G1 cannot stand on PD alone.** Holding the default pose, it tips over within about 1.3 s, even at 10× the `pd_params` gains. The tracker has to balance actively.

## Gotchas

- **Import `isaacgym` before `torch`.** Otherwise IsaacGym raises `ImportError`; the entry scripts already import it first. In pytest, other test modules import torch during collection, so IsaacGym code has to run in a subprocess (as `tests/test_import_sweep.py` does).
- **One IsaacGym sim per process.** `tests/smoke/g1_env/README.md` says so, and nothing in the code destroys a sim. Build one env per process.
- **Legacy develop-era `.pkl` motions must be converted** with `scripts.convert_legacy_motions` (pass the G1 `--char-file`). MotionLib, create_dataset and MotionScope only read MS files.
- **PyYAML reads `1e-4` (no decimal point) as a string.** The code casts learning rates and parc_2's loss thresholds with `float()`. For new numeric keys, write `1.0e-4` or cast.
- **parc_2's `load_mdm` loads the highest-numbered `.ckpt` when given a directory.** It also accepts a `.ckpt` file, and uses the EMA weights if they were saved. parc_0 points `mdm_model_path` at `p1_train_gen/checkpoints/`, so `final_model.ckpt` (one level up) is not used.
- **parc_2 can loop forever.** It regenerates until `top_k` candidates pass `max_contact_loss`, `max_pen_loss` and `max_total_loss`, which a barely trained MDM never does. The smoke configs set them to 1e9.
- **Caches are reused whenever the file exists, without checking the dataset.** This applies to `sampler.pkl` (parc_1; rebuilt only when it fails to load or its device differs) and `terrain.pkl` (parc_3, parc_4). Delete them after changing the data.
- **`model.pt` is only saved at checkpoints.** The tracker writes `model.pt` and `checkpoints/model_<iter>.pt` only every `iters_per_checkpoint` iterations (200 in the shipped agent configs), not at the end, so a short run leaves the iteration-0 model.
- **create_dataset rewrites motion files in place.** It re-saves every motion file it reads, and so do parc_1, parc_3 and parc_4.
