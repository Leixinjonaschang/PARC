# G1 end-to-end smoke (stages 0 to 4)

One tiny PARC iteration for the Unitree G1: setup, MDM training, kinematic motion generation,
tracker training with the future-pose agent, recording, and a dataset of the produced motions.
There is no G1 dataset on the reference machine, so the iteration starts from the synthetic G1
motions of `tests/fixtures/pipeline_g1_synthetic.py`. Every stage runs at tiny scale (porting
plan §7, with 32 envs for 5 iterations in stage 3); nothing here measures motion or tracking
quality.

`run_g1_e2e.py` runs every step in its own process (IsaacGym allows one sim per process), checks
its outputs and prints a summary. Everything goes to `$PARC_DATA_DIR/g1_e2e/`, which the first
step deletes and recreates; nothing is written to the repository.

## Run

Run from the repo root. Stages 1 to 4 use `cuda:0` (at most 32 envs); check `nvidia-smi` first.

```bash
export PARC_DATA_DIR=/path/to/an/existing/dir   # outputs: $PARC_DATA_DIR/g1_e2e/
export WANDB_MODE=disabled
export WANDB_DIR=$PARC_DATA_DIR                 # parc_1's offline wandb run: $WANDB_DIR/wandb/
uv run python tests/smoke/g1_e2e/run_g1_e2e.py
```

`--from_step` and `--to_step` run part of the chain (steps `data`, `stage0`, `stage1`, `stage2`,
`stage3`, `stage4`, `stage4b`, `dataset`); a step needs the outputs of the steps before it. Each
step's output is in `$PARC_DATA_DIR/g1_e2e/logs/<step>.log`. When several worktrees run IsaacGym
at once, give each its own `TORCH_EXTENSIONS_DIR` (gymtorch is JIT-built there).

## Steps

The commands the driver runs, from the repo root (`$D` is `$PARC_DATA_DIR/g1_e2e`):

| Step | Command | What the driver checks |
|---|---|---|
| data | `python tests/fixtures/pipeline_g1_synthetic.py $D/motions --num_motions_per_class 3`<br>`python -m scripts.create_dataset_yaml --config tests/smoke/g1_e2e/create_dataset_g1_e2e.yaml` | 6 MS motions (60 frames, classes `walk` and `step`) in `$D/motions.yaml`; it loads in MotionLib |
| stage0 | `python -m scripts.parc_0_setup_iter --config tests/smoke/g1_e2e/parc_0_setup_iter_g1_e2e.yaml` | the G1 xml in `mdm_config.yaml`, kin gen's `opt.char_model`, the 3 `create_dataset_config.yaml` and `p3_tracker/input_dm_env.yaml`; `tracker.yaml` uses `input_dm_env.yaml` (equal to `dm_env_g1.yaml`, keys in the same order) and `dm_agent_g1.yaml`; `phys_record.yaml` uses p3's `dm_env.yaml`, `agent_config.yaml`, `model.pt` |
| stage1 | `python -m scripts.parc_1_train_gen --config $D/iter_1/p1_train_gen/mdm_config.yaml` | `iter_1_start_motions.yaml` (6 motions), `sampler.pkl`, `sampler_stats.txt`, `checkpoints/model_1.ckpt`, `final_model.ckpt`; the losses include `HF_COLLISION_LOSS` and `TARGET_LOSS` |
| stage2 | `python -m scripts.parc_2_kin_gen --config $D/iter_1/p2_kin_gen/boxes_0_0/kin_gen_config.yaml` | EMA weights (`Using ema model...`); 2 `*_opt.pkl`, 2 `flipped/*_opt_flipped.pkl` and 2 raw motions in `ignore/raw/`; the opt files load in MotionLib with `joint_rot (N, 35, 4)` and `contacts (N, 36)` |
| stage3 | `python -m scripts.parc_3_tracker --config $D/iter_1/p3_tracker/tracker.yaml` | `FuturePoseDMPPOAgent`; 5 iterations with finite `Predictor_Loss` and `Train_Return`; `motions.yaml` with 10 motions (6 synthetic, 4 generated); `dm_env.yaml` keeps the `pd_params` order; `model.pt` has develop's generation-C keys (encoder 1620 → 32, predictor 743 → 32, actor input 775) |
| stage4 | `python -m scripts.parc_4_phys_record --config $D/iter_1/p4_phys_record/phys_record.yaml` | exit status 0, 4 envs (the 4 generated motions), the record summary; recorded files, if any, load in MotionLib with obs `(N, 2363)` |
| stage4b | `python -m scripts.run_tracker --mode record --visualize False --test_predictor True --rand_seed 0 --num_envs 4 --device cuda:0 --env_config $D/iter_1/p4_phys_record_no_et/record_env_no_et.yaml --agent_config $D/iter_1/p3_tracker/agent_config.yaml --model_file $D/iter_1/p3_tracker/model.pt` | `Predictor test mode`; 4 recorded MS files in `recorded_motions/boxes_0_49/` with obs `(98, 2363)`; they load in MotionLib |
| dataset | `python -m scripts.create_dataset_yaml --config tests/smoke/g1_e2e/create_dataset_g1_e2e_produced.yaml` | `$D/iter_1/produced_motions.yaml` over the generated and recorded motions; it loads in MotionLib |

Stage 4b is an extra check, not a pipeline stage. `record_env_no_et.yaml` is stage 4's
`record_env.yaml` with `enable_early_termination: false` and its own output paths (the driver
writes it). Without early termination an episode only ends at the end of its motion, which the
recorder counts as tracked, so even the untrained policy's recordings are kept. This exercises
`--mode record --test_predictor True` (porting plan G3.4) and gives recorded files for the
dataset step.

## Expected output

About 2.5 minutes on a shared L40S (2026-10-14; gymtorch already built). The summary of that run
(`$D` stands for the full path):

```
===== G1 end-to-end summary ($D) =====
data       12.2 s  6 synthetic G1 motions (walk, step) in motions.yaml, 11.80 s in MotionLib
stage0      0.2 s  G1 char file in mdm_config, kin_gen opt.char_model, 3 dataset configs and input_dm_env.yaml (= dm_env_g1.yaml, key order kept); tracker.yaml: env_config input_dm_env.yaml, agent_config dm_agent_g1.yaml
stage1     18.4 s  iter_1_start_motions.yaml (6 motions), sampler.pkl, checkpoints/model_1.ckpt, final_model.ckpt
stage2     21.5 s  2 *_opt.pkl + 2 flipped load in MotionLib: boxes_0_0_opt.pkl: 98 frames, joint_rot (98, 35, 4), contacts (98, 36)
stage3     42.5 s  5 iterations, Train_Return 2.59 -> 2.21, Predictor_Loss 0.00713 -> 0.00803; motions.yaml with 10 motions
stage4     21.8 s  exit 0, 4 envs (4 motions), 'Total successfull motions: 0', 0 recorded files
stage4b    20.7 s  4 recorded MS files in ['boxes_0_49'] load in MotionLib: boxes_0_0_opt_dm.pkl: 98 frames, joint_rot (98, 35, 4), contacts (98, 36), obs (98, 2363)
dataset     5.8 s  produced_motions.yaml: 8 motions (4 kinematic, 4 recorded), 25.87 s in MotionLib
total     143.1 s
G1 E2E OK
```

Training is not seeded, so the stage 3 numbers vary: another run had Train_Return 0.55 -> 1.52
and Predictor_Loss 0.010 -> 0.008. Other lines to look for in the logs:

- stage1: the loss tables include `HF_COLLISION_LOSS` and `TARGET_LOSS` (the target loss is a
  clamped negative dot product, so small negative values are normal).
- stage2: `loading path: .../p1_train_gen/checkpoints/model_1.ckpt`, `Using ema model...`,
  `Found 4 / 4 valid motions.` (the loss thresholds are 1e9). No wandb run is started.
- stage3: `Building DM_PPO agent (FuturePoseDMPPOAgent)` and the `OBS SHAPES` of porting plan
  appendix A: `char_obs` (266), `tar_obs` (6, 234), `tar_contacts` (6, 36), `char_contacts` (36),
  `hf` (441), 2363 in all.
- stage4: `env <i> failed to track motion boxes_0_<j>_opt[_flipped]` in each pass,
  `Successful motions at 0.1 ... 0.4 percent start time: 0`, `Total successfull motions: 0`.
- stage4b: `Predictor test mode: the actor uses the predicted future-pose latent.`, four
  `wrote motion data to .../recorded_motions/boxes_0_<j>_opt[_flipped]_dm.pkl`,
  `Total successfull motions: 4`.

Output layout under `$D`: `motions/{walk,step}/`, `motions.yaml`, `logs/`, and `iter_1/` with
`iter_1_start_motions.yaml`, `p1_train_gen/`, `p2_kin_gen/` (`boxes_0_0/`, `boxes_0_0/flipped/`,
`boxes_0_0/log/`, `ignore/raw/boxes_0_0/`), `p3_tracker/`, `p4_phys_record/`,
`p4_phys_record_no_et/` and `produced_motions.yaml`.

## Notes

- **Stage 3 trains 5 iterations, but `model.pt` holds the iteration-0 weights.** `dm_agent_g1.yaml`
  checkpoints every 200 iterations (`iters_per_checkpoint`), and the DM agent tests every 100
  (`iters_per_output`), so `Test_Return` stays at its iteration-0 value. Stages 4 and 4b record
  with the iteration-0 policy.
- **Stage 4 records nothing, as expected for an untrained policy.** The G1 falls within about half
  a second in every pass: `failed to track motion` for all 4 motions in the first pass and in the
  retries that start later in the motion (start time fractions 0.1 to 0.3). At 0.4 the rest of
  each 3.2 s motion is shorter than 2 s, and the recorder stops trying. It prints
  `Total successfull motions: 0`, leaves `recorded_motions/` empty and exits with status 0.
- The dataset step's `p4_phys_record/recorded_motions/` is empty (see above), so the recorded
  motions of `produced_motions.yaml` are the stage 4b ones.
- Decision-pending items the run touches but does not settle: the G1 joint limits are not enforced
  by PhysX (the env prints a warning; `limited="true"` in the MJCF), effort is unlimited,
  `key_bodies` uses `head_link`, stage 2's `load_mdm` takes the highest-numbered checkpoint in
  `checkpoints/` rather than `final_model.ckpt`, and kin gen runs in BOXES mode (as the G1
  parc_0 config sets it).
