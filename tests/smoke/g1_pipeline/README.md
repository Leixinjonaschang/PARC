# G1 pipeline smoke tests (stages 0 to 2)

Small runs of the PARC stages for the Unitree G1, plus humanoid stages 1 and 2. There
is no G1 dataset on this machine, so the G1 runs start from synthetic G1 motions
(`tests/fixtures/pipeline_g1_synthetic.py`). Run every command from the repo root.
Outputs go to `output/smoke_g1/` (G1) and `output/smoke/` (humanoid, shared with
`tests/smoke/README.md`); both are gitignored.

Set these for every command (`path_loader` only needs `DATA_DIR` to exist; the smoke
configs use repo-relative paths):

```bash
export PARC_DATA_DIR=/home/ubuntu/projects/parkour_sgs   # any existing directory
export WANDB_MODE=disabled
```

With `use_wandb: False`, parc_1 and the motion optimization still call
`wandb.init(mode="offline")`, which writes local runs under `$WANDB_DIR/wandb/`
(hence `WANDB_DIR=...` below). Nothing is uploaded.

Stages 1 and 2 run on `cuda:0`; check `nvidia-smi` first. Reference times are from a
shared L40S machine (2026-10-14).

## G1 synthetic motions

```bash
rm -rf output/smoke_g1
uv run python tests/fixtures/pipeline_g1_synthetic.py output/smoke_g1/motions --num_motions_per_class 3
```

Six 60-frame clips at 30 fps in `walk/` (flat floor, box beside the path) and `step/`
(0.15 m step up), with 36 contact bodies and `joint_rot` of shape (60, 35, 4).

## G1 stage 0: write the stage configs

```bash
uv run python -m scripts.parc_0_setup_iter --config tests/smoke/g1_pipeline/parc_0_setup_iter_g1_smoke.yaml
grep -rn "char_file\|char_model\|char_filepath" output/smoke_g1/iter_1 --include="*.yaml"
```

The config sets `char_file`, so parc_0 writes the G1 xml into the MDM config, kin gen's
`opt.char_model`, the three dataset configs, and `p3_tracker/input_dm_env.yaml` (a copy
of the tracker env config that `p3_tracker/tracker.yaml` points to).

This needs Lane A's `data/configs/tracker_config/dm_env_g1.yaml`. Without it, parc_0
writes the stage 1 and 2 configs and then fails with
`AssertionError: Config file does not exist: data/configs/tracker_config/dm_env_g1.yaml`.
To run stages 0 to 2 before Lane A is merged, use a copy of the config with
`input_tracker_config_path: "tests/smoke/parc_3_tracker_smoke.yaml"` (the humanoid
tracker; stage 3 is not run then).

## G1 stage 1: train the MDM

```bash
WANDB_DIR=output/smoke_g1 uv run python -m scripts.parc_1_train_gen --config output/smoke_g1/iter_1/p1_train_gen/mdm_config.yaml
```

About 35 s. Builds `output/smoke_g1/iter_1/iter_1_start_motions.yaml` (6 motions,
classes step and walk), then writes `sampler.pkl`, `sampler_stats.txt`,
`checkpoints/model_1.ckpt` and `final_model.ckpt` in `p1_train_gen/`. The loss tables
include `HF_COLLISION_LOSS` and `TARGET_LOSS`. A second run prints `loading sampler`
and reuses `sampler.pkl`; with another `device` in the config it prints
`rebuilding sampler: it was built on device cuda:0 but the config device is cpu`.

## G1 stage 2: generate kinematic motions

```bash
WANDB_DIR=output/smoke_g1 uv run python -m scripts.parc_2_kin_gen --config output/smoke_g1/iter_1/p2_kin_gen/boxes_0_0/kin_gen_config.yaml
uv run python tests/smoke/g1_pipeline/check_motions.py --char_file data/assets/unitree_g1/g1_mocap_29dof.xml output/smoke_g1/iter_1/p2_kin_gen
```

About 35 s. One BOXES terrain; the MDM (EMA weights, `Using ema model...`) generates 4
candidates for 3 s, and the best 2 are optimized for 10 iterations. Expect in
`output/smoke_g1/iter_1/p2_kin_gen/`: `boxes_0_0/boxes_0_{0,1}_opt.pkl`,
`boxes_0_0/flipped/boxes_0_{0,1}_opt_flipped.pkl`, `boxes_0_0/log/`, and the raw
motions in `ignore/raw/boxes_0_0/boxes_0_{0,1}.pkl`. `check_motions.py` loads each in
MotionLib and prints `checked 6 files`, with `joint_rot (98, 35, 4), contacts (98, 36)`.

## G1 datasets of stages 3 and 4 (CPU)

parc_3 and parc_4 first build their dataset yaml; that part runs without IsaacGym:

```bash
uv run python -m scripts.create_dataset_yaml --config output/smoke_g1/iter_1/p3_tracker/create_dataset_config.yaml
uv run python -m scripts.create_dataset_yaml --config output/smoke_g1/iter_1/p4_phys_record/create_dataset_config.yaml
```

Expect `p3_tracker/motions.yaml` with 10 motions in the classes step, walk and
boxes_0_0 (`ignore/` and `log/` are skipped), and `p4_phys_record/motions.yaml` with
the 4 generated motions.

## G1 stages 3 and 4 (after Lane A is merged)

Not run in Lane C1: they need the G1 tracker env.

```bash
uv run python -m scripts.parc_3_tracker --config output/smoke_g1/iter_1/p3_tracker/tracker.yaml
uv run python -m scripts.parc_4_phys_record --config output/smoke_g1/iter_1/p4_phys_record/phys_record.yaml
```

Stage 3 comes from `parc_3_tracker_g1_smoke.yaml` (G1 env and agent configs, 4 envs,
2e4 samples) and should write `dm_env.yaml`, `agent_config.yaml` and `model.pt` to
`p3_tracker/`. Stage 4 (from `data/configs/parc_4_phys_record_default.yaml`) records
one env per motion of `p4_phys_record/motions.yaml` into
`p4_phys_record/recorded_motions/`.

## Humanoid stages 1 and 2

Run S4 and S8 of `tests/smoke/README.md` first (dataset and MDM in `output/smoke/`), then:

```bash
WANDB_DIR=output/smoke uv run python -m scripts.parc_2_kin_gen --config tests/smoke/g1_pipeline/parc_2_kin_gen_humanoid_smoke.yaml
uv run python tests/smoke/g1_pipeline/check_motions.py --char_file data/assets/humanoid.xml output/smoke/p2_kin_gen
```

About 35 s. Expect `output/smoke/p2_kin_gen/raw/boxes_0_{0,1}.pkl` and in
`output/smoke/p2_kin_gen/opt/` the `_opt.pkl` files, `flipped/` and `log/`;
`check_motions.py` prints `checked 6 files` with `joint_rot (98, 14, 4), contacts (98, 15)`.
