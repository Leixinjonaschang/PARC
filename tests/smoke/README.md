# G0 humanoid baseline smoke tests

Small end-to-end runs of the humanoid pipeline on the bundled example motions
(`data/motion_terrains/`). Run every command from the repo root. All outputs go
to `output/smoke/` (gitignored). Delete that directory to start over:
`p3_tracker/terrain.pkl` and `p1_train_gen/sampler.pkl` are reused when present.

Set these for every command. The smoke configs only use repo-relative paths,
but `path_loader` requires `DATA_DIR` to be an existing directory.

```bash
export PARC_DATA_DIR=/home/ubuntu/projects/parkour_sgs   # any existing directory
export WANDB_MODE=disabled
```

S5 to S8 need a GPU (S5 to S7 also need IsaacGym). Check `nvidia-smi` first; the
configs use at most 64 envs. gymtorch is JIT-built per venv into a shared cache,
so when several worktrees run IsaacGym at once, give each its own
`TORCH_EXTENSIONS_DIR`.

Reference times are from an L40S (2026-10-14).

The G1 smoke tests have their own READMEs:

- [`g1_env/README.md`](g1_env/README.md): the G1 tracker env (build, observations, PD gains), training
  with the future-pose agent, test, record and debug visuals, plus a humanoid env regression check.
- [`g1_pipeline/README.md`](g1_pipeline/README.md): G1 stages 0 to 2 (setup, MDM, kinematic
  generation) and humanoid stages 1 and 2.
- [`g1_e2e/README.md`](g1_e2e/README.md): one G1 iteration end to end, stages 0 to 4 and the
  dataset of the produced motions, run by `run_g1_e2e.py`.

## S0 Static checks

```bash
uv run python -m compileall -q parc scripts
uv run pytest tests/test_import_sweep.py -v
```

## S3 Load the civilization example

```bash
uv run python -c "
from parc.anim.kin_char_model import KinCharModel
from parc.anim.motion_lib import MotionLib
cm = KinCharModel('cpu'); cm.load_char_file('data/assets/humanoid.xml')
ml = MotionLib.from_file('data/motion_terrains/civilization.pkl', cm, 'cpu', contact_info=True)
print(ml.num_motions(), ml.get_total_length())"
```

Expect `1 8.433...` (254 frames at 30 fps).

## S4 Build a dataset yaml

`create_dataset` re-saves every motion file it reads, so it runs on copies.
`sfu.pkl` is left out: it has exactly 15 frames, which the MDM sampler rejects.
`min_num_frames: 16` drops it, and a class left without motions is skipped with a
warning (`motion class 'sfu' has no usable motions, skipping it`), so adding it changes
nothing. Files without terrain or motion data are skipped with a warning as well.

```bash
mkdir -p output/smoke/motions/civilization output/smoke/motions/teaser
cp data/motion_terrains/civilization.pkl output/smoke/motions/civilization/
cp data/motion_terrains/TEASER_TERRAIN.pkl data/motion_terrains/dec2024_teaser_717_1_modified_opt.pkl \
   data/motion_terrains/dec2024_teaser_717_1_opt_dm.pkl output/smoke/motions/teaser/
uv run python -c "
from parc.util import path_loader
from parc.util.create_dataset import create_dataset_yaml_from_config
create_dataset_yaml_from_config(path_loader.load_config('tests/smoke/create_dataset_smoke.yaml'))"
```

Expect `output/smoke/motions.yaml` listing 4 motions in 2 classes (civilization, teaser).

## S5 Train the tracker (stage 3)

```bash
uv run python -m scripts.parc_3_tracker --config tests/smoke/parc_3_tracker_smoke.yaml
```

64 envs, 201 iterations, about 7 minutes. Expect `output/smoke/p3_tracker/`
with `dm_env.yaml`, `agent_config.yaml`, `log.txt`, `model.pt` and
`checkpoints/model_0000000{000,200}.pt`. Reference: Test_Return 3.94, 5.10, 5.85
at iterations 0, 100, 200 (a P3 re-run on develop_migration with all lanes merged:
3.67, 5.54, 6.24 in 7.4 minutes).

## S6 Test the trained tracker

Pass `--test_episodes`; the default is unbounded.

```bash
uv run python -m scripts.run_tracker --mode test --visualize False --num_envs 16 --test_episodes 32 \
  --env_config output/smoke/p3_tracker/dm_env.yaml \
  --agent_config output/smoke/p3_tracker/agent_config.yaml \
  --model_file output/smoke/p3_tracker/model.pt
```

About 15 s. Reference: Mean Return 6.2 over 48 episodes (P3 re-run: 7.2 over 39).

## S7 Record motions (stage 4)

```bash
uv run python -m scripts.parc_4_phys_record --config tests/smoke/parc_4_phys_record_smoke.yaml
```

About 25 s, exit status 0. parc_4 rebuilds `output/smoke/motions.yaml` and records one
env per motion (4) into `output/smoke/p4_phys_record/recorded_motions/`. Motions that
fail the first pass are retried from later start times (fractions 0.1 to 0.5) until the
rest of the motion is shorter than 2 s. The S5 policy tracks none of the 4 motions to
their end: `failed to track motion` in every pass, `Total successfull motions: 0`, and
`recorded_motions/` stays empty. Recorded motions are MS files named `<motion>_dm.pkl`
with the observations in `misc_data["obs"]` and `misc_data["obs_shapes"]`; names with a
number (`dec2024_teaser_717_...`) are moved into folders of 50 (`dec2024_teaser_700_749/`).
MotionScope shows the humanoid's recorded obs (Obs GUI). The G1 equivalents are in
`g1_env/README.md` (step 6) and `g1_e2e/README.md`.

## S8 Train the MDM (stage 1)

```bash
WANDB_DIR=output/smoke uv run python -m scripts.parc_1_train_gen --config tests/smoke/parc_1_train_gen_smoke.yaml
```

2 epochs of 2 iterations, about 20 s. Expect `output/smoke/p1_train_gen/` with
`sampler.pkl`, `sampler_stats.txt`, `checkpoints/model_1.ckpt` and
`final_model.ckpt`. With `use_wandb: False`, parc_1 calls
`wandb.init(mode="offline")`, which overrides `WANDB_MODE=disabled` and writes a
local offline run under `$WANDB_DIR/wandb/`. Nothing is uploaded.
