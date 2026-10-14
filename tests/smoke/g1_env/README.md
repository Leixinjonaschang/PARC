# G1 tracker env smoke checks (porting plan A5)

GPU checks of the Unitree G1 tracker env: env build, observation layout, training with the
future-pose agent, test, record, and debug visuals. A humanoid regression check is included. There is
no G1 dataset in the repo, so `make_g1_smoke_data.py` writes two synthetic standing motions.

Run every command from the repo root. Outputs go to `output/smoke/g1_env/` (gitignored); delete it to
start over. Set these for every command:

```bash
export PARC_DATA_DIR=/home/ubuntu/projects/parkour_sgs   # any existing directory
export WANDB_MODE=disabled
```

Check `nvidia-smi` first; the checks use at most 64 envs. IsaacGym allows one sim per process, so each
command runs on its own. When several worktrees run IsaacGym at once, give each its own
`TORCH_EXTENSIONS_DIR` (gymtorch is JIT-built there).

Reference values and times are from an L40S (2026-10-14), shared with other jobs.

## CPU unit tests

```bash
uv run pytest tests/test_env_dof_props.py tests/test_env_g1_config.py
```

## 1. Synthetic G1 data

```bash
uv run python tests/smoke/g1_env/make_g1_smoke_data.py
```

Writes `motions/synthetic/g1_sway.pkl` and `g1_reach.pkl` (MS files, 90 frames at 30 fps, CLAMP).
The legs hold the default pose with the root at the standing height (0.7919 m, feet on the
ground) on a flat terrain (dx 0.4). The waist and arms move sinusoidally within their joint limits.
Also writes `motions.yaml`.

## 2. Env build and observations (A5 a, b)

```bash
uv run python tests/smoke/g1_env/check_g1_env.py
```

About 40 s. The env prints its DOF table (29 rows: limits in radians, hasLimits, effort, stiffness,
damping, and the `pd_params` keys used), then:

```
CHECK total mass 34.3771 kg (expected 34.377)
CHECK 29 DOFs: limits = kinematic model (radians), stiffness/damping = pd_params
CHECK hasLimits (limits enforced by PhysX): 0 of 29 DOFs; effort: [3.4028234663852886e+38]
CHECK action dim 29, bounds = joint range mid -/+ 0.7 range
CHECK OBS SHAPES = appendix A, obs dim 2363, action dim 29
CHECK 150 random-action steps x 4 envs: obs and rewards finite, 72 resets, mean root height 0.424-0.768 m
CHECK G1 env OK
```

## 3. Debug visuals (A5 f)

IsaacGym's viewer needs a display (there was none on the reference machine). This dry run builds the
env with `visualize=True` and `debug_visuals=True` behind a gym proxy. The proxy stubs the viewer
calls and counts what one frame draws:

```bash
uv run python tests/smoke/g1_env/check_debug_visuals.py
```

Expect `per frame 1 add_lines call with 5292 lines (441 hf points x 4 envs), 36 contact bodies
coloured on the character and on the reference`. To look at it, run this on a machine with a display
after step 4:

```bash
uv run python -m scripts.run_tracker --mode test --visualize True --debug_visuals True --num_envs 4 \
  --env_config output/smoke/g1_env/p3_tracker/dm_env.yaml \
  --agent_config output/smoke/g1_env/p3_tracker/agent_config.yaml \
  --model_file output/smoke/g1_env/p3_tracker/model.pt
```

Red crosses mark the heightfield samples of every env. In the camera env, contact bodies go white to
red with contact force on the character, and green to red with the reference contacts.

## 4. Train (A5 c)

```bash
uv run python -m scripts.parc_3_tracker --config tests/smoke/g1_env/parc_3_tracker_g1_smoke.yaml
```

About 2 min: 64 envs, 10 iterations of `FuturePoseDMPPOAgent` (`dm_agent_g1.yaml`, predictor
enabled). Writes `p3_tracker/` with `dm_env.yaml`, `agent_config.yaml`, `log.txt`, `model.pt`
(iteration 0) and `checkpoints/`. Every iteration logs `Predictor_Loss`. Reference, iterations 0 to 9:
Train_Return 4.87 to 5.55, Critic_Loss 1.91 to 0.52, Predictor_Loss 0.010 to 0.024.

## 5. Test (A5 d)

```bash
for tp in False True; do
uv run python -m scripts.run_tracker --mode test --visualize False --num_envs 16 --test_episodes 32 \
  --rand_seed 0 --test_predictor $tp \
  --env_config output/smoke/g1_env/p3_tracker/dm_env.yaml \
  --agent_config output/smoke/g1_env/p3_tracker/agent_config.yaml \
  --model_file output/smoke/g1_env/p3_tracker/model.pt
done
```

About 40 s each. Reference: Mean Return 6.66 over 33 episodes, and 6.75 over 32 episodes with
`--test_predictor True` (which prints `Predictor test mode: ...`).

## 6. Record (A5 e)

Stage 4 on the step-4 tracker. Create the output directory first: parc_4 writes `record_env.yaml`
into it before creating it.

```bash
mkdir -p output/smoke/g1_env/p4_phys_record
uv run python -m scripts.parc_4_phys_record --config tests/smoke/g1_env/parc_4_phys_record_g1_smoke.yaml
```

About 40 s. The iteration-0 tracker tracks neither motion. Expect `failed to track motion` in the
first pass and in 4 retry passes (start time fractions 0.1 to 0.4), `Total successfull motions: 0`,
exit status 0, and an empty `p4_phys_record/recorded_motions/`.

To get recorded files without a trained tracker, this runs the same recorder with an oracle policy.
Its PD targets are the reference DOFs; it does not balance, so early termination is off. Then it loads
every recorded file with `MotionLib.from_file`:

```bash
uv run python tests/smoke/g1_env/check_g1_record.py
```

About 40 s. Expect 2 MS files in `record_check/recorded_motions/` (90 frames at 30 fps, obs
(90, 2363)). Over the first 0.5 s: mean |dof - ref| about 0.015 rad, root height within 1 mm, foot
contact 0.93. Then `CHECK G1 record OK`.

## 7. Humanoid regression

```bash
uv run python tests/smoke/g1_env/check_humanoid_env.py
```

Builds the civilization example and compares its action bounds and character DOF properties with
`tests/fixtures/env_humanoid_civilization_props.npz`, captured on develop_migration 89444d8 before
the Lane A changes. Expect max |delta| 2.38e-07 for the bounds and limits (float32 rounding;
limit 1e-6), identical gains, effort and hasLimits, and `CHECK humanoid env unchanged`. The
example still trains:

```bash
uv run python -m scripts.run_tracker --mode train --visualize False --num_envs 64 --max_samples 12288 \
  --rand_seed 0 --env_config data/motion_terrains/dm_env_civilization.yaml \
  --agent_config data/configs/tracker_config/dm_agent_default.yaml \
  --out_model_file output/smoke/g1_env/humanoid_train/model.pt --log_file output/smoke/g1_env/humanoid_train/log.txt
```

Reference: 6 iterations, Train_Return 3.77 to 4.27, Critic_Loss 2.98 to 0.68.

## What these checks showed about the G1 in IsaacGym

- **Joint limits are not enforced.** IsaacGym reads no ranges from MJCF joints without
  `limited="true"` (all G1 joints). `hasLimits` is read-only after loading, and the limits of DOFs
  without it read back as 0. The kinematic limits therefore only set the action bounds, and joints
  driven past them go past them. With `limited="true"` on the MJCF's default joint, IsaacGym reads
  the radian ranges correctly and PhysX stops the joints at their limits.
- **Effort is unlimited.** Every DOF has effort FLT_MAX; IsaacGym ignores `actuatorfrcrange`. The
  motors have no `gear`, so actuator `motor_effort` is 0. Only the torque, pd_exp and pd_1d control
  modes use it, as their torque limit.
- **PD alone does not keep the G1 standing.** Holding the default pose, it tips forward within about
  1.3 s, at `pd_params` gains and at 10 times those gains. The ankle pitch yields, though the
  zero-pose centre of mass (x = +0.02 m from the pelvis) lies inside the feet's support. The tracker
  has to balance actively.
