"""P3 integration: IGEnv's env offsets match IsaacGym's env origins (CPU PhysX, no GPU).

IGEnv._build_envs computed the offsets with a grid formula. With one env per row (2 or 3 envs in
the square style) IsaacGym lays the envs out along x, the formula along y (IGParkourEnv had the
same bug, fixed in Lane A). The check runs in a fresh interpreter because IsaacGym must be
imported before torch.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_MARKER = "ENV_OFFSETS_RESULT "

CHECK = r"""
import json, types
from isaacgym import gymapi
import torch
import parc.motion_tracker.envs.ig_env as ig_env

gym = gymapi.acquire_gym()
params = gymapi.SimParams()
params.up_axis = gymapi.UP_AXIS_Z
params.physx.use_gpu = False
params.use_gpu_pipeline = False
results = []
for style, num_envs in [("square", 1), ("square", 2), ("square", 3), ("square", 4), ("square", 10), ("x_line", 3)]:
    sim = gym.create_sim(-1, -1, gymapi.SIM_PHYSX, params)
    env = types.SimpleNamespace(_num_envs=num_envs, _env_style=style, _device="cpu", _env_spacing=2.0,
                                _gym=gym, _sim=sim, _get_env_spacing=lambda: 2.0,
                                _build_env=lambda env_id, env_ptr, config: None)
    ig_env.IGEnv._build_envs(env, {"env": {"env_spacing": 2.0}})
    origins = [[gym.get_env_origin(ptr).x, gym.get_env_origin(ptr).y, gym.get_env_origin(ptr).z] for ptr in env._envs]
    results.append({"style": style, "num_envs": num_envs, "offsets": env._env_offsets.tolist(), "origins": origins})
    gym.destroy_sim(sim)
print("ENV_OFFSETS_RESULT " + json.dumps(results), flush=True)
"""


def test_env_offsets_are_isaacgym_env_origins():
    proc = subprocess.run([sys.executable, "-c", CHECK], cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    if "No module named 'isaacgym'" in proc.stdout:
        pytest.skip("IsaacGym is not available")
    lines = [line for line in proc.stdout.splitlines() if line.startswith(RESULT_MARKER)]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:]
    results = json.loads(lines[-1][len(RESULT_MARKER):])

    for result in results:
        np.testing.assert_allclose(result["offsets"], result["origins"], err_msg=str(result))
    by_case = {(r["style"], r["num_envs"]): r["offsets"] for r in results}
    # the grid formula put the second of 2 or 3 envs at (0, 4)
    assert by_case[("square", 2)] == [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    assert by_case[("square", 3)][1] == [4.0, 0.0, 0.0]
    assert by_case[("square", 4)] == [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [4.0, 4.0, 0.0]]
    assert by_case[("x_line", 3)] == [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]
