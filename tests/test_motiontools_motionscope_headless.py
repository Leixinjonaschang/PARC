"""MotionScope GUI smoke test without a display (opt-in).

Runs tests/fixtures/motiontools_motionscope_harness.py in a subprocess. It initialises
polyscope with its headless EGL backend, which needs an NVIDIA driver, and clicks through
the motion, contact, terrain and optimization GUIs on synthetic G1 and humanoid motions.
CUDA is hidden, so the optimization runs on the CPU and EGL is the only GPU user.

    PARC_MOTIONSCOPE_HEADLESS=1 uv run pytest tests/test_motiontools_motionscope_headless.py

Not covered: the Isaac Gym GUI (needs a trained tracker), MDM generation (needs an MDM
checkpoint; the harness's path_mdm scenario does it given one), video recording and
keyboard/mouse input.
"""
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(FIXTURE_DIR))
import motiontools_synth as synth  # noqa: E402

import parc.util.file_io_helper as file_io_helper  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = FIXTURE_DIR / "motiontools_motionscope_harness.py"

pytestmark = pytest.mark.skipif(os.environ.get("PARC_MOTIONSCOPE_HEADLESS") != "1",
                                reason="opt-in (needs polyscope's EGL backend): set PARC_MOTIONSCOPE_HEADLESS=1")


def _run(tmp_path, scenario, cfg):
    (tmp_path / "data").symlink_to(REPO_ROOT / "data") # MotionScope uses repo-relative character paths
    env = dict(os.environ, MS_CFG=json.dumps(cfg), PARC_DATA_DIR=str(tmp_path), WANDB_MODE="disabled",
               WANDB_DIR=str(tmp_path), CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    proc = subprocess.run([sys.executable, str(HARNESS), scenario], cwd=str(tmp_path), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
    lines = [line for line in proc.stdout.splitlines() if not line.startswith("Loaded mesh")]
    tail = "\n".join(lines[-40:])
    assert proc.returncode == 0 and "HARNESS: DONE " + scenario in proc.stdout, tail
    assert "HARNESS: MISSING" not in proc.stdout, tail
    return proc.stdout


def _motions(tmp_path, char, num=1):
    return [str(synth.write_synthetic_motion(tmp_path / "{}_walk_{}.pkl".format(char, i), char=char, seed=i))
            for i in range(num)]


@pytest.mark.parametrize("char", ["g1", "humanoid"])
def test_editing_guis(tmp_path, char):
    first, second = _motions(tmp_path, char, num=2)
    out = _run(tmp_path, "basic", {"motions": [[first, char]], "load_motion": second})
    assert "{}_walk_1.pkl".format(char) in out.split("HARNESS: loaded motions:")[-1]
    if char == "g1":
        # D16 and the box-feet check of the contact helpers report instead of failing silently
        assert "D16" in out
        assert "Cannot label foot contacts" in out
        assert "HARNESS: body constraints: {6: 3, 14: 3, 28: 1, 35: 1}" in out
    else:
        assert "Cannot label" not in out
    assert (tmp_path / "output" / "_motions" / "new_motion.pkl").is_file()
    # "Save Current Motion" wrote the body constraints as numpy, not as BodyConstraint objects
    saved = re.search(r"HARNESS: saved MS motion: (\d+) body constraints, numpy (\w+); hf_mask_inds numpy (\w+)", out)
    assert saved is not None and int(saved.group(1)) > 0 and saved.groups()[1:] == ("True", "True"), out


def _recorded_motion(tmp_path, char):
    # a synthetic motion with obs/obs_shapes in its misc_data, as the tracker records them
    # (char_obs, tar_obs of 6 steps, contacts, heightfield); rotations are identity tan-norms
    char_model = synth.load_char(char)
    frames, terrain = synth.make_motion(char_model)
    num_frames, num_joints = frames.root_pos.shape[0], char_model.get_num_joints() - 1
    num_bodies, num_key_bodies = char_model.get_num_bodies(), 4 if char == "humanoid" else 5
    char_dim = 12 + 6 * num_joints + char_model.get_dof_size() + 3 * num_key_bodies
    shapes = OrderedDict([("char_obs", (char_dim,)), ("tar_obs", (6, 9 + 6 * num_joints + 3 * num_key_bodies)),
                          ("tar_contacts", (6, num_bodies)), ("char_contacts", (num_bodies,)), ("hf", (441,))])
    tan_norm = np.array([1, 0, 0, 0, 0, 1], dtype=np.float32)
    char_obs = np.zeros((num_frames,) + shapes["char_obs"], dtype=np.float32)
    char_obs[:, 0:6] = tan_norm
    char_obs[:, 12:12 + 6 * num_joints] = np.tile(tan_norm, num_joints)
    tar_obs = np.zeros((num_frames,) + shapes["tar_obs"], dtype=np.float32)
    tar_obs[..., 3:9] = tan_norm
    tar_obs[..., 9:9 + 6 * num_joints] = np.tile(tan_norm, num_joints)
    rest = [np.zeros((num_frames, int(np.prod(shapes[key]))), dtype=np.float32)
            for key in ("tar_contacts", "char_contacts", "hf")]
    obs = np.concatenate([char_obs, tar_obs.reshape(num_frames, -1)] + rest, axis=-1)
    obs_shapes = OrderedDict((key, {"use_normalizer": key in ("char_obs", "tar_obs"), "shape": torch.Size(shape)})
                             for key, shape in shapes.items())
    path = tmp_path / "{}_recorded_dm.pkl".format(char)
    file_io_helper.save_ms_file(str(path), motion_frames=frames, fps=30, loop_mode="CLAMP", terrain=terrain,
                                obs=obs, obs_shapes=obs_shapes)
    return str(path)


@pytest.mark.parametrize("char", ["humanoid", "g1"])
def test_recorded_obs(tmp_path, char):
    out = _run(tmp_path, "obs", {"motions": [[_recorded_motion(tmp_path, char), char]]})
    if char == "humanoid":
        assert "HARNESS: has_obs: True" in out
    else:
        # the obs viewer decodes the humanoid's layout only
        assert "HARNESS: has_obs: False" in out
        assert "Not showing the recorded obs" in out


@pytest.mark.parametrize("char", ["g1", "humanoid"])
def test_motion_stitcher(tmp_path, char):
    first, second = _motions(tmp_path, char, num=2)
    out = _run(tmp_path, "stitch", {"motions": [[first, char], [second, char]]})
    assert "_stitched" in out.split("HARNESS: loaded motions:")[-1]


def test_draw_every_gui(tmp_path):
    (motion,) = _motions(tmp_path, "g1")
    _run(tmp_path, "draw_all", {"motions": [[motion, "g1"]]})
