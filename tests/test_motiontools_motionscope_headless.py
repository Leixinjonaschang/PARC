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
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(FIXTURE_DIR))
import motiontools_synth as synth  # noqa: E402

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


@pytest.mark.parametrize("char", ["g1", "humanoid"])
def test_motion_stitcher(tmp_path, char):
    first, second = _motions(tmp_path, char, num=2)
    out = _run(tmp_path, "stitch", {"motions": [[first, char], [second, char]]})
    assert "_stitched" in out.split("HARNESS: loaded motions:")[-1]


def test_draw_every_gui(tmp_path):
    (motion,) = _motions(tmp_path, "g1")
    _run(tmp_path, "draw_all", {"motions": [[motion, "g1"]]})
