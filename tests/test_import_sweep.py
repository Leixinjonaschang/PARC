"""Static check: every module under parc/ and scripts/ imports cleanly.

IsaacGym refuses to load once torch has been imported, and other test modules
import torch at collection time. The sweep therefore runs in a fresh
interpreter (this file executed as a script), where it can import isaacgym
first. That also keeps the side effects of importing ~100 modules out of the
pytest process.

Run directly:  PARC_DATA_DIR=... uv run pytest tests/test_import_sweep.py -v
"""
import importlib
import json
import os
import pkgutil
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PACKAGES = ("parc", "scripts")
RESULT_MARKER = "IMPORT_SWEEP_RESULT "
SWEEP_TIMEOUT_S = 900  # first run may JIT-build gymtorch

# Modules that do real work when imported, so importing them is not a static check.
SKIP_MODULES = {
    # Builds a dataset yaml from hard-coded $DATA_DIR/releases_parc folders.
    "scripts.create_dataset_yaml",
    # Loads a hard-coded dataset into a MotionLib and analyses it.
    "scripts.find_interesting_motions",
    # Loads data/motion_terrains/sfu.pkl and prints its contents.
    "scripts.read_motion_data",
    # Loads ../Data/parkour_dataset/pre_auto_aug_dataset.yaml into a MotionLib.
    "scripts.motion_tests.compute_stats",
    # MotionScope entry point: imports global_header (see below) and builds the GUI.
    "scripts.run_motionscope",
    # MotionScope GUI. global_header reads motionscope_config.yaml, loads motions and
    # calls polyscope's ps.init(), which needs a display (and opens a window if there
    # is one). Every module below imports global_header.
    "parc.motionscope.include.global_header",
    "parc.motionscope.include.contact_editing_gui",
    "parc.motionscope.include.diffusion_gui",
    "parc.motionscope.include.ig_obs_gui",
    "parc.motionscope.include.io",
    "parc.motionscope.include.isaac_gym_gui",
    "parc.motionscope.include.kin_controller_gui",
    "parc.motionscope.include.motion_editing_gui",
    "parc.motionscope.include.optimization_gui",
    "parc.motionscope.include.path_planning_gui",
    "parc.motionscope.include.recording_gui",
    "parc.motionscope.include.terrain_gui",
}

# Sanity check that the walk actually reached the code we care about.
EXPECTED_MODULES = (
    "parc.motion_tracker.envs.ig_parkour.ig_parkour_env",
    "parc.motion_tracker.learning.dm_ppo_agent",
    "parc.motion_generator.mdm",
    "scripts.run_tracker",
    "scripts.parc_3_tracker",
)


def _namespace_subpackages(package):
    """Subdirectories without __init__.py that hold .py files (e.g. scripts/tools/).

    pkgutil.iter_modules does not report these, so they are added explicitly.
    """
    for base in package.__path__:
        for entry in sorted(os.listdir(base)):
            sub_dir = os.path.join(base, entry)
            if (entry.isidentifier() and entry != "__pycache__" and os.path.isdir(sub_dir)
                    and not os.path.exists(os.path.join(sub_dir, "__init__.py"))
                    and any(name.endswith(".py") for name in os.listdir(sub_dir))):
                yield package.__name__ + "." + entry


def _sweep():
    try:
        import isaacgym  # noqa: F401  (must be imported before torch)
    except ImportError as exc:
        return {"isaacgym_error": repr(exc)}

    imported, skipped, failures = [], [], {}

    def try_import(name):
        try:
            module = importlib.import_module(name)
        except BaseException as exc:  # SystemExit from scripts counts as a failure too
            failures[name] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            return None
        imported.append(name)
        return module

    def walk(package):
        names = [info.name for info in pkgutil.iter_modules(package.__path__, package.__name__ + ".")]
        names += list(_namespace_subpackages(package))
        for name in names:
            if name in SKIP_MODULES:
                skipped.append(name)
                continue
            module = try_import(name)
            if module is not None and hasattr(module, "__path__"):
                walk(module)

    for root in ROOT_PACKAGES:
        package = try_import(root)
        if package is not None:
            walk(package)

    return {"imported": imported, "skipped": skipped, "failures": failures}


def test_import_sweep():
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=SWEEP_TIMEOUT_S,
    )
    result_lines = [line for line in proc.stdout.splitlines() if line.startswith(RESULT_MARKER)]
    assert result_lines, "import sweep crashed (exit code {}):\n{}".format(
        proc.returncode, "\n".join(proc.stdout.splitlines()[-40:]))
    result = json.loads(result_lines[-1][len(RESULT_MARKER):])

    if "isaacgym_error" in result:
        pytest.skip("IsaacGym is not available: " + result["isaacgym_error"])

    failures = result["failures"]
    assert not failures, "{} module(s) failed to import:\n{}".format(
        len(failures), "\n".join("  {}: {}".format(name, err) for name, err in sorted(failures.items())))

    missing = [name for name in EXPECTED_MODULES if name not in result["imported"]]
    assert not missing, "sweep did not reach: {}".format(missing)

    stale_skips = sorted(SKIP_MODULES - set(result["skipped"]))
    assert not stale_skips, "skip list names modules that no longer exist: {}".format(stale_skips)

    print("imported {} modules, skipped {}".format(len(result["imported"]), len(result["skipped"])))


if __name__ == "__main__":
    print(RESULT_MARKER + json.dumps(_sweep()), flush=True)
