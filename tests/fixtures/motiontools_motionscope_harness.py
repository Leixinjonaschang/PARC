"""Drives the MotionScope GUI headless, for tests/test_motiontools_motionscope_headless.py.

polyscope can render without a display through its EGL backend (needs an NVIDIA driver).
This imports scripts/run_motionscope.py, and so global_header, with the `motions` (and MDM)
entries of motionscope_config.yaml replaced. It then runs frames of
run_motionscope.main_loop with every tree node forced open and the chosen buttons
"clicked". An exception in a frame is printed and exits with status 1, skipping imgui
cleanup.

    MS_CFG='{"motions": [["<motion.pkl>", "g1"]]}' python <this file> <scenario>

Run it from a directory where data/ is the repository's data/ directory (MotionScope
uses repo-relative character paths). Outputs such as output/_motions/ are written there.
MS_CFG keys: motions, load_mdm (default false), mdm (checkpoint path), load_motion (a
motion file for the "Load Motion" button).
"""
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import isaacgym  # noqa: F401,E402  (global_header imports the env builder; isaacgym before torch)
import polyscope as ps  # noqa: E402
import polyscope.imgui as psim  # noqa: E402

ps.set_allow_headless_backends(True)

SCENARIO = sys.argv[1]
CFG = json.loads(os.environ["MS_CFG"])

import parc.util.path_loader as path_loader  # noqa: E402

_orig_load_config = path_loader.load_config


def patched_load_config(path):
    cfg = _orig_load_config(path)
    if str(path).endswith("motionscope_config.yaml"):
        cfg["motions"] = CFG["motions"]
        cfg["load_mdm"] = CFG.get("load_mdm", False)
        if CFG.get("mdm"):
            cfg["mdm_filepaths"] = {"main": CFG["mdm"]}
    return cfg


path_loader.load_config = patched_load_config

# imgui patches: open every tree node; "click" the buttons in CLICK
CLICK = set()
CLICKED = []
_real_button = psim.Button
_real_tree_node = psim.TreeNode


def fake_button(label, *args, **kwargs):
    pressed = _real_button(label, *args, **kwargs)
    if label in CLICK:
        CLICKED.append(label)
        return True
    return pressed


def open_tree_node(label, *args, **kwargs):
    psim.SetNextItemOpen(True)
    return _real_tree_node(label, *args, **kwargs)


psim.Button = fake_button
psim.TreeNode = open_tree_node

try:
    import scripts.run_motionscope as app  # noqa: E402
except BaseException:
    traceback.print_exc(file=sys.stdout)
    print("HARNESS: FAILED during step: import")
    sys.stdout.flush()
    os._exit(1)

import parc.motionscope.include.global_header as g  # noqa: E402
import parc.motionscope.include.optimization_gui as optimization_gui  # noqa: E402

print("HARNESS: imported; current motion:", g.MotionManager().get_curr_motion().name,
      "char file:", g.g_char_filepath)

current_step = ["startup"]


def callback():
    try:
        app.main_loop()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        print("HARNESS: FAILED during step:", current_step[0])
        sys.stdout.flush()
        os._exit(1)


ps.set_user_callback(callback)


def frames(n, click=(), step=None):
    current_step[0] = step or ("click " + ", ".join(click) if click else "idle")
    CLICK.clear()
    CLICK.update(click)
    del CLICKED[:]
    for _ in range(n):
        ps.frame_tick()
    CLICK.clear()
    missing = set(click) - set(CLICKED)
    if missing:
        print("HARNESS: MISSING buttons never drawn:", sorted(missing))
    print("HARNESS: ok:", current_step[0])


def open_guis(*names):
    for name in names:
        setattr(g.MainVars(), name + "_gui_opened", True)


def loaded_motions():
    return list(g.MotionManager().get_loaded_motions().keys())


EDITING_GUIS = ("time", "motion", "contact", "terrain", "optimization", "path_planning", "recording")

if SCENARIO == "basic":
    open_guis(*EDITING_GUIS)
    frames(3, step="all editing GUIs open")
    frames(1, click=["Guess Terrain 2", "Gen Cave", "Resample motion frames for safe trajectory"],
           step="unavailable actions")
    frames(1, click=["Compute contact point constraints (for optimization)"])
    print("HARNESS: body constraints:",
          {b: len(c) for b, c in enumerate(g.OptimizationSettings().body_constraints) if len(c) > 0})
    g.OptimizationSettings().num_iters = 3
    g.OptimizationSettings().use_wandb = False
    g.OptimizationSettings().visualize_optimization = False
    frames(1, click=["Motion optimization"])
    assert "opt_motion" in loaded_motions(), loaded_motions()
    # animated optimization (motion_contact_optimization(yield_intermediate=True))
    g.MotionManager().set_curr_motion(os.path.basename(CFG["motions"][0][0]))
    g.OptimizationSettings().visualize_optimization = True
    frames(1, click=["Motion optimization"])
    frames(6, step="animated optimization frames")
    assert optimization_gui.g_opt_generator is None, "the optimization generator should be exhausted"
    frames(1, click=["Correct and Label Foot Contacts"])
    frames(1, click=["Correct and Label Foot Contacts, and Label Hand Contacts"])
    frames(1, click=["Regenerate Terrain"])
    frames(1, click=["Scale Motion Segment"])
    frames(1, click=["Cut frames [inclusive]"])
    frames(1, click=["Save Current Motion"])
    frames(1, click=["[LEGACY] Save Current Motion"])
    if CFG.get("load_motion"):
        g.g_motion_filepath = CFG["load_motion"]
    frames(1, click=["Load Motion"])
    frames(2)
elif SCENARIO == "stitch":
    open_guis("time", "motion")
    frames(2)
    frames(1, click=["Stitch selected motions"])
    frames(2)
elif SCENARIO == "path_mdm":
    open_guis("time", "path_planning", "mdm")
    frames(2)
    frames(1, click=["Generate Frames along path from current motion"])
    frames(2)
elif SCENARIO == "draw_all":
    open_guis(*EDITING_GUIS, "mdm", "ig_obs", "isaac_gym", "kin_controller", "help")
    frames(3, step="every GUI open")
else:
    raise ValueError(SCENARIO)

print("HARNESS: loaded motions:", loaded_motions())
print("HARNESS: DONE", SCENARIO)
sys.stdout.flush()
os._exit(0)
