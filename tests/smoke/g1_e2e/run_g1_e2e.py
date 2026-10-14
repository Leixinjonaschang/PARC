"""G1 end-to-end smoke: one tiny PARC iteration (stages 0 to 4) on synthetic G1 motions.

Runs every step in its own process (IsaacGym allows one sim per process), checks its outputs
and prints a summary with timings. Everything is written under $PARC_DATA_DIR/g1_e2e/, which
the first step deletes and recreates. See README.md for the steps and the expected output.

Run from the repo root, after checking nvidia-smi (stages 1 to 4 use cuda:0, at most 32 envs):
    PARC_DATA_DIR=<existing dir> WANDB_MODE=disabled \
        uv run python tests/smoke/g1_e2e/run_g1_e2e.py [--from_step stage3] [--to_step stage4]
"""
import argparse
import contextlib
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
E2E_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

G1_CHAR_FILE = "data/assets/unitree_g1/g1_mocap_29dof.xml"
G1_ENV_CONFIG = "data/configs/tracker_config/dm_env_g1.yaml"
G1_AGENT_CONFIG = "data/configs/tracker_config/dm_agent_g1.yaml"
G1_OBS_DIM = 2363  # porting plan, appendix A

STEPS = ["data", "stage0", "stage1", "stage2", "stage3", "stage4", "stage4b", "dataset"]
GPU_STEPS = {"stage1", "stage2", "stage3", "stage4", "stage4b"}


def quiet():
    # the in-process checks load the G1 and motion files, which print a line per mesh and per motion
    return contextlib.redirect_stdout(io.StringIO())


class E2E:
    def __init__(self, data_dir):
        self.data_dir = pathlib.Path(data_dir)
        self.root = self.data_dir / "g1_e2e"
        self.iter_dir = self.root / "iter_1"
        self.log_dir = self.root / "logs"
        self.p1 = self.iter_dir / "p1_train_gen"
        self.p2 = self.iter_dir / "p2_kin_gen"
        self.p3 = self.iter_dir / "p3_tracker"
        self.p4 = self.iter_dir / "p4_phys_record"
        self.p4b = self.iter_dir / "p4_phys_record_no_et"
        self.results = []

    # ------------------------------------------------------------------ helpers

    def run(self, step, args):
        """Runs `python <args>` from the repo root, tees its output to logs/<step>.log, returns it."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / (step + ".log")
        cmd = [sys.executable] + [str(a) for a in args]
        print("\n$ " + " ".join(cmd), flush=True)
        lines = []
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True, bufsize=1)
            for line in proc.stdout:
                log_file.write(line)
                lines.append(line.rstrip("\n"))
                if not line.startswith("Loaded mesh"):  # 36 per G1 model load
                    print("  | " + line, end="", flush=True)
            code = proc.wait()
        if code != 0:
            raise RuntimeError("{} exited with status {} (log: {})".format(" ".join(cmd), code, log_path))
        return lines

    def gpu_memory(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
            return out.stdout.strip()
        except OSError:
            return "nvidia-smi not found"

    @staticmethod
    def load_yaml(path):
        return yaml.safe_load(pathlib.Path(path).read_text())

    @staticmethod
    def log_values(lines, key):
        # values of one key in the "| key | value |" tables the agents print every iteration
        pattern = re.compile(r"^\|\s*" + re.escape(key) + r"\s*\|\s*(\S+)\s*\|$")
        return [m.group(1) for m in map(pattern.match, lines) if m]

    def check_motion_files(self, paths, expect_obs=False):
        import parc.anim.kin_char_model as kin_char_model
        import parc.anim.motion_lib as motion_lib
        import parc.util.file_io_helper as file_io_helper

        char_model = kin_char_model.KinCharModel("cpu")
        with quiet():
            char_model.load_char_file(str(REPO_ROOT / G1_CHAR_FILE))
        summaries = []
        for path in paths:
            data = file_io_helper.load_ms_file(str(path))
            motion = data.motion_data
            num_frames = motion.root_pos.shape[0]
            assert tuple(motion.joint_rot.shape) == (num_frames, char_model.get_num_joints() - 1, 4), path
            assert tuple(motion.body_contacts.shape) == (num_frames, char_model.get_num_bodies()), path
            assert data.terrain_data is not None, path
            if expect_obs:
                obs = data.misc_data[file_io_helper.OBS_KEY]
                assert obs.shape == (num_frames, G1_OBS_DIM), (path, obs.shape)
            with quiet():
                mlib = motion_lib.MotionLib.from_file(str(path), char_model, "cpu", contact_info=True)
            assert mlib.num_motions() == 1
            summaries.append("{}: {} frames, joint_rot {}, contacts {}{}".format(
                path.name, num_frames, tuple(motion.joint_rot.shape), tuple(motion.body_contacts.shape),
                ", obs {}".format(data.misc_data[file_io_helper.OBS_KEY].shape) if expect_obs else ""))
        return summaries

    def load_dataset(self, dataset_path):
        import parc.anim.kin_char_model as kin_char_model
        import parc.anim.motion_lib as motion_lib

        char_model = kin_char_model.KinCharModel("cpu")
        with quiet():
            char_model.load_char_file(str(REPO_ROOT / G1_CHAR_FILE))
            mlib = motion_lib.MotionLib.from_file(str(dataset_path), char_model, "cpu", contact_info=True)
        return mlib

    # ------------------------------------------------------------------ steps

    def step_data(self):
        if self.root.exists():
            shutil.rmtree(str(self.root))
        motion_dir = self.root / "motions"
        self.run("data_motions", [REPO_ROOT / "tests/fixtures/pipeline_g1_synthetic.py", motion_dir,
                                  "--num_motions_per_class", 3])
        self.run("data_dataset", ["-m", "scripts.create_dataset_yaml",
                                  "--config", E2E_DIR / "create_dataset_g1_e2e.yaml"])
        motions = self.load_yaml(self.root / "motions.yaml")["motions"]
        assert len(motions) == 6, motions
        mlib = self.load_dataset(self.root / "motions.yaml")
        return "6 synthetic G1 motions (walk, step) in motions.yaml, {:.2f} s in MotionLib".format(
            mlib.get_total_length())

    def step_stage0(self):
        self.run("stage0", ["-m", "scripts.parc_0_setup_iter", "--config", E2E_DIR / "parc_0_setup_iter_g1_e2e.yaml"])
        import parc.util.path_loader as path_loader

        mdm = self.load_yaml(self.p1 / "mdm_config.yaml")
        kin_gen = self.load_yaml(self.p2 / "boxes_0_0" / "kin_gen_config.yaml")
        tracker = self.load_yaml(self.p3 / "tracker.yaml")
        phys_record = self.load_yaml(self.p4 / "phys_record.yaml")
        datasets = [self.load_yaml(self.iter_dir / stage / "create_dataset_config.yaml")
                    for stage in ("p1_train_gen", "p3_tracker", "p4_phys_record")]
        env_copy = self.load_yaml(tracker["env_config"])

        assert mdm["char_file"] == G1_CHAR_FILE, mdm["char_file"]
        assert kin_gen["opt"]["char_model"] == G1_CHAR_FILE, kin_gen["opt"]["char_model"]
        assert all(d["char_filepath"] == G1_CHAR_FILE for d in datasets), [d["char_filepath"] for d in datasets]
        assert env_copy["env"]["char_file"] == G1_CHAR_FILE
        assert tracker["env_config"] == str(self.p3 / "input_dm_env.yaml"), tracker["env_config"]
        assert tracker["agent_config"] == G1_AGENT_CONFIG, tracker["agent_config"]
        # the env copy is dm_env_g1.yaml, in the same key order (pd_params: first match wins)
        source_env = path_loader.load_config(REPO_ROOT / G1_ENV_CONFIG)
        assert env_copy == source_env, "input_dm_env.yaml differs from " + G1_ENV_CONFIG
        assert list(env_copy["env"]["pd_params"]["stiffness"]) == list(source_env["env"]["pd_params"]["stiffness"])
        assert phys_record["env_file"] == str(self.p3 / "dm_env.yaml")
        assert phys_record["agent_file"] == str(self.p3 / "agent_config.yaml")
        assert kin_gen["procgen_mode"] == "BOXES" and kin_gen["num_new_motions"] == 1
        return ("G1 char file in mdm_config, kin_gen opt.char_model, 3 dataset configs and input_dm_env.yaml "
                "(= dm_env_g1.yaml, key order kept); tracker.yaml: env_config input_dm_env.yaml, "
                "agent_config dm_agent_g1.yaml")

    def step_stage1(self):
        lines = self.run("stage1", ["-m", "scripts.parc_1_train_gen", "--config", self.p1 / "mdm_config.yaml"])
        for name in ("sampler.pkl", "sampler_stats.txt", "final_model.ckpt", "checkpoints/model_1.ckpt"):
            assert (self.p1 / name).is_file(), name
        start_motions = self.load_yaml(self.iter_dir / "iter_1_start_motions.yaml")["motions"]
        assert len(start_motions) == 6, start_motions
        loss_keys = [k for k in ("HF_COLLISION_LOSS", "TARGET_LOSS") if any(k in line for line in lines)]
        assert loss_keys == ["HF_COLLISION_LOSS", "TARGET_LOSS"], loss_keys
        return "iter_1_start_motions.yaml (6 motions), sampler.pkl, checkpoints/model_1.ckpt, final_model.ckpt"

    def step_stage2(self):
        lines = self.run("stage2", ["-m", "scripts.parc_2_kin_gen",
                                    "--config", self.p2 / "boxes_0_0" / "kin_gen_config.yaml"])
        assert any("Using ema model" in line for line in lines), "kin gen did not use the EMA weights"
        opt_files = sorted((self.p2 / "boxes_0_0").glob("*_opt.pkl"))
        flipped_files = sorted((self.p2 / "boxes_0_0" / "flipped").glob("*_opt_flipped.pkl"))
        raw_files = sorted((self.p2 / "ignore" / "raw" / "boxes_0_0").glob("*.pkl"))
        assert len(opt_files) == 2 and len(flipped_files) == 2 and len(raw_files) == 2, (
            opt_files, flipped_files, raw_files)
        summaries = self.check_motion_files(opt_files + flipped_files)
        return "{} *_opt.pkl + {} flipped load in MotionLib: {}".format(
            len(opt_files), len(flipped_files), "; ".join(summaries[:1]))

    def step_stage3(self):
        lines = self.run("stage3", ["-m", "scripts.parc_3_tracker", "--config", self.p3 / "tracker.yaml"])
        assert any("(FuturePoseDMPPOAgent)" in line for line in lines), "not the future-pose agent"
        for name in ("dm_env.yaml", "agent_config.yaml", "log.txt", "model.pt", "motions.yaml"):
            assert (self.p3 / name).is_file(), name
        dm_env = self.load_yaml(self.p3 / "dm_env.yaml")
        assert dm_env["env"]["char_file"] == G1_CHAR_FILE
        assert dm_env["env"]["dm"]["motion_file"] == str(self.p3 / "motions.yaml")
        env_copy = self.load_yaml(self.p3 / "input_dm_env.yaml")
        assert list(dm_env["env"]["pd_params"]["stiffness"]) == list(env_copy["env"]["pd_params"]["stiffness"])
        assert len(self.load_yaml(self.p3 / "motions.yaml")["motions"]) == 10

        import torch
        state_dict = torch.load(str(self.p3 / "model.pt"), map_location="cpu")
        # develop's generation-C keys (tests/fixtures/learning_develop_genC_keys.json)
        assert tuple(state_dict["_model.obs_encoder._encoders.future_pose.fc.0.weight"].shape) == (512, 1620)
        assert tuple(state_dict["_model._predictor.fc.0.weight"].shape) == (512, 743)
        assert tuple(state_dict["_model._actor_layers.0.weight"].shape) == (2048, 775)

        iters = self.log_values(lines, "Iteration")
        predictor_loss = self.log_values(lines, "Predictor_Loss")
        train_return = self.log_values(lines, "Train_Return")
        assert len(iters) == 5 and len(predictor_loss) == 5, (iters, predictor_loss)
        assert all(v not in ("nan", "inf") for v in predictor_loss + train_return)
        return "5 iterations, Train_Return {} -> {}, Predictor_Loss {} -> {}; motions.yaml with 10 motions".format(
            train_return[0], train_return[-1], predictor_loss[0], predictor_loss[-1])

    def step_stage4(self):
        lines = self.run("stage4", ["-m", "scripts.parc_4_phys_record", "--config", self.p4 / "phys_record.yaml"])
        record_env = self.load_yaml(self.p4 / "record_env.yaml")
        assert record_env["env"]["char_file"] == G1_CHAR_FILE
        assert record_env["env"]["output_motion_dir"] == str(self.p4 / "recorded_motions")
        num_motions = len(self.load_yaml(self.p4 / "motions.yaml")["motions"])
        assert num_motions == 4, num_motions
        total = [line for line in lines if line.startswith("Total successfull motions:")]
        assert total, "no record summary"
        recorded = sorted((self.p4 / "recorded_motions").rglob("*.pkl"))
        if recorded:
            self.check_motion_files(recorded, expect_obs=True)
        return "exit 0, {} envs ({} motions), '{}', {} recorded files".format(
            num_motions, num_motions, total[-1], len(recorded))

    def step_stage4b(self):
        # Extra check: record mode with --test_predictor True and without early termination, so that the
        # untrained policy's recordings are kept (a motion counts as tracked when its end is reached).
        self.p4b.mkdir(parents=True, exist_ok=True)
        record_env = self.load_yaml(self.p4 / "record_env.yaml")
        record_env["env"]["enable_early_termination"] = False
        record_env["env"]["output_motion_dir"] = str(self.p4b / "recorded_motions")
        record_env["env"]["dm"]["terrain_save_path"] = str(self.p4b / "terrain.pkl")
        env_path = self.p4b / "record_env_no_et.yaml"
        env_path.write_text(yaml.safe_dump(record_env, sort_keys=False))
        num_motions = len(self.load_yaml(self.p4 / "motions.yaml")["motions"])
        lines = self.run("stage4b", ["-m", "scripts.run_tracker", "--mode", "record", "--visualize", "False",
                                     "--test_predictor", "True", "--rand_seed", 0,
                                     "--num_envs", num_motions, "--device", "cuda:0",
                                     "--env_config", env_path,
                                     "--agent_config", self.p3 / "agent_config.yaml",
                                     "--model_file", self.p3 / "model.pt"])
        assert any(line.startswith("Predictor test mode") for line in lines)
        recorded = sorted((self.p4b / "recorded_motions").rglob("*.pkl"))
        assert len(recorded) == num_motions, recorded
        summaries = self.check_motion_files(recorded, expect_obs=True)
        bins = sorted({p.parent.name for p in recorded})
        return "{} recorded MS files in {} load in MotionLib: {}".format(len(recorded), bins, summaries[0])

    def step_dataset(self):
        self.run("dataset", ["-m", "scripts.create_dataset_yaml",
                             "--config", E2E_DIR / "create_dataset_g1_e2e_produced.yaml"])
        dataset_path = self.iter_dir / "produced_motions.yaml"
        motions = self.load_yaml(dataset_path)["motions"]
        mlib = self.load_dataset(dataset_path)
        assert mlib.num_motions() == len(motions)
        num_recorded = sum("recorded_motions" in m["file"] for m in motions)
        return "produced_motions.yaml: {} motions ({} kinematic, {} recorded), {:.2f} s in MotionLib".format(
            len(motions), len(motions) - num_recorded, num_recorded, mlib.get_total_length())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from_step", choices=STEPS, default=STEPS[0])
    parser.add_argument("--to_step", choices=STEPS, default=STEPS[-1])
    args = parser.parse_args(argv)

    data_dir = os.environ.get("PARC_DATA_DIR")
    if not data_dir or not os.path.isdir(data_dir):
        print("set PARC_DATA_DIR to an existing directory (outputs go to $PARC_DATA_DIR/g1_e2e/)")
        return 2
    if os.environ.get("WANDB_MODE") is None:
        os.environ["WANDB_MODE"] = "disabled"

    e2e = E2E(data_dir)
    steps = STEPS[STEPS.index(args.from_step):STEPS.index(args.to_step) + 1]
    failed = None
    for step in steps:
        if step in GPU_STEPS:
            print("\nGPU memory used/total before {}: {}".format(step, e2e.gpu_memory()), flush=True)
        start = time.time()
        try:
            summary = getattr(e2e, "step_" + step)()
        except Exception as exc:
            failed = (step, "{}: {}".format(type(exc).__name__, exc))
            e2e.results.append((step, time.time() - start, "FAILED " + failed[1]))
            break
        e2e.results.append((step, time.time() - start, summary))

    print("\n===== G1 end-to-end summary ({}) =====".format(e2e.root))
    for step, seconds, summary in e2e.results:
        print("{:8s} {:6.1f} s  {}".format(step, seconds, summary))
    print("total    {:6.1f} s".format(sum(r[1] for r in e2e.results)))
    if failed is not None:
        print("FAILED at", failed[0])
        return 1
    print("G1 E2E OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
