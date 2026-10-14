from scripts import run_tracker
import argparse
import sys
import yaml
import os
from pathlib import Path
from parc.util.create_dataset import create_dataset_yaml_from_config
from parc.util import path_loader

def record_motions(config):
    device = config["device"]

    output_dir = Path(config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    model_file_path = Path(config["model_file"])
    agent_config_path = Path(config["agent_file"])
    env_config_path = Path(config["env_file"])

    # The motions to record: built from create_dataset_config when given, otherwise an
    # existing dataset yaml (dataset_file).
    if "create_dataset_config" in config:
        create_dataset_config = path_loader.load_config(config["create_dataset_config"])
        create_dataset_yaml_from_config(create_dataset_config)
        dataset_file_path = Path(create_dataset_config["save_path"])
    else:
        dataset_file_path = Path(config["dataset_file"])

    try:
        dataset = yaml.safe_load(dataset_file_path.read_text())
        num_motions = len(dataset["motions"])
        print("num_motions =", num_motions)
        num_envs = num_motions
    except IOError:
        print("error opening dataset file:", str(dataset_file_path))
        sys.exit(1)

    try:
        phys_record_env_config = path_loader.load_config(env_config_path)
        phys_record_env_config["env"]["dm"]["motion_file"] = str(dataset_file_path)
        phys_record_env_config["env"]["dm"]["terrain_save_path"] = str(output_dir / "terrain.pkl")
        phys_record_env_config["env"]["output_motion_dir"] = str(output_dir / "recorded_motions")
        phys_record_env_config_path = output_dir / "record_env.yaml"
        phys_record_env_config_path.write_text(yaml.safe_dump(phys_record_env_config))
    except IOError:
        print("could not find env config path:", str(env_config_path))
        sys.exit(1)

    record_argv = """run.py
--env_config {0}
--agent_config {1}
--model_file {2}
--num_envs {3}
--device {4}
--mode record
--visualize False
""".format(str(phys_record_env_config_path),
           str(agent_config_path),
           str(model_file_path),
           num_envs,
           device)

    print(record_argv)
    record_args_path = output_dir / "record_args.txt"
    record_args_path.write_text(record_argv[6:])

    run_tracker.main(record_argv.split())
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PARC stage 4: record physics-tracked motions.")
    parser.add_argument("--config", default="data/configs/parc_4_phys_record_default.yaml",
                        help="record config yaml (default: %(default)s)")
    cfg_path = parser.parse_args().config
    print("loading record config from", cfg_path)

    try:
        config = path_loader.load_config(cfg_path)
    except (IOError, AssertionError) as exc:
        print("error opening file:", cfg_path)
        print(exc)
        print("Current working directory:", os.getcwd())
        sys.exit(1)

    record_motions(config=config)
