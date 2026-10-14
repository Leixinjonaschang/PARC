from scripts import run_tracker
import argparse
import sys
import yaml
import os
from pathlib import Path
from parc.util.create_dataset import create_dataset_yaml_from_config
from parc.util import path_loader

def train_tracker(config):
    env_config_path = Path(config["env_config"])
    agent_config_path = Path(config["agent_config"])
    output_dir = Path(config["output_dir"])
    log_file = output_dir / "log.txt"
    out_model_file = output_dir / "model.pt"
    int_output_dir = output_dir / "checkpoints/"
    max_samples = config["max_samples"]
    num_envs = config["num_envs"]
    device = config["device"]
    #iter_index = config["iter_index"]

    in_model_file = config.get("in_model_file", None)
    if in_model_file == "None" or in_model_file == "none":
        in_model_file = None

    if "create_dataset_config" in config:
        create_dataset_config = path_loader.load_config(config["create_dataset_config"])
        create_dataset_yaml_from_config(create_dataset_config)

    dataset_file = config["dataset_file"]

    os.makedirs(output_dir, exist_ok=True)

    env_config = path_loader.load_config(env_config_path)
    env_config["env"]["dm"]["motion_file"] = dataset_file
    env_config["env"]["dm"]["terrain_save_path"] = str(output_dir / "terrain.pkl")

    agent_config = path_loader.load_config(agent_config_path)
    if in_model_file is not None:
        agent_config["normalizer_samples"] = 0
    

    new_env_config_path = output_dir / "dm_env.yaml"
    new_env_config_path.write_text(yaml.safe_dump(env_config))

    new_agent_config_path = output_dir / "agent_config.yaml"
    new_agent_config_path.write_text(yaml.safe_dump(agent_config))

    train_argv = """run.py
--env_config {0}
--agent_config {1}
--log_file {2}
--out_model_file {3}
--int_output_dir {4}
--max_samples {5}
--num_envs {6}
--device {7}
[MODEL_FILE]
--visualize False
""".format(str(new_env_config_path),
           str(new_agent_config_path),
           str(log_file),
           str(out_model_file),
           str(int_output_dir),
            max_samples,
            num_envs,
            device)
    
    if in_model_file is not None:
        train_argv = train_argv.replace("[MODEL_FILE]", "--model_file " + in_model_file)
    else:
        train_argv = train_argv.replace("[MODEL_FILE]", "")
    

    #with open()

    print(train_argv)
    train_args_path = output_dir / "train_args.txt"
    train_args_path.write_text(train_argv[6:])

    run_tracker.main(train_argv.split())
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PARC stage 3: train the tracking policy.")
    parser.add_argument("--config", default="data/configs/parc_3_tracker_default.yaml",
                        help="tracker training config yaml (default: %(default)s)")
    cfg_path = parser.parse_args().config
    print("loading tracker training config from", cfg_path)

    try:
        config = path_loader.load_config(cfg_path)
    except (IOError, AssertionError) as exc:
        print("error opening file:", cfg_path)
        print(exc)
        print("Current working directory:", os.getcwd())
        sys.exit(1)

    train_tracker(config=config)
