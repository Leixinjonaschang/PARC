import run
import sys
import yaml
import os
from pathlib import Path
from PARC.util.create_dataset import create_dataset_yaml_from_config

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

    if "create_dataset_config" in config:
        create_dataset_config = yaml.safe_load(Path(config["create_dataset_config"]).read_text())
        create_dataset_yaml_from_config(create_dataset_config)

    dataset_file = config["dataset_file"]

    os.makedirs(output_dir, exist_ok=True)

    env_config = yaml.safe_load(env_config_path.read_text())
    env_config["env"]["dm"]["motion_file"] = dataset_file
    env_config["env"]["dm"]["terrain_save_path"] = str(output_dir / "terrain.pkl")

    agent_config = yaml.safe_load(agent_config_path.read_text())
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

    run.main(train_argv.split())
    return

if __name__ == "__main__":
    

    if len(sys.argv) == 3:
        assert sys.argv[1] == "--config"
        cfg_path = sys.argv[2]
        print("loading tracker training config from", cfg_path)
    else:
        cfg_path = "PARC/tracker_default.yaml"

    try:
        with open(cfg_path, "r") as stream:
            config = yaml.safe_load(stream)
    except IOError:
        print("error opening file:", cfg_path)
        exit()

    train_tracker(config=config)