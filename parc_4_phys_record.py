import run
import sys
import yaml
import os
from pathlib import Path
from PARC.util.create_dataset import create_dataset_yaml_from_config

def record_motions(config):
    device = config["device"]

    output_dir = Path(config["output_dir"])
    
    model_file_path = Path(config["model_file"])
    agent_config_path = Path(config["agent_file"])
    env_config_path = Path(config["env_file"])

    if "create_dataset_config" in config:
        create_dataset_config = yaml.safe_load(Path(config["create_dataset_config"]).read_text())
        create_dataset_yaml_from_config(create_dataset_config)


    dataset_file_path = Path(create_dataset_config["save_path"])
    try:
        dataset = yaml.safe_load(dataset_file_path.read_text())
        num_motions = len(dataset["motions"])
        print("num_motions =", num_motions)
        num_envs = num_motions
    except IOError:
        print("error opening dataset file")
        exit()

    try:
        phys_record_env_config = yaml.safe_load(env_config_path.read_text())
        phys_record_env_config["env"]["dm"]["motion_file"] = str(dataset_file_path)
        phys_record_env_config["env"]["dm"]["terrain_save_path"] = str(output_dir / "terrain.pkl")
        phys_record_env_config["env"]["output_motion_dir"] = str(output_dir / "recorded_motions/ generated_motions")
        phys_record_env_config_path = output_dir / "record_env.yaml"
        phys_record_env_config_path.write_text(yaml.safe_dump(phys_record_env_config))
    except IOError:
        print("could not find env config path:", str(env_config_path))
        exit()

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
    

    #with open()

    print(record_argv)
    record_args_path = output_dir / "record_args.txt"
    record_args_path.write_text(record_argv[6:])

    run.main(record_argv.split())
    return

if __name__ == "__main__":
    

    if len(sys.argv) == 3:
        assert sys.argv[1] == "--config"
        cfg_path = sys.argv[2]
        print("loading record config from", cfg_path)
    else:
        cfg_path = "PARC/phys_record_default.yaml"

    try:
        with open(cfg_path, "r") as stream:
            config = yaml.safe_load(stream)
    except IOError:
        print("error opening file:", cfg_path)
        exit()

    record_motions(config=config)