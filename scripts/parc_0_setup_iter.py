import argparse
import yaml
import os
import datetime
import sys
from pathlib import Path
# Ensure the repository root is on the import path so "parc" resolves consistently
# across platforms (editable installs on Windows sometimes fail to place the repo
# ahead of site-packages when invoking this script directly).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))



# this script is used to set up the configs for the PARC iteration

from parc.util import path_loader



def setup_iter(config):
    output_dir = Path(config["output_dir"])

    write_train_gen = config["write_train_gen"]
    write_kin_gen = config["write_kin_gen"]
    write_tracker = config["write_tracker"]
    write_phys_record = config["write_phys_record"]

    ##### PARC 1 - TRAIN GEN #####
    input_mdm_config_path = Path(config["input_mdm_config_path"])
    input_mdm_model_path = config["input_mdm_model_path"]
    input_mdm_model_path = Path(input_mdm_model_path) if input_mdm_model_path is not None else None
    input_sampler_stats_path = config["input_sampler_stats_path"]
    input_sampler_stats_path = Path(input_sampler_stats_path) if input_sampler_stats_path is not None else None

    input_create_dataset_config_path = Path(config["input_create_dataset_config_path"])
    iter_start_dataset_path = Path(config["iter_start_dataset_path"])
    input_dataset_folder_paths = config["input_dataset_folder_paths"]

    ##### PARC 2 - KIN GEN #####
    input_kin_gen_config_path = Path(config["input_kin_gen_config_path"])
    kin_gen_num_batches_of_motions = config["kin_gen_num_batches_of_motions"] #10 # number of parallel jobs to run
    kin_gen_num_motions_per_batch = config["kin_gen_num_motions_per_batch"] # 50
    kin_gen_motion_id_offset = config["kin_gen_motion_id_offset"] #3500 # numbering starts from this number
    kin_gen_save_name = config["kin_gen_save_name"]
    kin_gen_procgen_mode = config["kin_gen_procgen_mode"]
    kin_gen_start_heading_mode = config["kin_gen_start_heading_mode"]

    ##### PARC 3 - TRACKER #####
    input_tracker_config_path = Path(config["input_tracker_config_path"])
    input_tracker_model_path = config["input_tracker_model_path"]
    input_tracker_model_path = Path(input_tracker_model_path) if input_tracker_model_path is not None else None

    ##### PARC 4 - PHYS RECORD #####
    input_phys_record_config_path = Path(config["input_phys_record_config_path"])


    # Ensure all input paths exist
    assert input_create_dataset_config_path.is_file(), str(input_create_dataset_config_path) + " is not a valid path"
    assert input_mdm_config_path.is_file(), str(input_mdm_config_path) + " is not a valid path"
    if input_mdm_model_path is not None:
        assert input_mdm_model_path.is_file(), str(input_mdm_model_path) + " is not a valid path"
    assert input_kin_gen_config_path.is_file(), str(input_kin_gen_config_path) + " is not a valid path"
    assert input_tracker_config_path.is_file(), str(input_tracker_config_path) + " is not a valid path"
    #assert


    output_train_gen_dir = output_dir / "p1_train_gen"
    output_kin_gen_dir = output_dir / "p2_kin_gen"
    output_tracker_dir = output_dir / "p3_tracker"
    output_phys_record_dir = output_dir / "p4_phys_record"



    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_train_gen_config_simple_path = output_train_gen_dir / "mdm_config.yaml"
    output_train_gen_config_path = output_train_gen_dir / ("mdm_config_" + timestamp + ".yaml")

    output_tracker_config_simple_path = output_tracker_dir / "tracker.yaml"
    output_tracker_config_path = output_tracker_dir / ("tracker_" + timestamp + ".yaml")

    output_phys_record_config_simple_path = output_phys_record_dir / "phys_record.yaml"
    output_phys_record_config_path = output_phys_record_dir / ("phys_record_" + timestamp + ".yaml")


    dataset_config = path_loader.load_config(input_create_dataset_config_path)


    # first create MDM training config
    print("********** CREATING PARC 1 TRAIN GEN CONFIGS **********")
    mdm_config = path_loader.load_config(input_mdm_config_path)
    mdm_config["motion_lib_file"] = str(iter_start_dataset_path)
    if input_mdm_model_path is not None:
        mdm_config["input_model_path"] = str(input_mdm_model_path)
    else:
        if "input_model_path" in mdm_config:
            del mdm_config["input_model_path"]
    mdm_config["output_dir"] = str(output_train_gen_dir)
    mdm_config["sampler_save_filepath"] = str(output_train_gen_dir / "sampler.pkl")
    mdm_config["sampler_stats_filepath"] = str(output_train_gen_dir / "sampler_stats.txt") if input_sampler_stats_path is None else str(input_sampler_stats_path)


    dataset_config["folder_paths"] = input_dataset_folder_paths
    dataset_config["save_path"] = str(iter_start_dataset_path)
    train_gen_dataset_config_path = output_train_gen_dir / "create_dataset_config.yaml"
    mdm_config["create_dataset_config"] = str(train_gen_dataset_config_path)

    if write_train_gen:
        os.makedirs(output_train_gen_dir, exist_ok=True)
        output_train_gen_config_simple_path.write_text(yaml.dump(mdm_config))
        output_train_gen_config_path.write_text(yaml.dump(mdm_config))
        train_gen_dataset_config_path.write_text(yaml.dump(dataset_config))


    # then create MDM generation configs
    print("********** CREATING PARC 2 KIN GEN CONFIGS **********")
    mdm_procgen_config = path_loader.load_config(input_kin_gen_config_path)

    for i in range(kin_gen_num_batches_of_motions):
        curr_motion_id_offset = kin_gen_motion_id_offset + i * kin_gen_num_motions_per_batch
        curr_kin_gen_folder_name = kin_gen_save_name + "_" + str(curr_motion_id_offset) + "_" + str(curr_motion_id_offset + kin_gen_num_motions_per_batch - 1)
        curr_opt_kin_gen_folder_path = output_kin_gen_dir / curr_kin_gen_folder_name
        curr_raw_kin_gen_folder_path = output_kin_gen_dir / "ignore" / "raw" / curr_kin_gen_folder_name
        

        output_kin_gen_config_path = curr_opt_kin_gen_folder_path / ("kin_gen_config" + ".yaml")
        opt_kin_gen_output_dir = str(curr_opt_kin_gen_folder_path)
        raw_kin_gen_output_dir = str(curr_raw_kin_gen_folder_path)
        mdm_procgen_config["mdm_model_path"] = str(output_train_gen_dir / "checkpoints")
        mdm_procgen_config["output_dir"] = raw_kin_gen_output_dir
        mdm_procgen_config["opt"]["output_dir"] = opt_kin_gen_output_dir
        mdm_procgen_config["opt"]["use_wandb"] = False
        mdm_procgen_config["motion_id_offset"] = curr_motion_id_offset
        mdm_procgen_config["num_new_motions"] = kin_gen_num_motions_per_batch
        mdm_procgen_config["save_name"] = kin_gen_save_name
        mdm_procgen_config["procgen_mode"] = kin_gen_procgen_mode
        mdm_procgen_config["first_heading_mode"] = kin_gen_start_heading_mode

        # then create tracker training config
        # need to make dataset file by merging opt_kin_gen_output_dir and input_dataset_path
        if write_kin_gen:
            os.makedirs(curr_opt_kin_gen_folder_path, exist_ok = True)
            os.makedirs(curr_raw_kin_gen_folder_path, exist_ok = True)
            output_kin_gen_config_path.write_text(yaml.safe_dump(mdm_procgen_config))

    print("********** CREATING PARC 3 TRACKER CONFIGS **********")
    tracker_config = path_loader.load_config(input_tracker_config_path)
    tracker_config["in_model_file"] = str(input_tracker_model_path)
    tracker_config["output_dir"] = str(output_tracker_dir)

    tracker_dataset_path = output_tracker_dir / "motions.yaml"
    tracker_create_data_config_path = output_tracker_dir / "create_dataset_config.yaml"
    tracker_config["create_dataset_config"] = str(tracker_create_data_config_path)
    dataset_config["folder_paths"].append(str(output_kin_gen_dir))
    dataset_config["save_path"] = str(tracker_dataset_path)
    tracker_config["dataset_file"] = str(tracker_dataset_path)
    if write_tracker:
        os.makedirs(output_tracker_dir, exist_ok=True)
        output_tracker_config_simple_path.write_text(yaml.dump(tracker_config))
        output_tracker_config_path.write_text(yaml.dump(tracker_config))
        tracker_create_data_config_path.write_text(yaml.dump(dataset_config))

    # then create record motions config
    print("********** CREATING PARC 4 PHYS RECORD CONFIGS **********")
    # We use the tracker_env_config
    phys_record_config = path_loader.load_config(input_phys_record_config_path)
    phys_record_config["env_file"] = str(output_tracker_dir / "dm_env.yaml")
    phys_record_config["agent_file"] = str(output_tracker_dir / "agent_config.yaml")
    phys_record_config["model_file"] = str(output_tracker_dir / "model.pt")
    phys_record_config["output_dir"] = str(output_phys_record_dir)
    phys_record_dataset_path = output_phys_record_dir / "motions.yaml"
    phys_record_create_data_config_path = output_phys_record_dir / "create_dataset_config.yaml"
    phys_record_config["create_dataset_config"] = str(phys_record_create_data_config_path)
    dataset_config["folder_paths"] = [str(output_kin_gen_dir)]
    dataset_config["save_path"] = str(phys_record_dataset_path)
    phys_record_config["dataset_file"] = str(phys_record_dataset_path)
    if write_phys_record:
        os.makedirs(output_phys_record_dir, exist_ok=True)
        output_phys_record_config_simple_path.write_text(yaml.dump(phys_record_config))
        output_phys_record_config_path.write_text(yaml.dump(phys_record_config))
        phys_record_create_data_config_path.write_text(yaml.dump(dataset_config))

    print("Finished writing config files for PARC iteration")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PARC stage 0: write the configs for a new iteration.")
    parser.add_argument("--config", default="data/configs/parc_0_setup_iter_config.yaml",
                        help="setup config yaml (default: %(default)s)")
    cfg_path = parser.parse_args().config
    print("loading setup config from", cfg_path)

    try:
        config = path_loader.load_config(cfg_path)
    except (IOError, AssertionError) as exc:
        print("error opening file:", cfg_path)
        print(exc)
        print("Current working directory:", os.getcwd())
        sys.exit(1)

    setup_iter(config=config)