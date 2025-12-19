import yaml
import os
import datetime
from pathlib import Path
# this script is used to set up the configs for the PARC iteration
output_dir = Path("../tests/parc/experiment_g1/iter_1/")

##### INPUT PATHS #####

## MDM CONFIG ###
input_mdm_config_path = Path("PARC/train_gen_default.yaml")
input_model_path = None # put path to model here after iteration 1
input_char_file = Path("data/assets/unitree_g1/g1_mocap_29dof.xml")

## KINEMATICS GENERATION CONFIG ###
input_kin_gen_config_path = Path("PARC/kin_gen_default.yaml")
kin_gen_num_batches_of_motions = 10 # number of parallel jobs to run
kin_gen_num_motions_per_batch = 50
kin_gen_motion_id_offset = 0 # numbering starts from this number
kin_gen_save_name = "boxes"

## TRACKER CONFIG ###
input_tracker_config_path = Path("PARC/tracker_default.yaml")
input_sampler_stats_path = None # put path to sampler stats file here (used for normalizing samples) after iter 1
input_tracker_model_path = None # put path to tracker here after iteration 1


## PHYS RECORD CONFIG ###
input_phys_record_config_path = Path("PARC/phys_record_default.yaml")

### CREATE DATASET CONFIG ###
input_create_dataset_config_path = Path("PARC/create_dataset_config.yaml")
iter_start_dataset_path = Path("../tests/parc/experiment_g1/iter_1/iter_1_start_motions.yaml")
input_dataset_folder_paths = ["../Data/initial_retargeted_g1_processed/"] # replace with path to your initial dataset, could be the downloaded dataset "parc_dataset_august_12/initial_aug/" if you want to reproduce results


write_train_gen = True
write_kin_gen = True
write_tracker = True
write_phys_record = True


# Ensure all input paths exist
assert input_create_dataset_config_path.is_file()
assert input_mdm_config_path.is_file()
if input_model_path is not None:
    assert input_model_path.is_file()
assert input_kin_gen_config_path.is_file()
assert input_tracker_config_path.is_file()
#assert

# create output directories
output_train_gen_dir = output_dir / "p1_train_gen"
output_kin_gen_dir = output_dir / "p2_kin_gen"
output_tracker_dir = output_dir / "p3_tracker"
output_phys_record_dir = output_dir / "p4_phys_record"


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

output_train_gen_config_path = output_train_gen_dir / ("mdm_config_" + timestamp + ".yaml")
output_tracker_config_path = output_tracker_dir / ("tracker_" + timestamp + ".yaml")
output_phys_record_config_path = output_phys_record_dir / ("phys_record_" + timestamp + ".yaml")


dataset_config = yaml.safe_load(input_create_dataset_config_path.read_text())


# first create MDM training config
print("********** CREATING PARC 1 TRAIN GEN CONFIGS **********")
mdm_config = yaml.safe_load(input_mdm_config_path.open("r"))
mdm_config["motion_lib_file"] = str(iter_start_dataset_path)
mdm_config["char_file"] = str(input_char_file)
if input_model_path is not None:
    mdm_config["input_model_path"] = str(input_model_path)
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
    output_train_gen_config_path.write_text(yaml.dump(mdm_config))
    train_gen_dataset_config_path.write_text(yaml.dump(dataset_config))


# then create MDM generation configs
print("********** CREATING PARC 2 KIN GEN CONFIGS **********")
mdm_procgen_config = yaml.safe_load(input_kin_gen_config_path.open('r'))

for i in range(kin_gen_num_batches_of_motions):
    curr_motion_id_offset = kin_gen_motion_id_offset + i * kin_gen_num_motions_per_batch
    curr_kin_gen_folder_name = kin_gen_save_name + "_" + str(curr_motion_id_offset) + "_" + str(curr_motion_id_offset + kin_gen_num_motions_per_batch - 1)
    curr_opt_kin_gen_folder_path = output_kin_gen_dir / curr_kin_gen_folder_name
    curr_raw_kin_gen_folder_path = output_kin_gen_dir / "ignore" / "raw" / curr_kin_gen_folder_name

    os.makedirs(curr_opt_kin_gen_folder_path, exist_ok = True)
    os.makedirs(curr_raw_kin_gen_folder_path, exist_ok = True)

    output_kin_gen_config_path = curr_opt_kin_gen_folder_path / ("kin_gen_config" + ".yaml")
    opt_kin_gen_output_dir = str(curr_opt_kin_gen_folder_path)
    raw_kin_gen_output_dir = str(curr_raw_kin_gen_folder_path)
    # TODO: when loading a model and given a folder, load the latest checkpoint from the folder
    mdm_procgen_config["mdm_model_path"] = str(output_train_gen_dir / "checkpoints")
    mdm_procgen_config["output_dir"] = raw_kin_gen_output_dir
    mdm_procgen_config["opt"]["output_dir"] = opt_kin_gen_output_dir
    mdm_procgen_config["opt"]["use_wandb"] = False
    mdm_procgen_config["opt"]["char_model"] = str(input_char_file)

    mdm_procgen_config["motion_id_offset"] = curr_motion_id_offset
    mdm_procgen_config["num_new_motions"] = kin_gen_num_motions_per_batch

    mdm_procgen_config["save_name"] = kin_gen_save_name

    # TODO: option to create multiple of these and a script to scp this to a server

    # then create tracker training config
    # need to make dataset file by merging opt_kin_gen_output_dir and input_dataset_path
    if write_kin_gen:
        output_kin_gen_config_path.write_text(yaml.safe_dump(mdm_procgen_config))

print("********** CREATING PARC 3 TRACKER CONFIGS **********")
tracker_config = yaml.safe_load(input_tracker_config_path.open('r'))

# Setup tracker env config with correct char file
tracker_env_config_path = Path(tracker_config["env_config"])
if not tracker_env_config_path.exists():
    tracker_env_config_path = Path("PARC") / tracker_env_config_path
if tracker_env_config_path.exists():
    env_config = yaml.safe_load(tracker_env_config_path.read_text())
    env_config["env"]["char_file"] = str(input_char_file)
    output_env_config_path = output_tracker_dir / "dm_env.yaml"
    os.makedirs(output_tracker_dir, exist_ok=True)
    output_env_config_path.write_text(yaml.dump(env_config))
    tracker_config["env_config"] = str(output_env_config_path)
else:
    print(f"Warning: Could not find env config at {tracker_env_config_path}")

# Setup tracker agent config
tracker_agent_config_path = Path(tracker_config["agent_config"])
if not tracker_agent_config_path.exists():
    tracker_agent_config_path = Path("PARC") / tracker_agent_config_path
if tracker_agent_config_path.exists():
    agent_config = yaml.safe_load(tracker_agent_config_path.read_text())
    output_agent_config_path = output_tracker_dir / "agent_config.yaml"
    os.makedirs(output_tracker_dir, exist_ok=True)
    output_agent_config_path.write_text(yaml.dump(agent_config))
    tracker_config["agent_config"] = str(output_agent_config_path)
else:
    print(f"Warning: Could not find agent config at {tracker_agent_config_path}")

if input_tracker_model_path is not None:
    tracker_config["in_model_file"] = str(input_tracker_model_path)
else:
    tracker_config["in_model_file"] = None

tracker_config["output_dir"] = str(output_tracker_dir)

tracker_dataset_path = output_tracker_dir / "motions.yaml"
tracker_create_data_config_path = output_tracker_dir / "create_dataset_config.yaml"
tracker_config["create_dataset_config"] = str(tracker_create_data_config_path)
dataset_config["folder_paths"].append(str(output_kin_gen_dir))
dataset_config["save_path"] = str(tracker_dataset_path)
tracker_config["dataset_file"] = str(tracker_dataset_path)
if write_tracker:
    os.makedirs(output_tracker_dir, exist_ok=True)
    output_tracker_config_path.write_text(yaml.dump(tracker_config))
    tracker_create_data_config_path.write_text(yaml.dump(dataset_config))

# then create record motions config
print("********** CREATING PARC 4 PHYS RECORD CONFIGS **********")
# We use the tracker_env_config
phys_record_config = yaml.safe_load(input_phys_record_config_path.open('r'))
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
    output_phys_record_config_path.write_text(yaml.dump(phys_record_config))
    phys_record_create_data_config_path.write_text(yaml.dump(dataset_config))