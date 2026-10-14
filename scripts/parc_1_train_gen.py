import argparse
import sys
from pathlib import Path
# Ensure the repository root is on the import path so "parc" resolves consistently
# across platforms (editable installs on Windows sometimes fail to place the repo
# ahead of site-packages when invoking this script directly).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import yaml
import wandb
import os
from shutil import copyfile
from parc.motion_generator.mdm import MDM
from parc.motion_generator.mdm_heightfield_contact_motion_sampler import MDMHeightfieldContactMotionSampler
from datetime import datetime

import pickle
from pathlib import Path
from parc.util.create_dataset import create_dataset_yaml_from_config
from parc.util import path_loader
import time

import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)




def train_mdm(config, input_mdm=None):
    use_wandb = config["use_wandb"]

    if "create_dataset_config" in config:
        create_dataset_config = path_loader.load_config(config["create_dataset_config"])
        create_dataset_yaml_from_config(create_dataset_config)

    # Since loading the sampler can be pretty slow
    sampler_file_path = Path(config["sampler_save_filepath"])
    try:
        sampler_save_dir = sampler_file_path.parent
        print("making new directory:", sampler_save_dir)
        os.makedirs(sampler_save_dir, exist_ok=True)
    except:
        print("could not make directory")

    if sampler_file_path.is_file():
        print("loading sampler: ", sampler_file_path)
        load_start_time = time.time()
        motion_sampler = pickle.load(sampler_file_path.open("rb"))
        load_end_time = time.time()
        print("loading sampler took:", load_end_time - load_start_time, "seconds.")
    else:
        print("creating sampler: ", sampler_file_path)
        create_start_time = time.time()
        motion_sampler = MDMHeightfieldContactMotionSampler(cfg=config)
        create_end_time = time.time()
        print("creating sampler took:", create_end_time - create_start_time, "seconds.")
        sampler_file_path.write_bytes(pickle.dumps(motion_sampler))
    
    config['seq_len'] = motion_sampler.get_seq_len()

    if input_mdm is None:
        if "input_model_path" in config:
            input_model_path = Path(config["input_model_path"])
            diffusion_model = MDM.load_checkpoint(str(input_model_path), device=config["device"])
            diffusion_model._use_wandb = use_wandb
            diffusion_model._batch_size = config["batch_size"]
            diffusion_model._epochs = config["epochs"]
            diffusion_model._iters_per_epoch = config["iters_per_epoch"]
            diffusion_model._epochs_per_checkpoint = config["epochs_per_checkpoint"]
        else:
            diffusion_model = MDM(cfg=config)
    else:
        diffusion_model = input_mdm

    output_dir = Path(config['output_dir'])
    checkpoint_dir = output_dir / "checkpoints"
    print("making new directory:", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("making new directory:", checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    

    #copy_cfg_path = output_dir / "mdm_cfg" + datetime.today().strftime('%Y-%m-%d') + ".yaml"
    #copyfile(cfg_path, copy_cfg_path)
    # with open(copy_cfg_path, "w") as f:
    #     yaml.safe_dump(config, f)


    if use_wandb:
        wandb.login()
        run = wandb.init(
            project = "train-mdm",
            config = config
        )
    else:
        run = wandb.init(
            project = "train-mdm",
            config = config,
            mode = 'offline'
        )
    
    diffusion_model.train(motion_sampler, checkpoint_dir, stats_filepath=config["sampler_stats_filepath"])


    output_model_path = output_dir / "final_model.ckpt"
    diffusion_model.save_checkpoint(output_model_path)
    print("saved diffusion model:", output_model_path)

    run.finish()

    return diffusion_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PARC stage 1: train the motion generator (MDM).")
    parser.add_argument("--config", default="data/configs/parc_1_train_gen_default.yaml",
                        help="mdm training config yaml (default: %(default)s)")
    cfg_path = Path(parser.parse_args().config)
    print("loading mdm training config from", cfg_path)

    try:
        config = path_loader.load_config(cfg_path)
    except (IOError, AssertionError) as exc:
        print("error opening file:", cfg_path)
        print(exc)
        print("Current working directory:", os.getcwd())
        sys.exit(1)

    train_mdm(config)