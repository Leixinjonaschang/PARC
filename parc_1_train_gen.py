import yaml
import wandb
import os
from shutil import copyfile
from diffusion.mdm import MDM
from diffusion.mdm_heightfield_contact_motion_sampler import MDMHeightfieldContactMotionSampler
from datetime import datetime
import sys
import pickle
from pathlib import Path
from PARC.util.create_dataset import create_dataset_yaml_from_config

def train_mdm(config, input_mdm=None):
    use_wandb = config["use_wandb"]
    
    if "create_dataset_config" in config:
        create_dataset_config = yaml.safe_load(Path(config["create_dataset_config"]).read_text())
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
        motion_sampler = pickle.load(sampler_file_path.open("rb"))
        motion_sampler.update_old_sampler(config)
        # If device still mismatched, rebuild sampler to guarantee correct device
        if str(motion_sampler._device) != str(config["device"]):
            print(f"Rebuilding sampler on device {config['device']} to match config")
            motion_sampler = MDMHeightfieldContactMotionSampler(cfg=config)
            sampler_file_path.write_bytes(pickle.dumps(motion_sampler))
    else:
        motion_sampler = MDMHeightfieldContactMotionSampler(cfg=config)
        sampler_file_path.write_bytes(pickle.dumps(motion_sampler))
    
    config['seq_len'] = motion_sampler.get_seq_len()

    if input_mdm is None:
        if "input_model_path" in config:
            input_model_path = Path(config["input_model_path"])
            diffusion_model = pickle.load(input_model_path.open("rb"))
            # Pass the new config to update_old_mdm so it can handle device updates
            diffusion_model.update_old_mdm(config)
            # Extra guard: ensure device matches config even if checkpoint device differs
            if "device" in config:
                diffusion_model.set_device(config["device"])
            diffusion_model._use_wandb = use_wandb

            # Allow overriding training parameters from the new config
            if "epochs" in config:
                print(f"Overriding epochs: {diffusion_model._epochs} -> {config['epochs']}")
                diffusion_model._epochs = config["epochs"]
            if "batch_size" in config:
                print(f"Overriding batch_size: {diffusion_model._batch_size} -> {config['batch_size']}")
                diffusion_model._batch_size = config["batch_size"]
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
            project="train-mdm",
            config=config
        )
    
    diffusion_model.train(motion_sampler, checkpoint_dir, stats_filepath=Path(config["sampler_stats_filepath"]))

    output_model_path = output_dir / "model.pkl"
    diffusion_model.save(output_model_path)
    print("saved diffusion model:", output_model_path)

    if use_wandb:
        run.finish()

    return diffusion_model

if __name__ == "__main__":

    if len(sys.argv) == 3:
        assert sys.argv[1] == "--config"
        cfg_path = Path(sys.argv[2])
        print("loading mdm training config from", cfg_path)
    else:
        cfg_path = Path("PARC/generator_default.yaml")
    
    try:
        config = yaml.safe_load(cfg_path.open("r"))
    except IOError:
        print("error opening file:", cfg_path)
        exit()

    train_mdm(config)