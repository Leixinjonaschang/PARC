import sys
sys.path.insert(1, sys.path[0] + ("/../.."))

import os
import torch
import pickle
import yaml
import time

import diffusion.mdm as mdm
import diffusion.gen_util as gen_util

import tools.procgen.mdm_path as mdm_path
import zmotion_editing_tools.motion_edit_lib as medit_lib

cpu_device = "cpu"
cuda_device = "cuda:0"

def load_mdm(mdm_path) -> mdm.MDM:
    with open(mdm_path, 'rb') as input_filestream:
        ret_mdm = pickle.load(input_filestream)
        if ret_mdm.use_ema:
            print('Using EMA model...')
            ret_mdm._denoise_model = ret_mdm._ema_denoise_model
        ret_mdm.update_old_mdm()
    return ret_mdm

if __name__ == "__main__":
    if len(sys.argv) == 3:
        assert sys.argv[1] == "--config"
        cfg_path = sys.argv[2]
        print("loading mdm validation config from", cfg_path)
    else:
        cfg_path = "tools/motion_tests/mdm_validation_test.yaml"
    
    try:
        with open(cfg_path, "r") as stream:
            config = yaml.safe_load(stream)
    except IOError:
        print("error opening file:", cfg_path)
        exit()

    with open(config["mdm_model"], "rb") as f:
        mdm_model = pickle.load(f)

    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    with open(config["input_terrains"], "r") as f:
        terrain_data = yaml.safe_load(f)
        terrain_files = terrain_data["terrains"]
        if isinstance(terrain_files, str):
            terrain_files = [terrain_files]

    mdm_path_settings = mdm_path.MDMPathSettings()
    mdm_path_config = config["mdm_path"]
    for key in mdm_path_config:
        setattr(mdm_path_settings, key, mdm_path_config[key])

    mdm_gen_settings = gen_util.MDMGenSettings()
    mdm_gen_config = config["mdm_gen"]
    for key in mdm_gen_config:
        setattr(mdm_gen_settings, key, mdm_gen_config[key])

    torch.manual_seed(0)
    for terrain_file in terrain_files:

        terrain_name = os.path.splitext(os.path.basename(terrain_file))[0]
        with open(terrain_file, "rb") as f:
            terrain_data = pickle.load(f)

            print("Generating for:", terrain_file)
            path_nodes = terrain_data["path_nodes"]
            terrain = terrain_data["terrain"]
            terrain.to_torch(cpu_device)

        start_time = time.time()
        
        gen_motion_frames, out_terrains, info = mdm_path.generate_frames_until_end_of_path(
            path_nodes=path_nodes,
            terrain=terrain,
            char_model=mdm_model._kin_char_model,
            mdm_model=mdm_model,
            prev_frames=None,
            mdm_path_settings=mdm_path_settings,
            mdm_gen_settings=mdm_gen_settings,
            add_noise_to_loss=False,
            verbose=True,
            slice_terrain=False)
        
        end_time = time.time()
        print("generation time:", end_time - start_time)
        
        zfill_num = len(str(len(gen_motion_frames)))
        for i in range(len(gen_motion_frames)):
            save_data = dict()
            frames, contacts = gen_motion_frames[i].get_mlib_format(mdm_model._kin_char_model)
            save_data["frames"] = frames
            save_data["contacts"] = contacts
            save_data["fps"] = 30
            save_data["loop_mode"] = "CLAMP"
            save_data["terrain"] = out_terrains[i]
            save_data["path_nodes"] = path_nodes.cpu()
            motion_data = medit_lib.MotionData(save_data)
            save_path = output_folder + terrain_name + "_" + str(i).zfill(zfill_num) + ".pkl"
            motion_data.save_to_file(save_path)