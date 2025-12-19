import sys
import os
import pickle
import yaml
import util.geom_util as geom_util
import util.terrain_util as terrain_util
import anim.kin_char_model as kin_char_model
import zmotion_editing_tools.motion_edit_lib as medit
from pathlib import Path
from typing import List
"""
Takes folders of motion data and creates a dataset .yaml file with proporitional sampling weights.
Motion classes are created based on the first layer of nested folders.
By default, all motion classes will have the same weighting: 1.0.
Optionally also computes processed terrain data (needed for training mdm).
"""
# TODO: backup dir to save motions before modifying them
# save_dir = "../tests/parc/april272025/iter_2/"
# yaml_name = "iter_2_end_motions"
# folder_paths = ["../tests/parc/april272025/iter_2/p4_phys_record/recorded_motions/",
#                 "../tests/parc/april272025/iter_1/p4_phys_record/recorded_motions/",
#                 "../Data/parkour_dataset_april/initial/"]
# char_filepath = "data/assets/humanoid.xml"

# compute_preprocessing_data = True

# cut_some_classes_in_half = True
# motion_classes_to_cut_in_half = [
#     "running",
#     "platform",
#     "vaulting",
#     "climbing",
#     "mid_climbing",
#     "stairs",
#     "jumping"]

# max_terrain_dim_x = 45
# max_terrain_dim_y = 45

def create_dataset_yaml(
    folder_paths: List[Path],
    save_path: Path,
    char_filepath: str,
    compute_preprocessing_data: bool,
    cut_some_classes_in_half: bool,
    motion_classes_to_cut_in_half: List[str],
    max_terrain_dim_x: int,
    max_terrain_dim_y: int
):
    motion_classes = []
    motion_class_proportions = dict()
    # Loop through first layer folders in
    for folder_path in folder_paths:
        folder_names = sorted([p for p in folder_path.iterdir() if p.is_dir()])
        
        for folder in folder_names:
            if "ignore" in str(folder) or folder.name == "log":
                continue
            motion_classes.append(folder.name)
            motion_class_proportions[folder.name] = 1.0
    print("MOTION CLASSES:")
    print(motion_classes)



    motion_class_proportions_sum = 0.0
    for key, val in motion_class_proportions.items():
        motion_class_proportions_sum += val

        assert key in motion_classes

    # collect all ".pkl" files in a folder (and subfolders) recursively
    dirs = []
    for folder_path in folder_paths:
        dirs.extend([p for p in folder_path.rglob("*") if (p.is_dir() and "ignore" not in str(p))])

    motions_dict = dict()
    motion_class_lengths = dict()

    for m_class in motion_classes:
        motions_dict[m_class] = []
        motion_class_lengths[m_class] = 0.0

    class MotionFile:
        def __init__(self, filepath, length):
            self.filepath = filepath
            self.length = length
            return

    motion_yaml = []

    z_buf = 3.0
    jump_buf = 0.8
    char_model = kin_char_model.KinCharModel(device="cpu")
    char_model.load_char_file(char_filepath)
    char_body_points = geom_util.get_char_point_samples(char_model)

    for dir in dirs:
        motion_files = list(dir.glob("*.pkl"))
        sorted_motion_files = sorted(motion_files)

        cut_motions_in_half = False
        if cut_some_classes_in_half:
            for class_name in motion_classes_to_cut_in_half:
                if class_name in str(dir):
                    cut_motions_in_half = True

        if cut_motions_in_half:
            sorted_motion_files = sorted_motion_files[::2]
        
        print("loading files in", dir)
        for i in range(len(sorted_motion_files)):
            motion_filepath = sorted_motion_files[i]
            #motion_filepath = os.path.join(dir, motion_file)

            #print("loading:", motion_filepath)
            
            motion_data = medit.load_motion_file(str(motion_filepath))
            motion_frames = motion_data.get_frames()
            num_frames = motion_frames.shape[0]
            fps = motion_data.get_fps()
            motion_len = num_frames / fps

            terrain = motion_data.get_terrain()
            if terrain.hf.shape[0] > max_terrain_dim_x or terrain.hf.shape[1] > max_terrain_dim_y:
                print("Large terrain excluded")
                print(motion_filepath)
                print(terrain.hf.shape)
                continue

            # filter out bad mdm generated motions
            if "loss" in motion_data._data and motion_data._data["loss"] > 20.0:
                print("BAD LOSS")
                print(motion_data["loss"])
                continue

            motion_class_found = False
            for m_class in motion_classes:
                if ("/" + m_class + "/") in str(motion_filepath):
                    motions_dict[m_class].append(MotionFile(motion_filepath, motion_len))
                    motion_class_lengths[m_class] += motion_len
                    motion_class_found = True
                    break
            assert motion_class_found, "no motion class found in " + motion_filepath
    #           motion_yaml.append({"file": motion_filepath, "weight": motion_len})

            if compute_preprocessing_data and not motion_data.has_hf_mask_inds():
                hf_mask_inds = terrain_util.compute_hf_extra_vals(
                    motion_frames=motion_frames,
                    terrain=terrain,
                    char_model=char_model,
                    char_body_points=char_body_points,
                    z_buf=z_buf,
                    jump_buf=jump_buf)

                motion_data.set_hf_mask_inds(hf_mask_inds)

                motion_data.set_terrain(terrain)

                motion_data.save_to_file(str(motion_filepath), True)


    total_length = 0.0
    for m_class in motion_classes:
        total_length += motion_class_lengths[m_class]

    motion_class_weight_factor = {}

    if total_length == 0:
        print("Warning: Total length of all motions is 0. Setting all weights to 0.")
        for m_class in motion_classes:
            motion_class_weight_factor[m_class] = 0.0
    else:
        for m_class in motion_classes:
            print(m_class, "total length:", motion_class_lengths[m_class])

            fraction = motion_class_lengths[m_class] / total_length
            print(m_class, "fraction:", fraction)
            intended_fraction = motion_class_proportions[m_class] / motion_class_proportions_sum
            print(m_class, "intended fraction", intended_fraction)
            
            if fraction > 0:
                print(m_class, "weight_factor", intended_fraction / fraction)
                motion_class_weight_factor[m_class] = intended_fraction / fraction
            else:
                print(m_class, "weight_factor set to 0 because fraction is 0")
                motion_class_weight_factor[m_class] = 0.0

    print("total length:", total_length)

    for m_class in motion_classes:
        curr_motions = motions_dict[m_class]
        weight_factor = motion_class_weight_factor[m_class]
        for motion in curr_motions:
            motion_yaml.append({"file": str(motion.filepath), 
                                "weight": motion.length * weight_factor})

    motion_yaml = {"motions": motion_yaml}

    save_path.write_text(yaml.dump(motion_yaml))

def create_dataset_yaml_from_config(config):
    create_dataset_yaml(
        folder_paths=[Path(p) for p in config["folder_paths"]],
        save_path=Path(config["save_path"]),
        char_filepath=config["char_filepath"],
        compute_preprocessing_data=config["compute_preprocessing_data"],
        cut_some_classes_in_half=config["cut_some_classes_in_half"],
        motion_classes_to_cut_in_half=config["motion_classes_to_cut_in_half"],
        max_terrain_dim_x=config["max_terrain_dim_x"],
        max_terrain_dim_y=config["max_terrain_dim_y"]
    )