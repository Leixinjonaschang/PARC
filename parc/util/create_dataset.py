from pathlib import Path
from typing import List

import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.util.file_io as file_io
import parc.util.file_io_helper as file_io_helper
import parc.util.geom_util as geom_util
import parc.util.motion_edit_lib as medit
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util

"""
Takes folders of motion data and creates a dataset .yaml file with proporitional sampling weights.
Motion classes are created based on the first layer of nested folders.
By default, all motion classes will have the same weighting: 1.0.
Optionally also computes processed terrain data (needed for training mdm).
"""
def create_dataset_yaml(
    folder_paths: List[Path],
    save_path: Path,
    char_filepath: str,
    compute_preprocessing_data: bool,
    override_old_hf_mask_inds: bool,
    cut_some_classes_in_half: bool,
    motion_classes_to_cut_in_half: List[str],
    max_terrain_dim_x: int,
    max_terrain_dim_y: int,
    min_num_frames: int
):
    motion_classes = []
    motion_class_proportions = dict()
    # Loop through first layer folders in
    for folder_path in folder_paths:
        folder_names = sorted([p for p in folder_path.iterdir() if p.is_dir()])
        
        for folder in folder_names:
            if "ignore" in str(folder):
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
            # load and save with file_io, not file_io_helper, because we want to preserve most of the file contents
            motion_filepath = sorted_motion_files[i]
            file_data = file_io.load_ms_file(str(motion_filepath))
            motion_frames = motion_util.MotionFrames.from_ms_motion_data(file_data.motion_data, device="cpu")
            num_frames = motion_frames.root_pos.shape[0]
            if num_frames < min_num_frames:
                print("excluding motion with too few frames:", num_frames, "<", min_num_frames)
                continue

            fps = file_data.motion_data.fps
            motion_len = num_frames / fps

            terrain = terrain_util.SubTerrain.from_ms_terrain_data(terrain_data=file_data.terrain_data, device="cpu")
            if terrain.hf.shape[0] > max_terrain_dim_x or terrain.hf.shape[1] > max_terrain_dim_y:
                print("Large terrain excluded")
                print(motion_filepath)
                print(terrain.hf.shape)
                continue

            motion_class_found = False
            for m_class in motion_classes:
                if (("/" + m_class + "/") in str(motion_filepath)) or (("\\" + m_class + "\\") in str(motion_filepath)):
                    motions_dict[m_class].append(MotionFile(motion_filepath, motion_len))
                    motion_class_lengths[m_class] += motion_len
                    motion_class_found = True
                    break
            assert motion_class_found, ("no motion class found in ", motion_filepath)

            if compute_preprocessing_data:
                if file_data.misc_data is None:
                    file_data.misc_data = {}
                if (not (file_io_helper.HF_MASK_INDS_KEY in file_data.misc_data) or override_old_hf_mask_inds):
                    hf_mask_inds, terrain = terrain_util.compute_hf_extra_vals(
                        motion_frames=motion_frames,
                        terrain=terrain,
                        char_model=char_model,
                        char_body_points=char_body_points,
                        z_buf=z_buf,
                        jump_buf=jump_buf)
                    file_data.misc_data[file_io_helper.HF_MASK_INDS_KEY] = hf_mask_inds
                    file_data.terrain_data.hf_maxmin = terrain.hf_maxmin.detach().cpu().numpy()

            file_io.save_ms_file(data=file_data, filepath=str(motion_filepath))

    total_length = 0.0
    for m_class in motion_classes:
        total_length += motion_class_lengths[m_class]

    motion_class_weight_factor = {}

    for m_class in motion_classes:
        print(m_class, "total length:", motion_class_lengths[m_class])
        assert motion_class_lengths[m_class] > 0.0, m_class + " length is: " + motion_class_lengths[m_class]

        fraction = motion_class_lengths[m_class] / total_length
        print(m_class, "fraction:", fraction)
        intended_fraction = motion_class_proportions[m_class] / motion_class_proportions_sum
        print(m_class, "intended fraction", intended_fraction)
        print(m_class, "weight_factor", intended_fraction / fraction)

        motion_class_weight_factor[m_class] = intended_fraction / fraction

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
    print("Creating datatset yaml:", config["save_path"])
    create_dataset_yaml(
        folder_paths=[Path(p) for p in config["folder_paths"]],
        save_path=Path(config["save_path"]),
        char_filepath=config["char_filepath"],
        compute_preprocessing_data=config["compute_preprocessing_data"],
        override_old_hf_mask_inds=config["override_old_hf_mask_inds"],
        cut_some_classes_in_half=config["cut_some_classes_in_half"],
        motion_classes_to_cut_in_half=config["motion_classes_to_cut_in_half"],
        max_terrain_dim_x=config["max_terrain_dim_x"],
        max_terrain_dim_y=config["max_terrain_dim_y"],
        min_num_frames = config["min_num_frames"]
    )