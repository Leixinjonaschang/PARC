import argparse
import sys
import os
import numpy as np
import torch
import yaml
import enum
import time
import re
from pathlib import Path
from parc.anim import kin_char_model
from parc.util import geom_util
from parc.util import terrain_util
from parc.util import motion_util

from parc.motion_generator import mdm
from parc.motion_generator import gen_util

from parc.motion_synthesis.procgen import astar
from parc.motion_synthesis.procgen import mdm_path
from parc.motion_synthesis.motion_opt import motion_optimization as motion_opt
from parc.util import motion_edit_lib as medit_lib
from parc.util import file_io
from parc.util import file_io_helper
from parc.util import path_loader
from parc.util import pickle_compat

cpu_device = "cpu"
cuda_device = "cuda:0" # default for the "device" config key (the MDM's device)


class ProcGenMode(enum.Enum):
    BOXES = 0
    PATHS = 1
    STAIRS = 2
    FILE = 3

class ProcGenBoxesSettings:
    num_boxes = 10
    min_box_h = -3.0
    max_box_h = 3.0
    box_max_len = 10
    box_min_len = 5
    max_box_angle = 6.28318530718
    min_box_angle = 0.0

class ProcGenPathsSettings:
    num_terrain_paths = 4
    maxpool_size = 1
    path_min_height = -2.8
    path_max_height = 3.0
    floor_height = -3.0

class ProcGenStairsSettings:
    min_stair_start_height = -3.0
    max_stair_start_height = 1.0
    min_step_height = 0.15
    max_step_height = 0.25
    num_stairs = 4
    min_stair_thickness = 2.0
    max_stair_thickness = 8.0


def _apply_config_to_settings(settings_cls, config_dict):
    settings = settings_cls()
    for key, value in config_dict.items():
        setattr(settings, key, value)
    return settings


def _load_terrain_file(terrain_path):
    # Returns (terrain, path_nodes or None) on the CPU. Accepts an MS file with terrain
    # data (e.g. this script's only_gen output) or a legacy {"terrain": SubTerrain,
    # "path_nodes": ...} pickle (develop's only_gen output, teaser_terrain.pkl).
    try:
        file_data = file_io_helper.load_ms_file(str(terrain_path), device=cpu_device)
    except file_io.MSFileFormatError:
        # pickle_compat remaps develop-era module paths and loads GPU tensors on the CPU
        legacy_data = pickle_compat.load(str(terrain_path), map_location=cpu_device)
        if not (isinstance(legacy_data, dict) and "terrain" in legacy_data):
            raise ValueError("{}: neither an MS file nor a legacy terrain pickle with a 'terrain' key"
                             .format(terrain_path))
        terrain = legacy_data["terrain"]
        terrain.update_old()
        terrain.to_torch(cpu_device)
        path_nodes = legacy_data.get("path_nodes")
    else:
        if file_data.terrain_data is None:
            raise ValueError("{}: MS file has no terrain data".format(terrain_path))
        terrain = terrain_util.SubTerrain.from_ms_terrain_data(file_data.terrain_data, device=cpu_device)
        path_nodes = (file_data.misc_data or {}).get(file_io_helper.PATH_NODES_KEY)

    if path_nodes is not None:
        path_nodes = torch.as_tensor(path_nodes).to(device=cpu_device)
    return terrain, path_nodes


def _load_input_terrains(input_terrain_path):
    suffix = os.path.splitext(input_terrain_path)[1]
    if suffix == ".pkl":
        terrain_paths = [input_terrain_path]
    elif suffix == ".yaml":
        input_terrains_yaml = path_loader.load_config(input_terrain_path)
        terrain_paths = input_terrains_yaml["terrains"]
    else:
        raise AssertionError("Unsupported terrain file format: %s" % suffix)

    terrains = []
    paths = []
    for terrain_path in terrain_paths:
        terrain, path_nodes = _load_terrain_file(terrain_path)
        terrains.append(terrain)
        paths.append(path_nodes)

    return terrains, paths


def _ensure_dir_with_slash(path):
    os.makedirs(path, exist_ok=True)
    return path if path.endswith("/") else path + "/"

def load_mdm(mdm_path, device=cuda_device) -> mdm.MDM:
    # mdm_path: a .ckpt file, or a directory whose highest-numbered .ckpt is loaded.
    # Generation uses the EMA weights when the checkpoint has them.

    resolved_path = path_loader.resolve_path(mdm_path)

    if not resolved_path.exists():
        assert False, f"MDM path does not exist: {resolved_path}"

    if not resolved_path.is_file():
        assert resolved_path.is_dir(), f"MDM path must be a file or directory: {resolved_path}"

        number_file_pairs = []

        for file in resolved_path.iterdir():
            if file.is_file() and file.suffix == ".ckpt":
                match = re.search(r'\d+', file.name)
                if match:
                    number = int(match.group())
                    number_file_pairs.append((number, file))

        if not number_file_pairs:
            assert False, f"No checkpoint files found in directory: {resolved_path}"

        resolved_path = max(number_file_pairs, key=lambda x: x[0])[1]

    print("loading path:", resolved_path)
    ret_mdm = mdm.MDM.load_checkpoint(str(resolved_path), device=device, use_ema_model=True)
    return ret_mdm

def mdm_procgen(config, input_mdm_model = None):
    use_opt = config["use_opt"]
    remove_hesitation = config["remove_hesitation"]
    max_contact_loss = config["max_contact_loss"]
    max_pen_loss = config["max_pen_loss"]
    max_total_loss = config["max_total_loss"]
    motion_id_offset = config["motion_id_offset"]
    num_new_motions = config["num_new_motions"]
    new_terrain_dim_x = config["new_terrain_dim_x"]
    new_terrain_dim_y = config["new_terrain_dim_y"]
    terrain_dx = config["dx"]
    terrain_dy = config["dy"]
    first_heading_mode = config.get("first_heading_mode", "auto")
    print("first_heading_mode:", first_heading_mode)
    procgen_mode = ProcGenMode[config["procgen_mode"]]
    simplify_terrain = config["simplify_terrain"]
    save_name = config.get("save_name", None)

    if procgen_mode == ProcGenMode.FILE:
        input_terrain_path = config["input_terrain_path"]
        input_terrains, _ = _load_input_terrains(input_terrain_path)
        num_input_terrains = len(input_terrains)

    astar_settings = _apply_config_to_settings(astar.AStarSettings, config["astar"])
    mdm_path_settings = _apply_config_to_settings(mdm_path.MDMPathSettings, config["mdm_path"])
    mdm_gen_settings = _apply_config_to_settings(gen_util.MDMGenSettings, config["mdm_gen"])
    boxes_settings = _apply_config_to_settings(ProcGenBoxesSettings, config["boxes"])
    paths_settings = _apply_config_to_settings(ProcGenPathsSettings, config["paths"])
    stairs_settings = _apply_config_to_settings(ProcGenStairsSettings, config["stairs"])

    only_gen = config["only_gen"]

    mdm_device = config.get("device", cuda_device)
    print("MDM device:", mdm_device)

    # We can also try using mdm models from different arbitrary checkpoints
    if input_mdm_model is None:
        mdm_model_path = Path(config["mdm_model_path"])
        mdm_model = load_mdm(mdm_model_path, device=mdm_device)
    else:
        mdm_model = input_mdm_model

    char_model = mdm_model._kin_char_model.get_copy(cpu_device)

    output_folder = _ensure_dir_with_slash(config["output_dir"])

    print("Output folder:", output_folder)

    opt_char_model = None
    body_points = None
    if use_opt:
        opt_cfg = config["opt"]
        opt_device = opt_cfg["device"]
        char_model_path = opt_cfg["char_model"]
        opt_output_folder_path = Path(opt_cfg["output_dir"])
        opt_flipped_output_folder_path = opt_output_folder_path / "flipped"
        opt_log_folder_path = opt_output_folder_path / "log"
        os.makedirs(opt_output_folder_path, exist_ok=True)
        os.makedirs(opt_flipped_output_folder_path, exist_ok=True)
        os.makedirs(opt_log_folder_path, exist_ok=True)

        num_opt_iters = opt_cfg["num_iters"]
        step_size = opt_cfg["step_size"]
        w_root_pos = opt_cfg["w_root_pos"]
        w_root_rot = opt_cfg["w_root_rot"]
        w_joint_rot = opt_cfg["w_joint_rot"]
        w_smoothness = opt_cfg["w_smoothness"]
        w_penetration = opt_cfg["w_penetration"]
        w_contact = opt_cfg["w_contact"]
        w_sliding = opt_cfg["w_sliding"]
        w_body_constraints = opt_cfg["w_body_constraints"]
        w_jerk = opt_cfg["w_jerk"]
        max_jerk = opt_cfg["max_jerk"]
        use_wandb = opt_cfg["use_wandb"]
        auto_compute_body_constraints = opt_cfg["auto_compute_body_constraints"]

        opt_char_model = kin_char_model.KinCharModel(opt_device)
        opt_char_model.load_char_file(char_model_path)

        char_point_sample_cfg = opt_cfg["char_point_samples"]
        sphere_num_subdivisions = char_point_sample_cfg["sphere_num_subdivisions"]
        box_num_slices = char_point_sample_cfg["box_num_slices"]
        box_dim_x = char_point_sample_cfg["box_dim_x"]
        box_dim_y = char_point_sample_cfg["box_dim_y"]
        capsule_num_circle_points = char_point_sample_cfg["capsule_num_circle_points"]
        capsule_num_sphere_subdivisions = char_point_sample_cfg["capsule_num_sphere_subdivisions"]
        capsule_num_cylinder_slices = char_point_sample_cfg["capsule_num_cylinder_slices"]
        body_points = geom_util.get_char_point_samples(
            opt_char_model,
            sphere_num_subdivisions=sphere_num_subdivisions,
            box_num_slices=box_num_slices,
            box_dim_x=box_dim_x,
            box_dim_y=box_dim_y,
            capsule_num_circle_points=capsule_num_circle_points,
            capsule_num_sphere_subdivisons=capsule_num_sphere_subdivisions,
            capsule_num_cylinder_slices=capsule_num_cylinder_slices)

        # The optimization runs on opt_device; the optimized frames are post-processed
        # (hesitation removal, hf mask, flipping) and saved on the CPU.
        opt_char_model_cpu = opt_char_model.get_copy(cpu_device)
        body_points_cpu = [points.to(device=cpu_device) for points in body_points]


    base_motion_name = save_name if save_name is not None else procgen_mode.name

    def build_motion_name(motion_idx, variant_idx=None):
        name = f"{base_motion_name}_{motion_idx}"
        if variant_idx is not None:
            name = f"{name}_{variant_idx}"
        return name

    first_start_time = time.time()

    for motion_idx in range(num_new_motions):
        motion_id = motion_idx + motion_id_offset

        def gen_motion_and_terrain(only_gen=False):
            start_time = time.time()

            path_nodes_3d = None
            terrain = None
            min_point_offset = None

            for terrain_attempt in range(1000):
                terrain = terrain_util.SubTerrain(
                    "terrain",
                    x_dim=new_terrain_dim_x,
                    y_dim=new_terrain_dim_y,
                    dx=terrain_dx,
                    dy=terrain_dy,
                    min_x=0.0,
                    min_y=0.0,
                    device=cpu_device)

                slice_terrain = True

                if procgen_mode == ProcGenMode.BOXES:
                    terrain_util.add_boxes_to_hf2(
                        terrain.hf,
                        box_max_height=boxes_settings.max_box_h,
                        box_min_height=boxes_settings.min_box_h,
                        hf_maxmin=None,
                        num_boxes=boxes_settings.num_boxes,
                        box_max_len=boxes_settings.box_max_len,
                        box_min_len=boxes_settings.box_min_len,
                        max_angle=boxes_settings.max_box_angle,
                        min_angle=boxes_settings.min_box_angle)

                elif procgen_mode == ProcGenMode.PATHS:
                    terrain_util.gen_paths_hf(
                        terrain,
                        num_paths=paths_settings.num_terrain_paths,
                        maxpool_size=paths_settings.maxpool_size,
                        floor_height=paths_settings.floor_height,
                        path_min_height=paths_settings.path_min_height,
                        path_max_height=paths_settings.path_max_height)

                elif procgen_mode == ProcGenMode.STAIRS:
                    terrain_util.add_stairs_to_hf(
                        terrain,
                        min_stair_start_height=stairs_settings.min_stair_start_height,
                        max_stair_start_height=stairs_settings.max_stair_start_height,
                        min_step_height=stairs_settings.min_step_height,
                        max_step_height=stairs_settings.max_step_height,
                        num_stairs=stairs_settings.num_stairs,
                        min_stair_thickness=stairs_settings.min_stair_thickness,
                        max_stair_thickness=stairs_settings.max_stair_thickness)

                elif procgen_mode == ProcGenMode.FILE:
                    input_terrain = input_terrains[motion_id % num_input_terrains]
                    start_dim_x = np.random.randint(0, input_terrain.dims[0].item() + 1 - new_terrain_dim_x)
                    start_dim_y = np.random.randint(0, input_terrain.dims[1].item() + 1 - new_terrain_dim_y)

                    terrain_slice = input_terrain.hf[
                        start_dim_x:start_dim_x + new_terrain_dim_x,
                        start_dim_y:start_dim_y + new_terrain_dim_y
                    ].clone()
                    terrain.hf[:, :] = terrain_slice

                    min_point_offset = input_terrain.get_point(
                        torch.tensor([start_dim_x, start_dim_y], dtype=torch.int64, device=cpu_device))

                    slice_terrain = False

                hf_orig = terrain.hf.clone()

                for _ in range(10):
                    start_node, end_node = astar.pick_random_start_end_nodes_on_edges(
                        terrain,
                        min_dist=astar_settings.min_start_end_xy_dist)

                    terrain.hf = hf_orig.clone()

                    if simplify_terrain:
                        terrain_util.flat_maxpool_2x2(terrain=terrain)

                        terrain_util.flatten_4x4_near_edge(
                            terrain=terrain,
                            grid_ind=start_node,
                            height=terrain.hf[start_node[0], start_node[1]].item())

                        terrain_util.flatten_4x4_near_edge(
                            terrain=terrain,
                            grid_ind=end_node,
                            height=terrain.hf[end_node[0], end_node[1]].item())

                    path_nodes_3d = astar.run_a_star_on_start_end_nodes(
                        terrain=terrain,
                        start_node=start_node,
                        end_node=end_node,
                        settings=astar_settings)

                    if path_nodes_3d is not False:
                        break

                if path_nodes_3d is not False:
                    break

            if path_nodes_3d is False or path_nodes_3d is None:
                raise AssertionError("something wrong with procgen")

            if only_gen:
                return terrain, path_nodes_3d
            
            best_motion_frames, best_motion_terrains, gen_info = mdm_path.generate_frames_until_end_of_path(
                path_nodes = path_nodes_3d,
                terrain = terrain,
                char_model = char_model,
                mdm_model = mdm_model,
                prev_frames = None,
                mdm_path_settings=mdm_path_settings,
                mdm_gen_settings=mdm_gen_settings,
                add_noise_to_loss=False,
                verbose=True, 
                slice_terrain=slice_terrain,
                first_heading_mode=first_heading_mode)
            
            end_time = time.time()
            print("Time to generate motion", motion_id, "=", end_time - start_time, "seconds.")

            gen_info["path_nodes"] = path_nodes_3d
            if procgen_mode == ProcGenMode.FILE:
                gen_info["min_point_offset"] = min_point_offset

            return best_motion_frames, best_motion_terrains, gen_info


        if only_gen:
            terrain, path_nodes = gen_motion_and_terrain(only_gen=True)

            new_motion_name = build_motion_name(motion_id)

            save_path = output_folder + new_motion_name + ".pkl"
            file_io_helper.save_ms_file(save_path, terrain=terrain, path_nodes=path_nodes)
            continue

        attempt_counter = 0
        while True:
            best_motion_frames, best_motion_terrains, gen_info = gen_motion_and_terrain()
            path_nodes = gen_info["path_nodes"]

            all_losses = gen_info["losses"]
            contact_losses = gen_info["contact_losses"]
            pen_losses = gen_info["pen_losses"]

            valid_indices = (contact_losses < max_contact_loss) & (pen_losses < max_pen_loss) & (all_losses < max_total_loss)

            valid_indices = torch.nonzero(valid_indices, as_tuple=False).flatten()
            num_valid_indices = valid_indices.numel()

            print("Found", num_valid_indices, "/", mdm_path_settings.mdm_batch_size, "valid motions.")
            if num_valid_indices < mdm_path_settings.top_k:
                attempt_counter += 1
                print("Could not find enough (" + str(mdm_path_settings.top_k) + ") valid generated motions")
                print("attempts so far:", attempt_counter)
            else:
                # All losses is already sorted
                all_losses = all_losses[valid_indices]
                best_motion_frames = [best_motion_frames[i] for i in valid_indices.tolist()]
                best_motion_terrains = [best_motion_terrains[i] for i in valid_indices.tolist()]
                break
        
        print("Generated good motions based on contact and penetration losses!")

        for j in range(mdm_path_settings.top_k):
            motion_frames = best_motion_frames[j]
            assert isinstance(motion_frames, motion_util.MotionFrames)
            motion_frames.set_device(cpu_device)
            # mdm_path returns each motion with a leading batch dim of 1; saving and
            # the optimization expect [num_frames, ...] (develop squeezed it as well)
            if len(motion_frames.root_pos.shape) == 3:
                motion_frames = motion_frames.squeeze(0)

            new_motion_name = build_motion_name(motion_id, j)
            save_path = output_folder + new_motion_name + ".pkl"

            extra_misc_data = dict()
            extra_misc_data["loss"] = all_losses[j].item()
            if procgen_mode == ProcGenMode.FILE:
                min_point_offset = gen_info["min_point_offset"]
            else:
                min_point_offset = None

            # the raw generated motion (develop and pre-rewrite upstream saved it too)
            file_io_helper.save_ms_file(filepath=save_path,
                                        motion_frames=motion_frames,
                                        fps=mdm_model._sequence_fps,
                                        loop_mode="CLAMP",
                                        terrain=best_motion_terrains[j],
                                        min_point_offset=min_point_offset,
                                        path_nodes=path_nodes,
                                        extra_misc_data=extra_misc_data)

            if use_opt:
                log_file = opt_log_folder_path / ("log_" + new_motion_name + ".txt")

                # the generated terrains are on the MDM's device
                terrain = best_motion_terrains[j].torch_copy()
                terrain.set_device(device=opt_device)
                src_frames = motion_frames.get_copy(new_device=opt_device)

                if auto_compute_body_constraints:
                    print("Computing approx body constraints...")
                    body_constraints = motion_opt.compute_approx_body_constraints(
                        root_pos=src_frames.root_pos,
                        root_rot=src_frames.root_rot,
                        joint_rot=src_frames.joint_rot,
                        contacts=src_frames.contacts,
                        char_model=opt_char_model,
                        terrain=terrain)
                    print("Finished computing approx body constraints.")
                else:
                    body_constraints = None

                opt_frames = motion_opt.motion_contact_optimization(
                    src_motion_frames=src_frames,
                    body_points=body_points,
                    terrain=terrain,
                    char_model=opt_char_model,
                    num_iters=num_opt_iters,
                    step_size=step_size,
                    w_root_pos=w_root_pos,
                    w_root_rot=w_root_rot,
                    w_joint_rot=w_joint_rot,
                    w_smoothness=w_smoothness,
                    w_penetration=w_penetration,
                    w_contact=w_contact,
                    w_sliding=w_sliding,
                    w_body_constraints=w_body_constraints,
                    w_jerk=w_jerk,
                    body_constraints=body_constraints,
                    max_jerk=max_jerk,
                    exp_name=new_motion_name,
                    use_wandb=use_wandb,
                    log_file=log_file)
                

                opt_frames.set_device(device=cpu_device)
                terrain.set_device(device=cpu_device)
                if remove_hesitation:
                    opt_frames = medit_lib.remove_hesitation_frames(opt_frames, opt_char_model_cpu)

                opt_save_path = opt_output_folder_path / (new_motion_name + "_opt.pkl")

                hf_mask_inds, terrain = terrain_util.compute_hf_extra_vals(motion_frames=opt_frames,
                                                                           terrain=terrain,
                                                                           char_model=opt_char_model_cpu,
                                                                           char_body_points=body_points_cpu)

                file_io_helper.save_ms_file(filepath=opt_save_path,
                                            motion_frames=opt_frames,
                                            fps=mdm_model._sequence_fps,
                                            loop_mode="CLAMP",
                                            terrain=terrain,
                                            hf_mask_inds=hf_mask_inds,
                                            body_constraints=body_constraints,
                                            min_point_offset=min_point_offset,
                                            path_nodes=path_nodes,
                                            extra_misc_data=extra_misc_data
                                            )
                
                
                # also save flipped version
                opt_flipped_save_path = opt_flipped_output_folder_path / (new_motion_name + "_opt_flipped.pkl")
                flipped_motion_frames = medit_lib.flip_motion_about_XZ_plane(motion_frames=opt_frames, char_model=opt_char_model_cpu)
                
                flipped_terrain = terrain.torch_copy()
                flipped_terrain.flip_by_XZ_axis()
                for i in range(len(hf_mask_inds)):
                    curr_inds = hf_mask_inds[i]
                    # only need to flip along y dim
                    curr_inds[:, 1] = flipped_terrain.hf.shape[1] - 1 - curr_inds[:, 1]

                file_io_helper.save_ms_file(filepath=opt_flipped_save_path,
                                            motion_frames=flipped_motion_frames,
                                            fps=mdm_model._sequence_fps,
                                            loop_mode="CLAMP",
                                            terrain=flipped_terrain,
                                            hf_mask_inds=hf_mask_inds,
                                            min_point_offset=min_point_offset,
                                            path_nodes=path_nodes,
                                            extra_misc_data=extra_misc_data
                                            )
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    end_time = time.time()
    print("Time to generate all motions =", end_time - first_start_time, "seconds.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PARC stage 2: generate kinematic motions with the MDM.")
    parser.add_argument("--config", default="data/configs/parc_2_kin_gen_default.yaml",
                        help="kin gen config yaml (default: %(default)s)")
    cfg_path = parser.parse_args().config
    print("loading mdm procgen config from", cfg_path)

    try:
        config = path_loader.load_config(cfg_path)
    except (IOError, AssertionError) as exc:
        print("error opening file:", cfg_path)
        print(exc)
        sys.exit(1)

    mdm_procgen(config)
