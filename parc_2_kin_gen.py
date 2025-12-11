import sys
import os
import numpy as np
import torch
import pickle
import yaml
import enum
import time
import re
from pathlib import Path
import util.torch_util as torch_util
import anim.kin_char_model as kin_char_model
import util.geom_util as geom_util
import util.terrain_util as terrain_util

import diffusion.mdm as mdm
import diffusion.gen_util as gen_util
import util.motion_util as motion_util
from util.motion_util import MotionFrames

import tools.procgen.astar as astar
import tools.procgen.mdm_path as mdm_path
import tools.motion_opt.motion_optimization as motion_opt
import zmotion_editing_tools.motion_edit_lib as medit_lib

cpu_device = "cpu"
cuda_device = "cuda:0"


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

def load_mdm(mdm_path: Path) -> mdm.MDM:

    if not mdm_path.is_file():
        assert mdm_path.is_dir()

        # get checkpoint with biggest number
        number_file_pairs = []

        for file in mdm_path.iterdir():
            if file.is_file() and file.suffix == ".pkl":
                match = re.search(r'\d+', file.name)
                if match:
                    number = int(match.group())
                    number_file_pairs.append((number, file))

        if not number_file_pairs:
            return None

        # Return the file with the largest number (latest checkpoint)
        mdm_path = max(number_file_pairs, key=lambda x: x[0])[1]

    print("loading path:", mdm_path)
    ret_mdm = pickle.load(mdm_path.open("rb"))
    if ret_mdm.use_ema:
        print('Using EMA model...')
        ret_mdm._denoise_model = ret_mdm._ema_denoise_model
    ret_mdm.update_old_mdm()
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
    dx = config["dx"]
    dy = config["dy"]
    procgen_mode = ProcGenMode[config["procgen_mode"]]
    simplify_terrain = config["simplify_terrain"]
    save_name = config.get("save_name", None)

    if procgen_mode == ProcGenMode.FILE:
        input_terrain_path = config["input_terrain_path"]

        input_terrains = []
        input_paths = []

        if os.path.splitext(input_terrain_path)[1] == ".pkl":
            with open(input_terrain_path, "rb") as f:
                input_terrains.append(pickle.load(f)["terrain"])
                input_terrains[0].to_torch(cpu_device)
        elif os.path.splitext(input_terrain_path)[1] == ".yaml":
            with open(input_terrain_path, "r") as f:
                input_terrains_yaml = yaml.safe_load(f)
                for curr_terrain_path in input_terrains_yaml["terrains"]:
                    with open(curr_terrain_path, "rb") as f2:
                        terrain_data = pickle.load(f2)
                        input_terrains.append(terrain_data["terrain"])
                        input_terrains[-1].to_torch(cpu_device)
                        input_paths.append(terrain_data["path_nodes"])
                        input_paths[-1] = input_paths[-1].to(device=cpu_device)
        else:
            assert False

        num_input_terrains = len(input_terrains)

    astar_settings = astar.AStarSettings()
    astar_config = config["astar"]
    for key in astar_config:
        setattr(astar_settings, key, astar_config[key])

    mdm_path_settings = mdm_path.MDMPathSettings()
    mdm_path_config = config["mdm_path"]
    for key in mdm_path_config:
        setattr(mdm_path_settings, key, mdm_path_config[key])

    mdm_gen_settings = gen_util.MDMGenSettings()
    mdm_gen_config = config["mdm_gen"]
    for key in mdm_gen_config:
        setattr(mdm_gen_settings, key, mdm_gen_config[key])

    boxes_settings = ProcGenBoxesSettings()
    boxes_config = config["boxes"]
    for key in boxes_config:
        setattr(boxes_settings, key, boxes_config[key])

    paths_settings = ProcGenPathsSettings()
    paths_config = config["paths"]
    for key in paths_config:
        setattr(paths_settings, key, paths_config[key])

    stairs_settings = ProcGenStairsSettings()
    stairs_config = config["stairs"]
    for key in stairs_config:
        setattr(stairs_settings, key, stairs_config[key])

    only_gen = config["only_gen"]

    # Get device for MDM model generation (from config or use default)
    mdm_device = config.get("device", cuda_device)
    print(f"Using device for MDM generation: {mdm_device}")

    # We can also try using mdm models from different arbitrary checkpoints
    if input_mdm_model is None:
        mdm_model_path = Path(config["mdm_model_path"])
        mdm_model = load_mdm(mdm_model_path)
    else:
        mdm_model = input_mdm_model
    # Move MDM model to specified device if needed
    if mdm_model is not None and str(mdm_model._device) != str(mdm_device):
        mdm_model.set_device(mdm_device)

    char_model = mdm_model._kin_char_model.get_copy(cpu_device)

    output_folder = config["output_dir"]
    if output_folder[-1] != "/":
        output_folder = output_folder + "/"
    os.makedirs(output_folder, exist_ok=True)

    print("Output folder:", output_folder)

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
    body_points = geom_util.get_char_point_samples(opt_char_model,
                                                   sphere_num_subdivisions=sphere_num_subdivisions,
                                                   box_num_slices=box_num_slices,
                                                   box_dim_x=box_dim_x,
                                                   box_dim_y=box_dim_y,
                                                   capsule_num_circle_points=capsule_num_circle_points,
                                                   capsule_num_sphere_subdivisons=capsule_num_sphere_subdivisions,
                                                   capsule_num_cylinder_slices=capsule_num_cylinder_slices)


    first_start_time = time.time()

    for motion_num in range(num_new_motions):
        motion_num = motion_num + motion_id_offset

        def gen_motion_and_terrain(only_gen=False):
            start_time = time.time()

            path_nodes_3d = False
            bug_counter = 0
            while path_nodes_3d is False and bug_counter < 1000:
                terrain = terrain_util.SubTerrain(
                    "terrain", 
                    x_dim = new_terrain_dim_x,
                    y_dim = new_terrain_dim_y,
                    dx = dx,
                    dy = dx,
                    min_x = 0.0,
                    min_y = 0.0,
                    device = cpu_device)

                slice_terrain = True
                
                if procgen_mode == ProcGenMode.BOXES:
                    terrain_util.add_boxes_to_hf2(terrain.hf,
                                                box_max_height = boxes_settings.max_box_h,
                                                box_min_height = boxes_settings.min_box_h,
                                                hf_maxmin = None,
                                                num_boxes = boxes_settings.num_boxes, 
                                                box_max_len = boxes_settings.box_max_len, 
                                                box_min_len = boxes_settings.box_min_len,
                                                max_angle = boxes_settings.max_box_angle,
                                                min_angle = boxes_settings.min_box_angle)
                    
                elif procgen_mode == ProcGenMode.PATHS:
                    terrain_util.gen_paths_hf(terrain,
                                            num_paths = paths_settings.num_terrain_paths, 
                                            maxpool_size = paths_settings.maxpool_size,
                                            floor_height = paths_settings.floor_height,
                                            path_min_height = paths_settings.path_min_height, 
                                            path_max_height = paths_settings.path_max_height)
                    
                elif procgen_mode == ProcGenMode.STAIRS:
                    terrain_util.add_stairs_to_hf(terrain,
                                                min_stair_start_height = stairs_settings.min_stair_start_height,
                                                max_stair_start_height = stairs_settings.max_stair_start_height,
                                                min_step_height = stairs_settings.min_step_height,
                                                max_step_height = stairs_settings.max_step_height,
                                                num_stairs = stairs_settings.num_stairs,
                                                min_stair_thickness = stairs_settings.min_stair_thickness,
                                                max_stair_thickness = stairs_settings.max_stair_thickness)

                elif procgen_mode == ProcGenMode.FILE:
                    # take a random 16x16 slice of the input terrain
                    input_terrain = input_terrains[motion_num % num_input_terrains]
                    start_dim_x = np.random.randint(0, input_terrain.dims[0].item() + 1 - new_terrain_dim_x)
                    start_dim_y = np.random.randint(0, input_terrain.dims[1].item() + 1 - new_terrain_dim_y)

                    terrain.hf[:, :] = input_terrain.hf[start_dim_x:start_dim_x + new_terrain_dim_x, start_dim_y:start_dim_y + new_terrain_dim_y].clone()

                    min_point_offset = input_terrain.get_point(torch.tensor([start_dim_x, start_dim_y], dtype=torch.int64, device=cpu_device))
                    
                    slice_terrain = False
                hf_orig = terrain.hf.clone()

                bug_counter2 = 0

                #if len(input_paths) > 0:
                #    path_nodes_3d = input_paths[motion_num % num_input_terrains].clone()

                while path_nodes_3d is False and bug_counter2 < 10:
                    start_node, end_node = astar.pick_random_start_end_nodes_on_edges(terrain,
                                                                                    min_dist=astar_settings.min_start_end_xy_dist)
                    
                    terrain.hf = hf_orig.clone()
                    
                    if simplify_terrain:
                        terrain_util.flat_maxpool_2x2(terrain=terrain)

                        terrain_util.flatten_4x4_near_edge(terrain=terrain,
                                                        grid_ind=start_node,
                                                        height=terrain.hf[start_node[0], start_node[1]].item())

                        terrain_util.flatten_4x4_near_edge(terrain=terrain,
                                                        grid_ind=end_node,
                                                        height=terrain.hf[end_node[0], end_node[1]].item())


                    path_nodes_3d = astar.run_a_star_on_start_end_nodes(
                        terrain = terrain, 
                        start_node = start_node, 
                        end_node = end_node, 
                        settings = astar_settings)
                    
                    bug_counter2 += 1
                
                bug_counter += 1

            if path_nodes_3d is False:
                assert False, "something wrong with procgen"

            if only_gen:
                return terrain, path_nodes_3d
            
            best_motion_frames, best_motion_terrains, info = mdm_path.generate_frames_until_end_of_path(
                path_nodes = path_nodes_3d,
                terrain = terrain,
                char_model = char_model,
                mdm_model = mdm_model,
                prev_frames = None, # TODO: sample from dataset? generate?
                mdm_path_settings=mdm_path_settings,
                mdm_gen_settings=mdm_gen_settings,
                add_noise_to_loss=False,
                verbose=True, 
                slice_terrain=slice_terrain)
            
            end_time = time.time()
            print("Time to generate motion", motion_num, "=", end_time - start_time, "seconds.")

            if procgen_mode == ProcGenMode.FILE:
                info["min_point_offset"] = min_point_offset

            return best_motion_frames, best_motion_terrains, info


        if only_gen:
            terrain, path_nodes = gen_motion_and_terrain(only_gen=True)
            # TODO: save terrain and path nodes to file
            save_data = dict()
            save_data["terrain"] = terrain
            save_data["path_nodes"] = path_nodes

            if save_name is None:
                new_motion_name = procgen_mode.name + "_" + str(motion_num)
            else:
                new_motion_name = save_name + "_" + str(motion_num)

            save_path = output_folder + new_motion_name + ".pkl"
            with open(save_path, "wb") as f:
                pickle.dump(save_data, f)
                print("Saved to:", save_data)
            continue

        attempt_counter = 0
        while True:
            best_motion_frames, best_motion_terrains, info = gen_motion_and_terrain()

            all_losses = info["losses"]
            contact_losses = info["contact_losses"]
            pen_losses = info["pen_losses"]

            valid_indices = (contact_losses < max_contact_loss) & (pen_losses < max_pen_loss) & (all_losses < max_total_loss)

            valid_indices = torch.nonzero(valid_indices).squeeze()
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
            motion_frames.set_device(cpu_device)
            mlib_motion_frames, mlib_contact_frames = motion_frames.get_mlib_format(char_model)

            # mlib_motion_frames = torch.cat([test_motion_frames[i][:-2], mlib_motion_frames.squeeze(0)], dim=0)
            # mlib_contact_frames = torch.cat([test_contact_frames[i][:-2], mlib_contact_frames.squeeze(0)], dim=0)
            mlib_motion_frames = mlib_motion_frames.squeeze(0)
            mlib_contact_frames = mlib_contact_frames.squeeze(0)

            if save_name is None:
                new_motion_name = procgen_mode.name + "_" + str(motion_num) + "_"+str(j)
            else:
                new_motion_name = save_name + "_" + str(motion_num) + "_"+str(j)
            save_path = output_folder + new_motion_name + ".pkl"
            kwargs = dict()
            kwargs["loss"] = all_losses[j].item()
            if procgen_mode == ProcGenMode.FILE:
                kwargs["min_point_offset"] = info["min_point_offset"]
            medit_lib.save_motion_data(motion_filepath = save_path,
                                       motion_frames = mlib_motion_frames,
                                       contact_frames = mlib_contact_frames,
                                       terrain = best_motion_terrains[j],
                                       fps = mdm_model._sequence_fps,
                                       loop_mode = "CLAMP",
                                       **kwargs)

            if use_opt:
                log_file = opt_log_folder_path / ("log_" + new_motion_name + ".txt")

                terrain = best_motion_terrains[j]
                src_frames = mlib_motion_frames.to(device=opt_device)
                contacts = mlib_contact_frames.to(device=opt_device)

                if auto_compute_body_constraints:
                    print("Computing approx body constraints...")
                    root_pos = src_frames[..., 0:3]
                    root_rot = torch_util.exp_map_to_quat(src_frames[..., 3:6])
                    joint_rot = opt_char_model.dof_to_rot(src_frames[..., 6:])
                    body_constraints = motion_opt.compute_approx_body_constraints(root_pos=root_pos,
                                                                            root_rot = root_rot,
                                                                            joint_rot=joint_rot,
                                                                            contacts=contacts,
                                                                            char_model=opt_char_model,
                                                                            terrain=terrain)
                    print("Finished computing approx body constraints.")
                    kwargs["opt:body_constraints"] = body_constraints
                else:
                    body_constraints = None

                opt_frames = motion_opt.motion_contact_optimization(src_frames=src_frames,
                                                    contacts=contacts,
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
                                                    log_file=log_file,
                                                    )
                
                if remove_hesitation:
                    opt_frames, mlib_contact_frames = medit_lib.remove_hesitation_frames(
                        opt_frames.cpu(), 
                        mlib_contact_frames,
                        char_model)
                    opt_frames = opt_frames.to(device=opt_device)

                opt_save_path = opt_output_folder_path / (new_motion_name + "_opt.pkl")

                hf_mask_inds = terrain_util.compute_hf_extra_vals(motion_frames=opt_frames, 
                                                                  terrain=terrain,
                                                                  char_model=opt_char_model,
                                                                  char_body_points=body_points)
                
                kwargs["hf_mask_inds"] = hf_mask_inds
                
                medit_lib.save_motion_data(motion_filepath = opt_save_path,
                                        motion_frames = opt_frames,
                                        contact_frames = mlib_contact_frames,
                                        terrain = terrain,
                                        fps = mdm_model._sequence_fps,
                                        loop_mode = "CLAMP",
                                        **kwargs)
                
                
                # also save flipped version
                opt_flipped_save_path = opt_flipped_output_folder_path / (new_motion_name + "_opt_flipped.pkl")
                flip_motion_frames, flip_contact_frames = medit_lib.flip_motion_about_XZ_plane(motion_frames=opt_frames,
                                                                                               char_model=opt_char_model,
                                                                                               contact_frames=mlib_contact_frames)
                
                flipped_terrain = terrain.torch_copy()
                flipped_terrain.flip_by_XZ_axis()
                for i in range(len(hf_mask_inds)):
                    curr_inds = hf_mask_inds[i]
                    # only need to flip along y dim
                    curr_inds[:, 1] = flipped_terrain.hf.shape[1] - 1 - curr_inds[:, 1]

                if "opt:body_constraints" in kwargs:
                    del kwargs["opt:body_constraints"]

                

                medit_lib.save_motion_data(motion_filepath=opt_flipped_save_path,
                                           motion_frames=flip_motion_frames,
                                           contact_frames=flip_contact_frames,
                                           terrain=flipped_terrain,
                                           fps = mdm_model._sequence_fps,
                                           loop_mode = "CLAMP",
                                           **kwargs)
            
        torch.cuda.empty_cache()
    
    end_time = time.time()
    print("Time to generate all motions =", end_time - first_start_time, "seconds.")
    return

if __name__ == "__main__":
    if len(sys.argv) == 3:
        assert sys.argv[1] == "--config"
        cfg_path = sys.argv[2]
        print("loading mdm procgen config from", cfg_path)
    else:
        cfg_path = "tools/procgen/mdm_procgen.yaml"
    
    try:
        with open(cfg_path, "r") as stream:
            config = yaml.safe_load(stream)
    except IOError:
        print("error opening file:", cfg_path)
        exit()

    mdm_procgen(config)