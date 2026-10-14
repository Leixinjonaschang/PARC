import os
import random
import sys
import yaml
from parc.util import torch_util
from parc.anim import kin_char_model
from parc.util import motion_edit_lib as medit_lib
from parc.util import geom_util
from parc.motion_synthesis.motion_opt import motion_optimization as moopt
import math
import time

def rand_float(f_min, f_max):
    return random.random() * (f_max - f_min) + f_min

def fetch_motion_files(motion_file):
    ext = os.path.splitext(motion_file)[1]
    if (ext == ".yaml"):
        motion_files = []
        motion_weights = []

        with open(motion_file, 'r') as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        motion_list = motion_config['motions']
        for motion_entry in motion_list:
            curr_file = motion_entry['file']
            curr_weight = motion_entry['weight']
            assert(curr_weight >= 0)

            motion_weights.append(curr_weight)
            motion_files.append(curr_file)
    else:
        motion_files = [motion_file]
        motion_weights = [1.0]

    return motion_files, motion_weights

if __name__ == "__main__":
    if len(sys.argv) == 3:
        assert sys.argv[1] == "--config"
        cfg_path = sys.argv[2]
        print("loading config from", cfg_path)
    else:
        cfg_path = "parc/motion_synthesis/motion_opt/config/motion_opt.yaml"

    try:
        with open(cfg_path, "r") as stream:
            cfg = yaml.safe_load(stream)
    except IOError:
        print("error opening file:", cfg_path)
        exit()

    motions_yaml_path = cfg["motions_yaml_path"]
    motion_filepaths, motion_weights = fetch_motion_files(motions_yaml_path)   
    
    motion_datas = []
    for filepath in motion_filepaths:
        motion_datas.append(medit_lib.load_motion_file(filepath))
    
    num_motions = len(motion_datas)

    device = cfg["device"]
    char_model_path = cfg["char_model"]
    output_folder_path = cfg["output_folder_path"]
    log_folder_path = output_folder_path + "log/"
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(log_folder_path, exist_ok=True)
    num_iters = cfg["num_iters"]
    step_size = cfg["step_size"]
    w_root_pos = cfg["w_root_pos"]
    w_root_rot = cfg["w_root_rot"]
    w_joint_rot = cfg["w_joint_rot"]
    w_smoothness = cfg["w_smoothness"]
    w_penetration = cfg["w_penetration"]
    w_contact = cfg["w_contact"]
    w_sliding = cfg["w_sliding"]
    w_body_constraints = cfg["w_body_constraints"]
    w_jerk = cfg["w_jerk"]
    max_jerk = cfg["max_jerk"]
    use_wandb = cfg["use_wandb"]

    auto_compute_body_constraints = cfg["auto_compute_body_constraints"]

    frame_stride = cfg["frame_stride"]

    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(char_model_path)

    char_point_sample_cfg = cfg["char_point_samples"]
    sphere_num_subdivisions = char_point_sample_cfg["sphere_num_subdivisions"]
    box_num_slices = char_point_sample_cfg["box_num_slices"]
    box_dim_x = char_point_sample_cfg["box_dim_x"]
    box_dim_y = char_point_sample_cfg["box_dim_y"]
    capsule_num_circle_points = char_point_sample_cfg["capsule_num_circle_points"]
    capsule_num_sphere_subdivisions = char_point_sample_cfg["capsule_num_sphere_subdivisions"]
    capsule_num_cylinder_slices = char_point_sample_cfg["capsule_num_cylinder_slices"]
    body_points = geom_util.get_char_point_samples(char_model,
                                                   sphere_num_subdivisions=sphere_num_subdivisions,
                                                   box_num_slices=box_num_slices,
                                                   box_dim_x=box_dim_x,
                                                   box_dim_y=box_dim_y,
                                                   capsule_num_circle_points=capsule_num_circle_points,
                                                   capsule_num_sphere_subdivisons=capsule_num_sphere_subdivisions,
                                                   capsule_num_cylinder_slices=capsule_num_cylinder_slices)

    start_time = time.time()
    for _ in range(num_motions):
        # change selection to be multinomial, based on length of each motion
        #file_index = random.randint(0, num_initial_motions-1)
        file_index = _
        input_motion_name = os.path.basename(os.path.splitext(motion_filepaths[file_index])[0])
        print("OPTIMIZING MOTION:", input_motion_name)
        print("MOTION", str(_) + "/"+str(num_motions))

        motion_data = motion_datas[file_index]
        terrain = motion_data.get_terrain().torch_copy()
        terrain.set_device(device)
        src_frames = motion_data.get_frames().clone().to(device=device)
        contacts = motion_data.get_contacts().clone().to(device=device)


        if auto_compute_body_constraints:
            print("Computing approx body constraints...")
            root_pos = src_frames[..., 0:3]
            root_rot = torch_util.exp_map_to_quat(src_frames[..., 3:6])
            joint_rot = char_model.dof_to_rot(src_frames[..., 6:])
            body_constraints = moopt.compute_approx_body_constraints(root_pos=root_pos,
                                                                     root_rot = root_rot,
                                                                     joint_rot=joint_rot,
                                                                     contacts=contacts,
                                                                     char_model=char_model,
                                                                     terrain=terrain)
            motion_data.set_opt_body_constraints(body_constraints)
            print("Finished computing approx body constraints.")

        if motion_data.has_opt_body_constraints():
            body_constraints = motion_data.get_opt_body_constraints()
            for b in range(len(body_constraints)):
                for constraint in body_constraints[b]:
                    constraint.start_frame_idx = int(math.ceil(constraint.start_frame_idx / frame_stride))
                    constraint.end_frame_idx = int(math.floor(constraint.end_frame_idx // frame_stride))
                    constraint.constraint_point = constraint.constraint_point.to(device=device)
        else:
            body_constraints = None

        src_frames = src_frames[::frame_stride]
        contacts = contacts[::frame_stride]
        output_motion_name = input_motion_name + "_opt"

        log_file = log_folder_path + "log_" + output_motion_name + ".txt"

        opt_frames = moopt.motion_contact_optimization(src_frames=src_frames, 
                                    contacts=contacts,
                                    body_points=body_points,
                                    terrain=terrain,
                                    char_model=char_model,
                                    num_iters=num_iters,
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
                                    max_jerk=max_jerk,
                                    body_constraints=body_constraints,
                                    exp_name=output_motion_name,
                                    use_wandb=use_wandb,
                                    log_file=log_file)
        
        output_filepath = output_folder_path + output_motion_name + ".pkl"
        

        new_motion_data = {
            'fps': motion_data.get_fps() // frame_stride,
            'loop_mode': "CLAMP",
            'frames': opt_frames,
            'contacts': contacts,
            'terrain': terrain
        }
        if body_constraints is not None:
            new_motion_data["opt:body_constraints"] = body_constraints
        
        new_motion_data = medit_lib.MotionData(new_motion_data)
        new_motion_data.save_to_file(output_filepath)

    end_time = time.time()

    print("Total optimization time for", num_motions, "motions:", end_time - start_time, "seconds.")