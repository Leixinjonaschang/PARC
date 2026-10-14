import os
import random
import torch
import yaml
from parc.anim import kin_char_model
from parc.util import motion_edit_lib as medit_lib
from parc.util import geom_util
from parc.util import terrain_util
from parc.util import torch_util
from parc.motion_synthesis.motion_opt import motion_optimization as moopt
import enum

"""
Script used to augment initial dataset with different terrain heights.
Also randomly rotates/squishes/stretches motions for a bit more spatial variation.
Motions have to then be post-processed with kinematic optimization, and then deepmimic for best results.
"""

class TerrainAugmentationType(enum.Enum):
    HEIGHT_SCALE = 0
    BOXES_ALONG_PATH = 1
    NONE = 2

def rand_float(f_min, f_max):
    return random.random() * (f_max - f_min) + f_min

if __name__ == "__main__":
    if len(sys.argv) == 3:
        assert sys.argv[1] == "--config"
        cfg_path = sys.argv[2]
        print("loading config from", cfg_path)
    else:
        cfg_path = "parc/motion_synthesis/motion_opt/config/platform_aug.yaml"

    try:
        with open(cfg_path, "r") as stream:
            cfg = yaml.safe_load(stream)
    except IOError:
        print("error opening file:", cfg_path)
        exit()

    motions_yaml_path = cfg["motions_yaml_path"]
    try:
        with open(motions_yaml_path, "r") as stream:
            motions = yaml.safe_load(stream)
            motions = motions["motions"]
    except IOError:
        print("error opening file:", motions_yaml_path)
        exit()       
    
    motion_datas = []
    motion_filepaths = []
    for elem in motions:
        motion_datas.append(medit_lib.load_motion_file(elem["file"]))
        motion_filepaths.append(elem["file"])
    

    num_initial_motions = len(motion_datas)
    num_variations_per_motion = [0] * num_initial_motions

    device = cfg["device"]
    char_model_path = cfg["char_model"]
    num_new_motions = cfg["num_new_motions"]
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
    use_wandb = cfg["use_wandb"]
    max_motion_len = cfg["max_motion_len"]
    max_heading_angle = cfg["max_heading_angle"]
    min_heading_angle = cfg["min_heading_angle"]
    terrain_padding_size = cfg["terrain_padding_size"]
    slice_terrain = cfg["slice_terrain"]

    x_scale = cfg["x_scale"]
    y_scale = cfg["y_scale"]
    sample_weight_by_length = cfg["sample_weight_by_length"]

    terrain_aug_type = TerrainAugmentationType[cfg["terrain_aug_type"]]

    motion_lengths = []
    for elem in motion_datas:
        curr_motion_length = elem.get_frames().shape[0] / elem.get_fps()
        motion_lengths.append(curr_motion_length)
    motion_lengths = torch.tensor(motion_lengths, dtype=torch.float32, device=device)

    if terrain_aug_type == TerrainAugmentationType.BOXES_ALONG_PATH:
        min_len = cfg["min_len"]
        max_len = cfg["max_len"]
        min_angle = cfg["min_angle"]
        max_angle = cfg["max_angle"]
        min_num_boxes = cfg["min_num_boxes"]
        max_num_boxes = cfg["max_num_boxes"]
        min_h = cfg["min_h"]
        max_h = cfg["max_h"]
    elif terrain_aug_type == TerrainAugmentationType.HEIGHT_SCALE:
        min_h_scale = cfg["min_h_scale"]
        max_h_scale = cfg["max_h_scale"]
        bad_h_range = cfg["bad_h_range"]

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
    if sample_weight_by_length:
        sample_weights = motion_lengths
    else:
        sample_weights = torch.ones_like(motion_lengths)
    file_indices = torch.multinomial(sample_weights, num_samples = num_new_motions, replacement=True)

    for _ in range(num_new_motions):
        # change selection to be multinomial, based on length of each motion
        #file_index = random.randint(0, num_initial_motions-1)
        file_index = file_indices[_]
        input_motion_name = os.path.basename(os.path.splitext(motion_filepaths[file_index])[0])
        print("AUGMENTING MOTION:", input_motion_name)
        print("MOTION", str(_) + "/"+str(num_new_motions))
        num_variations_per_motion[file_index] += 1

        motion_data = motion_datas[file_index]
        terrain = motion_data.get_terrain().torch_copy()
        terrain.set_device(device=device)
        src_frames = motion_data.get_frames().clone().to(device=device)
        contacts = motion_data.get_contacts().clone().to(device=device)

        fps = motion_data.get_fps()

        max_num_frames = int(round(fps * max_motion_len))
        src_num_frames = src_frames.shape[0]
        if src_num_frames > max_num_frames:
            # pick a random starting frame then slice motion and terrain
            start_frame_idx = random.randint(0, src_num_frames - max_num_frames - 1)
            src_frames = src_frames[start_frame_idx:start_frame_idx+max_num_frames]
            contacts = contacts[start_frame_idx:start_frame_idx+max_num_frames]
            
        # randomly rotate around the origin
        heading_angle = torch.tensor([rand_float(min_heading_angle, max_heading_angle)], dtype=torch.float32, device=device)
        heading_angle = heading_angle * torch.pi / 180.0
        src_frames[..., 0:2] = torch_util.rotate_2d_vec(src_frames[..., 0:2], heading_angle)
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        delta_heading_quat = torch_util.axis_angle_to_quat(z_axis, heading_angle)
        new_heading_quat = torch_util.quat_multiply(delta_heading_quat, torch_util.exp_map_to_quat(src_frames[..., 3:6]))
        src_frames[..., 3:6] = torch_util.quat_to_exp_map(new_heading_quat)

        # also randomly stretch/shrink motions along a random direction
        src_frames[..., 0] *= rand_float(x_scale[0], x_scale[1])
        src_frames[..., 1] *= rand_float(y_scale[0], y_scale[1])

        if terrain_padding_size > 0:
            terrain.pad(padding_size=terrain_padding_size)
        if slice_terrain:
            terrain, src_frames = terrain_util.slice_terrain_around_motion(src_frames, terrain, 
                                                                               padding = terrain.dxdy[0].item()*2)   

        # randomly change height
        if terrain_aug_type == TerrainAugmentationType.HEIGHT_SCALE:
            height_scale = rand_float(min_h_scale, max_h_scale)
            while bad_h_range[0] < height_scale and height_scale < bad_h_range[1]:
                height_scale = rand_float(min_h_scale, max_h_scale)
            terrain.hf = height_scale * terrain.hf
        elif terrain_aug_type == TerrainAugmentationType.BOXES_ALONG_PATH:
            num_boxes = random.randint(min_num_boxes, max_num_boxes)
            frame_inds = torch.randint(0, src_frames.shape[0],
                                    size=[num_boxes], device=device,
                                    dtype=torch.int64)
            print(src_frames.device, terrain.min_point.device, frame_inds.device)
            box_centers = src_frames[frame_inds, 0:2] - terrain.min_point
            box_centers = box_centers / terrain.dxdy
            terrain_util.add_boxes_to_hf_at_xy_points(
                box_centers=box_centers,
                hf=terrain.hf,
                min_h=min_h,
                max_h=max_h,
                min_len=min_len,
                max_len=max_len,
                min_angle=min_angle,
                max_angle=max_angle
            )

        output_motion_name = input_motion_name + "_aug" + str(num_variations_per_motion[file_index]).zfill(3)

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
                                    w_body_constraints=0.0,
                                    body_constraints=None,
                                    exp_name=output_motion_name,
                                    use_wandb=use_wandb,
                                    log_file=log_file)
        
        output_filepath = output_folder_path + output_motion_name + ".pkl"
        

        new_motion_data = {
            'fps': motion_data.get_fps(),
            'loop_mode': "CLAMP",
            'frames': opt_frames,
            'contacts': contacts,
            'terrain': terrain
        }
        new_motion_data = medit_lib.MotionData(new_motion_data)
        new_motion_data.save_to_file(output_filepath)