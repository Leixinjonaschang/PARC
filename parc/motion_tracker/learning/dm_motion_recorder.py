import copy
import os
import re
import shutil

import numpy as np
import torch

import parc.util.torch_util as torch_util
from parc.motion_tracker.learning.base_agent import AgentMode


def organize_recorded_dm_motions(source_dir, verbose=True):

    # Regex to extract [STRING], [NUMBER] from filenames
    pattern = re.compile(r"^(.*?)_(\d+)_.*\.pkl$")

    # Loop through all files in the directory
    for filename in os.listdir(source_dir):
        match = pattern.match(filename)
        if match:
            base_string, number_str = match.groups()
            number = int(number_str)

            # Compute bin range (e.g., 100–199)
            bin_start = (number // 50) * 50
            bin_end = bin_start + 50
            folder_name = f"{base_string}_{bin_start}_{bin_end-1}"

            # Create destination folder
            dest_dir = os.path.join(source_dir, folder_name)
            os.makedirs(dest_dir, exist_ok=True)

            # Move the file
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.move(src_path, dest_path)

            if verbose:
                print(f"Moved {filename} → {folder_name}/")
    return

def record_dm_motions(agent):
    agent.eval()
    agent.set_mode(AgentMode.TEST)

    env = agent.get_env()

    if not hasattr(env, "get_dm_env"):
        raise AttributeError("record_dm_motions requires an agent with a DM environment")

    env.set_rand_reset(False)
    env.set_demo_mode(True)
    env.set_rand_root_pos_offset_scale(0.0)
    env._episode_length = 1000.0

    record_obs = True

    def record_motions_helper(name_suffix: str, prev_successful_motions=None):
        agent._curr_obs, agent._curr_info = env.reset()

        env.build_agent_states_dict(name_suffix, record_obs=record_obs)
        env.write_agent_states()

        if prev_successful_motions is not None:
            for env_id in range(len(prev_successful_motions)):
                env.set_writing_env_state(env_id, not prev_successful_motions[env_id])

        while True:
            action, action_info = agent._decide_action(agent._curr_obs, agent._curr_info)

            next_obs, r, done, next_info = agent._step_env(action)

            agent._curr_obs, agent._curr_info = agent._reset_done_envs(done)

            if not env.is_writing_agent_states():
                print("done writing agent states")
                break

    record_motions_helper(name_suffix="_dm")

    successful_motions = copy.deepcopy(env.get_env_success_states())

    possible_start_time_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

    num_successful_motions = []
    num_successful_motions.append(sum(successful_motions))

    counter = 0
    num_envs = agent.get_num_envs()
    start_time_fraction_per_motion_id = [None] * num_envs
    for motion_id in range(num_envs):
        if successful_motions[motion_id]:
            start_time_fraction_per_motion_id[motion_id] = 0.0

    while not all(successful_motions) and counter < len(possible_start_time_fractions):
        start_time_fraction = possible_start_time_fractions[counter]

        for motion_id in range(num_envs):
            motion_length = (1.0 - start_time_fraction) * env.get_dm_env()._motion_lib.get_motion_length(motion_id).item()
            if motion_length < 2.0:
                successful_motions[motion_id] = True
        start_time_fraction_tensor = start_time_fraction * torch.ones(size=[num_envs], dtype=torch.float32, device=agent._device)
        env.get_dm_env().set_motion_start_time_fraction(start_time_fraction_tensor)
        record_motions_helper(name_suffix="_dm", prev_successful_motions=successful_motions)

        counter += 1

        new_successful_motions = copy.deepcopy(env.get_env_success_states())

        num_successful_motions.append(sum(new_successful_motions))
        for motion_id in range(num_envs):
            successful_motions[motion_id] = successful_motions[motion_id] or new_successful_motions[motion_id]

            if new_successful_motions[motion_id]:
                start_time_fraction_per_motion_id[motion_id] = start_time_fraction

    print("Organizing motions...")
    organize_recorded_dm_motions(env._output_motion_dir, verbose=False)
    print("Finished organizing motions.")

    print("Successful motions at 0 percent start time:", num_successful_motions[0])
    print("Success rate:", num_successful_motions[0] / len(successful_motions))

    print("len num_successful_motions:", len(num_successful_motions))

    for i in range(len(num_successful_motions) - 1):
        print(i)
        start_time_fraction = possible_start_time_fractions[i]
        print("Successful motions at", start_time_fraction, "percent start time:", num_successful_motions[i + 1])

    print("Total successfull motions:", sum(num_successful_motions))
    print("Total success rate:", sum(num_successful_motions) / len(successful_motions))

    exit()

    mlib = env.get_dm_env()._motion_lib
    motion_save_folder_path = "output/_motions/successful_ref_motions/"
    os.makedirs(motion_save_folder_path, exist_ok=True)
    for motion_id in range(num_envs):
        motion_length = mlib._motion_lengths[motion_id].item()
        start_time_fraction = start_time_fraction_per_motion_id[motion_id]
        if start_time_fraction is None:
            continue
        start_time = start_time_fraction * motion_length
        fps = mlib._motion_fps[motion_id].item()
        start_frame = int(np.floor(start_time * fps))

        motion_start_idx = mlib._motion_start_idx[motion_id].item()
        motion_end_idx = motion_start_idx + mlib._motion_num_frames[motion_id].item()
        motion_frames = mlib._motion_frames[motion_start_idx + start_frame:motion_end_idx]
        contact_frames = mlib._frame_contacts[motion_start_idx + start_frame:motion_end_idx]
        terrain = mlib._terrains[motion_id]

        import parc.util.motion_edit_lib as medit_lib
        motion_og_name = mlib.get_motion_names()[motion_id]
        motion_save_path = motion_save_folder_path + motion_og_name + "_success_ref.pkl"
        medit_lib.save_motion_data(motion_filepath=motion_save_path,
                                   motion_frames=motion_frames,
                                   contact_frames=contact_frames,
                                   terrain=terrain,
                                   fps=mlib._motion_fps[motion_id].item(),
                                   loop_mode="CLAMP")

    exit()

    num_augments_per_motion = 2
    radius = 0.35
    max_heading_angle_degrees = 35.0
    max_start_time_fraction = 0.5
    min_start_time_fraction = 0.0
    num_envs = env.get_num_envs()
    for i in range(num_augments_per_motion):

        root_pos_offset = 2.0 * torch.rand(size=[num_envs, 3], dtype=torch.float32, device=agent._device) - 1.0
        root_pos_offset *= radius
        root_pos_offset[..., 2] = 0.0

        root_heading_offset = 2.0 * torch.rand(size=[num_envs], dtype=torch.float32, device=agent._device) - 1.0
        root_heading_offset *= max_heading_angle_degrees * torch.pi / 180.0
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=agent._device).expand(num_envs, -1)
        root_rot_offset = torch_util.axis_angle_to_quat(z_axis, root_heading_offset)

        start_time_fraction = torch.rand(size=[num_envs], dtype=torch.float32, device=agent._device) * \
            (max_start_time_fraction - min_start_time_fraction) + min_start_time_fraction

        env.get_dm_env().set_root_pos_offset(root_pos_offset)
        env.get_dm_env().set_root_rot_offset(root_rot_offset)
        env.get_dm_env().set_motion_start_time_fraction(start_time_fraction)

        name_suffix = "_dm_aug" + str(i)
        record_motions_helper(name_suffix)
