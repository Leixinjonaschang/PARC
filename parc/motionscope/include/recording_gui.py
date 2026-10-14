import os

import imageio.v2 as imageio
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import send2trash

import parc.anim.motion_lib as motion_lib
import parc.motionscope.include.global_header as g
import parc.motionscope.include.ig_obs_gui as ig_obs_gui
import parc.util.motion_edit_lib as medit_lib
from parc.motionscope.include.singleton import SingletonClass


class RecordingSettings(SingletonClass):
    """
    Can hard code camera jsons here for recording comparison videos
    """

    # present_terrain
    view_json = '{"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[0.928424537181854,-0.37151899933815,-1.87303297871644e-10,-5.68922710418701,0.106584832072258,0.266355097293854,0.957963883876801,-1.2079918384552,-0.355901926755905,-0.889398097991943,0.286888986825943,-9.89432144165039,0.0,0.0,0.0,1.0],"windowHeight":1016,"windowWidth":1860}'
    
    # present terrain other angle
    view_json2 = '{"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[-0.954478859901428,-0.298272997140884,1.64057445406485e-09,5.38498306274414,0.112148329615593,-0.35887685418129,0.92662388086319,-0.525702238082886,-0.276387631893158,0.88444459438324,0.37599128484726,-12.1192407608032,0.0,0.0,0.0,1.0],"windowHeight":1016,"windowWidth":1860}'

    root_pos_spacing = 0.8

def cleanup_for_images_and_videos():
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()
    curr_motion.sequence.mesh.set_enabled(False)
    curr_motion.char.set_prev_state_enabled(False)
    g.OptimizationSettings().clear_body_constraints()

    main_vars.viewing_local_hf = False
    curr_motion.char.set_local_hf_enabled(main_vars.viewing_local_hf)

    g.g_dir_mesh.set_enabled(False)

    main_vars.mouse_ball_visible = False
    for mesh in g.g_mouse_ball_meshes:
        mesh.set_enabled(main_vars.mouse_ball_visible)

    curr_motion.char.set_body_points_enabled(False)

    ps.set_background_color([1.0, 1.0, 1.0])
    ps.get_surface_mesh("origin axes").set_enabled(False)
    ps.get_surface_mesh("selected pos flag").set_enabled(False)

    print("View json:")
    print(ps.get_view_as_json())
    return


def _write_video(video_filepath, frames, fps):
    if not frames:
        print("No frames captured; skipping video writing.")
        return

    video_dir = os.path.dirname(video_filepath)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    if os.path.exists(video_filepath):
        send2trash.send2trash(video_filepath)

    imageio.mimwrite(video_filepath, frames, fps=fps, quality=8)
    print("Video creation successful!")


def record_video(name=None):
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()
    fps = int(curr_motion.mlib._motion_fps[0].item())
    g.g_dir_mesh.set_enabled(False)
    main_vars.mouse_ball_visible = False
    for mesh in g.g_mouse_ball_meshes:
        mesh.set_enabled(False)

    curr_motion.sequence.mesh.set_enabled(False)
    curr_motion.char.set_prev_state_enabled(False)

    video_folder = "output/videos/"
    os.makedirs(video_folder, exist_ok=True)

    video_fps = fps # max(60, fps)  # 120

    num_frames = curr_motion.mlib._motion_num_frames[0].item() * (video_fps // fps)
    frames = []
    terrain = g.TerrainMeshManager().get_active_terrain(require=True)
    for curr_frame_idx in range(0, num_frames):
        main_vars.motion_time = curr_frame_idx * 1.0 / video_fps
        print("screenshotting at time:", main_vars.motion_time)
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(terrain))
        curr_motion.char.update_local_hf(terrain)

        if g.IGObsSettings().has_obs and g.IGObsSettings().record_obs:
            ig_obs_gui.animate_obs(main_vars.motion_time, g.IGObsSettings().overlay_obs_on_motion)

        frames.append(ps.screenshot_to_buffer())

    if name is None:
        video_filepath = video_folder + os.path.splitext(curr_motion.name)[0] + ".mp4"
    else:
        video_filepath = video_folder + name + ".mp4"
    print("writing video to:", video_filepath)

    _write_video(video_filepath, frames, video_fps)
    main_vars.paused = True
    return


def recording_gui():
    main_vars = g.MainVars()
    curr_motion = g.MotionManager().get_curr_motion()
    fps = int(curr_motion.mlib._motion_fps[0].item())

    if psim.Button("Clean up for images/videos"):
        cleanup_for_images_and_videos()

    changed, RecordingSettings().view_json = psim.InputText("view json", RecordingSettings().view_json)

    if psim.Button("Setup from view json"):

        ps.set_view_from_json(RecordingSettings().view_json)

    if psim.Button("Setup from view json 2"):
        ps.set_view_from_json(RecordingSettings().view_json2)

    if psim.Button("Print current camera view json"):
        print(ps.get_view_as_json())

    if psim.Button("Record Video"):
        record_video()

    if psim.Button("Record Rotating Video"):
        # Play the motion 3 times, while completing one full rotation
        # rotate camera around terrain
        radius = 10.0
        terrain = g.TerrainMeshManager().get_active_terrain(require=True)
        target = terrain.get_xyz_point(grid_inds=terrain.dims // 2).numpy()

        g.g_dir_mesh.set_enabled(False)
        main_vars.mouse_ball_visible = False
        for mesh in g.g_mouse_ball_meshes:
            mesh.set_enabled(False)

        curr_motion.sequence.mesh.set_enabled(False)
        curr_motion.char.set_prev_state_enabled(False)

        video_folder = "output/videos/"
        video_fps = max(60, fps)  # 120

        num_frames = curr_motion.mlib._motion_num_frames[0].item() * (video_fps // fps)

        frames = []
        for i in range(3):
            for curr_frame_idx in range(0, num_frames):

                rot_time = (curr_frame_idx + i * num_frames) / (3.0 * num_frames) * 2.0 * np.pi
                camera_location_x = target[0] + np.cos(rot_time) * radius
                camera_location_y = target[1] + np.sin(rot_time) * radius
                camera_location_z = target[2] + 4.0
                camera_location = np.array([camera_location_x, camera_location_y, camera_location_z])
                ps.look_at(camera_location=camera_location, target=target)

                main_vars.motion_time = curr_frame_idx * 1.0 / video_fps
                print("screenshotting (loop: " + str(i) + ") at time:", main_vars.motion_time)
                curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
                curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(terrain))
                curr_motion.char.update_local_hf(terrain)

                frames.append(ps.screenshot_to_buffer())

        video_filepath = video_folder + os.path.splitext(curr_motion.name)[0] + ".mp4"
        print("writing video to:", video_filepath)

        _write_video(video_filepath, frames, video_fps)
        main_vars.paused = True



    changed, RecordingSettings().root_pos_spacing = psim.InputFloat("root_pos_spacing", RecordingSettings().root_pos_spacing)

    return