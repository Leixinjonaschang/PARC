import os

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motionscope.include.file_browser as file_browser
import parc.motionscope.include.global_header as g
import parc.motionscope.ps_mdm_util as ps_mdm_util
import parc.util.geom_util as geom_util
import parc.util.motion_edit_lib as medit_lib
import parc.util.motion_util as motion_util
import parc.util.torch_util as torch_util
from parc.motionscope.include.singleton import SingletonClass


class MotionStitcherApp(SingletonClass):

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self.motion_A_match_start_idx = 0
        self.motion_A_match_end_idx = 30
        self.motion_B_match_start_idx = 0
        self.motion_B_match_end_idx = 30
        self.motion_A_search_start_idx = 0
        self.motion_A_search_end_idx = 60
        self.motion_B_search_start_idx = 0
        self.motion_B_search_end_idx = 60
        self.num_blend_frames = 30

        self.motion_a_index = 0
        self.motion_b_index = 1
        self.output_motion_name = "stitched_motion"
        #self.foot_body_names = "left_foot,right_foot"
        self.foot_body_names = "left_ankle_roll_link,right_ankle_roll_link"
        self.status_message = ""

        self._last_motion_pair = None
        self._initialized = True
        return

    def draw_ui(self):

        if not psim.TreeNode("Motion Stitcher"):
            return
        
        main_vars = g.MainVars()
        motion_manager = g.MotionManager()
        terrain_manager = g.TerrainMeshManager()
        loaded_motions = motion_manager.get_loaded_motions()

        if len(loaded_motions) < 2:
            psim.TextUnformatted("Load at least two motions to stitch.")
            psim.TreePop()
            return

        motion_names = list(loaded_motions.keys())

        self.motion_a_index = min(self.motion_a_index, len(motion_names) - 1)
        self.motion_b_index = min(self.motion_b_index, len(motion_names) - 1)

        opened = psim.BeginCombo("Motion A", motion_names[self.motion_a_index])
        if opened:
            for idx, name in enumerate(motion_names):
                is_selected = (idx == self.motion_a_index)
                if psim.Selectable(name, is_selected)[0]:
                    self.motion_a_index = idx
                    self.status_message = ""
                if is_selected:
                    psim.SetItemDefaultFocus()
            psim.EndCombo()

        opened = psim.BeginCombo("Motion B", motion_names[self.motion_b_index])
        if opened:
            for idx, name in enumerate(motion_names):
                is_selected = (idx == self.motion_b_index)
                if psim.Selectable(name, is_selected)[0]:
                    self.motion_b_index = idx
                    self.status_message = ""
                if is_selected:
                    psim.SetItemDefaultFocus()
            psim.EndCombo()

        selected_pair = (motion_names[self.motion_a_index], motion_names[self.motion_b_index])
        if selected_pair != self._last_motion_pair:
            default_name = f"{selected_pair[0]}_to_{selected_pair[1]}_stitched"
            self.output_motion_name = default_name
            self.status_message = ""
            self._last_motion_pair = selected_pair

        changed_name, new_name = psim.InputText("New motion name", self.output_motion_name)
        if changed_name:
            self.output_motion_name = new_name

        changed_feet, new_feet = psim.InputText("Foot body names (comma-separated)", self.foot_body_names)
        if changed_feet:
            self.foot_body_names = new_feet
            self.status_message = ""

        def _set_motion_to_frame(motion: ps_mdm_util.MDMMotionPS, frame_idx: int):
            fps = motion.mlib.get_motion_fps(0)
            num_frames = motion.mlib._motion_num_frames[0].item()
            frame_idx = int(np.clip(frame_idx, 0, num_frames - 1))
            time_val = frame_idx / fps
            motion.set_to_time(float(time_val))
            motion.char.forward_kinematics()

        motion_a = motion_manager.get_motion(selected_pair[0])
        motion_b = motion_manager.get_motion(selected_pair[1])

        motion_a_num_frames = motion_a.mlib._motion_num_frames[0].item()
        motion_b_num_frames = motion_b.mlib._motion_num_frames[0].item()

        self.motion_A_match_start_idx = int(np.clip(self.motion_A_match_start_idx, 0, motion_a_num_frames - 1))
        self.motion_A_match_end_idx = int(np.clip(self.motion_A_match_end_idx, 0, motion_a_num_frames - 1))
        self.motion_B_match_start_idx = int(np.clip(self.motion_B_match_start_idx, 0, motion_b_num_frames - 1))
        self.motion_B_match_end_idx = int(np.clip(self.motion_B_match_end_idx, 0, motion_b_num_frames - 1))

        self.motion_A_search_start_idx = int(np.clip(self.motion_A_search_start_idx, 0, motion_a_num_frames - 1))
        self.motion_A_search_end_idx = int(np.clip(self.motion_A_search_end_idx, 0, motion_a_num_frames - 1))
        self.motion_B_search_start_idx = int(np.clip(self.motion_B_search_start_idx, 0, motion_b_num_frames - 1))
        self.motion_B_search_end_idx = int(np.clip(self.motion_B_search_end_idx, 0, motion_b_num_frames - 1))

        if self.motion_A_match_start_idx > self.motion_A_match_end_idx:
            self.motion_A_match_start_idx, self.motion_A_match_end_idx = (
                self.motion_A_match_end_idx, self.motion_A_match_start_idx)
        if self.motion_B_match_start_idx > self.motion_B_match_end_idx:
            self.motion_B_match_start_idx, self.motion_B_match_end_idx = (
                self.motion_B_match_end_idx, self.motion_B_match_start_idx)
        if self.motion_A_search_start_idx > self.motion_A_search_end_idx:
            self.motion_A_search_start_idx, self.motion_A_search_end_idx = (
                self.motion_A_search_end_idx, self.motion_A_search_start_idx)
        if self.motion_B_search_start_idx > self.motion_B_search_end_idx:
            self.motion_B_search_start_idx, self.motion_B_search_end_idx = (
                self.motion_B_search_end_idx, self.motion_B_search_start_idx)

        if main_vars.using_sliders:
            changed, new_val = psim.SliderInt(
                "Motion A start frame",
                self.motion_A_match_start_idx,
                v_min=0,
                v_max=motion_a_num_frames - 1,
            )
            if changed:
                self.motion_A_match_start_idx = new_val
                self.status_message = ""
                _set_motion_to_frame(motion_a, self.motion_A_match_start_idx)

            changed, new_val = psim.SliderInt(
                "Motion A end frame",
                self.motion_A_match_end_idx,
                v_min=0,
                v_max=motion_a_num_frames - 1,
            )
            if changed:
                self.motion_A_match_end_idx = new_val
                self.status_message = ""
                _set_motion_to_frame(motion_a, self.motion_A_match_end_idx)

            changed, new_val = psim.SliderInt(
                "Motion B start frame",
                self.motion_B_match_start_idx,
                v_min=0,
                v_max=motion_b_num_frames - 1,
            )
            if changed:
                self.motion_B_match_start_idx = new_val
                self.status_message = ""
                _set_motion_to_frame(motion_b, self.motion_B_match_start_idx)

            changed, new_val = psim.SliderInt(
                "Motion B end frame",
                self.motion_B_match_end_idx,
                v_min=0,
                v_max=motion_b_num_frames - 1,
            )
            if changed:
                self.motion_B_match_end_idx = new_val
                self.status_message = ""
                _set_motion_to_frame(motion_b, self.motion_B_match_end_idx)
        else:
            changed, self.motion_A_match_start_idx = psim.InputInt(
                "Motion A start frame", self.motion_A_match_start_idx)
            if changed:
                self.motion_A_match_start_idx = int(np.clip(self.motion_A_match_start_idx, 0, motion_a_num_frames - 1))
                self.status_message = ""
                _set_motion_to_frame(motion_a, self.motion_A_match_start_idx)

            changed, self.motion_A_match_end_idx = psim.InputInt(
                "Motion A end frame", self.motion_A_match_end_idx)
            if changed:
                self.motion_A_match_end_idx = int(np.clip(self.motion_A_match_end_idx, 0, motion_a_num_frames - 1))
                self.status_message = ""
                _set_motion_to_frame(motion_a, self.motion_A_match_end_idx)

            changed, self.motion_B_match_start_idx = psim.InputInt(
                "Motion B start frame", self.motion_B_match_start_idx)
            if changed:
                self.motion_B_match_start_idx = int(np.clip(self.motion_B_match_start_idx, 0, motion_b_num_frames - 1))
                self.status_message = ""
                _set_motion_to_frame(motion_b, self.motion_B_match_start_idx)

            changed, self.motion_B_match_end_idx = psim.InputInt(
                "Motion B end frame", self.motion_B_match_end_idx)
            if changed:
                self.motion_B_match_end_idx = int(np.clip(self.motion_B_match_end_idx, 0, motion_b_num_frames - 1))
                self.status_message = ""
                _set_motion_to_frame(motion_b, self.motion_B_match_end_idx)

        if self.motion_A_match_start_idx > self.motion_A_match_end_idx:
            self.motion_A_match_start_idx = self.motion_A_match_end_idx
        if self.motion_B_match_start_idx > self.motion_B_match_end_idx:
            self.motion_B_match_start_idx = self.motion_B_match_end_idx
        if self.motion_A_search_start_idx > self.motion_A_search_end_idx:
            self.motion_A_search_start_idx = self.motion_A_search_end_idx
        if self.motion_B_search_start_idx > self.motion_B_search_end_idx:
            self.motion_B_search_start_idx = self.motion_B_search_end_idx

        psim.Separator()
        psim.TextUnformatted("Motion match search ranges")

        if main_vars.using_sliders:
            changed, new_val = psim.SliderInt(
                "Motion A search start",
                self.motion_A_search_start_idx,
                v_min=0,
                v_max=motion_a_num_frames - 1,
            )
            if changed:
                self.motion_A_search_start_idx = new_val
                self.status_message = ""

            changed, new_val = psim.SliderInt(
                "Motion A search end",
                self.motion_A_search_end_idx,
                v_min=0,
                v_max=motion_a_num_frames - 1,
            )
            if changed:
                self.motion_A_search_end_idx = new_val
                self.status_message = ""

            changed, new_val = psim.SliderInt(
                "Motion B search start",
                self.motion_B_search_start_idx,
                v_min=0,
                v_max=motion_b_num_frames - 1,
            )
            if changed:
                self.motion_B_search_start_idx = new_val
                self.status_message = ""

            changed, new_val = psim.SliderInt(
                "Motion B search end",
                self.motion_B_search_end_idx,
                v_min=0,
                v_max=motion_b_num_frames - 1,
            )
            if changed:
                self.motion_B_search_end_idx = new_val
                self.status_message = ""
        else:
            changed, self.motion_A_search_start_idx = psim.InputInt(
                "Motion A search start", self.motion_A_search_start_idx)
            if changed:
                self.motion_A_search_start_idx = int(np.clip(self.motion_A_search_start_idx, 0, motion_a_num_frames - 1))
                self.status_message = ""

            changed, self.motion_A_search_end_idx = psim.InputInt(
                "Motion A search end", self.motion_A_search_end_idx)
            if changed:
                self.motion_A_search_end_idx = int(np.clip(self.motion_A_search_end_idx, 0, motion_a_num_frames - 1))
                self.status_message = ""

            changed, self.motion_B_search_start_idx = psim.InputInt(
                "Motion B search start", self.motion_B_search_start_idx)
            if changed:
                self.motion_B_search_start_idx = int(np.clip(self.motion_B_search_start_idx, 0, motion_b_num_frames - 1))
                self.status_message = ""

            changed, self.motion_B_search_end_idx = psim.InputInt(
                "Motion B search end", self.motion_B_search_end_idx)
            if changed:
                self.motion_B_search_end_idx = int(np.clip(self.motion_B_search_end_idx, 0, motion_b_num_frames - 1))
                self.status_message = ""

        if psim.Button("Search for motion matching frames"):
            if motion_a.mlib._kin_char_model.get_num_joints() != motion_b.mlib._kin_char_model.get_num_joints():
                self.status_message = "Selected motions must use the same character model."
            else:
                fps_a = motion_a.mlib.get_motion_fps(0)
                fps_b = motion_b.mlib.get_motion_fps(0)
                if fps_a != fps_b:
                    self.status_message = "Selected motions must have the same FPS."
                else:
                    try:
                        match_idx_a, match_idx_b, _heading_diff, _root_pos_diff = medit_lib.search_for_matching_motion_frames(
                            mlib_A=motion_a.mlib,
                            mlib_B=motion_b.mlib,
                            m_A_start_frame_idx=self.motion_A_search_start_idx,
                            m_A_end_frame_idx=self.motion_A_search_end_idx,
                            m_B_start_frame_idx=self.motion_B_search_start_idx,
                            m_B_end_frame_idx=self.motion_B_search_end_idx,
                        )
                    except ValueError as err:
                        self.status_message = str(err)
                    else:
                        self.motion_A_match_end_idx = match_idx_a
                        self.motion_B_match_start_idx = match_idx_b
                        if self.motion_A_match_start_idx > self.motion_A_match_end_idx:
                            self.motion_A_match_start_idx = self.motion_A_match_end_idx
                        if self.motion_B_match_end_idx < self.motion_B_match_start_idx:
                            self.motion_B_match_end_idx = self.motion_B_match_start_idx
                        _set_motion_to_frame(motion_a, match_idx_a)
                        _set_motion_to_frame(motion_b, match_idx_b)
                        self.status_message = (
                            f"Matched frames: Motion A {match_idx_a}, Motion B {match_idx_b}")

        changed, self.num_blend_frames = psim.InputInt("Num blend frames", self.num_blend_frames)
        if changed:
            self.num_blend_frames = max(0, self.num_blend_frames)
            self.status_message = ""

        for motion_name in selected_pair:
            curr_motion = motion_manager.get_motion(motion_name)
            if psim.TreeNode("Full Seq Editor: " + motion_name):
                draw_full_seq_editor_ui(curr_motion)
                psim.TreePop()

        if selected_pair[0] == selected_pair[1]:
            psim.TextUnformatted("Select two different motions to stitch.")
        else:
            if psim.Button("Stitch selected motions"):
                success, message = self._stitch_selected_motions(selected_pair[0], selected_pair[1])
                self.status_message = message

        if self.status_message:
            psim.TextUnformatted(self.status_message)

        psim.TreePop()
        return

    def _stitch_selected_motions(self, motion_a_name: str, motion_b_name: str):
        motion_manager = g.MotionManager()
        motion_a = motion_manager.get_motion(motion_a_name)
        motion_b = motion_manager.get_motion(motion_b_name)

        char_a = motion_a.mlib._kin_char_model
        char_b = motion_b.mlib._kin_char_model
        if char_a.get_num_joints() != char_b.get_num_joints():
            return False, "Selected motions must use the same character model."

        fps_a = motion_a.mlib.get_motion_fps(0)
        fps_b = motion_b.mlib.get_motion_fps(0)
        if fps_a != fps_b:
            return False, "Selected motions must have the same FPS."

        frames_a = self._get_world_space_frames(motion_a)
        frames_b = self._get_world_space_frames(motion_b)

        num_frames_a = motion_a.mlib._motion_num_frames[0].item()
        num_frames_b = motion_b.mlib._motion_num_frames[0].item()

        start_idx_a = int(np.clip(self.motion_A_match_start_idx, 0, num_frames_a - 1))
        end_idx_a = int(np.clip(self.motion_A_match_end_idx, 0, num_frames_a - 1))
        if end_idx_a < start_idx_a:
            start_idx_a, end_idx_a = end_idx_a, start_idx_a

        start_idx_b = int(np.clip(self.motion_B_match_start_idx, 0, num_frames_b - 1))
        end_idx_b = int(np.clip(self.motion_B_match_end_idx, 0, num_frames_b - 1))
        if end_idx_b < start_idx_b:
            start_idx_b, end_idx_b = end_idx_b, start_idx_b

        slice_a = slice(start_idx_a, end_idx_a + 1)
        slice_b = slice(start_idx_b, end_idx_b + 1)

        segment_a = frames_a.get_slice(slice_a)
        segment_b = frames_b.get_slice(slice_b)

        if segment_a.root_pos is None or segment_a.root_pos.shape[0] == 0:
            return False, "Motion A selection does not contain any frames."
        if segment_b.root_pos is None or segment_b.root_pos.shape[0] == 0:
            return False, "Motion B selection does not contain any frames."

        if (segment_a.contacts is None) != (segment_b.contacts is None):
            segment_a.contacts = None
            segment_b.contacts = None

        target_pos = segment_a.root_pos[-1]
        target_rot = segment_a.root_rot[-1]

        foot_body_names = [name.strip() for name in self.foot_body_names.split(",") if name.strip()]
        if len(foot_body_names) == 0:
            return False, "Specify at least one foot body name to align foot heights."

        available_body_names = set(char_a.get_body_names())
        missing_names = [name for name in foot_body_names if name not in available_body_names]
        if missing_names:
            missing_list = ", ".join(missing_names)
            return False, f"Unknown foot body name(s): {missing_list}."

        foot_body_ids = [char_a.get_body_id(name) for name in foot_body_names]

        target_foot_height = self._compute_average_foot_height(segment_a, foot_body_ids, char_a, -1)
        if target_foot_height is None:
            return False, "Unable to compute foot heights for Motion A."

        if target_foot_height.device != segment_a.root_pos.device or target_foot_height.dtype != segment_a.root_pos.dtype:
            target_foot_height = target_foot_height.to(device=segment_a.root_pos.device, dtype=segment_a.root_pos.dtype)

        aligned_segment_b = self._align_segment(segment_b, target_pos, target_rot, char_a, foot_body_ids,
                                                target_foot_height)

        stitched_frames = self._blend_segments(segment_a, aligned_segment_b)
        if stitched_frames.root_pos.shape[0] < 2:
            return False, "Unable to stitch motions with fewer than two frames."

        new_motion_name = self.output_motion_name if self.output_motion_name else f"{motion_a_name}_to_{motion_b_name}_stitched"

        motion_manager.make_new_motion(
            motion_frames=stitched_frames,
            new_motion_name=new_motion_name,
            motion_fps=fps_a,
            new_char_model=char_a)

        return True, f"Created motion '{new_motion_name}'."

    def _get_world_space_frames(self, motion: ps_mdm_util.MDMMotionPS):
        motion_frames = motion.mlib.get_frames_for_id(0)
        device = motion_frames.root_pos.device

        rot_quat = motion.compute_rot_quat().to(device=device)
        origin = torch.zeros(3, dtype=motion_frames.root_pos.dtype, device=device)

        rotated_frames = medit_lib.rotate_motion(motion_frames, rot_quat, origin)
        offset = torch.tensor(motion.root_offset, dtype=rotated_frames.root_pos.dtype, device=device)
        rotated_frames.root_pos = rotated_frames.root_pos + offset
        rotated_frames.body_pos = None
        rotated_frames.body_rot = None

        return rotated_frames

    def _align_segment(self, segment: motion_util.MotionFrames, target_pos: torch.Tensor, target_rot: torch.Tensor,
                       char_model: kin_char_model.KinCharModel, foot_body_ids, target_foot_height: torch.Tensor):
        device = segment.root_pos.device
        start_pos = segment.root_pos[0]
        start_rot = segment.root_rot[0]
        start_heading_quat = torch_util.calc_heading_quat(start_rot)
        target_heading_quat = torch_util.calc_heading_quat(target_rot)

        rot_delta = torch_util.quat_multiply(target_heading_quat, torch_util.quat_inv(start_heading_quat))
        aligned = medit_lib.rotate_motion(segment, rot_delta.to(device=device), start_pos)

        translation = target_pos - aligned.root_pos[0]
        translation[2] = 0.0

        aligned_foot_height = self._compute_average_foot_height(aligned, foot_body_ids, char_model, 0)
        if aligned_foot_height is not None and target_foot_height is not None:
            if (aligned_foot_height.device != translation.device or
                    aligned_foot_height.dtype != translation.dtype):
                aligned_foot_height = aligned_foot_height.to(device=translation.device, dtype=translation.dtype)
            if target_foot_height.device != translation.device or target_foot_height.dtype != translation.dtype:
                target_foot_height = target_foot_height.to(device=translation.device, dtype=translation.dtype)
            translation[2] = target_foot_height - aligned_foot_height
        else:
            translation[2] = target_pos[2] - aligned.root_pos[0, 2]

        aligned.root_pos = aligned.root_pos + translation
        aligned.body_pos = None
        aligned.body_rot = None

        return aligned

    def _compute_average_foot_height(self, segment: motion_util.MotionFrames, foot_body_ids,
                                     char_model: kin_char_model.KinCharModel, frame_idx: int):
        if segment.joint_rot is None:
            return None

        try:
            body_pos, _ = char_model.forward_kinematics(segment.root_pos, segment.root_rot, segment.joint_rot)
        except Exception:
            return None

        if body_pos is None:
            return None

        if body_pos.ndim == 3:
            num_frames = body_pos.shape[0]
            if frame_idx >= num_frames or frame_idx < -num_frames:
                return None
            frame_body_pos = body_pos[frame_idx]
        elif body_pos.ndim == 2:
            frame_body_pos = body_pos
        else:
            return None

        heights = []
        for body_id in foot_body_ids:
            if body_id < 0 or body_id >= frame_body_pos.shape[0]:
                return None
            heights.append(frame_body_pos[body_id, 2])

        if len(heights) == 0:
            return None

        heights_tensor = torch.stack(heights)
        return torch.mean(heights_tensor)

    def _blend_segments(self, segment_a: motion_util.MotionFrames, segment_b: motion_util.MotionFrames):
        segments = []

        num_frames_a = segment_a.root_pos.shape[0]
        num_frames_b = segment_b.root_pos.shape[0]

        blend_frames = min(self.num_blend_frames, num_frames_a, num_frames_b)

        if blend_frames > 0:
            pre_len = num_frames_a - blend_frames
        else:
            pre_len = num_frames_a

        if pre_len > 0:
            segments.append(segment_a.get_slice(slice(0, pre_len)))

        if blend_frames > 0:
            blend_a = segment_a.get_slice(slice(num_frames_a - blend_frames, num_frames_a))
            blend_b = segment_b.get_slice(slice(0, blend_frames))

            weights = torch.linspace(1.0 / (blend_frames + 1), blend_frames / (blend_frames + 1), blend_frames,
                                     dtype=blend_a.root_pos.dtype, device=blend_a.root_pos.device)
            one_minus = 1.0 - weights

            root_pos = one_minus.unsqueeze(-1) * blend_a.root_pos + weights.unsqueeze(-1) * blend_b.root_pos
            root_rot = torch_util.slerp(blend_a.root_rot, blend_b.root_rot, weights)

            joint_rot = None
            if blend_a.joint_rot is not None and blend_b.joint_rot is not None:
                joint_rot = torch_util.slerp(blend_a.joint_rot, blend_b.joint_rot, weights.view(-1, 1))

            contacts = None
            if blend_a.contacts is not None and blend_b.contacts is not None:
                contacts = one_minus.unsqueeze(-1) * blend_a.contacts + weights.unsqueeze(-1) * blend_b.contacts

            blended = motion_util.MotionFrames(
                root_pos=root_pos,
                root_rot=root_rot,
                joint_rot=joint_rot,
                contacts=contacts)
            segments.append(blended)

            if num_frames_b - blend_frames > 0:
                segments.append(segment_b.get_slice(slice(blend_frames, num_frames_b)))
        else:
            segments.append(segment_b)

        stitched_frames = motion_util.cat_motion_frames(segments)
        stitched_frames.body_pos = None
        stitched_frames.body_rot = None

        return stitched_frames

def draw_full_seq_editor_ui(curr_motion: ps_mdm_util.MDMMotionPS):
    main_vars = g.MainVars()
    terrain_manager = g.TerrainMeshManager()

    _, main_vars.using_sliders = psim.Checkbox("use sliders", main_vars.using_sliders)
    _, curr_motion.editing_full_sequence = psim.Checkbox(
        "Editing full sequence", curr_motion.editing_full_sequence)

    if main_vars.using_sliders:
        changedx, curr_motion.root_offset[0] = psim.SliderFloat("offset x", curr_motion.root_offset[0], v_min=-10.0, v_max=10.0)
        changedy, curr_motion.root_offset[1] = psim.SliderFloat("offset y", curr_motion.root_offset[1], v_min=-10.0, v_max=10.0)
        changedz, curr_motion.root_offset[2] = psim.SliderFloat("offset z", curr_motion.root_offset[2], v_min=-10.0, v_max=10.0)
        changed_rot, curr_motion.root_heading_angle = psim.SliderFloat(
            "heading angle offset", curr_motion.root_heading_angle, v_min=-np.pi, v_max=np.pi)
    else:
        changedx, curr_motion.root_offset[0] = psim.InputFloat("offset x", curr_motion.root_offset[0])
        changedy, curr_motion.root_offset[1] = psim.InputFloat("offset y", curr_motion.root_offset[1])
        changedz, curr_motion.root_offset[2] = psim.InputFloat("offset z", curr_motion.root_offset[2])
        changed_rot, curr_motion.root_heading_angle = psim.InputFloat("heading angle offset", curr_motion.root_heading_angle)

    changed = changedx or changedy or changedz or changed_rot
    if changed:
        curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
        curr_motion.update_transforms(transform_full_sequence=curr_motion.editing_full_sequence)
        active_terrain = terrain_manager.get_active_terrain()
        if active_terrain is not None:
            curr_motion.char.update_local_hf(active_terrain)

    if psim.Button("Apply transforms to motion data"):
        curr_motion.apply_transforms_to_motion_data(vis_fps=2)

    if psim.Button("Flip Motion about XZ plane"):
        flipped_frames = medit_lib.flip_motion_about_XZ_plane(
            motion_frames=curr_motion.mlib.get_frames_for_id(0),
            char_model=curr_motion.mlib._kin_char_model)
        g.MotionManager().make_new_motion(
            motion_frames=flipped_frames,
            new_motion_name=curr_motion.name + "_flipped",
            motion_fps=curr_motion.mlib.get_motion_fps(0))

    if psim.Button("Fix Sinking"):
        mlib = curr_motion.mlib
        motion_frames = mlib.get_frames_for_id(0)
        root_pos = motion_frames.root_pos
        root_rot = motion_frames.root_rot
        joint_rot = motion_frames.joint_rot

        num_frames = root_pos.shape[0]
        if num_frames == 0:
            return

        end_root_pos = torch.stack((root_pos[0], root_pos[-1]), dim=0)
        end_root_rot = torch.stack((root_rot[0], root_rot[-1]), dim=0)
        end_joint_rot = torch.stack((joint_rot[0], joint_rot[-1]), dim=0)

        end_body_pos, end_body_rot = mlib._kin_char_model.forward_kinematics(
            end_root_pos, end_root_rot, end_joint_rot)
        lowest_points = mlib._kin_char_model.find_lowest_point(end_body_pos, end_body_rot)
        delta_z = lowest_points[0, 2] - lowest_points[1, 2]

        root_pos_delta = root_pos[-1] - root_pos[0]
        denom_sq = torch.dot(root_pos_delta, root_pos_delta)
        if denom_sq.item() > 1e-8:
            f = torch.sum((root_pos - root_pos[0]) * root_pos_delta, dim=-1) / denom_sq
        else:
            f = torch.linspace(0.0, 1.0, steps=num_frames, device=root_pos.device, dtype=root_pos.dtype)

        new_root_pos = root_pos.clone()
        new_root_pos[:, 2] = root_pos[:, 2] + delta_z * f

        fixed_frames = motion_util.MotionFrames(
            root_pos=new_root_pos,
            root_rot=root_rot,
            joint_rot=joint_rot,
            contacts=motion_frames.contacts)

        g.MotionManager().make_new_motion(
            motion_frames=fixed_frames,
            new_motion_name=curr_motion.name + "_nosink",
            motion_fps=mlib.get_motion_fps(0),
            loop_mode=mlib.get_motion_loop_mode_enum(0))
    return

def draw_editor_ui(curr_motion: ps_mdm_util.MDMMotionPS):
    if psim.TreeNode("Motion Editor"):
        if psim.TreeNode("Full Sequence Editing"):
            draw_full_seq_editor_ui(curr_motion)
            psim.TreePop()

        if psim.TreeNode("Spatiotemporal Editing"):

            changed_start, curr_motion.medit_start_frame = psim.InputInt("Edit start frame", curr_motion.medit_start_frame)
            if changed_start:
                curr_motion.medit_start_frame = max(curr_motion.medit_start_frame, 0)
                curr_motion.medit_start_frame = min(curr_motion.medit_start_frame, curr_motion.mlib._motion_frames.shape[0] - 1)

            changed_end, curr_motion.medit_end_frame = psim.InputInt("Edit end frame", curr_motion.medit_end_frame)
            if changed_end:
                curr_motion.medit_end_frame = max(curr_motion.medit_end_frame, 0)
                curr_motion.medit_end_frame = min(curr_motion.medit_end_frame, curr_motion.mlib._motion_frames.shape[0] - 1)
            if curr_motion.medit_end_frame < curr_motion.medit_start_frame:
                curr_motion.medit_end_frame = curr_motion.medit_start_frame

            _, curr_motion.medit_scale = psim.InputFloat("Edit scale", curr_motion.medit_scale)

            _, curr_motion.speed_scale = psim.InputFloat("Speed scale", curr_motion.speed_scale)

            if psim.Button("Scale Motion Segment"):
                new_frames = medit_lib.scale_motion_segment(
                    curr_motion.mlib.get_frames_for_id(0),
                    curr_motion.medit_scale,
                    curr_motion.medit_start_frame,
                    curr_motion.medit_end_frame)

                g.MotionManager().make_new_motion(
                    motion_frames=new_frames,
                    new_motion_name=curr_motion.name + "_scaled",
                    motion_fps=curr_motion.mlib._motion_fps[0].item(),
                    loop_mode=curr_motion.mlib.get_motion_loop_mode_enum(0))

            if psim.Button("Cut frames [inclusive]"):
                pre_cut_frames = curr_motion.mlib.get_frames_for_id(0).get_slice(slice(0, curr_motion.medit_start_frame))
                post_cut_frames = curr_motion.mlib.get_frames_for_id(0).get_slice(
                    slice(curr_motion.medit_end_frame + 1, curr_motion.mlib.get_motion_num_frames(0).item()))

                new_frames = motion_util.cat_motion_frames([pre_cut_frames, post_cut_frames])

                g.MotionManager().make_new_motion(
                    motion_frames=new_frames,
                    new_motion_name=curr_motion.name + "_cut",
                    motion_fps=curr_motion.mlib._motion_fps[0].item(),
                    loop_mode=curr_motion.mlib.get_motion_loop_mode_enum(0))

            if psim.Button("Find and remove hesitation frames"):
                new_motion_frames = medit_lib.remove_hesitation_frames(
                    motion_frames=curr_motion.mlib.get_frames_for_id(0),
                    char_model=curr_motion.mlib._kin_char_model,
                    verbose=True)

                g.MotionManager().make_new_motion(
                    motion_frames=new_motion_frames,
                    new_motion_name=curr_motion.name,
                    motion_fps=curr_motion.mlib._motion_fps[0].item())

            if psim.Button("Resample motion frames for safe trajectory"):
                new_motion_frames = medit_lib.resample_motion_for_safe_body_trajectories(
                    motion_frames=curr_motion.mlib.get_frames_for_id(0),
                    char_model=curr_motion.mlib._kin_char_model,
                    fps=curr_motion.mlib.get_motion_fps(0),
                    max_jerk=1000.0,
                    max_subdivisions=4
                )

                g.MotionManager().make_new_motion(
                    motion_frames=new_motion_frames,
                    new_motion_name=curr_motion.name + "_safer",
                    motion_fps=curr_motion.mlib._motion_fps[0].item())

            psim.TreePop()

        psim.TreePop()
    return

def draw_analytics_ui(curr_motion: ps_mdm_util.MDMMotionPS):
    main_vars = g.MainVars()
    terrain_manager = g.TerrainMeshManager()
    if psim.TreeNode("Motion Stats"):
        if psim.Button("Body derivatives"):
            mlib = curr_motion.mlib
            root_pos = mlib._frame_root_pos
            root_rot = mlib._frame_root_rot
            joint_rot = mlib._frame_joint_rot

            body_pos, body_rot = curr_motion.char.char_model.forward_kinematics(root_pos, root_rot, joint_rot)

            dt = 1.0 / mlib._motion_fps[0].item()

            body_vel = (body_pos[1:] - body_pos[:-1]) / dt
            body_acc = (body_vel[1:] - body_vel[:-1]) / dt
            body_jerk = (body_acc[1:] - body_acc[:-1]) / dt

            vel_mag = body_vel.norm(dim=-1)
            acc_mag = body_acc.norm(dim=-1)
            jerk_mag = body_jerk.norm(dim=-1)

            max_jerk, max_frame_idx = torch.max(jerk_mag, dim=0)

            print("Max speed:", torch.max(vel_mag))
            print("Max acceleration (magnitude):", torch.max(acc_mag))
            print("Max jerk (magnitude):", max_jerk)
            print("Max jerk frame idx:", max_frame_idx)

            print("Mean speed:", torch.mean(vel_mag))
            print("Mean acceleration (magnitude):", torch.mean(acc_mag))
            print("Mean jerk (magnitude):", torch.mean(jerk_mag))

            print("Std speed:", torch.std(vel_mag))
            print("Std acceleration (magnitude):", torch.std(acc_mag))
            print("Std jerk (magnitude):", torch.std(jerk_mag))

        if psim.Button("Compute motion contact loss"):
            from parc.motion_synthesis.procgen.mdm_path import compute_motion_loss

            mlib = curr_motion.mlib
            root_pos = mlib._frame_root_pos
            root_rot = mlib._frame_root_rot
            joint_rot = mlib._frame_joint_rot
            contacts = mlib._frame_contacts

            body_points = geom_util.get_char_point_samples(mlib._kin_char_model)

            motion_frames = motion_util.MotionFrames(root_pos=root_pos, root_rot=root_rot, joint_rot=joint_rot, contacts=contacts)
            motion_frames = motion_frames.unsqueeze(0)
            active_terrain = terrain_manager.get_active_terrain(require=True)
            losses = compute_motion_loss(motion_frames, None, active_terrain, mlib._kin_char_model, body_points, w_contact=1.0,
                                            w_pen=1.0, w_path=0.0)
            for key in losses:
                print(key + ":", losses[key].item())

        if psim.TreeNode("Plot positions"):
            mlib = curr_motion.mlib
            char_model = mlib._kin_char_model
            device = curr_motion.mlib._device
            dt = mlib._motion_dt[0].item()
            num_frames = mlib._motion_num_frames[0].item()
            motion_ids = torch.zeros(size=[num_frames], dtype=torch.int64, device=device)

            all_motion_times = torch.arange(start=0, end=num_frames, dtype=torch.float32, device=device) * dt
            curr_motion_time = torch.tensor([main_vars.motion_time], dtype=torch.float32, device=device)
            single_motion_id = torch.tensor([0], dtype=torch.int64, device=device)

            if mlib._contact_info:
                root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, contacts = mlib.calc_motion_frame(motion_ids, all_motion_times)
            else:
                root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = mlib.calc_motion_frame(motion_ids, all_motion_times)

            body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot, joint_rot)

            frame_idx0, frame_idx1, blend = curr_motion.mlib._calc_frame_blend(single_motion_id, curr_motion_time)
            offset = frame_idx0.item()

            if psim.TreeNode("Body positions"):
                body_pos_np = body_pos.cpu().numpy()
                for b in range(mlib._kin_char_model.get_num_joints()):
                    body_name = char_model.get_body_name(b)

                    if psim.TreeNode(body_name):

                        xyz_str = ["x", "y", "z"]

                        for i in range(3):
                            body_pos_1d = body_pos_np[:, b, i]

                            min_pos = np.min(body_pos_1d)
                            max_pos = np.max(body_pos_1d)

                            psim.PlotLines(
                                label=body_name + " " + xyz_str[i],
                                values=body_pos_1d.tolist(),
                                values_offset=offset,
                                scale_min=min_pos,
                                scale_max=max_pos
                            )

                        psim.TreePop()
                psim.TreePop()

            if psim.TreeNode("Joint DOFS"):
                joint_rot_np = joint_rot.cpu().numpy()

                for b in range(mlib._kin_char_model.get_num_joints()):
                    body_name = char_model.get_body_name(b)

                    if psim.TreeNode(body_name):
                        xyzw_str = ["x", "y", "z", "w"]
                        for i in range(4):
                            joint_rot_1d = joint_rot_np[:, b, i]
                            psim.PlotLines(
                                label=body_name + " " + xyzw_str[i],
                                values=joint_rot_1d.tolist(),
                                values_offset=offset,
                                scale_min=-1.0,
                                scale_max=1.0
                            )

                        psim.TreePop()
                psim.TreePop()
            psim.TreePop()
        psim.TreePop()

    if psim.TreeNode("Body trajectory visualization"):
        body_trajs = getattr(curr_motion, "ps_body_traj", [])
        has_body_trajs = len(body_trajs) > 0

        compute_label = "Compute body trajectories" if not has_body_trajs else "Recompute body trajectories"
        if psim.Button(compute_label):
            if has_body_trajs:
                for traj in body_trajs:
                    try:
                        traj.remove()
                    except Exception:
                        pass
            curr_motion._build_ps_body_traj(view_body_traj=True)
            body_trajs = getattr(curr_motion, "ps_body_traj", [])
            has_body_trajs = len(body_trajs) > 0

        if has_body_trajs and psim.Button("Delete body trajectories"):
            curr_motion.set_all_traj_enabled(False)
            for traj in body_trajs:
                try:
                    traj.remove()
                except Exception:
                    pass
            curr_motion.ps_body_traj = []
            curr_motion.body_traj_names = []
            if hasattr(curr_motion, "ps_body_traj_quantities"):
                delattr(curr_motion, "ps_body_traj_quantities")
            if hasattr(curr_motion, "body_traj_value_ranges"):
                delattr(curr_motion, "body_traj_value_ranges")
            body_trajs = []
            has_body_trajs = False

        if not has_body_trajs:
            psim.TextUnformatted("No body trajectories registered.")
        else:
            if psim.TreeNode("Color ranges"):
                speed_min = curr_motion.body_traj_speed_vmin
                speed_max = curr_motion.body_traj_speed_vmax
                changed_min, speed_min = psim.InputFloat("speed min", speed_min)
                changed_max, speed_max = psim.InputFloat("speed max", speed_max)
                if changed_min or changed_max:
                    if speed_min > speed_max:
                        if changed_min:
                            speed_max = speed_min
                        else:
                            speed_min = speed_max
                    curr_motion.body_traj_speed_vmin = speed_min
                    curr_motion.body_traj_speed_vmax = speed_max
                    curr_motion.set_body_traj_map_range("speed", speed_min, speed_max)

                acc_min = curr_motion.body_traj_acc_vmin
                acc_max = curr_motion.body_traj_acc_vmax
                changed_min, acc_min = psim.InputFloat("acc min", acc_min)
                changed_max, acc_max = psim.InputFloat("acc max", acc_max)
                if changed_min or changed_max:
                    if acc_min > acc_max:
                        if changed_min:
                            acc_max = acc_min
                        else:
                            acc_min = acc_max
                    curr_motion.body_traj_acc_vmin = acc_min
                    curr_motion.body_traj_acc_vmax = acc_max
                    curr_motion.set_body_traj_map_range("acc", acc_min, acc_max)

                jerk_min = curr_motion.body_traj_jerk_vmin
                jerk_max = curr_motion.body_traj_jerk_vmax
                changed_min, jerk_min = psim.InputFloat("jerk min", jerk_min)
                changed_max, jerk_max = psim.InputFloat("jerk max", jerk_max)
                if changed_min or changed_max:
                    if jerk_min > jerk_max:
                        if changed_min:
                            jerk_max = jerk_min
                        else:
                            jerk_min = jerk_max
                    curr_motion.body_traj_jerk_vmin = jerk_min
                    curr_motion.body_traj_jerk_vmax = jerk_max
                    curr_motion.set_body_traj_map_range("jerk", jerk_min, jerk_max)

                speed_range = curr_motion.get_body_traj_quantity_range("speed")
                acc_range = curr_motion.get_body_traj_quantity_range("acceleration")
                jerk_range = curr_motion.get_body_traj_quantity_range("jerk")
                psim.TextUnformatted(f"speed data range: [{speed_range[0]:.3f}, {speed_range[1]:.3f}]")
                psim.TextUnformatted(f"acceleration data range: [{acc_range[0]:.3f}, {acc_range[1]:.3f}]")
                psim.TextUnformatted(f"jerk data range: [{jerk_range[0]:.3f}, {jerk_range[1]:.3f}]")

                if psim.Button("View Speed"):
                    curr_motion.enable_traj_quantity("speed")
                if psim.Button("View Acc"):
                    curr_motion.enable_traj_quantity("acc")
                if psim.Button("View Jerk"):
                    curr_motion.enable_traj_quantity("jerk")

                psim.TreePop()

            if psim.TreeNode("Body visibility"):

                if psim.Button("View all trajs"):
                    for traj in body_trajs:
                        traj.set_enabled(True)

                if psim.Button("Hide all trajs"):
                    for traj in body_trajs:
                        traj.set_enabled(False)

                body_names = getattr(curr_motion, "body_traj_names", [])
                for body_name, traj in zip(body_names, body_trajs):
                    enabled = traj.is_enabled()
                    changed, enabled = psim.Checkbox(body_name, enabled)
                    if changed:
                        traj.set_enabled(enabled)
                psim.TreePop()

        psim.TreePop()
    return

def draw_char_state_ui(curr_motion: ps_mdm_util.MDMMotionPS):
    if psim.TreeNode("Character State"):

        if psim.Button("Set to Zero-Pose"):
            curr_motion.char.set_to_zero_pose()
            curr_motion.char.forward_kinematics()

        #root_pos_str = "root pos: " + np.array2string(curr_motion.char.motion_frames[0, -1, 0:3].cpu().numpy())
        #psim.TextUnformatted(root_pos_str)
        char_state = curr_motion.char.motion_frames
        root_pos = char_state.root_pos[-1].cpu().numpy()
        root_pos_changed, new_root_pos = psim.InputFloat3("root pos", root_pos)
        if root_pos_changed:
            new_root_pos = torch.from_numpy(np.array(new_root_pos)).to(dtype=torch.float32, device=curr_motion.device)
            curr_motion.char.set_root_pos(new_root_pos)


        #root_rot_str = "root rot: " + np.array2string(curr_motion.char.motion_frames[0, -1, 3:6].cpu().numpy())
        #psim.TextUnformatted(root_rot_str)
        root_rot = char_state.root_rot[-1].cpu().numpy()
        root_rot_changed, new_root_rot = psim.InputFloat4("root rot", root_rot)
        if root_rot_changed:
            new_root_rot = torch.from_numpy(np.array(new_root_rot)).to(dtype=torch.float32, device=curr_motion.device)
            curr_motion.char.set_root_rot_quat(new_root_rot)

        # root_vel_str = "root vel: " + np.array2string(curr_motion.char.get_root_vel().cpu().numpy())
        # psim.TextUnformatted(root_vel_str)

        heading_str = "heading :" + str(torch_util.calc_heading(char_state.root_rot[-1]).item())
        psim.TextUnformatted(heading_str)

        if psim.TreeNode("Body Positions"):
            for b in range(curr_motion.char.char_model.get_num_joints()):
                body_pos_str = curr_motion.char.char_model.get_body_name(b) + " pos: "
                body_pos = curr_motion.char.get_body_pos(b).cpu().numpy()
                body_pos_str += np.array2string(body_pos)
                psim.TextUnformatted(body_pos_str)
            psim.TreePop()

        any_dof_changed = False
        if psim.TreeNode("Joint DOFs"):
            joint_dofs = curr_motion.char.char_model.rot_to_dof(char_state.joint_rot[-1])
            #psim.TextUnformatted(np.array2string(joint_dofs.cpu().numpy()))

            for b in range(1, curr_motion.char.char_model.get_num_joints()):
                curr_joint_name = curr_motion.char.char_model.get_body_name(b)
                curr_joint_dof = curr_motion.char.char_model.get_joint(b).get_joint_dof(joint_dofs)

                if curr_joint_dof.shape[0] == 3:
                    changed, new_joint_dof = psim.InputFloat3(curr_joint_name, curr_joint_dof.cpu().numpy(), format="%.6f")
                    if changed:
                        curr_joint_dof[0] = new_joint_dof[0]
                        curr_joint_dof[1] = new_joint_dof[1]
                        curr_joint_dof[2] = new_joint_dof[2]
                        any_dof_changed = True
                elif curr_joint_dof.shape[0] == 1:
                    changed, new_joint_dof = psim.InputFloat(curr_joint_name, curr_joint_dof[0].item(), format="%.6f")
                    if changed:
                        curr_joint_dof[0] = new_joint_dof
                        any_dof_changed = True
                else:
                    psim.TextUnformatted(curr_joint_name + ": no dofs")

            psim.TreePop()

        if psim.TreeNode("Joint Limits"):
            if psim.Button("Rectangular Projection for all dofs"):
                joint_dofs = curr_motion.char.char_model.rot_to_dof(char_state.joint_rot[-1])
                joint_dofs[:] = curr_motion.char.char_model.apply_joint_dof_limits(joint_dofs)
                any_dof_changed = True

            for b in range(1, curr_motion.char.char_model.get_num_joints()):
                curr_joint_name = curr_motion.char.char_model.get_body_name(b)
                curr_joint = curr_motion.char.char_model.get_joint(b)

                if curr_joint.joint_type == kin_char_model.JointType.HINGE: # hinge
                    joint_limit_str = "min: " + str(curr_joint.limits[0].item()) + ", max: " + str(curr_joint.limits[1].item())
                    psim.TextUnformatted(curr_joint_name + " " + joint_limit_str)
                    if psim.Button(curr_joint_name + " Rectangular Projection"):
                        joint_dofs = curr_motion.char.motion_frames[0, -1, 6:]
                        curr_joint_dof = curr_joint.get_joint_dof(joint_dofs)
                        curr_joint_dof[:] = torch.clamp(curr_joint_dof, min=curr_joint.limits[0], max=curr_joint.limits[1])
                        any_dof_changed = True

                elif curr_joint.joint_type == kin_char_model.JointType.SPHERICAL:
                    joint_limit_str_x = "x_min: " + str(curr_joint.limits[0, 0].item()) + ", x_max: " + str(curr_joint.limits[0, 1].item())
                    joint_limit_str_y = "y_min: " + str(curr_joint.limits[1, 0].item()) + ", y_max: " + str(curr_joint.limits[1, 1].item())
                    joint_limit_str_z = "z_min: " + str(curr_joint.limits[2, 0].item()) + ", z_max: " + str(curr_joint.limits[2, 1].item())

                    psim.TextUnformatted(curr_joint_name + " " + joint_limit_str_x)
                    psim.TextUnformatted(curr_joint_name + " " + joint_limit_str_y)
                    psim.TextUnformatted(curr_joint_name + " " + joint_limit_str_z)
                else:
                    psim.TextUnformatted(curr_joint_name + ": no joint limits")

            psim.TreePop()

        if root_pos_changed or root_rot_changed or any_dof_changed:
            if any_dof_changed:
                char_state.joint_rot[-1] = curr_motion.char.char_model.dof_to_rot(joint_dofs)

            curr_motion.char.forward_kinematics()


        psim.TreePop()
    return


########## MOTION EDITING GUI ##########
def motion_editor_gui():
    settings = g.MotionEditorSettings()
    main_vars = g.MainVars()

    curr_motion = g.MotionManager().get_curr_motion()
    loaded_motions = g.MotionManager().get_loaded_motions()
    terrain_manager = g.TerrainMeshManager()

    if psim.TreeNode("Browse motion files"):
        browser_state = g.MotionFileBrowserState().get_state()
        selection_changed = file_browser.file_browser_widget(
            "motion_file_browser",
            browser_state,
            file_extensions=[".pkl"],
        )
        if selection_changed and browser_state.selected_path:
            g.g_motion_filepath = browser_state.selected_path

        changed, settings.load_terrain_with_motion = psim.Checkbox("load terrain with motion", settings.load_terrain_with_motion)
        changed, settings.load_mimickit_format = psim.Checkbox("load mimickit format", settings.load_mimickit_format)
        if psim.Button("Load Motion"):
            curr_motion.set_enabled(False, False)
            motion_name = os.path.basename(g.g_motion_filepath)
            if motion_name not in loaded_motions:

                if not settings.load_mimickit_format:
                    g.MotionManager().load_motion(filepath=g.g_motion_filepath,
                                            char_model_path=g.g_char_filepath,
                                            name=motion_name,
                                            update_terrain=settings.load_terrain_with_motion)
                else:
                    motion_data = medit_lib.load_motion_file(g.g_motion_filepath, device=main_vars.device, convert_to_class=False)
                    motion_frames = motion_util.MotionFrames.from_legacy_mlib_format(torch.from_numpy(motion_data['frames']), curr_motion.char.char_model)

                    motion_frames.contacts = torch.zeros(size=[motion_frames.root_pos.shape[0], curr_motion.char.char_model.get_num_contact_bodies()], dtype=torch.float32, device=main_vars.device)
                    
                    g.MotionManager().make_new_motion(
                        motion_frames=motion_frames,
                        new_motion_name=motion_name,
                        motion_fps=motion_data['fps']
                    )
            
            g.MotionManager().set_curr_motion(motion_name)
            curr_motion = g.MotionManager().get_curr_motion()
            main_vars.motion_time = 0.0
            curr_motion.char.set_to_time(main_vars.motion_time, main_vars.motion_dt, curr_motion.mlib)
            active_terrain = terrain_manager.get_active_terrain()
            if active_terrain is not None:
                curr_motion.update_transforms(shadow_height=curr_motion.char.get_hf_below_root(active_terrain))
                curr_motion.char.update_local_hf(active_terrain)
        psim.TreePop()

    if psim.TreeNode("Loaded motions"):
        motions_to_delete = set()
        for key in loaded_motions:
            if psim.TreeNode(key):
                selected = key == curr_motion.name
                changed, selected = psim.Checkbox("selected", selected)
                if changed:
                    curr_motion.deselect()
                    g.MotionManager().set_curr_motion(key, sequence_val=main_vars.viewing_motion_sequence)
                    curr_motion.select(g.MainVars().viewing_local_hf, g.MainVars().viewing_shadow)

                visible = loaded_motions[key].sequence.mesh.is_enabled()
                changed2, visible = psim.Checkbox("visible", visible)
                if changed2:
                    loaded_motions[key].set_enabled(main_vars.viewing_motion_sequence and visible, visible)

                if psim.Button("delete motion"):
                    if curr_motion.name == key:
                        print("can't delete current motion")
                    else:
                        motions_to_delete.add(key)
                psim.TreePop()

        for motion_name in motions_to_delete:
            loaded_motions[motion_name].remove()
            del loaded_motions[motion_name]

        if psim.Button("Hide all motions"):
            for key in loaded_motions:
                loaded_motions[key].set_disabled()

        if psim.Button("See all motions"):
            for key in loaded_motions:
                loaded_motions[key].set_enabled(main_vars.viewing_motion_sequence, True)

        if psim.Button("Make new blank motion"):
            g.MotionManager().make_new_blank_motion()
        psim.TreePop()

    if curr_motion is not None:
        psim.Separator()
        psim.TextUnformatted("Selected Motion: " + str(curr_motion.name))
        psim.Separator()
        draw_char_state_ui(curr_motion)
        draw_editor_ui(curr_motion)
        draw_analytics_ui(curr_motion)

        
        MotionStitcherApp().draw_ui()

    return
