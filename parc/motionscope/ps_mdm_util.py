import math
from typing import Optional

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motion_generator.mdm as mdm
import parc.motionscope.polyscope_util as ps_util
import parc.util.geom_util as geom_util
import parc.util.motion_util as motion_util
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util


class MDMCharacterPS(ps_util.CharacterPS):
    def __init__(self, name, color, char_model: kin_char_model.KinCharModel, 
                 local_hf_num_neg_x: int = 10,
                 local_hf_num_pos_x: int = 10,
                 local_hf_num_neg_y: int = 10,
                 local_hf_num_pos_y: int = 10,
                 dx: float = 0.4,
                 dy: float = 0.4,
                 device = "cpu",
                 history_length = 1):
        super().__init__(name, color, char_model, history_length)
        
        self.reconstruct_local_hf_points(
            local_hf_num_neg_x,
            local_hf_num_pos_x,
            local_hf_num_neg_y,
            local_hf_num_pos_y,
            dx, 
            dy,
            device)
        self.device = device
        
        self.center_ind = [2, 20]
        
        self.hf_z = None
        self._hf_xyz_points = None

        self._ps_surface_point_samples = None
        self._surface_points = None

        all_points = geom_util.get_char_point_samples(char_model)
        self.init_surface_point_samples(all_points)
        return
    
    def reconstruct_local_hf_points(self, 
            local_hf_num_neg_x: int,
            local_hf_num_pos_x: int,
            local_hf_num_neg_y: int,
            local_hf_num_pos_y: int,
            dx: float, 
            dy: float,
            device
        ):
        self.dx=dx
        self.dy=dy
        self.num_neg_x = local_hf_num_neg_x
        self.num_neg_y = local_hf_num_neg_y
        self.local_hf_points = geom_util.get_xy_grid_points(
            center=torch.zeros(size=(2,), dtype=torch.float32, device=device),
            dx=dx,
            dy=dy,
            num_x_neg=local_hf_num_neg_x,
            num_x_pos=local_hf_num_pos_x,
            num_y_neg=local_hf_num_neg_y,
            num_y_pos=local_hf_num_pos_y)
        
        num_points = self.local_hf_points.shape[0] * self.local_hf_points.shape[1]
        self._xyz_points = np.zeros(shape=[num_points, 3], dtype=np.float32)
        local_hf_ps_name = self.name + " local hf"
        self._local_hf_ps = ps.register_point_cloud(local_hf_ps_name, self._xyz_points)
        self._local_hf_ps.set_radius(0.001)
        self._local_hf_ps.set_color([0.8, 0.0, 0.0])
        return
    
    def get_global_hf_xy(self, terrain: terrain_util.SubTerrain):
        assert isinstance(terrain.hf, torch.Tensor)
        heading = torch_util.calc_heading(self.get_body_rot(0))
        xy_points = torch_util.rotate_2d_vec(self.local_hf_points, heading) + self.get_body_pos(0)[0:2]

        return xy_points
    
    def update_local_hf(self, terrain: terrain_util.SubTerrain, hf_z = None):
        xy_points = self.get_global_hf_xy(terrain)

        if hf_z is None:
            self.hf_z = terrain.get_hf_val_from_points(xy_points)
        else:
            self.hf_z = hf_z

        xyz_points = torch.cat([xy_points, self.hf_z.unsqueeze(-1)], dim=-1).view(-1, 3)
        self._xyz_points = xyz_points.cpu().numpy()

        self._local_hf_ps.update_point_positions(self._xyz_points)
        N, M = self.hf_z.shape
    
        # Create range of indices for rows and columns
        row_indices = torch.arange(N)
        col_indices = torch.arange(M)
        
        # Create a meshgrid of row and column indices
        rows, cols = torch.meshgrid(row_indices, col_indices, indexing='ij')

        self._local_hf_ps.add_scalar_quantity("i", values=rows.reshape(-1))
        self._local_hf_ps.add_scalar_quantity("j", values=cols.reshape(-1))
        return
    
    def set_local_hf_transparency(self, val):
        self._local_hf_ps.set_transparency(val)
        return
    
    def set_local_hf_enabled(self, val):
        self._local_hf_ps.set_enabled(val)
        return
    
    def get_hf_below_root(self, terrain: terrain_util.SubTerrain):
        ind = terrain.get_grid_index(self.get_body_pos(0)[0:2])
        z = terrain.hf[ind[0], ind[1]].item()
        return z
    
    def remove(self):
        super().remove()
        self._local_hf_ps.remove()
        return
    
    def deselect(self):
        self._local_hf_ps.set_enabled(False)
        return
    
    def select(self, viewing_local_hf, viewing_shadow):
        self._local_hf_ps.set_enabled(viewing_local_hf)
        self.set_shadow_enabled(viewing_shadow)
        return
    
    def has_surface_point_samples(self):
        return self._surface_points is not None
    
    def get_surface_point_samples(self, device=None):
        if device is None:
            return self._surface_points
        else:
            ret_body_points = []
            for b in range(len(self._surface_points)):
                ret_body_points.append(self._surface_points[b].to(device=device))
            return ret_body_points
        
    def get_transformed_surface_point_samples(self):
        transformed_body_points = []
        body_point_slices = []
        num_bodies = self.char_model.get_num_joints()
        num_points = 0
        for b in range(num_bodies):
            curr_surface_points = self._surface_points[b]
            curr_body_rot = self.motion_frames.body_rot[-1][b].unsqueeze(0)
            curr_body_pos = self.motion_frames.body_pos[-1][b]

            curr_surface_points = torch_util.quat_rotate(curr_body_rot, curr_surface_points) + curr_body_pos
            transformed_body_points.append(curr_surface_points)

            body_point_slices.append(slice(num_points, num_points + curr_surface_points.shape[0]))
            num_points += curr_surface_points.shape[0]

        transformed_surface_points = torch.cat(transformed_surface_points, dim=0)
        return transformed_surface_points, body_point_slices
        
    def init_surface_point_samples(self, all_points):
        num_bodies = len(all_points)
        self._surface_points = all_points
        assert num_bodies == self.char_model.get_num_joints()
        self._ps_surface_point_samples = []
        for b in range(num_bodies):
            name = self.name + self.char_model.get_body_name(b) + " point samples"
            ps_point_cloud = ps.register_point_cloud(name, all_points[b].cpu().numpy(), radius = 0.0005,
                                                     color = self.color,
                                                     enabled=False)
            self._ps_surface_point_samples.append(ps_point_cloud)

        self._surface_points_enabled = False
        return
    
    def update_surface_point_samples(self):
        # assuming forward kinematics is already called

        for b in range(self.char_model.get_num_joints()):
            body_pos = self.motion_frames.body_pos[-1][b]
            body_rot = self.motion_frames.body_rot[-1][b]

            pose = np.eye(4)
            pose[:3, 3] = body_pos.cpu().numpy()

            rot_mat = torch_util.quat_to_matrix(body_rot).cpu().numpy()
            pose[:3, :3] = rot_mat

            self._ps_surface_point_samples[b].set_transform(pose)
        return
    
    def set_surface_point_colors(self, b, colors):
        self._ps_surface_point_samples[b].add_color_quantity("sdf < 0", colors.cpu().numpy(), enabled=True)
        return

    def update_transforms(self, shadow_height=0):
        super().update_transforms(shadow_height)
        if self._ps_surface_point_samples is not None:
            self.update_surface_point_samples()
        return
    
    def get_surface_points_enabled(self):
        return self._surface_points_enabled
    
    def set_surface_points_enabled(self, val):
        for b in range(len(self._ps_surface_point_samples)):
            self._ps_surface_point_samples[b].set_enabled(val)
        self._surface_points_enabled = val
        return
    
    def get_normalized_local_hf(self, max_h: float):
        hf_z = self.hf_z
        hf_z = hf_z - self.get_body_pos(0)[2].item() # relative to root
        hf_z = torch.clamp(hf_z, min=-max_h, max=max_h) / max_h
        return hf_z
    
    def set_enabled(self, val):
        super().set_enabled(val)
        self.set_local_hf_enabled(val)
        return



class MDMMotionPS(ps_util.MotionPS):
    def __init__(self, name, mlib: motion_lib.MotionLib, char_color,
                 vis_fps: int = 15, start_time: float = 0.0, history_length=2, view_body_traj=False,
                 mdm_model: Optional[mdm.MDM] = None, active_terrain: Optional[terrain_util.SubTerrain] = None):
        self.name = name
        self.char_color = char_color
        self.mlib = mlib
        self.device = mlib._device

        if mdm_model is not None:
            self.char = MDMCharacterPS(
                name=name,
                color=char_color,
                char_model=mlib._kin_char_model,
                local_hf_num_neg_x=mdm_model._num_x_neg,
                local_hf_num_pos_x=mdm_model._num_x_pos,
                local_hf_num_neg_y=mdm_model._num_y_neg,
                local_hf_num_pos_y=mdm_model._num_y_pos,
                dx=mdm_model._dx,
                dy=mdm_model._dy,
                device=self.device,
                history_length=history_length)
        else:
            self.char = MDMCharacterPS(
                name=name,
                color=char_color, 
                char_model=mlib._kin_char_model,
                device=self.device,
                history_length=history_length)


        # have a character frame every 2/30 seconds
        motion_length = mlib._motion_lengths[0].item()
        #print("MOTION LENGTH:", motion_length)

        
        if motion_length > 7.0:
           vis_fps = 1
        num_frames = int(round(vis_fps * motion_length))
        if num_frames < 2:
            num_frames = 2

        if vis_fps == -1:
            num_frames = mlib._motion_frames.shape[0]
        self.sequence = ps_util.MotionSequencePS(name + " motion sequence", char_color, [0.0, 0.0, 0.0],
                                                 num_frames, 0.0, motion_length, self.mlib)

        self.root_offset = np.array([0.0, 0.0, 0.0])
        self.root_heading_angle = 0.0
        self.start_retarget_time = 0.0
        self.end_retarget_time = self.mlib._motion_lengths[0].item()

        self.editing_full_sequence = True
        self.medit_start_frame = 0
        self.medit_end_frame = 5
        self.medit_scale = 0.9
        self.speed_scale = 0.8

        self.body_traj_speed_vmin = 0.0
        self.body_traj_speed_vmax = 2.0
        self.body_traj_acc_vmin = 0.0
        self.body_traj_acc_vmax = 100.0
        self.body_traj_jerk_vmin = 0.0
        self.body_traj_jerk_vmax = 1000.0

        self._build_ps_body_traj(view_body_traj)

        self.set_to_time(start_time)

        if active_terrain is not None:
            self.update_transforms(shadow_height=self.char.get_hf_below_root(active_terrain))
            #     num_vis_frames = int(round(vis_fps * new_mlib._motion_lengths[0].item()))
            #     MotionManager().get_curr_motion().update_sequence(0.0, new_mlib._motion_lengths[0].item(), num_vis_frames)
            self.char.update_local_hf(active_terrain)
        return
    
    def _build_ps_body_traj(self, view_body_traj):
        self.ps_body_traj = []
        self.body_traj_names = []

        fps = self.mlib.get_motion_fps(0)
        body_pos = self.mlib.get_frames_for_id(0, compute_fk=True).body_pos

        body_vel = torch.zeros_like(body_pos)
        body_vel[1:] = (body_pos[1:] - body_pos[:-1]) * fps

        body_acc = torch.zeros_like(body_pos)
        body_acc[2:] = (body_vel[2:] - body_vel[1:-1]) * fps

        body_jerk = torch.zeros_like(body_pos)
        body_jerk[3:] = (body_acc[3:] - body_acc[2:-1]) * fps

        self.body_speed = torch.norm(body_vel, dim=-1)
        self.body_acc_mag = torch.norm(body_acc, dim=-1)
        self.body_jerk_mag = torch.norm(body_jerk, dim=-1)

        self.ps_body_traj_quantities = {
            "speed": self.body_speed,
            "acc": self.body_acc_mag,
            "jerk": self.body_jerk_mag
        }

        speed_range = self._compute_scalar_range(self.body_speed)
        acc_range = self._compute_scalar_range(self.body_acc_mag)
        jerk_range = self._compute_scalar_range(self.body_jerk_mag)

        self.body_traj_value_ranges = {
            "speed": speed_range,
            "acc": acc_range,
            "jerk": jerk_range,
        }

        if view_body_traj:
            for b in range(self.mlib._kin_char_model.get_num_bodies()):
                body_name = self.mlib._kin_char_model.get_body_name(b)
                nodes = body_pos[:, b].cpu().numpy()
                ps_body_traj = ps.register_curve_network(name=self.name + "_" + body_name+"_traj", nodes=nodes, edges='line', enabled=True, radius=0.0005)

                ps_body_traj.add_scalar_quantity(
                    name="speed",
                    values=self.body_speed[:, b].cpu().numpy(),
                    enabled=True,
                    cmap='reds',
                    vminmax=speed_range,
                )
                ps_body_traj.add_scalar_quantity(
                    name="acc",
                    values=self.body_acc_mag[:, b].cpu().numpy(),
                    enabled=False,
                    cmap='reds',
                    vminmax=acc_range,
                )
                ps_body_traj.add_scalar_quantity(
                    name="jerk",
                    values=self.body_jerk_mag[:, b].cpu().numpy(),
                    enabled=False,
                    cmap='reds',
                    vminmax=jerk_range,
                )

                self.ps_body_traj.append(ps_body_traj)
                self.body_traj_names.append(body_name)

            self.set_body_traj_map_range("speed", self.body_traj_speed_vmin, self.body_traj_speed_vmax)
            self.set_body_traj_map_range("acc", self.body_traj_acc_vmin, self.body_traj_acc_vmax)
            self.set_body_traj_map_range("jerk", self.body_traj_jerk_vmin, self.body_traj_jerk_vmax)

        self._all_traj_enabled = True
        return

    @staticmethod
    def _compute_scalar_range(values: torch.Tensor):
        if values.numel() == 0:
            return (0.0, 1.0)

        vmin = float(torch.min(values).item())
        vmax = float(torch.max(values).item())

        if math.isclose(vmin, vmax):
            delta = 1.0 if math.isclose(vmin, 0.0) else max(abs(vmin) * 0.1, 1e-3)
            vmax = vmin + delta

        return (vmin, vmax)

    def set_body_traj_map_range(self, quantity_name: str, vmin: float, vmax: float):
        if not hasattr(self, "ps_body_traj_quantities"):
            return
        
        assert quantity_name in self.ps_body_traj_quantities

        vminmax = [float(vmin), float(vmax)]
        for b in range(len(self.ps_body_traj)):
            self.ps_body_traj[b].add_scalar_quantity(quantity_name, self.ps_body_traj_quantities[quantity_name][:,b], vminmax=vminmax)
        self.body_traj_value_ranges[quantity_name] = vminmax
        return

    def get_body_traj_quantity_range(self, quantity_name: str):
        if hasattr(self, "body_traj_value_ranges"):
            return self.body_traj_value_ranges.get(quantity_name, (0.0, 1.0))
        return (0.0, 1.0)
    
    def enable_traj_quantity(self, quantity_name: str):
        for b in range(len(self.ps_body_traj)):
            ps_body_traj = self.ps_body_traj[b]
            ps_body_traj.add_scalar_quantity(
                name=quantity_name,
                values=self.ps_body_traj_quantities[quantity_name][:, b].cpu().numpy(),
                enabled=True,
                cmap='reds',
                vminmax=self.body_traj_value_ranges[quantity_name],
            )
        return
    
    def set_all_traj_enabled(self, val):
        self._all_traj_enabled = val
        for traj in self.ps_body_traj:
            traj.set_enabled(val)
        return

    def remove(self):
        self.char.remove()
        self.sequence.remove()
        for traj in getattr(self, "ps_body_traj", []):
            traj.remove()
        return
    
    def deselect(self):
        self.char.deselect()
        return
    
    def select(self, viewing_local_hf, viewing_shadow):
        self.char.select(viewing_local_hf, viewing_shadow)
        return
    
    def set_motion_sequence_colors(self, frame_colors, num_frames):
        motion_length = self.mlib._motion_lengths[0].item()
        self.sequence = ps_util.MotionSequencePS(self.name + " motion sequence", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 
                                                 num_frames, 0.0, motion_length, self.mlib,
                                                 frame_colors=frame_colors)
        
        return
    
    def set_disabled(self):
        self.char.set_local_hf_enabled(False)
        self.set_all_traj_enabled(False)
        return super().set_disabled()
    
    def only_view_char(self):
        self.char.set_enabled(True)
        self.sequence.set_enabled(False)
        self.char.set_local_hf_enabled(False)
        self.char.set_prev_state_enabled(False)
        self.char.set_shadow_enabled(False)
        self.char.set_body_points_enabled(False)
        self.set_all_traj_enabled(False)
        return

    