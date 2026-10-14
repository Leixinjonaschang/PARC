import math
import os
import platform

import cpuinfo
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

if platform.system() == "Linux":
    import parc.motion_tracker.envs.env_builder as env_builder
    import parc.motion_tracker.learning.agent_builder as agent_builder

import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import trimesh
import yaml

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.motion_generator.gen_util as gen_util
import parc.motion_generator.mdm as mdm
import parc.motionscope.polyscope_util as ps_util
import parc.motionscope.ps_mdm_util as ps_mdm_util
import parc.motion_synthesis.procgen.astar as astar
import parc.motion_synthesis.procgen.mdm_path as mdm_path
import parc.util.file_io as file_io
import parc.util.geom_util as geom_util
import parc.util.misc_util as misc_util
import parc.util.motion_edit_lib as medit_lib
import parc.util.motion_util as motion_util
import parc.util.path_loader as path_loader
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util
from parc.motionscope.include.file_browser import FileBrowserState
from parc.motionscope.include.singleton import SingletonClass

try:
    config_path = Path(__file__).resolve().parents[1] / "motionscope_config.yaml"
    input_config = path_loader.load_config(config_path)

    def _resolve_char_path(char_identifier: str) -> str:
        if char_identifier == "humanoid":
            return "data/assets/humanoid.xml"
        return char_identifier

    def _parse_motion_entry(entry):
        if not isinstance(entry, (list, tuple)):
            raise ValueError(f"Motion entry must be a list or tuple, got {type(entry)}")
        if len(entry) < 2:
            raise ValueError("Motion entry must contain at least motion and character identifiers")

        motion_path = str(entry[0])
        char_identifier = str(entry[1])
        return motion_path, _resolve_char_path(char_identifier)

    raw_motion_entries = input_config["motions"]
    g_motion_filepaths = [_parse_motion_entry(entry) for entry in raw_motion_entries]

    if len(g_motion_filepaths) == 0:
        raise ValueError("At least one motion must be specified in motionscope_config.yaml")

    g_motion_browser_path, g_char_filepath = g_motion_filepaths[0]

    g_use_legacy_file_format = input_config["use_legacy_file_format"]

    LOAD_MDM = input_config["load_mdm"]

    g_mdm_filepaths = input_config["mdm_filepaths"]

    g_retargeting_config_path = input_config.get("retargeting_config_path")
    g_retargeting_src_motion_path = input_config.get("retargeting_src_motion_path")
except Exception as exc:
    assert False, f"Failed to load MotionScope config: {exc}"



class MainVars(SingletonClass):
    device = "cpu"
    gpu_device = "cuda:0" # TODO: make all models use this device!!
    selected_grid_ind = torch.tensor([0, 0], dtype=torch.int64, device=device)
    mouse_world_pos = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    paused = False
    max_play_time = 99999.0
    looping = True
    motion_time = 0.0
    use_contact_info = input_config["use_contact_info"]
    local_hf_visibility = 1.0
    viewing_local_hf = True
    viewing_prev_state = False
    viewing_char = True
    viewing_motion_sequence = True
    viewing_shadow = True
    using_sliders = True
    seq_vis_fps = 15
    mouse_size = 0
    mouse_ball_visible = True
    loaded_target_xy = False#"target_xy" in motion_data._data
    motion_dt = 1.0 / 30.0
    curr_time = 0.0
    curr_dt = 1.0 / 30.0
    visualize_body_points = False

    save_path_with_motion = False
    save_motion_as_loop = False
    save_terrain_with_motion = True
    save_misc_data_with_motion = True

    time_gui_opened = True
    motion_gui_opened = False
    contact_gui_opened = False
    terrain_gui_opened = False
    mdm_gui_opened = False
    optimization_gui_opened = False
    path_planning_gui_opened = False
    recording_gui_opened = False
    ig_obs_gui_opened = False
    isaac_gym_gui_opened = False
    sampler_gui_opened = False
    kin_controller_gui_opened = False
    help_gui_opened = False

    saved_cam_params = None
    save_filepath = "output/_motions/new_motion.pkl"

    def set_motion_time(self, val):
        assert isinstance(val, float)
        self.motion_time = val
        return

    def update_time(self):
        next_curr_time = time.time()
        self.curr_dt = next_curr_time - self.curr_time
        self.curr_time = next_curr_time
        return

## MOTION EDITING GLOBALS ##
class MotionEditorSettings(SingletonClass):
    num_blend_frames = 5
    load_terrain_with_motion = True
    load_mimickit_format = False

class MotionFileBrowserState(SingletonClass):
    """Persistent state for the motion file browser widget."""

    def __init__(self) -> None:
        if hasattr(self, "browser_state"):
            return

        start_dir = os.path.dirname(g_motion_browser_path)
        selected_path = g_motion_browser_path
        self.browser_state = FileBrowserState(current_dir=start_dir, selected_path=selected_path)

    def get_state(self) -> FileBrowserState:
        return self.browser_state
    
    def set_state(self, path):
        dirname = os.path.dirname(path)
        self.browser_state = FileBrowserState(current_dir=dirname, selected_path=path)
        return

## TERRAIN EDITING GLOBALS ##
class TerrainEditorSettings(SingletonClass):
    dx = 0.1
    height = 0.6
    edit_modes = ["heightfield", "max", "min"]
    curr_edit_mode = 0
    mask_mode = True
    viewing_terrain = True
    viewing_max = False
    viewing_min = False
    terrain_padding = 0.4 # meters
    num_boxes = 10
    max_num_boxes = 8
    min_num_boxes = 2
    box_max_len = 10
    box_min_len = 5
    maxpool_size = 1
    box_heights = 0.6
    max_box_angle = 0.0 #2.0 * torch.pi
    min_box_angle = 0.0
    min_box_h = -2.0


    max_box_h = 2.0
    use_maxmin = False
    slice_min_i = 0
    slice_min_j = 0
    slice_max_i = 0
    slice_max_j = 0
    new_terrain_dim_x = 16
    new_terrain_dim_y = 16
    num_terrain_paths = 4
    path_min_height = -2.8
    path_max_height = 3.0
    floor_height = -3.0

    min_stair_start_height = -3.0
    max_stair_start_height = 1.0
    min_step_height = 0.15
    max_step_height = 0.25
    num_stairs = 4
    min_stair_thickness = 2.0
    max_stair_thickness = 8.0

class PathPlanningSettings(SingletonClass):
    waypoints = []
    waypoints_ps = []
    manual_placement_mode = False

    extend_path_mode = False

    astar_settings = astar.AStarSettings()

    path_nodes = None
    path_nodes_ps = []

    mdm_path_settings = mdm_path.MDMPathSettings()

    use_prev_frames = False

    curviness = 0.5

    nav_graph = None

    def clear_waypoints(self):
        for waypoint_ps in self.waypoints_ps:
            waypoint_ps.remove()

        self.waypoints_ps.clear()
        self.waypoints.clear()
        return

    def place_waypoint(self, node: torch.Tensor, color = None):
        terrain = TerrainMeshManager().get_active_terrain(require=True)

        z = terrain.get_hf_val(node)

        world_pos = terrain.get_point(node)
        world_pos = torch.cat([world_pos, z.unsqueeze(0)]).cpu().numpy()

        sphere_trimesh = trimesh.primitives.Sphere(0.1)

        if color is None:
            color = [0.1, 1.0, 0.1]

        name = "PathWaypoint" + str(len(self.waypoints))
        sphere_ps_mesh = ps.register_surface_mesh(name, sphere_trimesh.vertices, sphere_trimesh.faces)
        sphere_ps_mesh.set_color(color)
        transform = np.eye(4)
        transform[:3, 3] = world_pos
        sphere_ps_mesh.set_transform(transform)

        self.waypoints.append(node.cpu().numpy())
        self.waypoints_ps.append(sphere_ps_mesh)
        return
    
    def visualize_path_nodes(self, ps_name: str, nodes_3dpos: torch.Tensor):
        edges = []
        for i in range(nodes_3dpos.shape[0] - 1):
            edges.append([i, i+1])
        edges = np.array(edges)
        ps_path_node = ps.register_curve_network(ps_name, nodes=nodes_3dpos.cpu().numpy(), edges=edges, 
                                    enabled=True, color=[1.0, 0.0, 0.0], radius=0.0014)
        self.path_nodes_ps.append(ps_path_node)
        return
    
    def clear_path_nodes(self):
        for garbage in self.path_nodes_ps:
            garbage.remove()
        self.path_nodes_ps.clear()

        self.path_nodes = None
        return
    
    def run_astar(self):
        terrain = TerrainMeshManager().get_active_terrain(require=True)
        if len(self.waypoints) >= 2:
            path_vis_name="ASTARpath"

            for i in range(len(self.waypoints) - 1):
                start_node = self.waypoints[i]
                end_node = self.waypoints[i+1]
                path_nodes = astar.run_a_star_on_start_end_nodes(terrain = terrain,
                                                                start_node = start_node,
                                                                end_node = end_node,
                                                                settings = self.astar_settings,
                                                                nav_graph = self.nav_graph)
                if path_nodes is False:
                    print("no path found")
                else:
                    if self.extend_path_mode and self.path_nodes is not None:
                        self.path_nodes = torch.cat([self.path_nodes, path_nodes[1:]], dim=0)
                    else:
                        self.path_nodes = path_nodes
                    self.visualize_path_nodes(path_vis_name, self.path_nodes)
        else:
            print("Not enough waypoints")

# MDM
# note: we can load these from the mdm using mdm._cfg["heightmap"]["local_grid"]
g_mdm_model = None
class MDMSettings(SingletonClass):
    loaded_mdm_models = dict()
    current_mdm_key = "main"
    
    append_mdm_motion_to_prev_motion = False
    mdm_batch_size = 1
    local_hf_num_neg_x = 8
    local_hf_num_pos_x = 8
    local_hf_num_neg_y = 8
    local_hf_num_pos_y = 8
    sample_prev_states_only = False
    hide_batch_motions = True

    conv_layer_num = 0
    gen_settings = gen_util.MDMGenSettings()

    def select_mdm_helper(self, key) -> mdm.MDM:
        return self.loaded_mdm_models[key]

    def select_mdm(self, key):
        global g_mdm_model
        self.current_mdm_key = key
        g_mdm_model = self.select_mdm_helper(key)
        MDMSettings().local_hf_num_neg_x = g_mdm_model._num_x_neg
        MDMSettings().local_hf_num_pos_x = g_mdm_model._num_x_pos
        MDMSettings().local_hf_num_neg_y = g_mdm_model._num_y_neg
        MDMSettings().local_hf_num_pos_y = g_mdm_model._num_y_pos

        # TODO: reconstruct char local hf
        MotionManager().get_curr_motion().char.reconstruct_local_hf_points(
            local_hf_num_neg_x=g_mdm_model._num_x_neg,
            local_hf_num_pos_x=g_mdm_model._num_x_pos,
            local_hf_num_neg_y=g_mdm_model._num_y_neg,
            local_hf_num_pos_y=g_mdm_model._num_y_pos,
            dx = g_mdm_model._dx, 
            dy = g_mdm_model._dy,
            device = MainVars().device)

        MainVars().motion_dt = 1.0 / g_mdm_model._sequence_fps
        active_terrain = TerrainMeshManager().get_active_terrain(require=True)
        MotionManager().get_curr_motion().char.update_local_hf(active_terrain)
        return

## DEFAULT SETTINGS
MDMSettings().gen_settings.use_ddim = True
MDMSettings().gen_settings.use_cfg = False

class MDMSamplerSettings(SingletonClass):
    sampler = None

class ContactEditingSettings(SingletonClass):
    contact_eps = 1e-2
    selected_body_id = 0
    start_frame_idx=0
    end_frame_idx=0

class OptimizationSettings(SingletonClass):
    step_size = 1e-3
    num_iters = 1000
    w_root_pos = 1.0
    w_root_rot = 10.0
    w_joint_rot = 1.0
    w_smoothness = 10.0
    w_penetration = 1000.0
    w_contact = 1000.0
    w_sliding = 10.0
    w_body_constraints = 1000.0
    w_jerk = 100.0
    max_jerk = 1000.0
    body_constraints = None
    body_constraints_ps_meshes = OrderedDict()
    use_wandb = True
    auto_compute_body_constraints = False
    visualize_optimization = False
    opt_vis_meshes = []
    opt_vis_frame_idxs = []

    # TODO: separate body constraint dict for different motions
    def create_body_constraint_ps_mesh(self, body_id, start_frame_idx, end_frame_idx, position, 
                                       char_model: kin_char_model.KinCharModel):
        sphere_trimesh = trimesh.primitives.Sphere(0.025)

        body_name = char_model.get_body_name(body_id)
        name_str = body_name + ":" + str(start_frame_idx) + "->" + str(end_frame_idx) 
        sphere_ps_mesh = ps.register_surface_mesh(name_str, sphere_trimesh.vertices, sphere_trimesh.faces)
        sphere_ps_mesh.set_color([0.0, 0.0, 1.0])
        transform = np.eye(4)
        transform[:3, 3] = position
        sphere_ps_mesh.set_transform(transform)
        self.body_constraints_ps_meshes[name_str] = sphere_ps_mesh
        return
    
    def create_body_constraint_ps_meshes(self):
        if self.body_constraints is not None:
            for b in range(len(self.body_constraints)):
                for c_idx in range(len(self.body_constraints[b])):
                    curr_body_constraint = self.body_constraints[b][c_idx]
                    self.create_body_constraint_ps_mesh(b, curr_body_constraint.start_frame_idx, 
                                                        curr_body_constraint.end_frame_idx,
                                                        curr_body_constraint.constraint_point.cpu(),
                                                        MotionManager().get_curr_motion().char.char_model)
        return
    
    def clear_body_constraints(self):
        if self.body_constraints is not None:
            for b in range(len(self.body_constraints)):
                self.body_constraints[b] = []
            
            for key, ps_mesh in self.body_constraints_ps_meshes.items():
                ps_mesh.remove()
            self.body_constraints_ps_meshes.clear()

    def clear_opt_vis_meshes(self):
        for mesh_set in self.opt_vis_meshes:
            for mesh in mesh_set:
                mesh.remove()
        self.opt_vis_meshes = []
        self.opt_vis_frame_idxs = []

class IGObsSettings(SingletonClass):
    has_obs = False
    overlay_obs_on_motion = False
    view_tar_obs = True
    view_key_points = True
    record_obs = False

    def setup(self, obs, obs_shapes):
        self.has_obs = True
        self.obs = obs
        self.obs_shapes = obs_shapes

        self.char_obs, self.tar_obs, self.tar_contacts, self.char_contacts, self.hf_obs_points = extract_obs(self.obs, self.obs_shapes)

        ps.register_point_cloud("hf_obs_points", self.hf_obs_points[0], radius=0.001)

        self.hf_obs_points = torch.from_numpy(self.hf_obs_points).to(dtype=torch.float32, device=MainVars().device)

        self.root_pos_tar_obs, self.root_rot_tar_obs, self.joint_rot_tar_obs, self.key_pos_tar_obs = misc_util.inverse_tar_obs(torch.from_numpy(self.tar_obs))

        self.num_tar_obs = 6 # TODO dont hardcode
        self.ps_tar_char_meshes = []
        self.tar_char_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for i in range(self.num_tar_obs):
            ps_char_meshes = ps_util.create_char_mesh("tar_obs_" + str(i), color=self.tar_char_color, transparency=0.3, char_model=MotionManager().get_curr_motion().mlib._kin_char_model)
            self.ps_tar_char_meshes.append(ps_char_meshes)

        num_frames = self.key_pos_tar_obs.shape[0]
        self.key_pos_tar_obs = self.key_pos_tar_obs.view(num_frames, -1, 3)
        self.ps_tar_key_pos_pc = ps.register_point_cloud("tar_obs_key_pos", points = self.key_pos_tar_obs[0].numpy(),
                                                            color=[0.0, 0.0, 1.0], transparency=0.6, radius=0.003)


        self.proprio_char_obs = misc_util.extract_obs(torch.from_numpy(self.char_obs), False)
        self.proprio_char_color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        self.ps_proprio_char_mesh = ps_util.create_char_mesh("char_obs" + str(i), color=self.proprio_char_color, transparency=1.0, char_model=MotionManager().get_curr_motion().mlib._kin_char_model)
        self.proprio_key_pos_obs = self.proprio_char_obs["key_pos"].view(num_frames, -1, 3)
        self.ps_proprio_key_pos_pc = ps.register_point_cloud("proprio_key_pos", points=self.proprio_key_pos_obs[0].numpy(), 
                                                             color=[0.0, 0.7, 1.0], transparency=0.8, radius=0.004)
        return
    
    def SetViewTarObs(self, val):
        for i in range(self.num_tar_obs):
            for j in range(len(self.ps_tar_char_meshes[i])):
                self.ps_tar_char_meshes[i][j].set_enabled(val)
        self.ps_tar_key_pos_pc.set_enabled(val)
        return

ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("none")
ps.set_background_color([0.0, 0.0, 0.0])
ps.set_program_name("MotionScope")
ps.init()
g_ps_origin_axes = ps_util.create_origin_axis_mesh()

## LOAD MDM
def load_mdm(mdm_path) -> mdm.MDM:
    if LOAD_MDM:
        resolved_path = path_loader.resolve_path(mdm_path)
        if not resolved_path.exists():
            assert False, f"MDM path does not exist: {resolved_path}"

        if resolved_path.suffix == ".pkl":
            ret_mdm = pickle.load(resolved_path.open("rb"))
        else:
            ret_mdm = mdm.MDM.load_checkpoint(str(resolved_path), device=MainVars().gpu_device)
        if ret_mdm.use_ema:
            print('Using EMA model...')
            ret_mdm._denoise_model = ret_mdm._ema_denoise_model
        print("MDM uses heightmap: ", ret_mdm._use_heightmap_obs)
        print("MDM uses target: ", ret_mdm._use_target_obs)
        return ret_mdm
    else:
        return None

for key in g_mdm_filepaths:
    MDMSettings().loaded_mdm_models[key] = load_mdm(g_mdm_filepaths[key])

@dataclass
class TerrainMeshEntry:
    name: str
    terrain: Optional["terrain_util.SubTerrain"] = None
    verts: Optional[np.ndarray] = None
    tris: Optional[np.ndarray] = None
    hf_ps_mesh: Optional[ps.SurfaceMesh] = None
    hf_max_mesh: Optional[ps.SurfaceMesh] = None
    hf_min_mesh: Optional[ps.SurfaceMesh] = None
    hf_embree: Optional[object] = None

    def remove_registered_meshes(self):
        for mesh in (self.hf_ps_mesh, self.hf_max_mesh, self.hf_min_mesh):
            if mesh is not None:
                try:
                    ps.remove_surface_mesh(mesh.get_name())
                except Exception:
                    pass
        self.hf_ps_mesh = None
        self.hf_max_mesh = None
        self.hf_min_mesh = None

    def clear_cached_mesh_data(self):
        self.verts = None
        self.tris = None
        self.hf_embree = None

    def reset_mesh_state(self):
        self.remove_registered_meshes()
        self.clear_cached_mesh_data()


class TerrainMeshManager(SingletonClass):

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._terrains: Dict[str, TerrainMeshEntry] = {}
        self._active_name: Optional[str] = None
        self._initialized = True

    def _make_entry(self, name: str) -> TerrainMeshEntry:
        return TerrainMeshEntry(name=name)

    def _get_entry(self, name: Optional[str] = None, create: bool = True) -> Optional[TerrainMeshEntry]:
        if name is None:
            name = self._active_name
            if name is None:
                if not create:
                    return None
                name = "default"
                self._active_name = name
        if name not in self._terrains:
            if not create:
                return None
            self._terrains[name] = self._make_entry(name)
        return self._terrains[name]

    def _mesh_name(self, entry: TerrainMeshEntry, base_name: str) -> str:
        if entry.name == "default":
            return base_name
        return f"{base_name}::{entry.name}"

    def get_registered_terrain_names(self):
        return list(self._terrains.keys())

    def get_active_terrain_name(self) -> Optional[str]:
        return self._active_name

    def set_active_terrain(self, terrain_name: str, rebuild: bool = False):
        entry = self._get_entry(terrain_name, create=False)
        if entry is None:
            raise KeyError(f"No terrain registered with name '{terrain_name}'")
        self._active_name = terrain_name
        if rebuild:
            self.reset(terrain_name=terrain_name)
        return entry

    def register_terrain(self, terrain_name: str, terrain: "terrain_util.SubTerrain",
                         make_active: bool = True, rebuild: bool = True):
        entry = self._get_entry(terrain_name)
        entry.reset_mesh_state()
        entry.terrain = terrain
        if make_active:
            self._active_name = terrain_name
        if rebuild:
            self.reset(terrain_name=terrain_name)
        return entry

    def update_active_terrain(self, terrain: "terrain_util.SubTerrain", terrain_name: Optional[str] = None,
                              rebuild: bool = True):
        if terrain_name is not None:
            entry = self._get_entry(terrain_name)
            self._active_name = terrain_name
        else:
            entry = self._get_entry()
        entry.reset_mesh_state()
        entry.terrain = terrain
        if rebuild:
            self.reset(terrain_name=entry.name)
        return entry

    def get_active_entry(self) -> Optional[TerrainMeshEntry]:
        return self._get_entry(create=False)

    def require_active_entry(self) -> TerrainMeshEntry:
        entry = self.get_active_entry()
        if entry is None or entry.terrain is None:
            raise ValueError("No active terrain has been registered")
        return entry

    def get_entry(self, terrain_name: Optional[str] = None) -> Optional[TerrainMeshEntry]:
        """Return the terrain entry for the given name, if it exists."""
        return self._get_entry(terrain_name, create=False)

    def get_active_terrain(self, terrain_name: Optional[str] = None, require: bool = False):
        entry = self._get_entry(terrain_name, create=False)
        terrain = None if entry is None else entry.terrain
        if terrain is None and require:
            raise ValueError("No terrain available for the requested entry")
        return terrain

    @property
    def verts(self) -> Optional[np.ndarray]:
        entry = self._get_entry(create=False)
        return None if entry is None else entry.verts

    @verts.setter
    def verts(self, value: np.ndarray):
        entry = self._get_entry()
        entry.verts = value

    @property
    def tris(self) -> Optional[np.ndarray]:
        entry = self._get_entry(create=False)
        return None if entry is None else entry.tris

    @tris.setter
    def tris(self, value: np.ndarray):
        entry = self._get_entry()
        entry.tris = value

    @property
    def hf_ps_mesh(self):
        entry = self._get_entry(create=False)
        return None if entry is None else entry.hf_ps_mesh

    @hf_ps_mesh.setter
    def hf_ps_mesh(self, value):
        entry = self._get_entry()
        entry.hf_ps_mesh = value

    @property
    def hf_max_mesh(self):
        entry = self._get_entry(create=False)
        return None if entry is None else entry.hf_max_mesh

    @hf_max_mesh.setter
    def hf_max_mesh(self, value):
        entry = self._get_entry()
        entry.hf_max_mesh = value

    @property
    def hf_min_mesh(self):
        entry = self._get_entry(create=False)
        return None if entry is None else entry.hf_min_mesh

    @hf_min_mesh.setter
    def hf_min_mesh(self, value):
        entry = self._get_entry()
        entry.hf_min_mesh = value

    @property
    def hf_embree(self):
        entry = self._get_entry(create=False)
        return None if entry is None else entry.hf_embree

    @hf_embree.setter
    def hf_embree(self, value):
        entry = self._get_entry()
        entry.hf_embree = value

    def get_trimesh(self, terrain_name: Optional[str] = None):
        entry = self._get_entry(terrain_name, create=False)
        if entry is None or entry.verts is None or entry.tris is None:
            raise ValueError("Terrain mesh has not been built for the requested terrain")
        return trimesh.Trimesh(vertices=entry.verts, faces=entry.tris)

    def update_vertex_positions(self, verts, terrain_name: Optional[str] = None):
        entry = self._get_entry(terrain_name, create=False)
        if entry is None or entry.hf_ps_mesh is None:
            raise ValueError("Terrain mesh has not been built for the requested terrain")
        entry.verts = verts
        entry.hf_ps_mesh.update_vertex_positions(vertices=verts)
        return

    def reset(self, terrain_name: Optional[str] = None, terrain: Optional["terrain_util.SubTerrain"] = None):
        entry = self._get_entry(terrain_name)
        if terrain is not None:
            entry.terrain = terrain

        self.build_ps_hf_mesh(terrain_name=entry.name)
        if entry.hf_ps_mesh is not None:
            entry.hf_ps_mesh.set_enabled(TerrainEditorSettings().viewing_terrain)

        self.build_ps_max_mesh(terrain_name=entry.name)
        if entry.hf_max_mesh is not None:
            entry.hf_max_mesh.set_enabled(TerrainEditorSettings().viewing_max)

        self.build_ps_min_mesh(terrain_name=entry.name)
        if entry.hf_min_mesh is not None:
            entry.hf_min_mesh.set_enabled(TerrainEditorSettings().viewing_min)
        return

    def rebuild(self, terrain_name: Optional[str] = None, terrain: Optional["terrain_util.SubTerrain"] = None):
        entry = self._get_entry(terrain_name)
        if terrain is not None:
            entry.terrain = terrain
        self.build_ps_hf_mesh(terrain_name=entry.name)
        self.build_ps_max_mesh(terrain_name=entry.name)
        self.build_ps_min_mesh(terrain_name=entry.name)
        return

    def soft_rebuild(self, terrain_name: Optional[str] = None, terrain: Optional["terrain_util.SubTerrain"] = None):
        entry = self._get_entry(terrain_name)
        if terrain is not None:
            entry.terrain = terrain
        self.build_ps_hf_mesh(terrain_name=entry.name, print_time_stats=True)
        return

    def compute_hf_trimesh(self, terrain: Optional["terrain_util.SubTerrain"] = None):
        if terrain is None:
            entry = self._get_entry(create=False)
            if entry is not None and entry.terrain is not None:
                terrain = entry.terrain
        if terrain is None:
            raise ValueError("No terrain available to build mesh")

        if input_config["terrain_mesh_mode"] == "greedy":
            verts, tris = terrain_util.convert_heightfield_to_blocky_trimesh_optimized(
                terrain.hf,
                terrain.min_point[0].item(),
                terrain.min_point[1].item(),
                terrain.dxdy[0].item())
        elif input_config["terrain_mesh_mode"] == "simple":
            verts, tris = terrain_util.convert_heightfield_to_voxelized_trimesh(
                terrain.hf,
                terrain.min_point[0].item(),
                terrain.min_point[1].item(),
                terrain.dxdy[0].item())
        else:
            raise ValueError(f"Unsupported terrain mesh mode: {input_config['terrain_mesh_mode']}")

        hf_trimesh = trimesh.Trimesh(vertices=verts, faces=tris)

        return hf_trimesh

    def build_ps_hf_mesh(self, terrain_name: Optional[str] = None, 
                         terrain: Optional["terrain_util.SubTerrain"] = None,
                         print_time_stats = False):
        entry = self._get_entry(terrain_name)
        if terrain is not None:
            entry.terrain = terrain

        terrain_obj = entry.terrain
        if terrain_obj is None:
            return None

        if print_time_stats:
            start_time = time.time()
        hf_trimesh = self.compute_hf_trimesh(terrain=terrain_obj)
        verts = hf_trimesh.vertices
        tris = hf_trimesh.faces

        entry.verts = verts
        entry.tris = tris


        if print_time_stats:
            end_time = time.time()
            print("compute hf_trimesh time:", end_time - start_time, "seconds.")

            start_time = time.time()

        info = cpuinfo.get_cpu_info()
        if "intel" in info['brand_raw'].lower():
            hf_embree = trimesh.ray.ray_pyembree.RayMeshIntersector(hf_trimesh)
        else:
            hf_embree = trimesh.ray.ray_triangle.RayMeshIntersector(hf_trimesh)

        if print_time_stats:
            end_time = time.time()
            print("hf_embree construction time:", end_time - start_time, "seconds.")

            start_time = time.time()


        old_mesh = entry.hf_ps_mesh
        if old_mesh is not None:
            try:
                ps.remove_surface_mesh(old_mesh.get_name())
            except Exception:
                pass

        if print_time_stats:
            end_time = time.time()
            print("remove surface mesh time:", end_time - start_time, "seconds.")

            start_time = time.time()

        mesh_name = self._mesh_name(entry, "heightfield")
        hf_ps_mesh = ps.register_surface_mesh(mesh_name, verts, tris)

        heights = verts[..., 2]
        max_h = np.max(heights)
        min_h = np.min(heights)
        if max_h > min_h + 1e-3:
            heights = (heights - min_h) / (max_h - min_h)
            hf_ps_mesh.add_scalar_quantity("height", heights, enabled=True)
        else:
            hf_ps_mesh.set_color([0.5, 0.5, 0.5])

        if print_time_stats:
            end_time = time.time()
            print("register surface mesh time:", end_time - start_time, "seconds")

        entry.hf_ps_mesh = hf_ps_mesh
        entry.hf_embree = hf_embree
        return hf_ps_mesh

    def build_ps_max_mesh(self, terrain_name: Optional[str] = None, terrain: Optional["terrain_util.SubTerrain"] = None):
        entry = self._get_entry(terrain_name)
        if terrain is not None:
            entry.terrain = terrain

        terrain_obj = entry.terrain
        if terrain_obj is None:
            return None

        true_hf = torch.ones_like(terrain_obj.hf_mask, dtype=torch.bool)
        verts, tris = terrain_util.convert_hf_mask_to_flat_voxels(
            true_hf,
            terrain_obj.hf_maxmin[..., 0],
            terrain_obj.min_point[0].item(),
            terrain_obj.min_point[1].item(),
            terrain_obj.dxdy[0].item(),
            voxel_w_scale=0.8)

        old_mesh = entry.hf_max_mesh
        if old_mesh is not None:
            try:
                ps.remove_surface_mesh(old_mesh.get_name())
            except Exception:
                pass

        mesh_name = self._mesh_name(entry, "heightfield max")
        hf_max_ps_mesh = ps.register_surface_mesh(mesh_name, verts, tris)
        hf_max_ps_mesh.set_color([1.0, 0.2, 0.2])
        hf_max_ps_mesh.set_transparency(0.2)
        entry.hf_max_mesh = hf_max_ps_mesh
        return hf_max_ps_mesh

    def build_ps_min_mesh(self, terrain_name: Optional[str] = None, terrain: Optional["terrain_util.SubTerrain"] = None):
        entry = self._get_entry(terrain_name)
        if terrain is not None:
            entry.terrain = terrain

        terrain_obj = entry.terrain
        if terrain_obj is None:
            return None

        true_hf = torch.ones_like(terrain_obj.hf_mask, dtype=torch.bool)
        verts, tris = terrain_util.convert_hf_mask_to_flat_voxels(
            true_hf,
            terrain_obj.hf_maxmin[..., 1],
            terrain_obj.min_point[0].item(),
            terrain_obj.min_point[1].item(),
            terrain_obj.dxdy[0].item(),
            voxel_w_scale=0.8)

        old_mesh = entry.hf_min_mesh
        if old_mesh is not None:
            try:
                ps.remove_surface_mesh(old_mesh.get_name())
            except Exception:
                pass

        mesh_name = self._mesh_name(entry, "heightfield min")
        hf_min_ps_mesh = ps.register_surface_mesh(mesh_name, verts, tris)
        hf_min_ps_mesh.set_color([0.2, 0.2, 1.0])
        hf_min_ps_mesh.set_transparency(0.2)
        entry.hf_min_mesh = hf_min_ps_mesh
        return hf_min_ps_mesh

    def set_terrain_mesh_enabled(self, val, terrain_name: Optional[str] = None):
        entry = self._get_entry(terrain_name, create=False)
        if entry is None or entry.hf_ps_mesh is None:
            return
        entry.hf_ps_mesh.set_enabled(val)
        return

    def set_all_terrain_meshes_enabled(self, val: bool):
        for entry in self._terrains.values():
            if entry.hf_ps_mesh is not None:
                entry.hf_ps_mesh.set_enabled(val)
        return

    def remove_terrain(self, terrain_name: str) -> bool:
        entry = self._terrains.pop(terrain_name, None)
        if entry is None:
            return False

        entry.remove_registered_meshes()
        entry.clear_cached_mesh_data()
        entry.terrain = None

        if self._active_name == terrain_name:
            self._active_name = None
            for other_name, other_entry in self._terrains.items():
                if other_entry.terrain is not None:
                    self._active_name = other_name
                    break

        return True



def create_mouse_ball_ps_meshes(size):

    dim = size * 2 + 1
    mouse_ball_trimesh = trimesh.primitives.Sphere(0.05)

    meshes = []

    for i in range(dim):
        for j in range(dim):
            name_str = "mouse " + str(i) + "," + str(j)
            mouse_ball_ps_mesh = ps.register_surface_mesh(name_str, mouse_ball_trimesh.vertices,
                                                        mouse_ball_trimesh.faces)
            mouse_ball_ps_mesh.set_color([1.0, 0.0, 0.0])
            meshes.append(mouse_ball_ps_mesh)
    return meshes

g_mouse_ball_meshes = create_mouse_ball_ps_meshes(MainVars().mouse_size)


def update_mouse_ball_ps_meshes(size):
    global g_mouse_ball_meshes
    for mesh in g_mouse_ball_meshes:
        ps.remove_surface_mesh(mesh.get_name())
    g_mouse_ball_meshes = create_mouse_ball_ps_meshes(size)
    return

def build_selected_pos_flag_mesh() -> ps.SurfaceMesh:

    flag_mesh = ps_util.create_vector_mesh([0.0, 0.0, 1.0], name="selected pos flag", color=[1.0, 0.0, 1.0])

    return flag_mesh
g_flag_mesh = build_selected_pos_flag_mesh()

def update_selected_pos_flag_mesh(xyz: torch.Tensor):
    global g_flag_mesh

    transform = np.eye(4)
    transform[:3, 3] = xyz.cpu().numpy()
    g_flag_mesh.set_transform(transform)
    return

def update_selected_pos_flag_mesh_xy(xy: torch.Tensor):
    global g_flag_mesh

    terrain = TerrainMeshManager().get_active_terrain(require=True)
    xyz = terrain.get_xyz_point(terrain.get_grid_index(xy))

    transform = np.eye(4)
    transform[:3, 3] = xyz.cpu().numpy()
    g_flag_mesh.set_transform(transform)
    return

class MotionManager(SingletonClass):
    loaded_motions = OrderedDict()
    ik_motions = OrderedDict()
    curr_motion = None
    curr_ik_motion = None

    def load_motion(self, filepath, char_model_path: Union[kin_char_model.KinCharModel, str], name,
                    root_offset = None, vis_fps = 15, update_terrain=False,
                    color =  [0.2, 0.2, 0.8]) -> ps_mdm_util.MDMMotionPS:
        if isinstance(char_model_path, kin_char_model.KinCharModel):
            char_model = char_model_path
        else:
            char_model = kin_char_model.KinCharModel(MainVars().device)
            char_model.load_char_file(char_model_path)

        mlib = motion_lib.MotionLib.from_file(motion_file=filepath, 
                                              char_model=char_model, 
                                              device=MainVars().device, 
                                              contact_info=MainVars().use_contact_info)
        if root_offset is not None:
            mlib._motion_frames[:, 0:2] += root_offset
            mlib._frame_root_pos[:, 0:2] += root_offset

        terrain_manager = TerrainMeshManager()

        
        if mlib._terrains[0] is None:
            new_terrain = terrain_util.SubTerrain(x_dim=16, y_dim=16, dx=0.4, dy=0.4, min_x = -3.0, min_y=-3.0, device=MainVars().device)
        else:
            new_terrain = mlib._terrains[0].torch_copy()
        terrain_manager.register_terrain(name, new_terrain, make_active=update_terrain, rebuild=True)

        motion = ps_mdm_util.MDMMotionPS(name, mlib, char_color=color, vis_fps=vis_fps, mdm_model=g_mdm_model)
        motion.set_to_time(0.0)



        active_terrain = terrain_manager.get_active_terrain()
        if active_terrain is not None:
            motion.char.update_local_hf(active_terrain)

        self.loaded_motions[name] = motion
        return motion

    def add_motion(self, motion: ps_mdm_util.MDMMotionPS, name):
        self.loaded_motions[name] = motion
        return
    
    def set_curr_motion(self, name, sequence_val=True, char_val=True):
        self.curr_motion = self.loaded_motions[name]
        self.curr_motion.set_enabled(sequence_val, char_val)
        terrain_manager = TerrainMeshManager()
        if name in terrain_manager.get_registered_terrain_names():
            terrain_manager.set_active_terrain(name)
        return
    
    def get_curr_motion(self) -> ps_mdm_util.MDMMotionPS:
        return self.curr_motion
    
    def get_loaded_motions(self):
        return self.loaded_motions
    
    def get_motion(self, name) -> ps_mdm_util.MDMMotionPS:
        return self.loaded_motions[name]
    
    def make_new_motion(self, 
                        motion_frames: motion_util.MotionFrames,
                        new_motion_name: str,
                        motion_fps: int, 
                        vis_fps: int = 2,
                        loop_mode = motion_lib.LoopMode.CLAMP,
                        new_color = [0.8, 0.2, 0.2],
                        new_char_model = None,
                        view_seq=True,
                        history_length=2):
        MainVars().use_contact_info=True
        if new_char_model is None:
            new_char_model = self.get_curr_motion().char.char_model
        new_mlib = motion_lib.MotionLib.from_frames(
            frames=motion_frames, 
            char_model=new_char_model, 
            device=MainVars().device,
            loop_mode=loop_mode,
            fps=motion_fps,
            contact_info=MainVars().use_contact_info)

        # TODO: fix
        #MotionManager().get_curr_motion().set_enabled(False)
        dt = 1.0 / motion_fps
        MainVars().motion_time = dt * (self.get_curr_motion().char._history_length - 1.0)
        #MainVars().motion_time = new_mlib._motion_lengths[0].item()


        self.get_curr_motion().deselect()
        new_motion = ps_mdm_util.MDMMotionPS(new_motion_name, new_mlib, new_color,
                            vis_fps=vis_fps, start_time=MainVars().motion_time,
                            history_length=history_length, mdm_model=g_mdm_model)
        self.add_motion(new_motion, new_motion_name)
        self.set_curr_motion(new_motion_name, sequence_val=view_seq)

        self.get_curr_motion().select(MainVars().viewing_local_hf, MainVars().viewing_shadow)
        return new_motion

    def make_new_blank_motion(self,
                              name: str = "blank_motion",
                              motion_fps: int = 30, 
                              vis_fps: int = 4,
                              loop_mode = motion_lib.LoopMode.CLAMP,
                              new_color = [0.2, 0.8, 0.2],
                              char_model: Optional[kin_char_model.KinCharModel] = None,
                              history_length=2):
        
        if char_model is None:
            char_model = self.get_curr_motion().char.char_model
        
        blank_frames = motion_util.MotionFrames()
        blank_frames.init_blank_frames(char_model, history_length=5, batch_size=1)
        blank_frames = blank_frames.squeeze(0)

        return self.make_new_motion(
            motion_frames=blank_frames,
            new_motion_name=name,
            motion_fps=motion_fps,
            vis_fps=vis_fps,
            loop_mode=loop_mode,
            new_color=new_color,
            new_char_model=char_model,
            history_length=history_length)






# TODO: remove this
def get_motion(name) -> ps_mdm_util.MDMMotionPS:
    return MotionManager().get_loaded_motions()[name]

g_dir_mesh = ps_util.create_vector_mesh([1.0, 0.0, 0.0], name="direction", radius = 0.02)




def extract_obs(obs, obs_shapes):
    # NOTE: hard coded for parkour dm env
    num_frames = obs.shape[0]

    char_obs_start = 0
    char_obs_end = obs_shapes["char_obs"]["shape"][0]
    char_obs = obs[:, char_obs_start:char_obs_end]

    tar_obs_start = char_obs_end
    tar_obs_end = tar_obs_start + obs_shapes["tar_obs"]["shape"][0] * obs_shapes["tar_obs"]["shape"][1]
    tar_obs = np.reshape(obs[:, tar_obs_start:tar_obs_end], newshape=[num_frames, obs_shapes["tar_obs"]["shape"][0], -1])

    tar_contacts_start = tar_obs_end
    tar_contacts_end = tar_contacts_start + obs_shapes["tar_contacts"]["shape"][0] * obs_shapes["tar_contacts"]["shape"][1]
    tar_contacts = np.reshape(obs[:, tar_contacts_start:tar_contacts_end], newshape=[num_frames, obs_shapes["tar_contacts"]["shape"][0], -1])

    char_contacts_start = tar_contacts_end
    char_contacts_end = char_contacts_start + obs_shapes["char_contacts"]["shape"][0]
    char_contacts = obs[:, char_contacts_start:char_contacts_end]

    hf_start = char_contacts_end
    hf_end = hf_start + obs_shapes["hf"]["shape"][0]
    hf = obs[:, hf_start:hf_end]
    num_points = hf.shape[1]

    ray_points_behind = 2
    ray_points_ahead = 60
    ray_num_left = 3
    ray_num_right = 3
    ray_dx = 0.05
    ray_angle = 0.26179938779

    ray_xy_points = geom_util.get_xy_points_cone(
        center=torch.zeros(size=(2,), dtype=torch.float32, device="cpu"),
        dx=ray_dx,
        num_neg=ray_points_behind,
        num_pos=ray_points_ahead,
        num_rays_neg=ray_num_left,
        num_rays_pos=ray_num_right,
        angle_between_rays=ray_angle).numpy()
    
    assert ray_xy_points.shape[0] == num_points

    hf_points = np.zeros(shape=[num_frames, num_points, 3])
    hf_points[:, :, 0:2] = np.expand_dims(ray_xy_points, 0)
    hf_points[:, :, 2] = hf

    return char_obs, tar_obs, tar_contacts, char_contacts, hf_points


def update_dir_mesh():
    global g_dir_mesh

    transform = np.eye(4)
    transform[:3, 3] = MotionManager().get_curr_motion().char.get_body_pos(0).cpu()

    dir = MainVars().mouse_world_pos - MotionManager().get_curr_motion().char.get_body_pos(0)
    dir[2] = 0.0
    norm = torch.norm(dir)

    if norm < 0.05:
        g_dir_mesh.set_transparency(0.0)
        return

    else:
        dir = dir / norm
        angle = torch.atan2(dir[1], dir[0])
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=MainVars().device)
        quat = torch_util.axis_angle_to_quat(z_axis, angle).squeeze(0)
        rot_mat = torch_util.quat_to_matrix(quat)
        
        transform[:3, :3] = rot_mat.cpu().numpy()
        g_dir_mesh.set_transform(transform)
        g_dir_mesh.set_transparency(1.0)

        return

terrain_manager = TerrainMeshManager()

for file_idx in range(len(g_motion_filepaths)):
    elem = g_motion_filepaths[file_idx]
    motion_filepath = elem[0]
    char_filepath = elem[1]
    
    if g_use_legacy_file_format:
        g_motion_data = medit_lib.load_motion_file(motion_filepath)
        g_motion_data.set_device(MainVars().device)
        MainVars().motion_dt = 1.0 / g_motion_data.get_fps()
        if "opt:body_constraints" in g_motion_data._data:
            OptimizationSettings().body_constraints = g_motion_data._data["opt:body_constraints"]
            OptimizationSettings().create_body_constraint_ps_meshes()

        if "path_nodes" in g_motion_data._data:
            PathPlanningSettings().path_nodes = g_motion_data._data["path_nodes"]
            PathPlanningSettings().visualize_path_nodes(MotionManager().get_curr_motion().name + "_path", PathPlanningSettings().path_nodes)

        if "loss" in g_motion_data._data:
            print("Loss:", g_motion_data._data["loss"])

        if "obs" in g_motion_data._data:
            IGObsSettings().setup(g_motion_data._data["obs"], g_motion_data._data["obs_shapes"])

        # Get Terrain
        if g_motion_data.has_terrain():
            terrain = g_motion_data.get_terrain()
        else:
            terrain = terrain_util.SubTerrain(x_dim=16, y_dim=16, dx=0.4, dy=0.4, min_x = -3.0, min_y=-3.0, device=MainVars().device)
        terrain_manager.update_active_terrain(terrain)
    else:
        MotionManager().load_motion(motion_filepath, char_filepath, os.path.basename(motion_filepath), update_terrain=file_idx==0)

        misc_data = file_io.load_ms_file(motion_filepath).misc_data

        if misc_data is not None:
            print("misc data is not none")
            if "path_nodes" in misc_data:
                if isinstance(misc_data["path_nodes"], np.ndarray):
                    misc_data["path_nodes"] = torch.from_numpy(misc_data["path_nodes"])
                PathPlanningSettings().path_nodes = misc_data["path_nodes"].to(dtype=torch.float32, device=MainVars().device)
                PathPlanningSettings().visualize_path_nodes("ASTARpath", PathPlanningSettings().path_nodes)
MotionManager().set_curr_motion(os.path.basename(g_motion_browser_path))

g_kin_controller = None

if LOAD_MDM:
    MDMSettings().select_mdm("main")