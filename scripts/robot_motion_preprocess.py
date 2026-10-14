"""Turn retargeted robot motions (GMR output) into PARC motion files (MS format).

GMR retargets a PARC motion (the *source* motion, usually of data/assets/humanoid.xml)
to a robot and writes a pickle (often a NumPy 2 pickle) holding

    frames  [N, 6 + dof]: root position, root rotation (exp map), joint DOFs
            in the robot's DOF order (the order of the joints in its MJCF)
    fps

This script turns each such file into an MS motion file (parc/util/file_io.py):

1. Terrain: taken from the source motion.  Without one, a flat terrain around the
   motion is created.
2. Contacts: copied from the source motion by body name (--contact_bodies, default
   the feet and hands).  Names go through KinCharModel's body-name aliases, so
   left_foot is left_ankle_roll_link on the G1.  Without usable source contacts (no
   source, a different frame count, or --geometric_contacts), foot contacts are
   detected from the feet's collision geoms: a foot is in contact when its lowest
   point is within --contact_eps of the terrain and it moves slower than
   --contact_vel_thresh.
3. Height fix: the motion is shifted vertically so that the soles of the feet that
   are in contact during the first --height_fix_frames frames rest on the terrain,
   on average.  The sole is --sole_offset below the foot body's origin; by default
   the offset comes from the feet's collision geoms (0.035 m on the G1, 0.05 m on
   the humanoid).
4. Body constraints (misc_data "opt:body_constraints") for the feet and hands: one per
   run of contact frames, at the body's mean position over the run.
5. hf_mask_inds and the terrain's hf_maxmin, for heightfield augmentation.

Source motions are found by file name in --source_dir: first in the subdirectory that
matches the input file's place under --input, then directly in --source_dir, each
time trying the input file name and the name without --name_suffix (the suffix the
retargeting step appended, default "_to_unitree_g1").  Source motions may be MS
files or legacy (develop-era) motion pickles.  A file without a source motion fails
unless --allow_missing_source is given; without --source_dir and --source, every
file is processed on a flat terrain.

Usage, from the repository root:

    python -m scripts.robot_motion_preprocess \\
        --input <GMR output .pkl or directory> --output <.pkl or directory> \\
        --char_file data/assets/unitree_g1/g1_mocap_29dof.xml \\
        --source_dir <directory of the source PARC motions>

A directory input is processed recursively and mirrored under --output.

Ported from develop's tools/motion_file_proc/robot_motion_preprocess.py (2f2c6cb).
It writes MS files instead of legacy pickles, maps source contacts by body name
instead of fixed humanoid columns, and fixes develop's foot geometry (doc/porting_plan.md
DB8): the geometric contact test used the bounding box of the first foot geom, which on
the G1 is a visual mesh with 1 m half-extents, so every slow foot counted as in
contact; and the height fix used a 0.05 m sole offset (the humanoid's), which leaves
the G1's feet 1.5 cm above the terrain.  --sole_offset 0.05 reproduces develop's heights.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Make "parc" and "scripts" importable when this file is run as a plain script.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.file_io as file_io
import parc.util.file_io_helper as file_io_helper
import parc.util.geom_util as geom_util
import parc.util.motion_edit_lib as motion_edit_lib
import parc.util.motion_util as motion_util
import parc.util.path_loader as path_loader
import parc.util.pickle_compat as pickle_compat
import parc.util.terrain_util as terrain_util
import parc.util.torch_util as torch_util
from scripts.convert_legacy_motions import normalize_fps, normalize_loop_mode

DEFAULT_CHAR_FILE = "data/assets/unitree_g1/g1_mocap_29dof.xml"
DEFAULT_SOURCE_CHAR_FILE = "data/assets/humanoid.xml"
DEFAULT_NAME_SUFFIX = "_to_unitree_g1"

FOOT_NAMES = ("left_foot", "right_foot")
HAND_NAMES = ("left_hand", "right_hand")
CONSTRAINT_BODY_NAMES = FOOT_NAMES + HAND_NAMES
ALL_BODIES = "all"

# Used when the feet have no collision geoms to derive the sole offset from.
FALLBACK_SOLE_OFFSET = 0.05
# Flat terrain for motions without a source terrain: motionscope's defaults.
FLAT_TERRAIN_DX = 0.1
FLAT_TERRAIN_PADDING = 0.4
# Contact values above this start a body constraint (as in motion_optimization).
CONSTRAINT_CONTACT_THRESHOLD = 0.9

COLLISION_GEOM_TYPES = (kin_char_model.GeomType.SPHERE,
                        kin_char_model.GeomType.BOX,
                        kin_char_model.GeomType.CAPSULE)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn retargeted robot motions ({frames, fps} pickles, e.g. GMR output) into PARC "
                    "MS motion files: copy terrain and contacts from the source motions, fix the height, "
                    "and add body constraints and hf_mask_inds.",
        epilog="Paths may use $DATA_DIR.  Example: python -m scripts.robot_motion_preprocess "
               "--input retargeted_g1/ --output g1_motions/ --source_dir initial/")
    io = parser.add_argument_group("input and output")
    io.add_argument("--input", required=True,
                    help="retargeted motion .pkl, or a directory of them (searched recursively)")
    io.add_argument("--output", required=True,
                    help="output .pkl for a file input, output directory for a directory input")
    io.add_argument("--char_file", "--robot_model", dest="char_file", default=DEFAULT_CHAR_FILE,
                    help="MJCF of the robot the motions were retargeted to (default: %(default)s)")
    io.add_argument("--device", default="cpu", help="torch device (default: %(default)s)")

    source = parser.add_argument_group("source motions (terrain and contacts)")
    source_choice = source.add_mutually_exclusive_group()
    source_choice.add_argument("--source_dir", "--terrain_dir", dest="source_dir", default=None,
                               help="directory of the source motions (MS or legacy .pkl), matched by file name")
    source_choice.add_argument("--source", default=None,
                               help="source motion of a single --input file")
    source.add_argument("--source_char_file", default=DEFAULT_SOURCE_CHAR_FILE,
                        help="MJCF of the source motions' character, which names their contact columns "
                             "(default: %(default)s)")
    source.add_argument("--name_suffix", default=DEFAULT_NAME_SUFFIX,
                        help="suffix the retargeting step appended to the source file name; stripped when "
                             "looking for the source motion ('' to disable; default: %(default)s)")
    source.add_argument("--allow_missing_source", action="store_true",
                        help="process files without a source motion in --source_dir on a flat terrain with "
                             "geometric contacts, instead of failing them")

    contacts = parser.add_argument_group("contacts")
    contacts.add_argument("--contact_bodies", nargs="+", default=list(CONSTRAINT_BODY_NAMES), metavar="NAME",
                          help="bodies whose contacts are copied from the source motion, by name or alias "
                               "(default: {}); 'all' copies every source body the robot also "
                               "has".format(" ".join(CONSTRAINT_BODY_NAMES)))
    contacts.add_argument("--geometric_contacts", action="store_true",
                          help="ignore source contacts and detect foot contacts from the foot geometry")
    contacts.add_argument("--contact_eps", type=float, default=0.01,
                          help="geometric contacts: max height of the foot's lowest point above the terrain "
                               "in m (default: %(default)s)")
    contacts.add_argument("--contact_vel_thresh", type=float, default=0.2,
                          help="geometric contacts: max foot speed in m/s (default: %(default)s)")

    height = parser.add_argument_group("height fix")
    height.add_argument("--sole_offset", type=float, default=None,
                        help="distance from the foot body's origin down to the sole in m (default: derived "
                             "from the feet's collision geoms; develop used 0.05)")
    height.add_argument("--height_fix_frames", type=int, default=10,
                        help="number of initial frames whose foot contacts set the height (0 disables the "
                             "height fix; default: %(default)s)")
    return parser.parse_args(argv)


def resolve_cli_path(path_like) -> Path:
    if path_loader.DATA_PLACEHOLDER in str(path_like):
        return path_loader.resolve_path(path_like)
    return Path(path_like).expanduser().absolute()


def resolve_char_file(path_like, default) -> Path:
    # default character files live in the repository, wherever the script is run from
    path = repo_root / default if path_like == default else resolve_cli_path(path_like)
    if not path.is_file():
        raise FileNotFoundError("character file not found: {}".format(path))
    return path


def load_char_model(char_file: Path, device) -> kin_char_model.KinCharModel:
    char_model = kin_char_model.KinCharModel(device)
    char_model.load_char_file(str(char_file))
    return char_model


def find_body_ids(char_model: kin_char_model.KinCharModel, names: Sequence[str]) -> Dict[str, int]:
    ids = {}
    for name in names:
        if char_model.has_body(name):
            ids[name] = char_model.get_body_id(name)
        else:
            print("Warning: the character has no '{}' body".format(name))
    return ids


# ---------------------------------------------------------------------------
# Foot geometry
# ---------------------------------------------------------------------------
def collision_geoms(char_model: kin_char_model.KinCharModel, body_id: int) -> List[kin_char_model.Geom]:
    # Meshes are skipped: KinCharModel gives them unit dims, and on the G1 they are the
    # visual meshes, while the feet's collision shapes are small spheres.
    return [geom for geom in char_model.get_geoms(body_id) if geom._shape_type in COLLISION_GEOM_TYPES]


def _geom_quat(geom: kin_char_model.Geom, device) -> Optional[torch.Tensor]:
    quat = getattr(geom, "_quat", None)
    if quat is None:
        return None
    quat = torch.as_tensor(np.asarray(quat, dtype=np.float32), device=device)
    if torch.equal(quat, torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)):
        return None
    return quat


def lowest_point_z(geoms: Sequence[kin_char_model.Geom], body_pos: torch.Tensor, body_rot: torch.Tensor) -> torch.Tensor:
    """World z of the lowest point of ``geoms`` attached to a body at body_pos [N, 3], body_rot [N, 4]."""
    num = body_pos.shape[0]
    device = body_pos.device
    lowest = []
    for geom in geoms:
        offset = geom._offset.to(device).unsqueeze(0).expand(num, -1)
        if geom._shape_type == kin_char_model.GeomType.SPHERE:
            center = body_pos + torch_util.quat_rotate(body_rot, offset)
            lowest.append(center[:, 2] - geom._dims.reshape(-1)[0].item())
        elif geom._shape_type == kin_char_model.GeomType.BOX:
            box_rot, box_offset = body_rot, geom._offset.to(device)
            quat = _geom_quat(geom, device)
            if quat is not None:  # express the box in its own frame
                box_rot = torch_util.quat_mul(body_rot, quat.unsqueeze(0).expand(num, -1))
                box_offset = torch_util.quat_rotate(torch_util.quat_conjugate(quat).unsqueeze(0),
                                                    box_offset.unsqueeze(0))[0]
            corners = geom_util.get_box_points_batch(body_pos, box_rot, geom._dims.to(device), box_offset)
            lowest.append(torch.min(corners[..., 2], dim=-1).values)
        elif geom._shape_type == kin_char_model.GeomType.CAPSULE:
            # capsules are stored as a start point (offset) and a start-to-end vector (dims)
            end = offset + geom._dims.to(device).unsqueeze(0)
            start_z = (body_pos + torch_util.quat_rotate(body_rot, offset))[:, 2]
            end_z = (body_pos + torch_util.quat_rotate(body_rot, end))[:, 2]
            lowest.append(torch.minimum(start_z, end_z) - geom._radius)
        else:
            assert False, "unsupported collision geom: {}".format(geom._shape_type)
    return torch.min(torch.stack(lowest, dim=-1), dim=-1).values


def derive_sole_offset(char_model: kin_char_model.KinCharModel, foot_ids: Sequence[int], device) -> Optional[float]:
    """Mean distance from the foot bodies' origins down to their lowest collision point, at rest."""
    offsets = []
    for body_id in foot_ids:
        geoms = collision_geoms(char_model, body_id)
        if len(geoms) == 0:
            continue
        origin = torch.zeros(size=(1, 3), dtype=torch.float32, device=device)
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        offsets.append(-lowest_point_z(geoms, origin, identity).item())
    if len(offsets) == 0:
        return None
    return float(np.mean(offsets))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class SourceMotion:
    def __init__(self, path: Path, file_format: str,
                 terrain: Optional[terrain_util.SubTerrain],
                 contacts: Optional[torch.Tensor],
                 fps):
        self.path = path
        self.file_format = file_format
        self.terrain = terrain
        self.contacts = contacts  # [N, columns], or None
        self.fps = fps


def _legacy_terrain(terrain, device) -> terrain_util.SubTerrain:
    terrain.update_old()  # terrains saved before hf_maxmin existed lack it
    terrain.to_torch(device)
    return terrain


def _to_tensor(value, device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)


def load_robot_motion(path: Path, char_model: kin_char_model.KinCharModel, device):
    """Read a retargeted motion {frames [N, 6 + dof], fps[, loop_mode]}."""
    data = pickle_compat.load(path, map_location=device)
    if file_io.is_ms_container(data):
        raise ValueError("already an MS motion file; expected a retargeted motion {'frames', 'fps'}")
    if not isinstance(data, dict) or data.get("frames") is None:
        found = sorted(map(str, data.keys())) if isinstance(data, dict) else type(data).__name__
        raise ValueError("expected a dict with 'frames' [N, {}] and 'fps', found {}".format(
            6 + char_model.get_dof_size(), found))

    frames = _to_tensor(data["frames"], device)
    if frames.dim() == 3 and frames.shape[0] == 1:
        frames = frames[0]
    num_cols = 6 + char_model.get_dof_size()
    if frames.dim() != 2 or frames.shape[1] != num_cols:
        raise ValueError("frames have shape {}, but the character has {} DOFs, so [N, {}] is expected "
                         "(wrong --char_file?)".format(tuple(frames.shape), char_model.get_dof_size(), num_cols))
    if data.get("fps") is None:
        print("  Warning: no fps in the file; assuming {}".format(normalize_fps(None)))

    # develop's fallback: a terrain stored in the input itself (legacy SubTerrain)
    terrain = data.get("terrain")
    if terrain is not None:
        terrain = _legacy_terrain(terrain, device)
    return frames.contiguous(), normalize_fps(data.get("fps")), normalize_loop_mode(data.get("loop_mode")), terrain


def load_source_motion(path: Path, device) -> SourceMotion:
    raw = pickle_compat.load(path, map_location=device)
    if file_io.is_ms_container(raw):
        ms_data = file_io_helper.load_ms_file(str(path), device=device)
        terrain = None
        if ms_data.terrain_data is not None:
            terrain = terrain_util.SubTerrain.from_ms_terrain_data(ms_data.terrain_data, device=device)
        contacts = fps = None
        if ms_data.motion_data is not None:
            contacts = ms_data.motion_data.body_contacts
            fps = ms_data.motion_data.fps
        return SourceMotion(path, "MS", terrain, contacts, fps)

    if file_io.is_legacy_motion_dict(raw):
        terrain = raw.get("terrain")
        if terrain is not None:
            terrain = _legacy_terrain(terrain, device)
        contacts = raw.get("contacts")
        if contacts is not None:
            contacts = _to_tensor(contacts, device)
            if contacts.dim() == 3 and contacts.shape[0] == 1:
                contacts = contacts[0]
        return SourceMotion(path, "legacy", terrain, contacts, raw.get("fps"))

    raise ValueError("source {} is not a motion file (neither an MS file nor a legacy motion dict)".format(path))


def find_source_file(input_file: Path, input_root: Optional[Path], source_dir: Path, name_suffix: str) -> Optional[Path]:
    """develop's lookup: the input's subdirectory of source_dir, then source_dir; name with and without the suffix."""
    names = [input_file.name]
    stem = input_file.stem
    if name_suffix and stem.endswith(name_suffix) and len(stem) > len(name_suffix):
        names.append(stem[:-len(name_suffix)] + input_file.suffix)

    search_dirs = [source_dir]
    if input_root is not None:
        rel_dir = input_file.parent.relative_to(input_root)
        if rel_dir != Path("."):
            search_dirs.insert(0, source_dir / rel_dir)

    for search_dir in search_dirs:
        for name in names:
            candidate = search_dir / name
            if candidate.is_file():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------
def source_contact_names(source_char_model: kin_char_model.KinCharModel, num_columns: int) -> Optional[List[str]]:
    # MS files store contacts per contact body; develop-era files per body.
    if num_columns == source_char_model.get_num_contact_bodies():
        return [source_char_model.get_contact_body_name(i) for i in range(num_columns)]
    if num_columns == source_char_model.get_num_bodies():
        return list(source_char_model.get_body_names())
    return None


def map_source_contacts(src_contacts: torch.Tensor, src_names: Sequence[str],
                        source_char_model: kin_char_model.KinCharModel,
                        char_model: kin_char_model.KinCharModel,
                        body_names: Sequence[str]) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
    """Copy source contact columns to the robot's bodies by name; returns [N, robot bodies] and the pairs mapped."""
    copy_all = list(body_names) == [ALL_BODIES]
    names = src_names if copy_all else body_names

    contacts = torch.zeros(size=(src_contacts.shape[0], char_model.get_num_bodies()),
                           dtype=torch.float32, device=src_contacts.device)
    mapped = []
    for name in names:
        if not source_char_model.has_body(name) or source_char_model.resolve_body_name(name) not in src_names:
            print("  Warning: the source motion has no contacts for '{}'".format(name))
            continue
        if not char_model.has_body(name):
            if not copy_all:
                print("  Warning: the robot has no body '{}'; its source contacts are dropped".format(name))
            continue
        column = list(src_names).index(source_char_model.resolve_body_name(name))
        body_id = char_model.get_body_id(name)
        contacts[:, body_id] = torch.maximum(contacts[:, body_id], src_contacts[:, column])
        mapped.append((src_names[column], char_model.get_body_name(body_id)))
    return contacts, mapped


def compute_foot_contacts(frames: torch.Tensor, char_model: kin_char_model.KinCharModel, fps,
                          terrain: terrain_util.SubTerrain, foot_ids: Sequence[int], sole_offset: float,
                          contact_eps: float, contact_vel_thresh: float) -> torch.Tensor:
    """Foot contacts from geometry: lowest collision point near the terrain and the foot moving slowly."""
    motion = motion_util.motion_frames_from_mlib_format(frames, char_model)
    body_pos, body_rot = motion.body_pos, motion.body_rot
    num_frames = frames.shape[0]

    body_vel = torch.zeros_like(body_pos)
    if num_frames > 1:
        body_vel[:-1] = (body_pos[1:] - body_pos[:-1]) * fps
        body_vel[-1] = body_vel[-2]

    contacts = torch.zeros(size=(num_frames, char_model.get_num_bodies()), dtype=torch.float32, device=frames.device)
    for body_id in foot_ids:
        geoms = collision_geoms(char_model, body_id)
        if len(geoms) > 0:
            sole_z = lowest_point_z(geoms, body_pos[:, body_id], body_rot[:, body_id])
        else:
            sole_z = body_pos[:, body_id, 2] - sole_offset
        terrain_z = terrain.get_hf_val_from_points(body_pos[:, body_id, 0:2])
        is_close = (sole_z - terrain_z) < contact_eps
        is_slow = torch.norm(body_vel[:, body_id], dim=-1) < contact_vel_thresh
        contacts[:, body_id] = torch.logical_and(is_close, is_slow).to(dtype=torch.float32)
    return contacts


def apply_height_fix(frames: torch.Tensor, contacts: torch.Tensor, char_model: kin_char_model.KinCharModel,
                     terrain: terrain_util.SubTerrain, foot_ids: Sequence[int], sole_offset: float,
                     num_fix_frames: int) -> Tuple[Optional[float], int]:
    """Shift frames (in place) so the soles of the feet in contact in the first frames rest on the terrain.

    Returns the vertical offset applied (None if there were no foot contacts) and the
    number of foot-frames it was computed from.
    """
    num_fix_frames = min(frames.shape[0], num_fix_frames)
    if num_fix_frames <= 0:
        return None, 0
    body_pos = motion_util.motion_frames_from_mlib_format(frames[:num_fix_frames], char_model).body_pos

    height_diffs = []
    for body_id in foot_ids:
        is_contact = contacts[:num_fix_frames, body_id] > 0.5
        if is_contact.any():
            sole_z = body_pos[:, body_id, 2] - sole_offset
            terrain_z = terrain.get_hf_val_from_points(body_pos[:, body_id, 0:2][is_contact])
            height_diffs.append(sole_z[is_contact] - terrain_z)
    if len(height_diffs) == 0:
        return None, 0

    height_diffs = torch.cat(height_diffs)
    offset = -torch.mean(height_diffs)
    frames[..., 2] += offset
    return offset.item(), height_diffs.shape[0]


def extract_consecutive_trues(bool_tensor: torch.Tensor) -> List[torch.Tensor]:
    # Same grouping as develop and motion_optimization.compute_approx_body_constraints,
    # including its quirk: a final run of a single frame is dropped.
    bool_tensor = bool_tensor.flatten()
    indices = torch.nonzero(bool_tensor, as_tuple=True)[0]
    if len(indices) == 0:
        return []
    breaks = torch.nonzero(indices[1:] - indices[:-1] > 1, as_tuple=True)[0] + 1
    breaks = [0] + breaks.tolist()
    groups = [indices[breaks[i]:breaks[i + 1]].clone() for i in range(len(breaks) - 1)]
    if breaks[-1] < indices.shape[0] - 1:
        groups.append(indices[breaks[-1]:].clone())
    return groups


def compute_body_constraints(frames: torch.Tensor, contacts: torch.Tensor, char_model: kin_char_model.KinCharModel,
                             body_ids: Sequence[int]) -> List[List[moopt.BodyConstraint]]:
    """One constraint per run of contact frames, at the body origin's mean position over the run.

    develop's lightweight version of compute_approx_body_constraints: no geom offset
    and no projection of the points onto the terrain.
    """
    body_pos = motion_util.motion_frames_from_mlib_format(frames, char_model).body_pos
    body_constraints = [[] for _ in range(char_model.get_num_bodies())]
    for body_id in body_ids:
        constraints = []
        for group in extract_consecutive_trues(contacts[:, body_id] > CONSTRAINT_CONTACT_THRESHOLD):
            constraints.append(moopt.BodyConstraint(start_frame_idx=group[0].item(),
                                                    end_frame_idx=group[-1].item(),
                                                    constraint_point=torch.mean(body_pos[group, body_id], dim=0)))
        body_constraints[body_id] = constraints
    return body_constraints


def create_flat_terrain(frames: torch.Tensor, char_model: kin_char_model.KinCharModel, device) -> terrain_util.SubTerrain:
    # create_terrain_for_motion sizes the grid as if the first root position were at the
    # xy origin, so build it for the motion moved there and move the terrain back.
    anchor = frames[0, 0:2].clone()
    root_pos = frames[:, 0:3].clone()
    root_pos[:, 0:2] -= anchor
    terrain = motion_edit_lib.create_terrain_for_motion(motion_util.MotionFrames(root_pos=root_pos), char_model,
                                                        char_points=None, dx=FLAT_TERRAIN_DX,
                                                        padding=FLAT_TERRAIN_PADDING)
    terrain.set_device(device)
    terrain.min_point = terrain.min_point + anchor
    return terrain


def compute_hf_extra_vals(frames: torch.Tensor, char_model: kin_char_model.KinCharModel,
                          terrain: terrain_util.SubTerrain):
    # the point samples develop's robot_motion_preprocess used (parc_2_kin_gen's)
    body_points = geom_util.get_char_point_samples(char_model,
                                                   sphere_num_subdivisions=0,
                                                   box_num_slices=2,
                                                   box_dim_x=2,
                                                   box_dim_y=2,
                                                   capsule_num_circle_points=4,
                                                   capsule_num_sphere_subdivisons=0,
                                                   capsule_num_cylinder_slices=2)
    motion = motion_util.motion_frames_from_mlib_format(frames, char_model)
    return terrain_util.compute_hf_extra_vals(motion_frames=motion, terrain=terrain, char_model=char_model,
                                              char_body_points=body_points)


def preprocess_motion_file(input_path: Path, output_path: Path, source_path: Optional[Path],
                           char_model: kin_char_model.KinCharModel,
                           source_char_model: Optional[kin_char_model.KinCharModel],
                           foot_ids: Dict[str, int], constraint_ids: Dict[str, int],
                           sole_offset: float, args: argparse.Namespace) -> None:
    device = args.device
    frames, fps, loop_mode, terrain = load_robot_motion(input_path, char_model, device)
    num_frames = frames.shape[0]
    print("  {} frames at {} fps".format(num_frames, fps))

    source = None
    if source_path is not None:
        source = load_source_motion(source_path, device)
        print("  source: {} ({})".format(source_path, source.file_format))
        if source.terrain is not None:
            terrain = source.terrain
    if terrain is None:
        terrain = create_flat_terrain(frames, char_model, device)
        print("  terrain: no source terrain; flat terrain {} x {}".format(*terrain.hf.shape))
    else:
        print("  terrain: {} x {} cells, dx {:.3f}".format(terrain.hf.shape[0], terrain.hf.shape[1],
                                                         terrain.dxdy[0].item()))

    # contacts
    contacts = None
    if source is not None and source.contacts is not None and not args.geometric_contacts:
        src_contacts = source.contacts
        src_names = source_contact_names(source_char_model, src_contacts.shape[-1])
        if src_contacts.dim() != 2 or src_contacts.shape[0] != num_frames:
            print("  Warning: the source contacts have shape {}, not {} frames; detecting contacts "
                  "geometrically".format(tuple(src_contacts.shape), num_frames))
        elif src_names is None:
            print("  Warning: the source contacts have {} columns, but {} has {} bodies; detecting contacts "
                  "geometrically (wrong --source_char_file?)".format(src_contacts.shape[1], args.source_char_file,
                                                                   source_char_model.get_num_bodies()))
        else:
            if source.fps is not None and normalize_fps(source.fps) != fps:
                print("  Warning: the source motion has {} fps, the retargeted motion {}".format(source.fps, fps))
            contacts, mapped = map_source_contacts(src_contacts, src_names, source_char_model, char_model,
                                                   args.contact_bodies)
            if len(mapped) == 0:
                print("  Warning: none of the source contacts maps to the robot; detecting contacts geometrically")
                contacts = None
            else:
                print("  contacts: from the source, " + ", ".join("{} -> {}".format(a, b) for a, b in mapped))
    if contacts is None:
        contacts = compute_foot_contacts(frames, char_model, fps, terrain, list(foot_ids.values()), sole_offset,
                                         args.contact_eps, args.contact_vel_thresh)
        print("  contacts: geometric, feet in contact in {} foot-frames".format(
            int(contacts[:, list(foot_ids.values())].sum().item())))

    offset, num_used = apply_height_fix(frames, contacts, char_model, terrain, list(foot_ids.values()),
                                        sole_offset, args.height_fix_frames)
    if offset is None:
        print("  height fix: no foot contacts in the first {} frames; not applied".format(args.height_fix_frames))
    else:
        print("  height fix: {:+.4f} m (from {} foot-frames)".format(offset, num_used))

    body_constraints = compute_body_constraints(frames, contacts, char_model, list(constraint_ids.values()))
    hf_mask_inds, terrain = compute_hf_extra_vals(frames, char_model, terrain)
    print("  body constraints: {}; hf_mask_inds: {} frames".format(
        sum(len(c) for c in body_constraints), len(hf_mask_inds)))

    motion_frames = motion_util.MotionFrames(
        root_pos=frames[:, 0:3],
        root_rot=torch_util.quat_pos(torch_util.exp_map_to_quat(frames[:, 3:6])),
        joint_rot=torch_util.quat_pos(char_model.dof_to_rot(frames[:, 6:])),
        contacts=contacts[:, char_model.get_contact_body_ids().to(device=contacts.device)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_io_helper.save_ms_file(str(output_path), motion_frames=motion_frames, fps=fps, loop_mode=loop_mode,
                                terrain=terrain, hf_mask_inds=hf_mask_inds, body_constraints=body_constraints)
    print("  saved {}".format(output_path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    input_path = resolve_cli_path(args.input)
    output_path = resolve_cli_path(args.output)
    if not input_path.exists():
        raise FileNotFoundError("input not found: {}".format(input_path))
    if args.source is not None and not input_path.is_file():
        raise ValueError("--source needs a single --input file; use --source_dir for directories")
    if args.height_fix_frames < 0:
        raise ValueError("--height_fix_frames must be >= 0")

    char_file = resolve_char_file(args.char_file, DEFAULT_CHAR_FILE)
    print("Robot: {}".format(char_file))
    char_model = load_char_model(char_file, args.device)

    source_char_model = None
    if (args.source is not None or args.source_dir is not None) and not args.geometric_contacts:
        source_char_model = load_char_model(resolve_char_file(args.source_char_file, DEFAULT_SOURCE_CHAR_FILE),
                                            args.device)

    foot_ids = find_body_ids(char_model, FOOT_NAMES)
    constraint_ids = find_body_ids(char_model, CONSTRAINT_BODY_NAMES)
    for name, body_id in constraint_ids.items():
        print("  {} -> {}".format(name, char_model.get_body_name(body_id)))

    sole_offset = args.sole_offset
    if sole_offset is None:
        sole_offset = derive_sole_offset(char_model, list(foot_ids.values()), args.device)
        if sole_offset is None:
            sole_offset = FALLBACK_SOLE_OFFSET
            print("Warning: the feet have no collision geoms; using a sole offset of {} m".format(sole_offset))
        else:
            print("Sole offset: {:.4f} m (from the feet's collision geoms)".format(sole_offset))
    else:
        print("Sole offset: {:.4f} m (--sole_offset)".format(sole_offset))

    if input_path.is_dir():
        if output_path == input_path:
            raise ValueError("--output must differ from the input directory")
        # skip earlier outputs when --output lies inside --input
        input_files = sorted(p for p in input_path.rglob("*.pkl") if p.is_file() and output_path not in p.parents)
        jobs = [(p, output_path / p.relative_to(input_path)) for p in input_files]
        input_root = input_path
    else:
        if output_path.is_dir():
            output_path = output_path / input_path.name
        jobs = [(input_path, output_path)]
        input_root = None
    if len(jobs) == 0:
        print("No .pkl files found in {}".format(input_path))
        return 1

    source_dir = None if args.source_dir is None else resolve_cli_path(args.source_dir)
    if source_dir is not None and not source_dir.is_dir():
        raise FileNotFoundError("--source_dir not found: {}".format(source_dir))
    single_source = None if args.source is None else resolve_cli_path(args.source)
    if single_source is not None and not single_source.is_file():
        raise FileNotFoundError("--source not found: {}".format(single_source))

    failures = []
    for i, (src_file, dst_file) in enumerate(jobs):
        print("[{}/{}] {}".format(i + 1, len(jobs), src_file))
        source_path = single_source
        if source_dir is not None:
            source_path = find_source_file(src_file, input_root, source_dir, args.name_suffix)
            if source_path is None:
                # usually a naming mismatch; a flat terrain would be wrong for most motions
                message = "no source motion found in {}".format(source_dir)
                if not args.allow_missing_source:
                    failures.append((src_file, FileNotFoundError(message + " (--allow_missing_source processes "
                                                                 "it on a flat terrain)")))
                    print("  Failed: " + message)
                    continue
                print("  Warning: " + message)
        try:
            preprocess_motion_file(src_file, dst_file, source_path, char_model, source_char_model,
                                   foot_ids, constraint_ids, sole_offset, args)
        except Exception as exc:
            failures.append((src_file, exc))
            print("  Failed: {}: {}".format(type(exc).__name__, exc))

    print("Processed {} of {} files.".format(len(jobs) - len(failures), len(jobs)))
    for path, exc in failures:
        print(" - failed {}: {}: {}".format(path, type(exc).__name__, exc))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
