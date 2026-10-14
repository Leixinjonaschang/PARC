"""Convert legacy (develop-era) motion pickles into MS files.

A legacy motion file is a flat dict, as written by develop's
``save_motion_data`` or ``tools/motion_file_proc/robot_motion_preprocess.py``:

    frames               [N, 6 + dof]: root position, root rotation (exp map), joint DOFs
    contacts             [N, bodies]
    terrain              a pickled util.terrain_util.SubTerrain
    fps, loop_mode       30 and "CLAMP" when missing (develop's loader defaults)
    hf_mask_inds, opt:body_constraints, path_nodes, min_point_offset, ...  -> misc_data

Files are read with parc.util.pickle_compat, which remaps develop's module paths
(util.terrain_util.SubTerrain, tools.motion_opt.motion_optimization.BodyConstraint,
...), numpy._core, and tensors pickled on a GPU.  Terrain-only files (no frames)
become MS files with motion_data=None.  Files that are already MS files are
copied unchanged; other pickles are skipped.

Usage, from the repository root:

    python -m scripts.convert_legacy_motions \\
        --inputs <legacy dir> [<legacy dir> ...] --outputs <new dir> [<new dir> ...] \\
        --char-file data/assets/unitree_g1/g1_mocap_29dof.xml

Each input directory is converted recursively into the matching output
directory, which must be a different directory.  Relative paths are resolved
against the repository root, and "$DATA_DIR" is expanded as in the configs.
"""
import argparse
import enum
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Make "parc" importable when this file is run as a plain script.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.anim.motion_lib as motion_lib
import parc.util.file_io as file_io
import parc.util.file_io_helper as file_io_helper
import parc.util.path_loader as path_loader
import parc.util.pickle_compat as pickle_compat
import parc.util.torch_util as torch_util


DEFAULT_CHAR_FILE = "data/assets/humanoid.xml"
DEFAULT_FPS = 30
DEFAULT_LOOP_MODE = "CLAMP"

# Legacy keys stored in MSMotionData / MSTerrainData; every other key goes to misc_data.
LEGACY_FIELDS = {"frames", "contacts", "terrain", "fps", "loop_mode"}

CONVERTED = "converted"
COPIED = "copied"  # already an MS file


class NotAMotionFileError(ValueError):
    """The pickle is neither a legacy motion file nor an MS file."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy (develop-era) motion pickles into the MS file format."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input directories that contain legacy .pkl motion files (recursively processed).",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        required=True,
        help="Output directories that will receive the converted motion files in MS format.",
    )
    parser.add_argument(
        "--char-file",
        default=DEFAULT_CHAR_FILE,
        help="Path to the character MuJoCo XML description used for DOF to quaternion conversion.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pkl"],
        help="File extensions to convert (case-insensitive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be converted without writing outputs.",
    )
    return parser.parse_args(argv)


def resolve_path(path_like: str, *, base_dir: Path) -> Path:
    if path_loader.DATA_PLACEHOLDER in str(path_like):
        path = path_loader.resolve_path(path_like)
    else:
        path = Path(path_like).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def collect_motion_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    norm_exts = {ext.lower() for ext in extensions}
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in norm_exts
        )
    )


def _to_numpy(value: Any, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return value if dtype is None else value.astype(dtype, copy=False)


def _squeeze_motion_dim(array: np.ndarray) -> np.ndarray:
    # develop's MotionLib accepted [1, N, D] arrays and dropped the leading dim
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    return array


def normalize_fps(value: Any) -> Union[int, float]:
    if value is None:
        return DEFAULT_FPS
    if isinstance(value, (torch.Tensor, np.ndarray, np.generic)):
        value = value.item()
    fps = float(value)
    if not fps > 0.0:
        raise ValueError("invalid fps: {!r}".format(value))
    return int(fps) if fps.is_integer() else fps


def normalize_loop_mode(value: Any) -> str:
    if value is None:
        return DEFAULT_LOOP_MODE
    if isinstance(value, enum.Enum):
        name = value.name
    elif isinstance(value, (int, np.integer)):
        name = motion_lib.LoopMode(int(value)).name
    else:
        name = str(value)
    if name not in motion_lib.LoopMode.__members__:
        raise ValueError("invalid loop_mode: {!r}".format(value))
    return name


def convert_frames(frames: Any, char_model: kin_char_model.KinCharModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = _squeeze_motion_dim(_to_numpy(frames, np.float32))
    num_cols = 6 + char_model.get_dof_size()
    if frames.ndim != 2 or frames.shape[1] != num_cols:
        raise ValueError(
            "legacy frames have shape {}, but the character has {} DOFs, so [N, {}] is expected "
            "(wrong --char-file?)".format(frames.shape, char_model.get_dof_size(), num_cols))

    frames = torch.from_numpy(frames)
    root_pos = frames[:, 0:3].contiguous()
    root_rot = torch_util.quat_pos(torch_util.exp_map_to_quat(frames[:, 3:6]))
    joint_rot = torch_util.quat_pos(char_model.dof_to_rot(frames[:, 6:]))
    return root_pos.numpy(), root_rot.numpy(), joint_rot.numpy()


def convert_contacts(contacts: Any, num_frames: int, char_model: kin_char_model.KinCharModel) -> np.ndarray:
    contacts = _squeeze_motion_dim(_to_numpy(contacts, np.float32))
    num_contact_bodies = char_model.get_num_contact_bodies()
    num_bodies = char_model.get_num_bodies()
    if contacts.ndim != 2 or contacts.shape[0] != num_frames:
        raise ValueError("legacy contacts have shape {}, expected [{}, {}]".format(
            contacts.shape, num_frames, num_contact_bodies))
    if contacts.shape[1] == num_contact_bodies:
        return contacts
    if contacts.shape[1] == num_bodies:
        # per-body contacts; MS files store contact bodies only
        contact_body_ids = char_model.get_contact_body_ids().cpu().numpy()
        return np.ascontiguousarray(contacts[:, contact_body_ids])
    raise ValueError("legacy contacts have {} columns, but the character has {} contact bodies "
                     "(wrong --char-file?)".format(contacts.shape[1], num_contact_bodies))


def convert_terrain(terrain: Any) -> file_io.MSTerrainData:
    terrain.update_old()  # terrains saved before hf_maxmin existed lack it
    terrain.to_torch("cpu")  # numpy_copy() expects torch fields; older files store numpy
    terrain_np = terrain.numpy_copy()

    dxdy = np.asarray(terrain_np.dxdy, dtype=np.float64).reshape(-1)
    if not np.allclose(dxdy, dxdy[0]):
        raise ValueError("terrain cells are {}; the MS format stores a single dx".format(dxdy.tolist()))

    return file_io.MSTerrainData(
        hf=terrain_np.hf,
        hf_maxmin=terrain_np.hf_maxmin,
        min_point=terrain_np.min_point,
        dx=float(dxdy[0]),
    )


def _convert_misc_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        return {k: _convert_misc_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_misc_value(elem) for elem in value]
    if isinstance(value, tuple):
        return tuple(_convert_misc_value(elem) for elem in value)
    if isinstance(value, set):
        return sorted(_convert_misc_value(elem) for elem in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy_copy"):
        terrain_np = value.numpy_copy()
        return {
            "hf": terrain_np.hf,
            "hf_maxmin": terrain_np.hf_maxmin,
            "min_point": terrain_np.min_point,
            "dxdy": terrain_np.dxdy,
        }
    return value


def build_misc_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    misc_data = {}
    for key, value in raw_data.items():
        if key in LEGACY_FIELDS or value is None:
            continue
        if key == file_io_helper.HF_MASK_INDS_KEY:
            misc_data[key] = file_io_helper.numpify_hf_mask_inds(value)
        elif key == file_io_helper.BODY_CONSTRAINTS_KEY:
            misc_data[key] = file_io_helper.numpify_body_constraints(value)
        else:
            misc_data[key] = _convert_misc_value(value)
    return misc_data


def legacy_to_ms_file(raw_data: Dict[str, Any], char_model: kin_char_model.KinCharModel) -> file_io.MSFileData:
    """Convert one legacy motion dict (as unpickled) into MSFileData."""
    if not file_io.is_legacy_motion_dict(raw_data):
        raise NotAMotionFileError("not a legacy motion dict")

    motion_data = None
    if raw_data.get("frames") is not None:
        root_pos, root_rot, joint_rot = convert_frames(raw_data["frames"], char_model)
        body_contacts = None
        if raw_data.get("contacts") is not None:
            body_contacts = convert_contacts(raw_data["contacts"], root_pos.shape[0], char_model)
        motion_data = file_io.MSMotionData(
            root_pos=root_pos,
            root_rot=root_rot,
            joint_rot=joint_rot,
            body_contacts=body_contacts,
            fps=normalize_fps(raw_data.get("fps")),
            loop_mode=normalize_loop_mode(raw_data.get("loop_mode")),
        )

    terrain_data = None
    if raw_data.get("terrain") is not None:
        terrain_data = convert_terrain(raw_data["terrain"])

    if motion_data is None and terrain_data is None:
        raise NotAMotionFileError("legacy dict has neither frames nor terrain")

    return file_io.MSFileData(
        motion_data=motion_data,
        terrain_data=terrain_data,
        misc_data=build_misc_data(raw_data),
    )


def convert_motion_file(
    src_path: Path,
    dst_path: Path,
    *,
    char_model: kin_char_model.KinCharModel,
) -> str:
    """Convert src_path into dst_path; returns CONVERTED or COPIED (already MS).

    Raises NotAMotionFileError for pickles that are not motion files.
    """
    raw_data = pickle_compat.load(src_path, map_location="cpu")

    if file_io.is_ms_container(raw_data):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_path), str(dst_path))
        return COPIED
    if not file_io.is_legacy_motion_dict(raw_data):
        raise NotAMotionFileError("not a motion file: unpickled a {}".format(type(raw_data).__name__))

    ms_file = legacy_to_ms_file(raw_data, char_model)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    file_io.save_ms_file(ms_file, str(dst_path))
    return CONVERTED


def process_directory_pair(
    input_root: Path,
    output_root: Path,
    *,
    extensions: Sequence[str],
    char_model: kin_char_model.KinCharModel,
    dry_run: bool,
) -> Tuple[Dict[str, int], List[Tuple[Path, Exception]]]:
    motion_files = collect_motion_files(input_root, extensions)
    counts = {CONVERTED: 0, COPIED: 0, "skipped": 0}
    failures: List[Tuple[Path, Exception]] = []

    for src_path in motion_files:
        rel_path = src_path.relative_to(input_root)
        dst_path = (output_root / rel_path).with_suffix(".pkl")

        if dry_run:
            print(f"[DRY RUN] convert {src_path} -> {dst_path}")
            continue

        try:
            status = convert_motion_file(src_path, dst_path, char_model=char_model)
        except NotAMotionFileError as exc:
            counts["skipped"] += 1
            print(f"Skipped {src_path}: {exc}")
            continue
        except Exception as exc:
            failures.append((src_path, exc))
            print(f"Failed to convert {src_path}: {type(exc).__name__}: {exc}")
            continue

        counts[status] += 1
        if status == COPIED:
            print(f"Copied {src_path} -> {dst_path} (already an MS file)")
        else:
            print(f"Converted {src_path} -> {dst_path}")

    return counts, failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if len(args.inputs) != len(args.outputs):
        raise ValueError("Number of input and output directories must match.")

    char_file = resolve_path(args.char_file, base_dir=repo_root)
    if not char_file.is_file():
        raise FileNotFoundError(f"Character file not found: {char_file}")

    char_model = kin_char_model.KinCharModel(device="cpu")
    char_model.load_char_file(str(char_file))

    totals = {CONVERTED: 0, COPIED: 0, "skipped": 0}
    all_failures: List[Tuple[Path, Exception]] = []

    for input_dir, output_dir in zip(args.inputs, args.outputs):
        input_root = resolve_path(input_dir, base_dir=repo_root)
        output_root = resolve_path(output_dir, base_dir=repo_root)

        if not input_root.is_dir():
            print(f"Skipping missing input directory: {input_root}")
            continue
        if output_root == input_root:
            raise ValueError(f"Output directory must differ from the input directory: {input_root}")

        counts, failures = process_directory_pair(
            input_root,
            output_root,
            extensions=args.extensions,
            char_model=char_model,
            dry_run=args.dry_run,
        )
        for key, count in counts.items():
            totals[key] += count
        all_failures.extend(failures)

    if args.dry_run:
        print("Dry run complete.")
        return 0

    if all_failures:
        print("Conversion completed with errors:")
        for path, exc in all_failures:
            print(f" - {path}: {type(exc).__name__}: {exc}")
    else:
        print("Conversion completed without errors.")

    print("Total converted files: {}, copied MS files: {}, skipped non-motion files: {}, failed: {}".format(
        totals[CONVERTED], totals[COPIED], totals["skipped"], len(all_failures)))
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
