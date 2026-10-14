import sys

sys.path.insert(1, sys.path[0] + ("/.."))
import argparse
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.util.file_io as file_io
import parc.util.motion_edit_lib as motion_edit_lib
import parc.util.torch_util as torch_util

DEFAULT_INPUT_FOLDERS: Sequence[str] = [
    "data/terrains/legacy/temp/"
]

DEFAULT_OUTPUT_FOLDERS: Sequence[str] = [
    "data/terrains/"
]

DEFAULT_CHAR_FILE = "parc/data/assets/humanoid.xml"
LEGACY_FIELDS = {"frames", "contacts", "terrain", "fps", "loop_mode"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert legacy motion files saved with MotionData into the new MSFileData format."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=list(DEFAULT_INPUT_FOLDERS),
        help="Input directories that contain legacy .pkl motion files (recursively processed).",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=list(DEFAULT_OUTPUT_FOLDERS),
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
    return parser.parse_args()


def resolve_path(path_like: str, *, base_dir: Path) -> Path:
    path = Path(path_like)
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


def build_misc_data(raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    misc_items = {
        key: _convert_misc_value(raw_data[key])
        for key in raw_data.keys() - LEGACY_FIELDS
    }
    return misc_items or None


def motion_to_ms_file(
    raw_data: Dict[str, Any],
    motion_data: motion_edit_lib.MotionData,
    char_model: kin_char_model.KinCharModel,
) -> file_io.MSFileData:
    frames = motion_data.get_frames()
    if frames.ndim != 2 or frames.shape[1] < 6:
        raise ValueError("Legacy motion frames must be a 2D tensor with at least 6 columns.")

    root_pos = frames[:, 0:3]
    root_rot_exp = frames[:, 3:6]
    joint_dof = frames[:, 6:]

    root_rot = torch_util.quat_pos(torch_util.exp_map_to_quat(root_rot_exp))
    joint_rot = torch_util.quat_pos(char_model.dof_to_rot(joint_dof))

    body_contacts = None
    if motion_data.has_contacts():
        body_contacts = motion_data.get_contacts().detach().cpu().numpy()

    motion_payload = file_io.MSMotionData(
        root_pos=root_pos.detach().cpu().numpy(),
        root_rot=root_rot.detach().cpu().numpy(),
        joint_rot=joint_rot.detach().cpu().numpy(),
        body_contacts=body_contacts,
        fps=int(motion_data.get_fps()),
        loop_mode=str(motion_data.get_loop_mode()),
    )

    terrain_data = None
    if motion_data.has_terrain():
        terrain_np = motion_data.get_terrain().numpy_copy()
        terrain_data = file_io.MSTerrainData(
            hf=terrain_np.hf,
            hf_maxmin=terrain_np.hf_maxmin,
            min_point=terrain_np.min_point,
            dx=float(terrain_np.dxdy[0]),
        )

    misc_data = build_misc_data(raw_data)

    return file_io.MSFileData(
        motion_data=motion_payload,
        terrain_data=terrain_data,
        misc_data=misc_data,
    )


def convert_motion_file(
    src_path: Path,
    dst_path: Path,
    *,
    char_model: kin_char_model.KinCharModel,
) -> None:
    raw_data = motion_edit_lib.load_motion_file(
        motion_filepath=str(src_path), convert_to_class=False
    )
    motion_data = motion_edit_lib.MotionData(copy.deepcopy(raw_data), device="cpu")
    ms_file = motion_to_ms_file(raw_data, motion_data, char_model)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    file_io.save_ms_file(ms_file, str(dst_path))


def process_directory_pair(
    input_root: Path,
    output_root: Path,
    *,
    extensions: Sequence[str],
    char_model: kin_char_model.KinCharModel,
    dry_run: bool,
) -> Tuple[int, List[Tuple[Path, Exception]]]:
    motion_files = collect_motion_files(input_root, extensions)
    failures: List[Tuple[Path, Exception]] = []

    for src_path in motion_files:
        rel_path = src_path.relative_to(input_root)
        dst_path = (output_root / rel_path).with_suffix(".pkl")

        if dry_run:
            print(f"[DRY RUN] convert {src_path} -> {dst_path}")
            continue

        try:
            convert_motion_file(src_path, dst_path, char_model=char_model)
            print(f"Converted {src_path} -> {dst_path}")
        except Exception as exc:
            failures.append((src_path, exc))
            print(f"Failed to convert {src_path}: {exc}")

    converted_count = (len(motion_files) - len(failures)) if not dry_run else 0
    return converted_count, failures


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if len(args.inputs) != len(args.outputs):
        raise ValueError("Number of input and output directories must match.")

    char_file = resolve_path(args.char_file, base_dir=repo_root)
    if not char_file.is_file():
        raise FileNotFoundError(f"Character file not found: {char_file}")

    char_model = kin_char_model.KinCharModel(device="cpu")
    char_model.load_char_file(str(char_file))

    total_converted = 0
    all_failures: List[Tuple[Path, Exception]] = []

    for input_dir, output_dir in zip(args.inputs, args.outputs):
        input_root = resolve_path(input_dir, base_dir=repo_root)
        output_root = resolve_path(output_dir, base_dir=repo_root)

        if not input_root.is_dir():
            print(f"Skipping missing input directory: {input_root}")
            continue

        converted, failures = process_directory_pair(
            input_root,
            output_root,
            extensions=args.extensions,
            char_model=char_model,
            dry_run=args.dry_run,
        )
        total_converted += converted
        all_failures.extend(failures)

    if args.dry_run:
        print("Dry run complete.")
        return

    if all_failures:
        print("Conversion completed with errors:")
        for path, exc in all_failures:
            print(f" - {path}: {exc}")
    else:
        print("Conversion completed without errors.")

    print(f"Total converted files: {total_converted}")


if __name__ == "__main__":
    main()