"""Preview the structure of a PARC .pkl file: an MS motion file or a legacy (develop-era) pickle.

MS files (parc/util/file_io.py) are shown section by section (motion_data,
terrain_data, misc_data); legacy pickles key by key.  Legacy files are read with
parc.util.pickle_compat, which maps develop's module paths (util.terrain_util, ...),
NumPy 2's numpy._core and GPU tensors.  With --char-file, the joint DOFs derived
from joint_rot (KinCharModel.rot_to_dof) are listed too.

Usage, from anywhere:

    python scripts/tools/pkl_preview.py <file.pkl> [--char-file data/assets/unitree_g1/g1_mocap_29dof.xml]
"""
import argparse
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

# Make "parc" importable when this file is run as a plain script.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

import parc.anim.kin_char_model as kin_char_model
import parc.util.file_io as file_io
import parc.util.pickle_compat as pickle_compat

# Suppress numpy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

MS = "MS"
LEGACY = "legacy"
OTHER = "other"


def load_pkl(file_path):
    """Returns (kind, data): (MS, MSFileData), (LEGACY, dict) or (OTHER, object)."""
    data = pickle_compat.load(file_path)
    if file_io.is_ms_container(data):
        return MS, file_io.load_ms_file(file_path)
    if isinstance(data, dict):
        return LEGACY, data
    return OTHER, data


def load_char_model(char_file):
    char_model = kin_char_model.KinCharModel("cpu")
    char_model.load_char_file(str(char_file))
    return char_model


def joint_dofs(joint_rot, char_model):
    """[N, joints - 1, 4] quaternions -> [N, dofs], or None (with a message) if they don't fit the model."""
    joint_rot = torch.as_tensor(np.asarray(joint_rot), dtype=torch.float32)
    num_joints = char_model.get_num_joints() - 1
    if joint_rot.dim() < 2 or joint_rot.shape[-2] != num_joints:
        print(f"  --char-file has {num_joints} joints, joint_rot has shape {tuple(joint_rot.shape)}; "
              f"no DOFs shown (wrong --char-file?)")
        return None
    return char_model.rot_to_dof(joint_rot).numpy()


def describe(value):
    """(type name, info) for one value: shapes for arrays, lengths for lists, summaries for terrains."""
    type_str = type(value).__name__
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
            # Handle scalar wrapped in array
            return type_str, f"Value: {value.item()} (Scalar Array)"
        return type_str, f"Shape: {tuple(value.shape)}, Dtype: {value.dtype}"
    if isinstance(value, (list, tuple)):
        info = f"Length: {len(value)}"
        if len(value) > 0:
            info += f", Item Type: {type(value[0]).__name__}"
            if all(isinstance(x, list) for x in value):
                info += f", {sum(len(x) for x in value)} items in total"
        return type_str, info
    if isinstance(value, (bool, int, float, str, np.generic)):
        return type_str, f"Value: {value}"
    if value is None:
        return type_str, "None"
    if hasattr(value, "hf") and hasattr(value, "dxdy"):  # a SubTerrain
        return type_str, (f"hf {tuple(np.shape(value.hf))}, dx {np.asarray(value.dxdy).reshape(-1)[0]:.4g}, "
                          f"min_point {np.round(np.asarray(value.min_point, dtype=np.float64), 4).tolist()}")
    # Generic object
    str_val = str(value)
    return type_str, str_val[:47] + "..." if len(str_val) > 50 else str_val


def print_table(items):
    # Table Header
    print(f"{'Key':<25} | {'Type':<20} | {'Info (Shape/Value/Len)'}")
    print("-" * 80)
    for key, value in items:
        type_str, info = describe(value)
        print(f"{str(key):<25} | {type_str:<20} | {info}")


def preview_ms(ms_data, char_model=None):
    for section, data in [("motion_data", ms_data.motion_data), ("terrain_data", ms_data.terrain_data)]:
        print(f"\n[{section}]")
        if data is None:
            print("  None")
            continue
        items = list(asdict(data).items())
        if section == "motion_data" and char_model is not None:
            dofs = joint_dofs(data.joint_rot, char_model)
            if dofs is not None:
                items.append(("joint_dof (rot_to_dof)", dofs))
        print_table(items)

    print("\n[misc_data]")
    if ms_data.misc_data is None:
        print("  None")
    elif len(ms_data.misc_data) == 0:
        print("  (empty)")
    else:
        print_table(sorted(ms_data.misc_data.items(), key=lambda item: str(item[0])))


def preview_pkl(file_path, char_file=None):
    """
    Load and preview the structure of a PARC .pkl file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False

    print(f"\nPreviewing file: {os.path.abspath(file_path)}")
    print("=" * 80)

    kind, data = load_pkl(file_path)
    char_model = None if char_file is None else load_char_model(char_file)

    if kind == MS:
        print("Format: MS motion file (motion_data / terrain_data / misc_data)")
        preview_ms(data, char_model)
    elif kind == LEGACY:
        print("Format: legacy pickle (top-level keys)")
        # Sort keys for consistent output
        print_table(sorted(data.items(), key=lambda item: str(item[0])))
        if char_model is not None and data.get("frames") is not None:
            print(f"(legacy frames are [N, 6 + DOFs]: the DOFs are frames[:, 6:]; "
                  f"the character has {char_model.get_dof_size()})")
    else:
        print(f"Root object is not a dict, it is: {type(data)}")
        # If it's not a dict, try to print a summary
        print(describe(data)[1])

    print("=" * 80)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preview the structure of a PARC .pkl file (MS or legacy).")
    parser.add_argument("file_path", help="path to the .pkl file")
    parser.add_argument("--char-file", default=None,
                        help="character .xml; shows the joint DOFs of MS files (rot_to_dof)")
    args = parser.parse_args(argv)
    return 0 if preview_pkl(args.file_path, args.char_file) else 1


if __name__ == "__main__":
    sys.exit(main())
