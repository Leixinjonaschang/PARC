"""Print the data of one frame of a PARC motion .pkl: an MS motion file or a legacy (develop-era) pickle.

MS files store rotations as quaternions (x, y, z, w): root_rot [N, 4] and joint_rot
[N, joints - 1, 4].  With --char-file, the frame's joint DOFs are derived from
joint_rot (KinCharModel.rot_to_dof), in the order of frames[:, 6:] of legacy files
and GMR output.  Legacy files are read with parc.util.pickle_compat.

Usage, from anywhere:

    python scripts/tools/pkl_preview_example.py <file.pkl> [frame_idx] [--char-file <character .xml>]

frame_idx defaults to 0; negative values count from the end.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

# Make "parc" and "scripts" importable when this file is run as a plain script.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

import parc.util.file_io_helper as file_io_helper
from scripts.tools.pkl_preview import LEGACY, MS, describe, joint_dofs, load_char_model, load_pkl

# Suppress numpy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def format_array(arr, precision=4):
    """Format a numpy array/torch tensor for readable printing."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)

    if arr.size == 0:
        return "[]"

    # If scalar
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        return f"{arr.item():.{precision}f}"

    # If 1D array
    if arr.ndim == 1:
        items = [f"{x:.{precision}f}" for x in arr]
        if len(items) > 10:
             return f"[{', '.join(items[:5])}, ..., {', '.join(items[-5:])}] (Size: {arr.size})"
        return f"[{', '.join(items)}]"

    return f"Array shape={arr.shape}, dtype={arr.dtype}"


def resolve_frame_idx(frame_idx, num_frames):
    """Python-style index into num_frames frames, or None if out of bounds."""
    if -num_frames <= frame_idx < num_frames:
        return frame_idx % num_frames
    return None


def print_hf_mask_inds(value, frame_idx):
    if value is not None and resolve_frame_idx(frame_idx, len(value)) is not None:
        frame_mask = value[frame_idx]
        if frame_mask is not None:
            if isinstance(frame_mask, torch.Tensor):
                frame_mask = frame_mask.cpu().numpy()
            print(f"  Occupied Grid Indices (Count: {len(frame_mask)}):")
            if len(frame_mask) > 0:
                print(f"    First 5 indices: {np.asarray(frame_mask)[:5].tolist()}")
        else:
            print("  No mask indices for this frame.")
    else:
        print("  None or out of bounds.")


def print_ms_frame(ms_data, frame_idx, char_model=None):
    motion_data = ms_data.motion_data
    if motion_data is None:
        print("\nNo motion_data (terrain-only file).")
    else:
        num_frames = motion_data.root_pos.shape[0]
        idx = resolve_frame_idx(frame_idx, num_frames)
        if idx is None:
            print(f"\nFrame index {frame_idx} out of bounds. Total frames: {num_frames}")
            return False
        print(f"\n[motion_data] frame {idx} of {num_frames} ({motion_data.fps} fps, {motion_data.loop_mode})")
        print(f"  Root Position (x, y, z): {format_array(motion_data.root_pos[idx])}")
        print(f"  Root Rotation (quat x, y, z, w): {format_array(motion_data.root_rot[idx])}")
        joint_rot = np.asarray(motion_data.joint_rot[idx])
        print(f"  Joint Rotations: {joint_rot.shape[0]} quaternions (x, y, z, w)")
        if char_model is not None:
            dofs = joint_dofs(motion_data.joint_rot[idx:idx + 1], char_model)
            if dofs is not None:
                print(f"  Joint DOFs ({dofs.shape[-1]} dims, rot_to_dof):")
                print(f"    {format_array(dofs[0])}")
        else:
            print("    (pass --char-file to see them as joint DOFs)")
        if motion_data.body_contacts is not None:
            contact_frame = np.asarray(motion_data.body_contacts[idx])
            print(f"  Contact States ({len(contact_frame)} bodies):")
            print(f"    {format_array(contact_frame)}")
        frame_idx = idx

    terrain_data = ms_data.terrain_data
    print("\n[terrain_data]")
    if terrain_data is None:
        print("  None")
    else:
        print(f"  HF Shape: {tuple(terrain_data.hf.shape)}, dx: {float(terrain_data.dx):.4g}, "
              f"min_point: {format_array(terrain_data.min_point)}")

    misc_data = ms_data.misc_data or {}
    print("\n[misc_data]")
    for key in sorted(misc_data, key=str):
        value = misc_data[key]
        print(f"\nKey: '{key}'")
        if key == file_io_helper.HF_MASK_INDS_KEY:
            print_hf_mask_inds(value, frame_idx)
        elif key == file_io_helper.BODY_CONSTRAINTS_KEY:
            active = [(body_id, c["start_frame_idx"], c["end_frame_idx"])
                      for body_id, constraints in enumerate(value) for c in constraints
                      if c["start_frame_idx"] <= frame_idx <= c["end_frame_idx"]]
            print(f"  Constraints active at this frame (body id, start, end): {active}")
        else:
            print(f"  {describe(value)[1]}")
    return True


def print_frame_data(data, frame_idx):
    """
    Print detailed data for a specific frame of a legacy file, preserving original keys.
    """
    sorted_keys = sorted(data.keys(), key=str)

    for key in sorted_keys:
        value = data[key]
        print(f"\nKey: '{key}'")

        # 1. Frames Data
        if key == 'frames':
            if resolve_frame_idx(frame_idx, len(value)) is not None:
                frame_data = value[frame_idx]

                # Assuming standard PARC format:
                # [0:3] Root Pos, [3:6] Root Rot (Exp Map), [6:] Joint DOFs
                root_pos = frame_data[0:3]
                root_rot = frame_data[3:6]
                joint_dofs = frame_data[6:]

                print(f"  Root Position (x, y, z): {format_array(root_pos)}")
                print(f"  Root Rotation (Exp Map): {format_array(root_rot)}")
                print(f"  Joint DOFs ({len(joint_dofs)} dims):")
                print(f"    {format_array(joint_dofs)}")
            else:
                print(f"  Frame index {frame_idx} out of bounds. Total frames: {len(value)}")

        # 2. Contact Data
        elif key == 'contacts':
            if resolve_frame_idx(frame_idx, len(value)) is not None:
                contact_frame = value[frame_idx]
                print(f"  Contact States ({len(contact_frame)} bodies):")
                print(f"    {format_array(contact_frame)}")
            else:
                print(f"  Frame index out of bounds for contacts.")

        # 3. HF Mask Indices
        elif key == 'hf_mask_inds':
            print_hf_mask_inds(value, frame_idx)

        # 4. Terrain (SubTerrain object)
        elif key == 'terrain':
            print(f"  {describe(value)[1]}")

        # 5. Metadata (scalars, strings, etc.)
        elif isinstance(value, (int, float, str, bool)):
             print(f"  Value: {value}")

        # 6. Other Arrays
        elif isinstance(value, (np.ndarray, list)) and hasattr(value, '__len__'):
             frames = data.get('frames')
             num_frames = 0 if frames is None else len(frames)
             if len(value) == num_frames and resolve_frame_idx(frame_idx, num_frames) is not None:
                 # Likely time-series data matching frame count
                 print(f"  Frame Value: {value[frame_idx]}")
             else:
                 # Metadata array or something else
                 print(f"  {describe(value)[1]}")

        # 7. Fallback
        else:
             print(f"  {describe(value)[1]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preview one frame of a PARC .pkl motion file (MS or legacy).")
    parser.add_argument("file_path", type=str, help="Path to the .pkl file")
    parser.add_argument("frame_idx", type=int, nargs='?', default=0,
                        help="Index of the frame to view (default: 0; negative counts from the end)")
    parser.add_argument("--char-file", default=None,
                        help="character .xml; shows the joint DOFs of MS files (rot_to_dof)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        return 1

    kind, data = load_pkl(args.file_path)
    char_model = None if args.char_file is None else load_char_model(args.char_file)

    print(f"\n{'='*80}")
    print(f" VIEWING DATA FOR FRAME: {args.frame_idx}")
    print(f"{'='*80}")
    if kind == MS:
        ok = print_ms_frame(data, args.frame_idx, char_model)
    elif kind == LEGACY:
        print_frame_data(data, args.frame_idx)
        ok = True
    else:
        print(f"Root object is not a dict, it is: {type(data)}")
        ok = False
    print(f"{'='*80}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
