import pickle
import numpy as np
import sys
import os
import torch
import argparse
import warnings

# Suppress numpy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add PARC directory to sys.path to allow unpickling of custom classes
current_dir = os.path.dirname(os.path.abspath(__file__))
parc_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if parc_dir not in sys.path:
    sys.path.insert(0, parc_dir)

# Handle numpy version compatibility issues
try:
    import numpy.core
    sys.modules['numpy._core'] = np.core
    if hasattr(np.core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
except ImportError:
    pass

def format_array(arr, precision=4):
    """Format a numpy array/torch tensor for readable printing."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    
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

def print_frame_data(data, frame_idx):
    """
    Print detailed data for a specific frame, preserving original keys.
    """
    print(f"\n{'='*80}")
    print(f" VIEWING DATA FOR FRAME: {frame_idx}")
    print(f"{'='*80}")

    sorted_keys = sorted(data.keys())
    
    for key in sorted_keys:
        value = data[key]
        print(f"\nKey: '{key}'")
        
        # 1. Frames Data
        if key == 'frames':
            if frame_idx < len(value):
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
            if frame_idx < len(value):
                contact_frame = value[frame_idx]
                print(f"  Contact States ({len(contact_frame)} bodies):")
                print(f"    {format_array(contact_frame)}")
            else:
                print(f"  Frame index out of bounds for contacts.")

        # 3. HF Mask Indices
        elif key == 'hf_mask_inds':
            if value is not None and frame_idx < len(value):
                frame_mask = value[frame_idx]
                if frame_mask is not None:
                    if isinstance(frame_mask, torch.Tensor):
                        frame_mask = frame_mask.cpu().numpy()
                    print(f"  Occupied Grid Indices (Count: {len(frame_mask)}):")
                    if len(frame_mask) > 0:
                        print(f"    First 5 indices: {frame_mask[:5].tolist()}")
                else:
                    print("  No mask indices for this frame.")
            else:
                print("  None or out of bounds.")

        # 4. Terrain (SubTerrain object)
        elif key == 'terrain':
            print(f"  {value}")
            # Optionally print some properties if it's a SubTerrain object
            if hasattr(value, 'hf') and hasattr(value.hf, 'shape'):
                 print(f"  HF Shape: {value.hf.shape}")

        # 5. Metadata (scalars, strings, etc.)
        elif isinstance(value, (int, float, str, bool)):
             print(f"  Value: {value}")

        # 6. Other Arrays
        elif isinstance(value, (np.ndarray, list)) and hasattr(value, '__len__'):
             if len(value) > frame_idx and len(value) == len(data.get('frames', [])):
                 # Likely time-series data matching frame count
                 try:
                     print(f"  Frame Value: {value[frame_idx]}")
                 except:
                     print(f"  {value}")
             else:
                 # Metadata array or something else
                 if isinstance(value, np.ndarray):
                    print(f"  Shape: {value.shape}, Dtype: {value.dtype}")
                 else:
                    print(f"  List Length: {len(value)}")
        
        # 7. Fallback
        else:
             print(f"  {value}")

    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Preview specific frame data from a PARC .pkl motion file.")
    parser.add_argument("file_path", type=str, help="Path to the .pkl file")
    parser.add_argument("frame_idx", type=int, nargs='?', default=0, help="Index of the frame to view (default: 0)")
    
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        return

    try:
        with open(args.file_path, 'rb') as f:
            data = pickle.load(f)
        
        print_frame_data(data, args.frame_idx)

    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
