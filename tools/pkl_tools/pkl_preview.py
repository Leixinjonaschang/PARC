import pickle
import numpy as np
import sys
import os
import torch
import warnings

# Suppress numpy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add PARC directory to sys.path to allow unpickling of custom classes (e.g. util.*)
# The script is located at PARC/tools/pkl_tools/pkl_preview.py
# We need to add the path that contains the 'util' package, which is the 'PARC' directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
parc_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if parc_dir not in sys.path:
    sys.path.insert(0, parc_dir)

# Handle numpy version compatibility issues (similar to MotionLib)
try:
    import numpy.core
    sys.modules['numpy._core'] = np.core
    if hasattr(np.core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
except ImportError:
    pass

def preview_pkl(file_path):
    """
    Load and preview the structure of a PARC training data .pkl file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"\nPreviewing file: {os.path.abspath(file_path)}")
    print("=" * 80)

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            print(f"Root object is not a dict, it is: {type(data)}")
            # If it's not a dict, try to print a summary
            if hasattr(data, 'shape'):
                print(f"Shape: {data.shape}")
            else:
                print(data)
            return

        # Sort keys for consistent output
        keys = sorted(data.keys())
        
        # Table Header
        print(f"{'Key':<25} | {'Type':<20} | {'Info (Shape/Value/Len)'}")
        print("-" * 80)

        for key in keys:
            value = data[key]
            type_str = type(value).__name__
            
            info = ""
            if isinstance(value, (np.ndarray, torch.Tensor)):
                if value.ndim == 0 or (value.ndim == 1 and value.size == 1):
                    # Handle scalar wrapped in array
                    try:
                        val = value.item()
                        info = f"Value: {val} (Scalar Array)"
                    except:
                         info = f"Shape: {value.shape}, Dtype: {value.dtype}"
                else:
                    info = f"Shape: {value.shape}, Dtype: {value.dtype}"
            
            elif isinstance(value, (list, tuple)):
                info = f"Length: {len(value)}"
                if len(value) > 0:
                    first_item_type = type(value[0]).__name__
                    info += f", Item Type: {first_item_type}"
            
            elif isinstance(value, (int, float, bool, str)):
                info = f"Value: {value}"
            
            elif value is None:
                info = "None"
                
            else:
                # Generic object
                str_val = str(value)
                if len(str_val) > 50:
                    info = str_val[:47] + "..."
                else:
                    info = str_val

            print(f"{key:<25} | {type_str:<20} | {info}")
            
        print("=" * 80)

    except Exception as e:
        print(f"Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python pkl_preview.py <path_to_pkl_file>")
        print("Example: python pkl_preview.py ../../Data/initial/running/run_000.pkl\n")
    else:
        preview_pkl(sys.argv[1])

