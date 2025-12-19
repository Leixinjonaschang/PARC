import pickle
import os
import sys
import argparse
import warnings
import numpy as np

# === Numpy Version Compatibility Patch ===
# This is crucial because some pickle files reference 'numpy._core' which might not exist in current environment,
# causing the Unpickler to fail to find the class and fallback to DummyObject for standard arrays.
try:
    # Try to access numpy.core (older versions) or map _core to it
    if hasattr(np, 'core'):
        sys.modules['numpy._core'] = np.core
        if hasattr(np.core, 'multiarray'):
             sys.modules['numpy._core.multiarray'] = np.core.multiarray
    # If using very new numpy (2.0+), things might be different, but usually this backward compat helps.
except Exception as e:
    pass
# =========================================

# 1. 定义一个通用的占位对象，用于替换无法加载的类
class DummyObject:
    def __init__(self, *args, **kwargs):
        pass
    
    def __setstate__(self, state):
        # 某些对象被 pickle 保存时，__setstate__ 可能会接收 tuple 而不是 dict
        if isinstance(state, dict):
            self.__dict__ = state
        else:
            self._raw_state = state
            # 尝试从 tuple 中恢复属性
            if isinstance(state, tuple):
                for item in state:
                    if isinstance(item, dict):
                        if not hasattr(self, '__dict__'):
                             self.__dict__ = {}
                        self.__dict__.update(item)

    def __getstate__(self):
        return self.__dict__

    def __repr__(self):
        keys = list(self.__dict__.keys()) if hasattr(self, '__dict__') else "No __dict__"
        return f"<DummyObject keys={keys}>"

# 2. 自定义 Unpickler
class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Special handling for numpy if the patch above didn't catch it
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        
        try:
            return super().find_class(module, name)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # print(f"Warning: Could not load {module}.{name}, using DummyObject")
            return DummyObject

# Helper to manually reconstruct numpy array from state tuple if needed
def reconstruct_numpy_from_state(state):
    """
    Attempt to reconstruct a numpy array from its __setstate__ tuple.
    State format typically: (version, shape, dtype, is_fortran, rawdata)
    """
    if isinstance(state, (tuple, list)) and len(state) >= 5:
        try:
            version = state[0]
            shape = tuple(state[1])
            dtype = state[2]
            is_fortran = state[3]
            rawdata = state[4]
            
            if isinstance(version, int) and isinstance(dtype, np.dtype):
                # Create empty array
                arr = np.ndarray(shape=shape, dtype=dtype, order='F' if is_fortran else 'C')
                # Fill data
                if isinstance(rawdata, (bytes, bytearray)):
                    arr.data = rawdata
                return arr
        except Exception as e:
            # print(f"Failed to reconstruct numpy array: {e}")
            pass
    return None

# 3. 数据清洗函数：递归将 DummyObject 转换为 dict
def sanitize_data(data):
    """
    递归地将数据中的 DummyObject 转换为标准字典或列表，
    确保保存后的 pickle 文件不依赖于任何自定义类。
    """
    if isinstance(data, DummyObject):
        # If this DummyObject holds a numpy array state, try to reconstruct it
        if hasattr(data, '_raw_state'):
            arr = reconstruct_numpy_from_state(data._raw_state)
            if arr is not None:
                return arr
            
            # If not array, continue with extraction
            return sanitize_data(data._raw_state)
            
        # 优先使用 __dict__
        if hasattr(data, '__dict__') and data.__dict__:
            return sanitize_data(data.__dict__)
            
        return {} # 无法提取内容，返回空字典

    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    
    if isinstance(data, (list, tuple)):
        # 将 tuple 也转换为 list，通用性更好
        return [sanitize_data(x) for x in data]
    
    # Numpy 数组、基础类型保持不变
    return data

def extract_frames_fps(file_path, output_path=None):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    data = None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        try:
            with open(file_path, 'rb') as f:
                data = RobustUnpickler(f).load()
        except Exception as e2:
            print(f"Error: Failed to load {file_path}. Reason: {e2}")
            return
    except Exception as e:
        try:
            with open(file_path, 'rb') as f:
                 data = RobustUnpickler(f).load()
        except Exception as e2:
             print(f"Error: Failed to load {file_path}. Reason: {e2}")
             return

    if data is None:
        return

    extracted_data = {}
    
    # 提取 frames 和 fps
    if isinstance(data, dict):
        if 'frames' in data: extracted_data['frames'] = data['frames']
        if 'fps' in data: extracted_data['fps'] = data['fps']
    else:
        # 对象属性提取
        if hasattr(data, 'frames'): extracted_data['frames'] = data.frames
        elif hasattr(data, '__dict__') and 'frames' in data.__dict__: extracted_data['frames'] = data.__dict__['frames']
        
        if hasattr(data, 'fps'): extracted_data['fps'] = data.fps
        elif hasattr(data, '__dict__') and 'fps' in data.__dict__: extracted_data['fps'] = data.__dict__['fps']

        if hasattr(data, '_raw_state') and isinstance(data._raw_state, tuple):
             for item in data._raw_state:
                 if isinstance(item, dict):
                     if 'frames' in item: extracted_data['frames'] = item['frames']
                     if 'fps' in item: extracted_data['fps'] = item['fps']

    if not extracted_data:
        print(f"Skipping {file_path}: No 'frames' or 'fps' found.")
        return

    # === 关键步骤：清洗数据，移除 DummyObject ===
    cleaned_extracted_data = sanitize_data(extracted_data)
    
    # 检查 frames 是否被提取为 dict
    if 'frames' in cleaned_extracted_data:
        frames_val = cleaned_extracted_data['frames']
        
        # 如果 frames 依然不是 ndarray，再次尝试搜寻
        if not isinstance(frames_val, np.ndarray):
             # 情况 1: frames 是个 dict，尝试解包
            if isinstance(frames_val, dict):
                found_array = False
                
                # A. 先检查常见的 key
                potential_keys = ['data', 'npy', 'numpy', '_data', 'motion']
                for k in potential_keys:
                    if k in frames_val and isinstance(frames_val[k], (np.ndarray, list)):
                        cleaned_extracted_data['frames'] = frames_val[k]
                        found_array = True
                        break
                
                # B. 如果是 dict 且包含了 array (sanitize_data 递归处理后 _raw_state 可能已经变成了 array)
                # 比如 {'_raw_state': array([...])}，因为 reconstruct_numpy_from_state 可能生效了
                if not found_array:
                    # 遍历 values 找 array
                     arrays = [v for v in frames_val.values() if isinstance(v, np.ndarray)]
                     if len(arrays) >= 1:
                         # 优先找形状像 frames 的 (N, D)
                         cleaned_extracted_data['frames'] = arrays[0] 
                         found_array = True

            # 情况 2: 确保 frames 最终是 numpy 数组 (如果是 list)
            if isinstance(cleaned_extracted_data['frames'], list):
                 # 简单的 list 可能不是 frames，但如果它很大，可能是
                 cleaned_extracted_data['frames'] = np.array(cleaned_extracted_data['frames'])
             
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_cleaned{ext}"
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(cleaned_extracted_data, f)
        print(f"Success: Extracted frames/fps to {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames and fps from pickle files, handling missing dependencies (like 'util').")
    parser.add_argument("--input", help="Input .pkl file or directory")
    parser.add_argument("--output", help="Output .pkl file (if input is file) or directory (if input is directory)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        count = 0
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.endswith(".pkl") and "_cleaned" not in file:
                    src = os.path.join(root, file)
                    dest = None
                    if args.output:
                        rel_path = os.path.relpath(src, args.input)
                        dest = os.path.join(args.output, rel_path)
                    extract_frames_fps(src, dest)
                    count += 1
        if count == 0:
            print(f"No .pkl files found in {args.input}")
    else:
        extract_frames_fps(args.input, args.output)
