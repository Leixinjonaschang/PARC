"""Extract {frames, fps} from PARC motion files, e.g. as retargeting (GMR) input.

frames is [N, 6 + DOFs]: root position, root rotation (exp map), joint DOFs.

* Legacy (develop-era) pickles are read without PARC, so this also runs where PARC
  is not installed (e.g. GMR's environment, NumPy 2).  Classes that cannot be
  imported (util.terrain_util.SubTerrain, ...) become placeholder objects; only
  frames and fps are kept.  Torch tensors pickled on a GPU are loaded on the CPU.
* MS files store rotations as quaternions; frames are rebuilt with the character
  model (--char-file, default data/assets/humanoid.xml) through
  MotionFrames.get_legacy_mlib_format, which needs PARC (parc.*, torch).

Usage:

    python scripts/tools/pkl_extract_frames_fps.py --input <.pkl or directory> [--output <.pkl or directory>]
        [--char-file <character .xml>]

Without --output, each file is written next to its input as <name>_cleaned.pkl.
Directories are processed recursively, skipping files named *_cleaned*.
"""
import argparse
import importlib.util
import io
import os
import pickle
import sys
import warnings

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_CHAR_FILE = os.path.join(REPO_ROOT, "data", "assets", "humanoid.xml")
MS_CONTAINER_KEYS = ("motion_data", "terrain_data", "misc_data")

# NumPy 2 pickles reference numpy._core.*, which NumPy 1.x does not have.  Remap
# inside the unpickler only: a global sys.modules alias breaks extensions built
# against NumPy 2 (doc/porting_plan.md DB6).
REMAP_NUMPY_CORE = importlib.util.find_spec("numpy._core") is None


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


def _load_storage_on_cpu(data):
    # torch.storage._load_from_bytes restores storages on the device they were pickled on
    import torch
    try:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(io.BytesIO(data), map_location="cpu")


# 2. 自定义 Unpickler
class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if REMAP_NUMPY_CORE and (module == "numpy._core" or module.startswith("numpy._core.")):
            module = "numpy.core" + module[len("numpy._core"):]

        if module == "torch.storage" and name == "_load_from_bytes":
            if importlib.util.find_spec("torch") is None:
                return DummyObject
            return _load_storage_on_cpu

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


def is_ms_container(data):
    return isinstance(data, dict) and any(key in data for key in MS_CONTAINER_KEYS)


def extract_legacy(data):
    """frames and fps from a legacy dict or object, without PARC."""
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
        return None

    # === 关键步骤：清洗数据，移除 DummyObject ===
    cleaned_extracted_data = sanitize_data(extracted_data)

    # 检查 frames 是否被提取为 dict
    if 'frames' in cleaned_extracted_data:
        frames_val = cleaned_extracted_data['frames']

        # torch tensors (frames saved from a tensor) become numpy arrays
        if hasattr(frames_val, 'detach') and hasattr(frames_val, 'cpu'):
            frames_val = frames_val.detach().cpu().numpy()
            cleaned_extracted_data['frames'] = frames_val

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

    return cleaned_extracted_data


class MSExtractor:
    """frames and fps from MS files; imports PARC (and loads the character) on first use."""

    def __init__(self, char_file):
        self._char_file = char_file
        self._char_model = None

    def __call__(self, file_path):
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        import torch
        import parc.anim.kin_char_model as kin_char_model
        import parc.util.file_io as file_io
        import parc.util.motion_util as motion_util

        if self._char_model is None:
            self._char_model = kin_char_model.KinCharModel("cpu")
            self._char_model.load_char_file(self._char_file)

        motion_data = file_io.load_ms_file(file_path).motion_data
        if motion_data is None:
            return None
        num_joints = self._char_model.get_num_joints() - 1
        if motion_data.joint_rot.ndim != 3 or motion_data.joint_rot.shape[1] != num_joints:
            raise ValueError("joint_rot has shape {}, but {} has {} joints (wrong --char-file?)".format(
                motion_data.joint_rot.shape, self._char_file, num_joints))

        motion_frames = motion_util.MotionFrames(
            root_pos=torch.as_tensor(motion_data.root_pos, dtype=torch.float32),
            root_rot=torch.as_tensor(motion_data.root_rot, dtype=torch.float32),
            joint_rot=torch.as_tensor(motion_data.joint_rot, dtype=torch.float32))
        frames, _ = motion_frames.get_legacy_mlib_format(self._char_model)
        return {"frames": frames.numpy(), "fps": motion_data.fps}


WRITTEN = "written"
SKIPPED = "skipped"
FAILED = "failed"


def extract_frames_fps(file_path, output_path=None, ms_extractor=None):
    """Returns WRITTEN, SKIPPED (no frames, e.g. a terrain-only file) or FAILED."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return FAILED

    try:
        with open(file_path, 'rb') as f:
            data = RobustUnpickler(f).load()
    except Exception as e:
        print(f"Error: Failed to load {file_path}. Reason: {e}")
        return FAILED

    if is_ms_container(data):
        if ms_extractor is None:
            ms_extractor = MSExtractor(DEFAULT_CHAR_FILE)
        try:
            extracted_data = ms_extractor(file_path)
        except Exception as e:
            print(f"Error: Failed to read MS file {file_path}. Reason: {type(e).__name__}: {e}")
            return FAILED
    else:
        extracted_data = extract_legacy(data)

    if not extracted_data:
        print(f"Skipping {file_path}: No 'frames' or 'fps' found.")
        return SKIPPED

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_cleaned{ext}"

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(extracted_data, f)
        print(f"Success: Extracted frames/fps to {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return FAILED
    return WRITTEN


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract frames and fps from PARC motion files (legacy pickles without PARC, MS files with it).")
    parser.add_argument("--input", required=True, help="Input .pkl file or directory")
    parser.add_argument("--output", help="Output .pkl file (if input is file) or directory (if input is directory)")
    parser.add_argument("--char-file", default=DEFAULT_CHAR_FILE,
                        help="character .xml of MS input files (default: %(default)s)")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ms_extractor = MSExtractor(args.char_file)

    if os.path.isdir(args.input):
        results = []
        for root, dirs, files in sorted(os.walk(args.input)):
            for file in sorted(files):
                if file.endswith(".pkl") and "_cleaned" not in file:
                    src = os.path.join(root, file)
                    dest = None
                    if args.output:
                        rel_path = os.path.relpath(src, args.input)
                        dest = os.path.join(args.output, rel_path)
                    results.append(extract_frames_fps(src, dest, ms_extractor))
        if len(results) == 0:
            print(f"No .pkl files found in {args.input}")
            return 1
        print(f"Written {results.count(WRITTEN)}, skipped {results.count(SKIPPED)}, failed {results.count(FAILED)}.")
        return 1 if FAILED in results else 0
    return 1 if extract_frames_fps(args.input, args.output, ms_extractor) == FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
