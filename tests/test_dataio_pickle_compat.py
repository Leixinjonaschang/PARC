"""parc.util.pickle_compat: develop-era module paths, NumPy 2 pickles, GPU tensors."""
import importlib.util
import io
import pickle
import pickletools
import sys
from dataclasses import asdict
from unittest import mock

import numpy as np
import pytest
import torch
import torch.serialization

import parc.motion_synthesis.motion_opt.motion_optimization as moopt
import parc.util.pickle_compat as pickle_compat
import parc.util.terrain_util as terrain_util


def _rewrite_modules(data: bytes, old: str, new: str) -> bytes:
    """Rewrite module names in pickle bytes (``old`` and its submodules -> ``new``).

    Handles GLOBAL (protocols 0-3) and length-prefixed SHORT_BINUNICODE strings
    (protocols >= 4).  FRAME headers are dropped since their lengths change;
    framing is optional for unpicklers.
    """
    def rename(module):
        if module == old or module.startswith(old + "."):
            return new + module[len(old):]
        return None

    out = bytearray()
    last = 0
    for op, arg, pos in pickletools.genops(data):
        if op.name == "FRAME":
            out += data[last:pos]
            last = pos + 9
        elif op.name == "GLOBAL":
            module, name = arg.split(" ", 1)
            renamed = rename(module)
            if renamed is not None:
                out += data[last:pos] + b"c" + renamed.encode() + b"\n" + name.encode() + b"\n"
                last = pos + 1 + len(module.encode()) + 1 + len(name.encode()) + 1
        elif op.name == "SHORT_BINUNICODE":
            renamed = rename(arg)
            if renamed is not None:
                encoded = renamed.encode()
                out += data[last:pos] + b"\x8c" + bytes([len(encoded)]) + encoded
                last = pos + 2 + len(arg.encode())
    out += data[last:]
    assert bytes(out) != data, "nothing to rewrite"
    return bytes(out)


def _numpy2_pickle(obj, protocol):
    # NumPy 2 pickles reference numpy._core.* where NumPy 1.x uses numpy.core.*
    return _rewrite_modules(pickle.dumps(obj, protocol=protocol), "numpy.core", "numpy._core")


def _develop_pickle(obj, protocol=4):
    # develop pickles name classes by develop's flat module paths
    data = pickle.dumps(obj, protocol=protocol)
    data = _rewrite_modules(data, "parc.util", "util")
    return _rewrite_modules(data, "parc.motion_synthesis.motion_opt", "tools.motion_opt")


def _legacy_modules_importable():
    return any(importlib.util.find_spec(m) is not None for m in ("util", "tools"))


@pytest.mark.parametrize("old, new", pickle_compat.MODULE_REMAP)
def test_remap_targets_exist(old, new):
    assert importlib.util.find_spec(new) is not None, (old, new)


@pytest.mark.parametrize("module, expected", [
    ("util.terrain_util", "parc.util.terrain_util"),
    ("util", "parc.util"),
    ("util.arg_parser", "parc.motion_tracker.util.arg_parser"),
    ("anim.kin_char_model", "parc.anim.kin_char_model"),
    ("tools.motion_opt.motion_optimization", "parc.motion_synthesis.motion_opt.motion_optimization"),
    ("tools.procgen.astar", "parc.motion_synthesis.procgen.astar"),
    ("zmotion_editing_tools.motion_edit_lib", "parc.util.motion_edit_lib"),
    ("diffusion.mdm", "parc.motion_generator.mdm"),
    ("diffusion.motion_generator", "parc.motion_generator.base_generator"),
    ("learning.normalizer", "parc.motion_tracker.learning.normalizer"),
    ("learning.nets.net_builder", "parc.util.nets.net_builder"),
    ("envs.ig_parkour.dm_env", "parc.motion_tracker.envs.ig_parkour.dm_env"),
    # not legacy paths: unchanged
    ("utility", "utility"),
    ("utils.x", "utils.x"),
    ("tools.motion_file_proc.pkl_preview", "tools.motion_file_proc.pkl_preview"),
    ("parc.util.terrain_util", "parc.util.terrain_util"),
    ("numpy.core.multiarray", "numpy.core.multiarray"),
    ("torch._utils", "torch._utils"),
    ("collections", "collections"),
])
def test_remap_module(module, expected):
    assert pickle_compat.remap_module(module) == expected


def test_remap_numpy_core_only_without_numpy_private_core():
    numpy_has_private_core = importlib.util.find_spec("numpy._core") is not None
    assert pickle_compat.REMAP_NUMPY_CORE == (not numpy_has_private_core)
    expected = "numpy._core.multiarray" if numpy_has_private_core else "numpy.core.multiarray"
    assert pickle_compat.remap_module("numpy._core.multiarray") == expected


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_numpy2_pickle(protocol):
    obj = {
        "frames": np.arange(12, dtype=np.float32).reshape(4, 3),
        "inds": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "fps": np.float64(30.0),
    }
    data = _numpy2_pickle(obj, protocol)
    assert b"numpy._core" in data
    numpy_alias_before = sys.modules.get("numpy._core")

    if pickle_compat.REMAP_NUMPY_CORE:
        with pytest.raises(ModuleNotFoundError, match="numpy._core"):
            pickle.loads(data)

    loaded = pickle_compat.loads(data)
    assert loaded["frames"].dtype == np.float32 and loaded["inds"].dtype == np.int64
    np.testing.assert_array_equal(loaded["frames"], obj["frames"])
    np.testing.assert_array_equal(loaded["inds"], obj["inds"])
    assert loaded["fps"] == 30.0
    # remapped inside the unpickler only: no global numpy._core alias (DB6)
    assert sys.modules.get("numpy._core") is numpy_alias_before


def _pickle_as_if_on_device(obj, location):
    # torch pickles a storage's device with torch.serialization.location_tag
    with mock.patch.object(torch.serialization, "location_tag", lambda storage: location):
        return pickle.dumps(obj)


def test_cuda_pickled_tensor_loads_on_cpu():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    data = _pickle_as_if_on_device({"t": tensor, "inds": [torch.tensor([[1, 2]])]}, "cuda:99")
    assert b"cuda:99" in data

    with pytest.raises(RuntimeError):  # no such GPU (or no CUDA at all)
        pickle.loads(data)

    loaded = pickle_compat.loads(data)
    assert loaded["t"].device.type == "cpu" and loaded["inds"][0].device.type == "cpu"
    assert torch.equal(loaded["t"], tensor)
    assert torch.equal(loaded["inds"][0], torch.tensor([[1, 2]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_cuda_tensor_map_location():
    gpu_data = pickle.dumps(torch.arange(3, dtype=torch.float32, device="cuda:0"))
    loaded = pickle_compat.loads(gpu_data)
    assert loaded.device.type == "cpu" and torch.equal(loaded, torch.arange(3, dtype=torch.float32))
    assert pickle_compat.loads(gpu_data, map_location=None).device == torch.device("cuda:0")

    cpu_data = pickle.dumps(torch.ones(2))
    assert pickle_compat.loads(cpu_data, map_location="cuda:0").device == torch.device("cuda:0")


def test_develop_module_paths():
    terrain = terrain_util.SubTerrain(x_dim=3, y_dim=2, dx=0.1, dy=0.1, device="cpu")
    terrain.hf[1, 1] = 0.5
    constraint = moopt.BodyConstraint(start_frame_idx=1, end_frame_idx=4,
                                      constraint_point=torch.tensor([0.1, 0.2, 0.3]))
    data = _develop_pickle({"terrain": terrain, "opt:body_constraints": [[constraint]]})
    assert b"util.terrain_util" in data and b"tools.motion_opt.motion_optimization" in data
    assert b"parc." not in data

    if not _legacy_modules_importable():
        with pytest.raises(ModuleNotFoundError):
            pickle.loads(data)

    loaded = pickle_compat.loads(data)
    assert isinstance(loaded["terrain"], terrain_util.SubTerrain)
    assert torch.equal(loaded["terrain"].hf, terrain.hf)
    loaded_constraint = loaded["opt:body_constraints"][0][0]
    assert isinstance(loaded_constraint, moopt.BodyConstraint)
    assert asdict(loaded_constraint).keys() == asdict(constraint).keys()
    assert (loaded_constraint.start_frame_idx, loaded_constraint.end_frame_idx) == (1, 4)
    assert torch.equal(loaded_constraint.constraint_point, constraint.constraint_point)


def test_missing_legacy_module_error():
    data = b"cutil.mp_util\nMPUtil\n."  # util.mp_util was removed upstream
    with pytest.raises(ModuleNotFoundError, match=r"util\.mp_util.*parc\.util\.mp_util"):
        pickle_compat.loads(data)


def test_missing_legacy_class_error():
    data = b"cutil.terrain_util\nNoSuchClass\n."
    with pytest.raises(AttributeError, match=r"parc\.util\.terrain_util.*NoSuchClass"):
        pickle_compat.loads(data)


def test_load_from_path_or_file(tmp_path):
    path = tmp_path / "obj.pkl"
    obj = {"a": np.arange(3), "b": torch.ones(2)}
    path.write_bytes(pickle.dumps(obj))

    for loaded in (pickle_compat.load(path), pickle_compat.load(str(path))):
        np.testing.assert_array_equal(loaded["a"], obj["a"])
        assert torch.equal(loaded["b"], obj["b"])
    with open(path, "rb") as f:
        assert torch.equal(pickle_compat.load(f)["b"], obj["b"])
    assert pickle_compat.load(io.BytesIO(pickle.dumps([1, 2]))) == [1, 2]
