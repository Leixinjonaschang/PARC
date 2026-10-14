"""Unpickling compatibility layer for pickles written by older code.

Plain ``pickle.load`` fails on three kinds of files this project still has to
read:

* develop-era files pickle classes under the old flat module layout, for
  example ``util.terrain_util.SubTerrain`` or
  ``tools.motion_opt.motion_optimization.BodyConstraint``;
* files written under NumPy 2 (for example GMR retargeting output) reference
  ``numpy._core``, which NumPy 1.x does not have;
* torch tensors pickled on a GPU are restored on that GPU (``cuda:1``, ...),
  which fails on machines without it.

:class:`CompatUnpickler` fixes all three while unpickling, through
:meth:`CompatUnpickler.find_class` only:

* legacy module paths are rewritten to their ``parc.*`` location (see
  :data:`MODULE_REMAP` and :func:`remap_module`);
* ``numpy._core*`` is rewritten to ``numpy.core*``, but only when the
  installed NumPy has no ``numpy._core`` (NumPy < 2);
* ``torch.storage._load_from_bytes`` is replaced by a loader that honours
  ``map_location`` (default ``"cpu"``).

Nothing is aliased in ``sys.modules``.  A global ``numpy._core`` alias (as
develop's ``anim/motion_lib.py`` installed) shadows the real module for any
extension built against NumPy 2 that is imported later, which then crashes
(opencv >= 4.10 segfaults).
"""
import functools
import importlib.util
import io
import os
import pickle
from typing import Any, BinaryIO, Tuple, Union

import torch

# (legacy module or package, port module or package).  A rule applies to the
# module itself and to all of its submodules; the most specific rule wins.
# Targets follow appendix B of doc/porting_plan.md.  Develop modules that
# upstream deleted (util.mp_util, util.tb_logger, learning.mp_optimizer, ...)
# have no target; unpickling them raises ModuleNotFoundError.
MODULE_REMAP: Tuple[Tuple[str, str], ...] = (
    ("util.arg_parser", "parc.motion_tracker.util.arg_parser"),
    ("diffusion.motion_generator", "parc.motion_generator.base_generator"),
    ("learning.nets", "parc.util.nets"),
    ("tools.motion_opt", "parc.motion_synthesis.motion_opt"),
    ("tools.procgen", "parc.motion_synthesis.procgen"),
    ("zmotion_editing_tools", "parc.util"),
    ("diffusion", "parc.motion_generator"),
    ("learning", "parc.motion_tracker.learning"),
    ("envs", "parc.motion_tracker.envs"),
    ("anim", "parc.anim"),
    ("util", "parc.util"),
)

NUMPY_CORE_REMAP: Tuple[str, str] = ("numpy._core", "numpy.core")

_RULES = tuple(sorted(MODULE_REMAP, key=lambda rule: rule[0].count("."), reverse=True))


def _numpy_has_private_core() -> bool:
    try:
        return importlib.util.find_spec(NUMPY_CORE_REMAP[0]) is not None
    except ValueError:
        # Present in sys.modules without a __spec__: lookups already resolve.
        return True


# NumPy >= 2 has numpy._core (and only a deprecated numpy.core shim), so
# NumPy 2 pickles load as they are there.
REMAP_NUMPY_CORE: bool = not _numpy_has_private_core()


def _replace_prefix(module: str, old: str, new: str):
    if module == old or module.startswith(old + "."):
        return new + module[len(old):]
    return None


def remap_module(module: str) -> str:
    """Return the module to import for ``module`` as recorded in a pickle."""
    for old, new in _RULES:
        target = _replace_prefix(module, old, new)
        if target is not None:
            return target
    if REMAP_NUMPY_CORE:
        target = _replace_prefix(module, *NUMPY_CORE_REMAP)
        if target is not None:
            return target
    return module


def _load_storage_from_bytes(data: bytes, map_location):
    # Stand-in for torch.storage._load_from_bytes, which calls torch.load
    # without map_location and so restores the storage on the device it was
    # pickled from.
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)


class CompatUnpickler(pickle.Unpickler):
    """``pickle.Unpickler`` that loads develop-era, NumPy 2 and GPU pickles.

    ``map_location`` is passed to ``torch.load`` for every pickled torch
    storage; it accepts the same values (``"cpu"``, ``"cuda:0"``, a
    ``torch.device``, a dict or a callable).  ``None`` keeps the device the
    tensor was pickled on.  Other keyword arguments go to ``pickle.Unpickler``.
    """

    def __init__(self, file: BinaryIO, map_location: Any = "cpu", **kwargs):
        super().__init__(file, **kwargs)
        self._map_location = map_location

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return functools.partial(_load_storage_from_bytes, map_location=self._map_location)

        target = remap_module(module)
        if target == module:
            return super().find_class(module, name)

        try:
            return super().find_class(target, name)
        except ModuleNotFoundError as exc:
            # Only rewrite the error when the remapped module itself is
            # missing, not when importing it fails on another dependency.
            missing = exc.name or ""
            if missing != target and not target.startswith(missing + "."):
                raise
            raise ModuleNotFoundError(
                "Cannot unpickle {}.{}: legacy module '{}' maps to '{}', which does not exist "
                "(removed upstream or not ported yet).".format(module, name, module, target),
                name=target) from exc
        except AttributeError as exc:
            raise AttributeError(
                "Cannot unpickle {}.{}: '{}' (remapped from legacy module '{}') has no attribute '{}'."
                .format(module, name, target, module, name)) from exc


PathOrFile = Union[str, os.PathLike, BinaryIO]


def load(file: PathOrFile, map_location: Any = "cpu", **kwargs):
    """Unpickle one object from ``file`` (a binary file object or a path)."""
    if isinstance(file, (str, os.PathLike)):
        with open(file, "rb") as f:
            return CompatUnpickler(f, map_location=map_location, **kwargs).load()
    return CompatUnpickler(file, map_location=map_location, **kwargs).load()


def loads(data: bytes, map_location: Any = "cpu", **kwargs):
    """Unpickle one object from the bytes-like object ``data``."""
    return CompatUnpickler(io.BytesIO(data), map_location=map_location, **kwargs).load()
