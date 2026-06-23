"""Native compiled MuJoCoUni extension boundary.

This package stays lower-level than ``mujoco_uni.runtime``. Application code
should normally import ``mujoco_uni.batch_env`` or ``mujoco_uni.runtime``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path


def _preload_mujoco_library() -> object | None:
    import mujoco

    mujoco_dir = Path(mujoco.__file__).resolve().parent
    candidates = sorted(mujoco_dir.glob("libmujoco*"))
    if not candidates:
        return None
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    return ctypes.CDLL(str(candidates[0]), mode=mode)


_MUJOCO_LIBRARY_HANDLE = _preload_mujoco_library()

try:
    from . import _batch_env as _batch_env
except ImportError as exc:  # pragma: no cover - optional local extension.
    _BATCH_IMPORT_ERROR = exc
    _batch_env = None  # type: ignore[assignment]
    NativeBatchEnvPool = None  # type: ignore[assignment]
    SUPPORTED_FIELDS: tuple[str, ...] = ()
else:
    _BATCH_IMPORT_ERROR = None
    NativeBatchEnvPool = _batch_env.BatchEnvPool
    SUPPORTED_FIELDS = tuple(_batch_env.SUPPORTED_FIELDS)


def batch_available() -> bool:
    return _batch_env is not None


def batch_import_error() -> ImportError | None:
    return _BATCH_IMPORT_ERROR


def require_native():
    if _batch_env is None:
        raise ImportError("MuJoCoUni native batch extension has not been built") from (
            _BATCH_IMPORT_ERROR
        )
    return _batch_env


__all__ = [
    "NativeBatchEnvPool",
    "SUPPORTED_FIELDS",
    "batch_available",
    "batch_import_error",
    "require_native",
]
