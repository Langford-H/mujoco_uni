"""Standalone MuJoCoUni batched executor package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .version import (
    MUJOCO_DEFAULT_VERSION,
    MUJOCO_MAX_VERSION_EXCLUSIVE,
    MUJOCO_MIN_VERSION,
    MUJOCO_VERSION,
    MUJOCO_VERSION_SPEC,
    __version__,
)

if TYPE_CHECKING:
    from .batch_env import SUPPORTED_FIELDS, BatchEnvPool
    from .compiled import batch_available, batch_import_error
    from .runtime import available_backends, batch_diagnostics
    from .version_manager import (
        canonical_mujoco_version,
        default_mujoco_version,
        discover_mujoco_envs,
        prepare_env,
        run_in_env,
        select_default_env,
        select_env,
    )

__all__ = [
    "__version__",
    "BatchEnvPool",
    "MUJOCO_DEFAULT_VERSION",
    "MUJOCO_MAX_VERSION_EXCLUSIVE",
    "MUJOCO_MIN_VERSION",
    "MUJOCO_VERSION",
    "MUJOCO_VERSION_SPEC",
    "SUPPORTED_FIELDS",
    "available_backends",
    "batch_available",
    "batch_diagnostics",
    "batch_import_error",
    "canonical_mujoco_version",
    "default_mujoco_version",
    "discover_mujoco_envs",
    "prepare_env",
    "run_in_env",
    "select_default_env",
    "select_env",
]


def __getattr__(name: str):
    if name in {"BatchEnvPool", "SUPPORTED_FIELDS"}:
        from . import batch_env

        return getattr(batch_env, name)
    if name in {"available_backends", "batch_diagnostics"}:
        from . import runtime

        return getattr(runtime, name)
    if name in {"batch_available", "batch_import_error"}:
        from . import compiled

        return getattr(compiled, name)
    if name in {
        "canonical_mujoco_version",
        "default_mujoco_version",
        "discover_mujoco_envs",
        "prepare_env",
        "run_in_env",
        "select_default_env",
        "select_env",
    }:
        from . import version_manager

        return getattr(version_manager, name)
    raise AttributeError(name)
