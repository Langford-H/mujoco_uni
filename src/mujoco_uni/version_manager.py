"""MuJoCo solver-version selection and uv environment helpers.

This module intentionally does not import ``mujoco`` at module import time.
MuJoCo is a native package; selecting a solver version must happen before the
target Python process imports it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mujoco_uni.version import (
    MUJOCO_DEFAULT_VERSION,
    MUJOCO_MAX_VERSION_EXCLUSIVE,
    MUJOCO_MIN_VERSION,
    MUJOCO_VERSION_SPEC,
    __version__,
)

SUPPORTED_MUJOCO_MINOR_ORDER = ("3.8", "3.10", "3.9", "3.7", "3.6", "3.5")
CANONICAL_MUJOCO_VERSION_BY_MINOR = {
    "3.5": "3.5.0",
    "3.6": "3.6.0",
    "3.7": "3.7.0",
    "3.8": "3.8.0",
    "3.9": "3.9.0",
    "3.10": "3.10.0",
}


def _is_auto_request(version: str | None) -> bool:
    return version is None or str(version).strip().lower() == "auto"


@dataclass(frozen=True)
class MuJoCoEnv:
    """A Python environment with a usable MuJoCo runtime installed."""

    version: str
    env_dir: Path
    python: Path
    source: str = "versioned-env"

    @property
    def minor(self) -> str:
        return minor_version(self.version)


def parse_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:\.|$)", str(version).strip())
    if match is None:
        raise ValueError(f"Unsupported MuJoCo version string: {version!r}")
    patch = 0 if match.group(3) is None else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), patch


def format_version(parts: tuple[int, int, int]) -> str:
    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def minor_version(version: str) -> str:
    major, minor, _ = parse_version(version)
    return f"{major}.{minor}"


def is_supported_mujoco_version(version: str) -> bool:
    parsed = parse_version(version)
    return parse_version(MUJOCO_MIN_VERSION) <= parsed < parse_version(MUJOCO_MAX_VERSION_EXCLUSIVE)


def canonical_mujoco_version(version: str | None) -> str:
    """Return an exact installable version for a user request.

    ``None`` and ``"auto"`` resolve to the package default. Two-component
    requests such as ``"3.10"`` resolve through the canonical patch table.
    Three-component requests are preserved after range validation.
    """

    if _is_auto_request(version):
        version = MUJOCO_DEFAULT_VERSION
    requested = str(version).strip()
    parts = parse_version(requested)
    if not is_supported_mujoco_version(format_version(parts)):
        raise ValueError(f"mujoco {requested!r} is outside supported range {MUJOCO_VERSION_SPEC}")
    if re.match(r"^\d+\.\d+$", requested):
        key = f"{parts[0]}.{parts[1]}"
        if key not in CANONICAL_MUJOCO_VERSION_BY_MINOR:
            raise ValueError(f"No canonical MuJoCo patch version is configured for {key}")
        return CANONICAL_MUJOCO_VERSION_BY_MINOR[key]
    return format_version(parts)


def versioned_env_name(version: str) -> str:
    major, minor, _ = parse_version(canonical_mujoco_version(version))
    return f".venv-mj{major}{minor}"


def _python_path(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _run(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=None if cwd is None else str(cwd),
        check=True,
        text=True,
        capture_output=capture,
    )


def _mujoco_version_from_python(python: Path) -> str | None:
    code = (
        "import json, mujoco; "
        "print(json.dumps({'version': getattr(mujoco, '__version__', '')}))"
    )
    try:
        result = _run([python, "-c", code], capture=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        version = str(json.loads(result.stdout)["version"])
    except (KeyError, json.JSONDecodeError, TypeError):
        return None
    return version if is_supported_mujoco_version(version) else None


def _env_from_dir(env_dir: Path, *, source: str = "versioned-env") -> MuJoCoEnv | None:
    python = _python_path(env_dir).absolute()
    if not python.exists():
        return None
    version = _mujoco_version_from_python(python)
    if version is None:
        return None
    return MuJoCoEnv(version=version, env_dir=env_dir, python=python, source=source)


def discover_mujoco_envs(
    roots: Iterable[str | os.PathLike[str]] | None = None,
    *,
    include_current: bool = True,
) -> list[MuJoCoEnv]:
    """Discover versioned uv environments containing supported MuJoCo builds."""

    search_roots = [Path.cwd()] if roots is None else [Path(root) for root in roots]
    envs: list[MuJoCoEnv] = []
    seen_python: set[Path] = set()

    if include_current:
        current_python = Path(sys.executable).absolute()
        version = _mujoco_version_from_python(current_python)
        if version is not None:
            envs.append(
                MuJoCoEnv(
                    version=version,
                    env_dir=current_python.parents[1],
                    python=current_python,
                    source="current-python",
                )
            )
            seen_python.add(current_python)

    for root in search_roots:
        if not root.exists():
            continue
        for env_dir in sorted(root.glob(".venv-mj*")):
            env = _env_from_dir(env_dir)
            if env is None or env.python in seen_python:
                continue
            envs.append(env)
            seen_python.add(env.python)
    return envs


def _minor_preference_index(version: str) -> int:
    minor = minor_version(version)
    try:
        return SUPPORTED_MUJOCO_MINOR_ORDER.index(minor)
    except ValueError:
        return len(SUPPORTED_MUJOCO_MINOR_ORDER)


def _patch_sort_key(version: str) -> int:
    return parse_version(version)[2]


def select_default_env(envs: Sequence[MuJoCoEnv]) -> MuJoCoEnv | None:
    """Select a default env using the requested minor-version preference order."""

    if not envs:
        return None
    return sorted(
        envs,
        key=lambda env: (_minor_preference_index(env.version), -_patch_sort_key(env.version)),
    )[0]


def default_mujoco_version(envs: Sequence[MuJoCoEnv] | None = None) -> str:
    selected = (
        select_default_env(list(envs))
        if envs is not None
        else select_default_env(discover_mujoco_envs())
    )
    return selected.version if selected is not None else MUJOCO_DEFAULT_VERSION


def select_env(envs: Sequence[MuJoCoEnv], requested: str | None = None) -> MuJoCoEnv | None:
    """Select an existing environment by explicit request or default order."""

    if _is_auto_request(requested):
        return select_default_env(envs)

    requested_text = str(requested).strip()
    requested_parts = parse_version(requested_text)
    requested_minor = f"{requested_parts[0]}.{requested_parts[1]}"
    exact = bool(re.match(r"^\d+\.\d+\.\d+", requested_text))
    matches = [
        env
        for env in envs
        if (env.version == format_version(requested_parts) if exact else env.minor == requested_minor)
    ]
    return select_default_env(matches)


def _default_package_source() -> str:
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").exists():
        return str(root)
    return f"mujoco-uni=={__version__}"


def _env_search_roots(env_dir: str | os.PathLike[str] | None) -> list[Path]:
    return [Path(env_dir).parent] if env_dir is not None else [Path.cwd()]


def _install_base(python: Path) -> list[str | Path]:
    return ["uv", "pip", "install", "--python", python]


def verify_env(env: MuJoCoEnv) -> None:
    code = f"""
import mujoco
import mujoco_uni
from mujoco_uni.compiled import MUJOCO_BUILD_VERSION
assert mujoco.__version__ == {env.version!r}, mujoco.__version__
assert MUJOCO_BUILD_VERSION == {env.version!r}, MUJOCO_BUILD_VERSION
assert mujoco_uni.MUJOCO_VERSION_SPEC == {MUJOCO_VERSION_SPEC!r}
"""
    _run([env.python, "-c", code])


def prepare_env(
    version: str | None = None,
    *,
    env_dir: str | os.PathLike[str] | None = None,
    project: str | os.PathLike[str] | None = None,
    package_source: str | os.PathLike[str] | None = None,
    python_version: str | None = None,
    reinstall: bool = False,
) -> MuJoCoEnv:
    """Create or refresh a uv env for one exact MuJoCo solver version."""

    selected: MuJoCoEnv | None = None
    if _is_auto_request(version):
        if env_dir is not None:
            selected = _env_from_dir(Path(env_dir))
        if selected is None:
            selected = select_default_env(
                discover_mujoco_envs(_env_search_roots(env_dir), include_current=True)
            )
    exact_version = selected.version if selected is not None else canonical_mujoco_version(version)
    resolved_env_dir = (
        selected.env_dir
        if selected is not None and env_dir is None
        else Path(env_dir)
        if env_dir is not None
        else Path.cwd() / versioned_env_name(exact_version)
    )
    python = _python_path(resolved_env_dir)
    if not python.exists():
        requested_python = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
        _run(["uv", "venv", resolved_env_dir, "--python", requested_python])

    install_base = _install_base(python)
    _run(
        [
            *install_base,
            f"mujoco=={exact_version}",
            "numpy",
            "pybind11",
            "wheel",
            "setuptools",
        ]
    )

    package_spec = str(package_source) if package_source is not None else _default_package_source()
    package_cmd: list[str | os.PathLike[str]] = [
        *install_base,
        "--no-deps",
        "--no-build-isolation",
    ]
    if reinstall:
        package_cmd.append("--force-reinstall")
    package_path = Path(package_spec)
    if package_path.exists():
        # MuJoCoUni's native extension is tied to the MuJoCo runtime installed
        # in the target environment. Avoid reusing a cached local wheel that may
        # have been compiled against another MuJoCo version.
        package_cmd.append("--no-cache")
    package_cmd.append(package_spec)
    _run(package_cmd)

    if project is not None:
        _run([*install_base, "-e", Path(project)])

    env = MuJoCoEnv(version=exact_version, env_dir=resolved_env_dir, python=python)
    verify_env(env)
    return env


def command_for_env(env: MuJoCoEnv, command: Sequence[str]) -> list[str]:
    if not command:
        raise ValueError("command must not be empty")
    first = command[0]
    if first in {"python", "python3"}:
        return [str(env.python), *command[1:]]
    return [*map(str, command)]


def run_in_env(
    command: Sequence[str],
    *,
    version: str | None = None,
    env_dir: str | os.PathLike[str] | None = None,
    project: str | os.PathLike[str] | None = None,
    package_source: str | os.PathLike[str] | None = None,
    python_version: str | None = None,
    prepare: bool = True,
    reinstall: bool = False,
) -> int:
    """Run a command under a selected MuJoCo versioned environment."""

    if prepare:
        env = prepare_env(
            version,
            env_dir=env_dir,
            project=project,
            package_source=package_source,
            python_version=python_version,
            reinstall=reinstall,
        )
    else:
        if env_dir is not None:
            exact_env = _env_from_dir(Path(env_dir))
            envs = [] if exact_env is None else [exact_env]
        else:
            envs = discover_mujoco_envs(_env_search_roots(None), include_current=True)
        selected = select_env(envs, version)
        if selected is None:
            raise RuntimeError("No matching MuJoCo environment found; rerun without --no-prepare")
        env = selected
        verify_env(env)
    return subprocess.run(command_for_env(env, command), check=False).returncode


__all__ = [
    "CANONICAL_MUJOCO_VERSION_BY_MINOR",
    "MuJoCoEnv",
    "SUPPORTED_MUJOCO_MINOR_ORDER",
    "canonical_mujoco_version",
    "command_for_env",
    "default_mujoco_version",
    "discover_mujoco_envs",
    "is_supported_mujoco_version",
    "prepare_env",
    "run_in_env",
    "select_default_env",
    "select_env",
    "versioned_env_name",
    "verify_env",
]
