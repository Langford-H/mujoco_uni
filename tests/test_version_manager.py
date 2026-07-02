from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import mujoco_uni.version_manager as version_manager
from mujoco_uni.version_manager import (
    MuJoCoEnv,
    canonical_mujoco_version,
    command_for_env,
    default_mujoco_version,
    discover_mujoco_envs,
    run_in_env,
    select_default_env,
    select_env,
    versioned_env_name,
)


def _env(version: str) -> MuJoCoEnv:
    return MuJoCoEnv(
        version=version,
        env_dir=Path(f"/tmp/.venv-mj{version.replace('.', '')}"),
        python=Path(f"/tmp/.venv-mj{version.replace('.', '')}/bin/python"),
    )


def test_default_version_prefers_requested_minor_order() -> None:
    envs = [_env("3.10.0"), _env("3.9.0"), _env("3.8.1"), _env("3.5.0")]
    assert select_default_env(envs) == envs[2]
    assert default_mujoco_version(envs) == "3.8.1"

    envs_without_38 = [_env("3.10.0"), _env("3.9.0"), _env("3.5.0")]
    assert select_default_env(envs_without_38) == envs_without_38[0]


def test_canonical_version_and_env_name() -> None:
    assert canonical_mujoco_version(None) == "3.8.0"
    assert canonical_mujoco_version("auto") == "3.8.0"
    assert canonical_mujoco_version("3.10") == "3.10.0"
    assert canonical_mujoco_version("3.8.1") == "3.8.1"
    assert versioned_env_name("3.10.0") == ".venv-mj310"
    assert versioned_env_name("3.8.1") == ".venv-mj38"

    with pytest.raises(ValueError):
        canonical_mujoco_version("3.4.0")
    with pytest.raises(ValueError):
        canonical_mujoco_version("3.11.0")


def test_select_env_accepts_minor_or_exact_requests() -> None:
    envs = [_env("3.8.1"), _env("3.10.0"), _env("3.8.0")]
    assert select_env(envs, "auto") == envs[0]
    assert select_env(envs, "3.10") == envs[1]
    assert select_env(envs, "3.8.0") == envs[2]
    assert select_env(envs, "3.9") is None


def test_command_for_env_replaces_python_entrypoint() -> None:
    env = _env("3.10.0")
    assert command_for_env(env, ["python", "script.py"]) == [str(env.python), "script.py"]
    assert command_for_env(env, ["python3", "-c", "print(1)"]) == [
        str(env.python),
        "-c",
        "print(1)",
    ]
    assert command_for_env(env, ["echo", "ok"]) == ["echo", "ok"]

    with pytest.raises(ValueError):
        command_for_env(env, [])


def test_discover_mujoco_envs_keeps_venv_python_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlinks are not available")

    env_dir = tmp_path / ".venv-mj310"
    bin_dir = env_dir / "bin"
    bin_dir.mkdir(parents=True)
    base_python = tmp_path / "base-python"
    base_python.write_text("")
    venv_python = bin_dir / "python"
    try:
        venv_python.symlink_to(base_python)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")

    probed: list[Path] = []

    def fake_mujoco_version_from_python(python: Path) -> str:
        probed.append(python)
        return "3.10.0"

    monkeypatch.setattr(version_manager, "_mujoco_version_from_python", fake_mujoco_version_from_python)

    envs = discover_mujoco_envs([tmp_path], include_current=False)

    assert envs == [
        MuJoCoEnv(
            version="3.10.0",
            env_dir=env_dir,
            python=venv_python.absolute(),
        )
    ]
    assert probed == [venv_python.absolute()]


def test_run_in_env_no_prepare_uses_explicit_env_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "custom-env"
    bin_dir = env_dir / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("")

    monkeypatch.setattr(
        version_manager,
        "_mujoco_version_from_python",
        lambda _python: "3.10.0",
    )
    monkeypatch.setattr(version_manager, "verify_env", lambda _env: None)
    recorded: list[list[str]] = []

    def fake_subprocess_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        del check
        recorded.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(version_manager.subprocess, "run", fake_subprocess_run)

    result = run_in_env(
        ["python", "train.py"],
        version="3.10",
        env_dir=env_dir,
        prepare=False,
    )

    assert result == 7
    assert recorded == [[str(python), "train.py"]]
