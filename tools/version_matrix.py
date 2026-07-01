from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSIONS = ("3.8.0", "3.10.0")


def _python_path(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _copy_source(dst: Path) -> Path:
    source = dst / "source"

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "dist",
            "__pycache__",
            "mujoco_uni.egg-info",
        }
        for name in names:
            if fnmatch.fnmatch(name, "_batch_env*.so"):
                ignored.add(name)
            if fnmatch.fnmatch(name, "*.pyc"):
                ignored.add(name)
        return ignored.intersection(names)

    shutil.copytree(ROOT, source, ignore=ignore)
    return source


def _smoke_code(expected_version: str) -> str:
    mismatch_version = "3.10.0" if not expected_version.startswith("3.10.") else "3.8.0"
    return f"""
import numpy as np
import mujoco as mj
import subprocess
import sys

import mujoco_uni
from mujoco_uni.batch_env import BatchEnvPool
from mujoco_uni.compiled import MUJOCO_BUILD_VERSION

assert mj.__version__ == {expected_version!r}, mj.__version__
assert MUJOCO_BUILD_VERSION == {expected_version!r}, MUJOCO_BUILD_VERSION
assert mujoco_uni.MUJOCO_VERSION_SPEC == ">=3.8,<3.11"

model = mj.MjModel.from_xml_string(\"\"\"
<mujoco>
  <worldbody>
    <body name="box">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
\"\"\")
data = mj.MjData(model)
state = np.zeros(mj.mj_stateSize(model, int(mj.mjtState.mjSTATE_FULLPHYSICS)))
mj.mj_getState(model, data, state, int(mj.mjtState.mjSTATE_FULLPHYSICS))
states = np.vstack([state, state])

with BatchEnvPool(model, nbatch=2, nthread=1) as pool:
    assert pool.nbatch == 2
    assert pool.nstate == state.shape[0]
    sensor = pool.forward(states)
    stepped = pool.step(states, nstep=1)
    assert sensor.shape == (2, model.nsensordata)
    assert stepped.shape == (2, state.shape[0])

mismatch_code = \"\"\"
import mujoco
mujoco.__version__ = {mismatch_version!r}
try:
    from mujoco_uni.batch_env import BatchEnvPool
except ImportError as exc:
    print(exc)
    raise SystemExit(0)
raise SystemExit("expected build/runtime mismatch watchdog to fail")
\"\"\"
result = subprocess.run(
    [sys.executable, "-c", mismatch_code],
    check=False,
    capture_output=True,
    text=True,
)
assert result.returncode == 0, result.stdout + result.stderr
assert "native batch extension was built against mujoco" in result.stdout

print("ok mujoco", {expected_version!r}, "build", MUJOCO_BUILD_VERSION)
"""


def run_version(version: str, *, keep_envs: bool, run_pytest: bool) -> None:
    tmp = Path(tempfile.mkdtemp(prefix=f"mujoco_uni_mj{version.replace('.', '')}_"))
    try:
        source = _copy_source(tmp)
        env_dir = tmp / ".venv"
        _run(["uv", "venv", str(env_dir), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"])
        python = _python_path(env_dir)
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                f"mujoco=={version}",
                "numpy",
                "pybind11",
                "pytest",
                "wheel",
                "setuptools",
            ]
        )
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--force-reinstall",
                "--no-deps",
                "--no-build-isolation",
                str(source),
            ]
        )
        _run([str(python), "-c", _smoke_code(version)])
        if run_pytest:
            _run(
                [
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_batch_env.py",
                    "tests/test_batch_env_parity.py",
                ],
                cwd=source,
            )
    finally:
        if keep_envs:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MuJoCoUni across MuJoCo versions.")
    parser.add_argument("--versions", nargs="+", default=list(DEFAULT_VERSIONS))
    parser.add_argument("--keep-envs", action="store_true")
    parser.add_argument("--pytest", action="store_true", help="Run the Python test suite per version.")
    args = parser.parse_args()

    for version in args.versions:
        print(f"\n== MuJoCo {version} ==", flush=True)
        run_version(version, keep_envs=bool(args.keep_envs), run_pytest=bool(args.pytest))


if __name__ == "__main__":
    main()
