# MuJoCoUni

MuJoCoUni is the standalone UniLab batch-executor layer for official MuJoCo.
It provides the `BatchEnvPool` API used by UniLab without modifying MuJoCo
solver, contact, integrator, or source-tree internals.

This repository is intentionally small:

- MuJoCo remains the solver dependency.
- MuJoCoUni owns batching, model cloning, per-thread `mjData`, and Python API
  validation.
- UniLab owns tasks, rollout logic, domain randomization payloads, and training.
- Distributed orchestration, if needed, lives above MuJoCoUni.

## Version Policy

MuJoCoUni has its own package version, independent of the MuJoCo solver version.

Current release:

```text
mujoco-uni==0.1.0
mujoco==3.8.0
```

The public metadata is available from Python:

```python
import mujoco_uni

print(mujoco_uni.__version__)
print(mujoco_uni.MUJOCO_VERSION)
print(mujoco_uni.MUJOCO_VERSION_SPEC)
```

MuJoCoUni fails fast if the loaded official `mujoco` package does not match the
solver version targeted by this release, or if the required MuJoCo Python model
pointer helpers are unavailable.

## Package Layout

The structure mirrors the DrakeUni split between runtime code and compiled
executor code:

```text
src/mujoco_uni/
  __init__.py
  version.py
  batch_env.py              # Stable public facade
  runtime/
    batch.py                # Python API, validation, compatibility behavior
  compiled/
    batch_env.cc            # Native pybind11 executor
    threadpool.h/.cc        # Local thread pool
    _batch_env*.so          # Generated extension after local build
```

Application code should normally import:

```python
from mujoco_uni.batch_env import BatchEnvPool, SUPPORTED_FIELDS
```

## Execution Model

`BatchEnvPool` accepts one `mujoco.MjModel` or a compatible sequence of
`mujoco.MjModel` objects. At construction time it reads the official MuJoCo
model pointer from Python, copies the model with `mj_copyModel`, and stores
pool-owned models internally.

The hot path is native C++ calling the official MuJoCo C API. The pool uses:

- one pool-owned `mjModel` per environment row,
- one reusable `mjData` per worker thread,
- disjoint environment chunks,
- disjoint output rows,
- one synchronization point per batch operation.

There is no MPI or OpenMP inside MuJoCoUni. Large-scale multi-process,
multi-socket, or multi-node collection should be coordinated by the caller.

## Public API

The stable facade is `mujoco_uni.batch_env`.

Exported symbols:

- `BatchEnvPool`
- `SUPPORTED_FIELDS`
- `batch_available`
- `batch_import_error`

Supported randomization/model fields:

```text
body_mass
body_ipos
body_iquat
body_inertia
dof_armature
gravity
geom_friction
kp
kd
```

Core `BatchEnvPool` behavior:

- construct from one model or a sequence of length `1` / `nbatch`,
- `step` over the full pool and return final state, optionally with final
  sensor data,
- `forward` over the full pool and return sensor data,
- sparse `reset` with optional model-field randomization,
- site Jacobian queries,
- hfield height sampling,
- multi-ray queries when supported by the native extension,
- non-owning model views through `get_model`, `get_models`, and
  `get_all_models`.

Returned model views remain valid only while the pool is alive.

## Installation

For development beside UniLab:

```bash
cd /path/to/mujoco_uni
uv sync
uv pip install --force-reinstall --no-deps --no-build-isolation -e .
```

For a UniLab checkout using this sibling repository:

```toml
[project.optional-dependencies]
mujoco = ["mujoco-uni==0.1.0"]

[tool.uv.sources]
mujoco-uni = { path = "../mujoco_uni", editable = true }
```

Then UniLab should import through its compatibility/backend layer, which in turn
imports:

```python
from mujoco_uni.batch_env import BatchEnvPool, SUPPORTED_FIELDS
```

## Rebuilding The Native Extension

Editable installs generate a local extension such as:

```text
src/mujoco_uni/compiled/_batch_env.cpython-313-darwin.so
```

That artifact is tied to the active Python environment, platform, and MuJoCo
patch version. Rebuild after switching virtual environments, Python versions, or
MuJoCo versions:

```bash
uv pip install "mujoco==3.8.0" pybind11 wheel
uv pip install --force-reinstall --no-deps --no-build-isolation -e .
```

## Validation

Run the standalone checks:

```bash
uv run ruff check .
uv run pytest -q
```

Recommended UniLab integration checks:

```bash
cd ../UniLab
uv run pytest \
  tests/base/test_mujoco_uni_package.py \
  tests/base/test_mujoco_batch_env_randomization.py \
  tests/base/test_mujoco_batch_env_jacobian.py \
  tests/base/backend/test_mujoco_site_jacobian.py \
  tests/envs/locomotion/go2w/test_go2w_height_scan.py \
  -q
```

Recommended training smoke:

```bash
cd ../UniLab
uv run python scripts/train_rsl_rl.py \
  task=go2_joystick_flat/mujoco \
  algo.seed=1 \
  algo.num_envs=256 \
  algo.num_steps_per_env=24 \
  algo.max_iterations=2 \
  algo.save_interval=100 \
  training.no_play=true \
  training.logger=tensorboard \
  training.device=cpu \
  training.log_root=/tmp/unilab_mujoco_uni_smoke
```

## Thread-Safety Notes

Built-in MuJoCo sensors are read from `mjData.sensordata` and are supported by
the batch executor.

Custom sensors, plugins, and global MuJoCo callbacks are thread-safety-sensitive.
If `nthread > 1`, any global mutable callback/plugin state is the caller's
responsibility.

## Non-Goals

MuJoCoUni does not:

- alter MuJoCo source code,
- fork MuJoCo solver logic,
- split a single MuJoCo solve across MPI ranks,
- add OpenMP to MuJoCo,
- own UniLab task YAMLs, rollout code, reward functions, or DrakeUni behavior.

## License

Apache-2.0.
