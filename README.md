# MuJoCoUni Standalone Executor

This package is the UniLab batch-executor layer for official MuJoCo.
MuJoCoUni has its own release version (`mujoco-uni==0.1.0`); this release
targets and pins the compatible solver dependency (`mujoco==3.8.0`). It links
against the official `mujoco` Python package and C API, then exposes the legacy
`BatchEnvPool` contract as `mujoco_uni.batch_env`.

The package is split like DrakeUni:

- `mujoco_uni.batch_env`: stable public facade.
- `mujoco_uni.runtime`: Python runtime/API layer.
- `mujoco_uni.compiled`: native-extension area. It contains the C++ source
  files, the Python loader boundary, and the built `_batch_env` extension after
  local build.

The package does not modify MuJoCo solver, contact, integrator, or source-tree
internals. It clones caller-provided `mujoco.MjModel` objects into a
pool-owned model set, allocates one `mjData` per worker thread, and parallelizes
across independent environment rows.

No MPI or OpenMP runtime is used inside this executor. Scaling is local to the
process: disjoint environment chunks, disjoint output rows, and one wait at each
batch boundary. Multi-process or multi-node orchestration should live above this
package.

Built-in MuJoCo sensors are collected from `mjData.sensordata`. Custom sensors,
plugins, and global MuJoCo callbacks are user responsibility: callback state must
be thread-safe when `nthread > 1`.

When using an editable install, the generated
`src/mujoco_uni/compiled/_batch_env*.so` is tied to the Python environment and
MuJoCo patch version that built it. Rebuild the extension after switching
virtualenvs or MuJoCo versions:

```bash
uv pip install "mujoco==3.8.0" pybind11 wheel
uv pip install --force-reinstall --no-deps --no-build-isolation -e .
```
