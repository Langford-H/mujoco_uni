# MuJoCoUni Roadmap

MuJoCoUni is the thin UniLab executor layer above official MuJoCo. It should
stay separate from MuJoCo solver internals while giving UniLab a stable,
high-throughput backend.

## v0.1: Standalone MuJoCoUni Baseline

Purpose: separate MuJoCoUni from the MuJoCo source tree.

Scope:

- standalone `mujoco_uni` Python package,
- official `mujoco==3.8.0` dependency,
- `BatchEnvPool` API compatible with current UniLab usage,
- local C++ thread pool for batched independent environments,
- pool-owned copied `mjModel` objects,
- reusable per-worker `mjData`,
- batched `step`, `forward`, `reset`, sensor collection, randomization fields,
  site Jacobian, hfield height, and multi-ray queries.

## v0.2: Explicit MuJoCo Version Selection

Purpose: make solver version choice explicit and manageable.

Plan:

- keep MuJoCoUni package version independent from MuJoCo solver version,
- add a compatibility table:

```text
MuJoCoUni version | supported mujoco version | status
```

- support one official MuJoCo solver version per Python environment/process,
- let UniLab select the intended MuJoCo/MuJoCoUni target at launch,
- document the `uv` environment pattern for switching solver versions,
- avoid requiring users to store multiple MuJoCo source repositories,
- fail fast when the loaded `mujoco` package does not match the native adapter.

Target user model:

```text
env-mj38  -> mujoco==3.8.x  -> mujoco-uni adapter for 3.8
env-mj310 -> mujoco==3.10.x -> mujoco-uni adapter for 3.10
```

## v0.3: MuJoCo mjThreadPool Adaptation

Purpose: evaluate official MuJoCo within-step threading for large or
contact-heavy environments.

Plan:

- investigate MuJoCo 3.10/newer `mjThreadPool` support,
- keep current outer `BatchEnvPool` parallelism as the default,
- add optional inner solver threading only when it helps,
- prevent oversubscription between outer batch workers and inner MuJoCo solver
  workers,
- expose clear knobs:

```text
local_workers
inner_solver_threads
inner_threadpool = off | auto | on
```

Default expectation:

```text
many small envs:
  use outer BatchEnvPool workers

few large/contact-heavy envs:
  consider MuJoCo inner mjThreadPool
```

## v0.4: MPI / Multi-Node Rollout Layer

Purpose: scale rollout collection across nodes while keeping UniLab simple.

Plan:

- keep MPI above local `BatchEnvPool`,
- make each rank own one local MuJoCoUni pool,
- partition environments across ranks,
- synchronize rollout progress at control boundaries,
- move actions/reset commands/randomization seeds from UniLab to ranks,
- move observations/rewards/done flags/sensor data from ranks back to UniLab,
- expose simple user-facing controls:

```text
num_nodes
ranks_per_node
envs_per_rank
workers_per_rank
```

Target shape:

```text
UniLab rollout bridge
        |
        +-- rank 0 -> BatchEnvPool
        +-- rank 1 -> BatchEnvPool
        +-- rank 2 -> BatchEnvPool
        +-- rank N -> BatchEnvPool
```

## Long-Term Direction

- keep MuJoCoUni thin,
- keep official MuJoCo as the solver owner,
- keep UniLab as the user-facing training and CPU/GPU bridge,
- keep MuJoCoUni and DrakeUni structurally similar where useful,
- treat tests and benchmarks as release gates for each feature version.
