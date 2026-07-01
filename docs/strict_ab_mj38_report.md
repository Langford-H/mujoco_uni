# Strict A/B Performance Check: MuJoCoUni v0.1 mj38

Date: 2026-06-30

## Purpose

The first full benchmark table showed surprisingly large standalone wins in
some cases, especially Franka. That is suspicious because source-tree
separation should not inherently speed up `mj_step` or `mj_forward`: both
standalone MuJoCoUni and legacy compact MuJoCoUni call MuJoCo 3.8 C APIs.

This stricter check asks a narrower question:

> Does standalone MuJoCoUni show a consistent speedup over legacy compact
> MuJoCoUni when measured in a cleaner A/B setup?

## Harness

Script:

```text
tools/strict_ab_mj38.py
```

Key controls:

- fresh subprocess for every sample,
- randomized standalone/legacy run order,
- same generated state array for standalone and legacy,
- C++ `BatchEnvPool` path only,
- no Python serial or multiprocessing baseline in the timing run,
- same MuJoCo solver version,
- same benchmark model files,
- summary reports median and p10-p90, not just one mean.

The MuJoCo solver library was checked:

```text
official mujoco libmujoco.3.8.0.dylib sha256:
611e23005d21eeff83b94a17d18727130050ac00406711ec8e7ae56687c5d8c6

legacy compact libmujoco.3.8.0.dylib sha256:
611e23005d21eeff83b94a17d18727130050ac00406711ec8e7ae56687c5d8c6
```

So the solver dylib itself is byte-identical in this local comparison.

## Focused Suspicious Case

Command:

```text
uv run python tools/strict_ab_mj38.py \
  --models Franka Go1 \
  --ops step forward \
  --nenv 4096 \
  --nstep 50 \
  --nthread 10 \
  --rounds 3 \
  --warmup 4 \
  --repeat 12 \
  --seed 20260630 \
  --output docs/strict_ab_mj38_results.json
```

Result:

| Model | Op | Standalone median | Legacy median | Ratio |
|---|---|---:|---:|---:|
| Franka | forward | 202573 | 206304 | 0.982x |
| Franka | step | 273844 | 269316 | 1.017x |
| Go1 | forward | 891594 | 854018 | 1.044x |
| Go1 | step | 704483 | 724721 | 0.972x |

Interpretation:

- The huge Franka step speedup from the full benchmark did not reproduce.
- Franka step is essentially tied in this focused run.
- Go1 is mixed: forward slightly faster standalone, step slightly slower.

## Broader Sweep

Command:

```text
uv run python tools/strict_ab_mj38.py \
  --models Go1 Allegro Franka Humanoid \
  --ops step forward \
  --nenv 4096 \
  --nstep 50 \
  --nthread 10 \
  --rounds 3 \
  --warmup 3 \
  --repeat 8 \
  --seed 20260631 \
  --output docs/strict_ab_mj38_all_models_results.json
```

Result:

| Model | Op | Standalone median | Legacy median | Ratio |
|---|---|---:|---:|---:|
| Allegro | forward | 1222546 | 1182528 | 1.034x |
| Allegro | step | 1212557 | 1225462 | 0.989x |
| Franka | forward | 181252 | 185370 | 0.978x |
| Franka | step | 173906 | 194966 | 0.892x |
| Go1 | forward | 836735 | 776133 | 1.078x |
| Go1 | step | 536691 | 657157 | 0.817x |
| Humanoid | forward | 171562 | 179202 | 0.957x |
| Humanoid | step | 149078 | 125949 | 1.184x |

Interpretation:

- The broader sweep is mixed.
- There is no consistent standalone speedup.
- There is also no catastrophic standalone regression, but some individual
  model/op samples are slower.
- The old full-run benchmark should be treated as a regression smoke, not as a
  speedup claim.

## Paired Step Check

The previous strict runs still aggregate many samples over time, so later runs
can be slower simply because the machine has already been working. A paired
step check was added to reduce this artifact:

- standalone and legacy are measured as an immediate pair,
- the first side of the pair is randomized,
- the ratio is computed inside each pair,
- a short cooldown is inserted between pairs.

Command:

```text
uv run python tools/strict_ab_mj38.py \
  --paired \
  --models Franka Go1 \
  --ops step \
  --nenv 4096 \
  --nstep 50 \
  --nthread 10 \
  --rounds 3 \
  --warmup 3 \
  --repeat 8 \
  --cooldown-s 3 \
  --seed 20260701 \
  --output docs/strict_ab_mj38_paired_step_results.json
```

Result:

| Model | Op | Median pair ratio | p10-p90 | Pairs |
|---|---|---:|---:|---:|
| Franka | step | 1.029x | 0.919x-1.029x | 3 |
| Go1 | step | 0.977x | 0.956x-0.977x | 3 |

Interpretation:

- The paired result still does not show a reliable standalone speedup.
- Franka is approximately tied in median but noisy.
- Go1 is slightly slower in standalone for this paired step run.
- This supports the conservative conclusion: separation should be treated as
  architecture/version-control work, not as a direct performance optimization.

## Source Difference Check

Compared against upstream `unilabsim/mujoco_uni` branch `uni/v3.8.0`, commit:

```text
2df646715a4b1a66cd907dbcdaa2f823fa2ff692
```

Observed source differences:

- `batch_env.cc` removes MuJoCo source-tree-only Python wrapper headers
  (`errors.h`, `raw.h`, `structs.h`) and uses official `mujoco.MjModel`
  `_address` / `_from_model_ptr`.
- `get_model` lifetime handling uses cached Python wrappers.
- `threadpool.*` removes the Abseil attribute dependency and fixes a
  sign-compare/type issue.

No obvious algorithmic `mj_step` / `mj_forward` loop change was found in the
diff. That matches the benchmark conclusion: separation itself should not be
credited as a speed mechanism.

## Conclusion

The strict result is:

```text
Standalone MuJoCoUni v0.1 preserves comparable throughput.
It does not prove a speedup over legacy compact MuJoCoUni.
```

Important limitation:

```text
This strict run isolates Python import/process state, but it does not isolate
physical machine state.
```

That means the result is still affected by repeated-run fatigue:

- CPU thermal state,
- frequency scaling,
- scheduler placement,
- memory pressure,
- cache state,
- benchmark order,
- the fact that later samples can be slower simply because the machine has
  already been doing heavy work.

Therefore the local strict run should **not** be used to rank standalone and
legacy by small percentages. It is only strong enough to say there is no clear
large regression and no credible evidence that source-tree separation itself
creates speed.

The correct v0.1 performance claim should be conservative:

```text
No obvious throughput regression in local strict A/B checks.
```

## Suggested Next Checks

- Run the strict harness on Linux x86_64, where CPU pinning and performance
  counters are easier.
- Compare wheel-vs-wheel builds, not editable local build vs PyPI legacy wheel.
- Increase rounds to 10+ on the target CPU server.
- Add final-state hash checks in the strict harness so every timed step also
  proves identical output.
- Capture CPU frequency / thermal / power-state metadata.
- Keep Python serial and multiprocessing baselines in a separate benchmark run,
  not mixed with standalone-vs-legacy A/B timing.
