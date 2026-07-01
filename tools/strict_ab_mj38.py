from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python() -> str:
    return sys.executable


def _mode_env(mode: str, legacy_site: Path, legacy_shim: Path) -> dict[str, str]:
    env = os.environ.copy()
    if mode == "legacy":
        prefix = f"{legacy_shim}{os.pathsep}{legacy_site}"
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = prefix if not existing else f"{prefix}{os.pathsep}{existing}"
    return env


def _extract_worker_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("STRICT_AB_JSON="):
            return json.loads(line.removeprefix("STRICT_AB_JSON="))
    raise RuntimeError(f"worker did not emit STRICT_AB_JSON; stdout was:\n{stdout}")


def _run_worker(args: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        [_python(), str(Path(__file__).resolve()), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "worker failed\n"
            f"cmd: {_python()} {Path(__file__).resolve()} {' '.join(args)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return _extract_worker_json(proc.stdout)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _throughput(op: str, seconds: float, nenv: int, nstep: int) -> float:
    if op == "step":
        return nenv * nstep / seconds
    return nenv / seconds


def _print_summary(results: dict[str, Any]) -> None:
    print("\nStrict A/B Summary")
    print("==================")
    print(
        "Each row aggregates fresh subprocess samples. Higher throughput is better; "
        "ratio is standalone / legacy."
    )
    print()
    print(
        "| Model | Op | Standalone median | Legacy median | Ratio | "
        "Standalone p10-p90 | Legacy p10-p90 |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for key in sorted(results["comparisons"]):
        item = results["comparisons"][key]
        standalone = item["standalone"]
        legacy = item["legacy"]
        print(
            f"| {item['model']} | {item['op']} | "
            f"{standalone['throughput_median']:.0f} | "
            f"{legacy['throughput_median']:.0f} | "
            f"{item['throughput_median_ratio']:.3f}x | "
            f"{standalone['throughput_p10']:.0f}-{standalone['throughput_p90']:.0f} | "
            f"{legacy['throughput_p10']:.0f}-{legacy['throughput_p90']:.0f} |"
        )


def _print_paired_summary(results: dict[str, Any]) -> None:
    print("\nPaired Strict A/B Summary")
    print("=========================")
    print(
        "Each ratio is computed inside an immediate standalone/legacy pair. "
        "Higher ratio means standalone faster."
    )
    print()
    print("| Model | Op | Median pair ratio | p10-p90 | Pairs |")
    print("|---|---|---:|---:|---:|")
    for key in sorted(results["paired_comparisons"]):
        item = results["paired_comparisons"][key]
        print(
            f"| {item['model']} | {item['op']} | "
            f"{item['ratio']['median']:.3f}x | "
            f"{item['ratio']['p10']:.3f}x-{item['ratio']['p90']:.3f}x | "
            f"{item['count']} |"
        )


def _measure_sample(
    *,
    bench_root: Path,
    legacy_site: Path,
    legacy_shim: Path,
    state_file: Path,
    model: str,
    op: str,
    mode: str,
    nenv: int,
    nstep: int,
    nthread: int,
    warmup: int,
    repeat: int,
    round_id: int,
) -> dict[str, Any]:
    sample = _run_worker(
        [
            "--worker",
            "measure",
            "--bench-root",
            str(bench_root),
            "--model",
            model,
            "--op",
            op,
            "--mode",
            mode,
            "--nenv",
            str(nenv),
            "--nstep",
            str(nstep),
            "--nthread",
            str(nthread),
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
            "--state-file",
            str(state_file),
        ],
        env=_mode_env(mode, legacy_site, legacy_shim),
    )
    sample["round"] = round_id
    return sample


def _parent_main(args: argparse.Namespace) -> None:
    bench_root = args.bench_root.resolve()
    legacy_site = args.legacy_site.resolve()
    legacy_shim = args.legacy_shim.resolve()
    output = args.output.resolve()
    data_dir = Path(tempfile.mkdtemp(prefix="mujoco-uni-strict-ab-"))

    if not bench_root.exists():
        raise FileNotFoundError(f"benchmark root not found: {bench_root}")
    if not legacy_site.exists():
        raise FileNotFoundError(f"legacy site not found: {legacy_site}")
    if not legacy_shim.exists():
        raise FileNotFoundError(f"legacy shim not found: {legacy_shim}")

    state_files: dict[str, Path] = {}
    for model in args.models:
        path = data_dir / f"{model}_{args.nenv}_seed{args.seed}.npz"
        _run_worker(
            [
                "--worker",
                "prepare",
                "--bench-root",
                str(bench_root),
                "--model",
                model,
                "--nenv",
                str(args.nenv),
                "--seed",
                str(args.seed),
                "--state-file",
                str(path),
            ],
            env=_mode_env("standalone", legacy_site, legacy_shim),
        )
        state_files[model] = path

    if args.paired:
        _parent_paired_main(args, bench_root, legacy_site, legacy_shim, output, state_files, started=None)
        return

    jobs: list[dict[str, str]] = []
    for round_id in range(args.rounds):
        round_jobs: list[dict[str, str]] = []
        for model in args.models:
            for op in args.ops:
                for mode in ("standalone", "legacy"):
                    round_jobs.append(
                        {
                            "round": str(round_id),
                            "model": model,
                            "op": op,
                            "mode": mode,
                        }
                    )
        random.Random(args.seed + round_id).shuffle(round_jobs)
        jobs.extend(round_jobs)

    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, job in enumerate(jobs, start=1):
        print(
            f"[{idx:03d}/{len(jobs):03d}] {job['mode']} "
            f"{job['model']} {job['op']} round={job['round']}"
        )
        sample = _run_worker(
            [
                "--worker",
                "measure",
                "--bench-root",
                str(bench_root),
                "--model",
                job["model"],
                "--op",
                job["op"],
                "--mode",
                job["mode"],
                "--nenv",
                str(args.nenv),
                "--nstep",
                str(args.nstep),
                "--nthread",
                str(args.nthread),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--state-file",
                str(state_files[job["model"]]),
            ],
            env=_mode_env(job["mode"], legacy_site, legacy_shim),
        )
        sample["round"] = int(job["round"])
        samples.append(sample)

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for sample in samples:
        key = (sample["model"], sample["op"], sample["mode"])
        grouped.setdefault(key, []).append(sample["seconds_mean"])

    comparisons: dict[str, Any] = {}
    for model in args.models:
        for op in args.ops:
            item: dict[str, Any] = {"model": model, "op": op}
            for mode in ("standalone", "legacy"):
                seconds = grouped[(model, op, mode)]
                stats = _summary(seconds)
                throughputs = [
                    _throughput(op, value, args.nenv, args.nstep) for value in seconds
                ]
                throughput_stats = _summary(throughputs)
                item[mode] = {
                    "seconds": stats,
                    "throughput_mean": throughput_stats["mean"],
                    "throughput_median": throughput_stats["median"],
                    "throughput_p10": throughput_stats["p10"],
                    "throughput_p90": throughput_stats["p90"],
                }
            item["throughput_median_ratio"] = (
                item["standalone"]["throughput_median"]
                / item["legacy"]["throughput_median"]
            )
            comparisons[f"{model}:{op}"] = item

    result = {
        "config": {
            "bench_root": str(bench_root),
            "legacy_site": str(legacy_site),
            "legacy_shim": str(legacy_shim),
            "models": args.models,
            "ops": args.ops,
            "nenv": args.nenv,
            "nstep": args.nstep,
            "nthread": args.nthread,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "elapsed_s": time.perf_counter() - started,
        },
        "samples": samples,
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _print_summary(result)
    print(f"\nWrote: {output}")


def _parent_paired_main(
    args: argparse.Namespace,
    bench_root: Path,
    legacy_site: Path,
    legacy_shim: Path,
    output: Path,
    state_files: dict[str, Path],
    *,
    started: float | None,
) -> None:
    started = time.perf_counter() if started is None else started
    rng = random.Random(args.seed)
    pair_jobs: list[dict[str, Any]] = []
    for round_id in range(args.rounds):
        for model in args.models:
            for op in args.ops:
                order = ["standalone", "legacy"]
                rng.shuffle(order)
                pair_jobs.append(
                    {
                        "round": round_id,
                        "model": model,
                        "op": op,
                        "order": order,
                    }
                )
        rng.shuffle(pair_jobs)

    samples: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for idx, job in enumerate(pair_jobs, start=1):
        print(
            f"[pair {idx:03d}/{len(pair_jobs):03d}] "
            f"{job['model']} {job['op']} order={'/'.join(job['order'])} "
            f"round={job['round']}"
        )
        pair_samples: dict[str, dict[str, Any]] = {}
        for mode in job["order"]:
            sample = _measure_sample(
                bench_root=bench_root,
                legacy_site=legacy_site,
                legacy_shim=legacy_shim,
                state_file=state_files[job["model"]],
                model=job["model"],
                op=job["op"],
                mode=mode,
                nenv=args.nenv,
                nstep=args.nstep,
                nthread=args.nthread,
                warmup=args.warmup,
                repeat=args.repeat,
                round_id=job["round"],
            )
            pair_samples[mode] = sample
            samples.append(sample)
        standalone_t = pair_samples["standalone"]["seconds_mean"]
        legacy_t = pair_samples["legacy"]["seconds_mean"]
        standalone_tp = _throughput(job["op"], standalone_t, args.nenv, args.nstep)
        legacy_tp = _throughput(job["op"], legacy_t, args.nenv, args.nstep)
        pairs.append(
            {
                "round": job["round"],
                "model": job["model"],
                "op": job["op"],
                "order": job["order"],
                "standalone_seconds_mean": standalone_t,
                "legacy_seconds_mean": legacy_t,
                "standalone_throughput": standalone_tp,
                "legacy_throughput": legacy_tp,
                "throughput_ratio": standalone_tp / legacy_tp,
            }
        )
        if args.cooldown_s > 0 and idx != len(pair_jobs):
            time.sleep(args.cooldown_s)

    paired_comparisons: dict[str, Any] = {}
    for model in args.models:
        for op in args.ops:
            selected = [
                pair
                for pair in pairs
                if pair["model"] == model and pair["op"] == op
            ]
            ratios = [pair["throughput_ratio"] for pair in selected]
            paired_comparisons[f"{model}:{op}"] = {
                "model": model,
                "op": op,
                "count": len(ratios),
                "ratio": _summary(ratios),
            }

    result = {
        "config": {
            "bench_root": str(bench_root),
            "legacy_site": str(legacy_site),
            "legacy_shim": str(legacy_shim),
            "models": args.models,
            "ops": args.ops,
            "nenv": args.nenv,
            "nstep": args.nstep,
            "nthread": args.nthread,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "paired": True,
            "cooldown_s": args.cooldown_s,
            "elapsed_s": time.perf_counter() - started,
        },
        "samples": samples,
        "pairs": pairs,
        "paired_comparisons": paired_comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _print_paired_summary(result)
    print(f"\nWrote: {output}")


def _worker_prepare(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.bench_root))
    import numpy as np
    import run_benchmarks as bench

    model = bench.load_model(args.model)
    states = bench.get_random_state(model, args.nenv, np.random.default_rng(args.seed))
    Path(args.state_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.state_file, states=states)
    print(
        "STRICT_AB_JSON="
        + json.dumps(
            {
                "worker": "prepare",
                "model": args.model,
                "state_file": str(args.state_file),
                "shape": list(states.shape),
            }
        )
    )


def _worker_measure(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.bench_root))
    import mujoco
    import numpy as np
    import run_benchmarks as bench

    from mujoco_uni.batch_env import BatchEnvPool

    model = bench.load_model(args.model)
    states = np.load(args.state_file)["states"]
    if states.shape[0] != args.nenv:
        raise ValueError(f"state file has {states.shape[0]} envs, expected {args.nenv}")

    with BatchEnvPool(model, nbatch=args.nenv, nthread=args.nthread) as pool:
        if args.op == "step":
            def run_once():
                return pool.step(states, nstep=args.nstep)
        elif args.op == "forward":
            def run_once():
                return pool.forward(states, chunk_size=4)
        else:
            raise ValueError(f"unsupported op: {args.op}")

        for _ in range(args.warmup):
            run_once()
        times = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            run_once()
            times.append(time.perf_counter() - start)

    arr = np.asarray(times, dtype=np.float64)
    out = {
        "worker": "measure",
        "mode": args.mode,
        "model": args.model,
        "op": args.op,
        "mujoco_file": mujoco.__file__,
        "mujoco_version": mujoco.__version__,
        "batch_pool_module": BatchEnvPool.__module__,
        "model_nq": int(model.nq),
        "model_nv": int(model.nv),
        "model_nu": int(model.nu),
        "nenv": args.nenv,
        "nstep": args.nstep,
        "nthread": args.nthread,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seconds_mean": float(arr.mean()),
        "seconds_median": float(np.median(arr)),
        "seconds_min": float(arr.min()),
        "seconds_max": float(arr.max()),
        "seconds_std": float(arr.std()),
    }
    print("STRICT_AB_JSON=" + json.dumps(out))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["prepare", "measure"])
    parser.add_argument("--bench-root", type=Path, default=Path("/tmp/mujoco_uni_bench"))
    parser.add_argument("--legacy-site", type=Path, default=Path("/tmp/mujoco_uni_legacy_site"))
    parser.add_argument("--legacy-shim", type=Path, default=Path("/tmp/mujoco_uni_legacy_shim"))
    parser.add_argument("--output", type=Path, default=_repo_root() / "docs" / "strict_ab_mj38_results.json")
    parser.add_argument("--models", nargs="+", default=["Franka", "Go1"])
    parser.add_argument("--ops", nargs="+", choices=["step", "forward"], default=["step", "forward"])
    parser.add_argument("--model", default="Franka")
    parser.add_argument("--op", choices=["step", "forward"], default="step")
    parser.add_argument("--mode", choices=["standalone", "legacy"], default="standalone")
    parser.add_argument("--nenv", type=int, default=4096)
    parser.add_argument("--nstep", type=int, default=50)
    parser.add_argument("--nthread", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--cooldown-s", type=float, default=0.0)
    parser.add_argument("--state-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker == "prepare":
        if args.state_file is None:
            raise ValueError("--state-file is required for prepare worker")
        _worker_prepare(args)
    elif args.worker == "measure":
        if args.state_file is None:
            raise ValueError("--state-file is required for measure worker")
        _worker_measure(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
