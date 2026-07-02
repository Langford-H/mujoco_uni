"""Command-line interface for MuJoCo solver-version selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mujoco_uni.version import MUJOCO_VERSION_SPEC
from mujoco_uni.version_manager import (
    SUPPORTED_MUJOCO_MINOR_ORDER,
    discover_mujoco_envs,
    prepare_env,
    run_in_env,
    select_default_env,
)


def _add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mujoco", default="auto", help="MuJoCo version, for example 3.8 or 3.10.0.")
    parser.add_argument("--env-dir", default=None, help="Versioned uv env directory.")
    parser.add_argument("--project", default=None, help="Optional project to install editable, such as UniLab.")
    parser.add_argument("--package-source", default=None, help="MuJoCoUni source path or package spec.")
    parser.add_argument("--python", default=None, help="Python version used when creating a new uv env.")
    parser.add_argument("--reinstall", action="store_true", help="Force reinstall/rebuild mujoco-uni.")


def _env_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "env_dir": args.env_dir,
        "project": args.project,
        "package_source": args.package_source,
        "python_version": args.python,
        "reinstall": args.reinstall,
    }


def _env_dict(env) -> dict[str, str]:
    return {
        "version": env.version,
        "env_dir": str(env.env_dir),
        "python": str(env.python),
        "source": env.source,
    }


def _cmd_versions(args: argparse.Namespace) -> int:
    roots = [Path(root) for root in (args.root or [Path.cwd()])]
    envs = discover_mujoco_envs(roots, include_current=not args.no_current)
    selected = select_default_env(envs)
    if args.json:
        print(
            json.dumps(
                {
                    "version_spec": MUJOCO_VERSION_SPEC,
                    "preference": list(SUPPORTED_MUJOCO_MINOR_ORDER),
                    "default": None if selected is None else _env_dict(selected),
                    "envs": [_env_dict(env) for env in envs],
                },
                indent=2,
            )
        )
        return 0

    print(f"MuJoCoUni supports mujoco{MUJOCO_VERSION_SPEC}")
    print("default preference:", " > ".join(SUPPORTED_MUJOCO_MINOR_ORDER))
    if selected is None:
        print("default: 3.8.0 (no prepared env found)")
    else:
        print(f"default: {selected.version} ({selected.python})")
    for env in envs:
        marker = "*" if selected == env else " "
        print(f"{marker} {env.version:7} {env.source:15} {env.python}")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    env = prepare_env(
        args.mujoco,
        **_env_kwargs(args),
    )
    print(f"prepared mujoco {env.version}: {env.python}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if not args.command:
        raise SystemExit("mujoco-uni run requires a command after --")
    return run_in_env(
        args.command,
        version=args.mujoco,
        **_env_kwargs(args),
        prepare=not args.no_prepare,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mujoco-uni")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    versions = subparsers.add_parser("versions", help="List discovered MuJoCo versioned envs.")
    versions.add_argument("--root", action="append", default=None)
    versions.add_argument("--no-current", action="store_true")
    versions.add_argument("--json", action="store_true")
    versions.set_defaults(func=_cmd_versions)

    prepare = subparsers.add_parser("prepare", help="Create or refresh a versioned MuJoCo env.")
    _add_env_args(prepare)
    prepare.set_defaults(func=_cmd_prepare)

    run = subparsers.add_parser("run", help="Run a command under a selected MuJoCo version.")
    _add_env_args(run)
    run.add_argument("--no-prepare", action="store_true", help="Use only already-discovered envs.")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
