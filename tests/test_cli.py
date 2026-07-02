from __future__ import annotations

from pathlib import Path

from mujoco_uni.cli import build_parser


def test_versions_root_override_does_not_keep_cwd_default(tmp_path: Path) -> None:
    args = build_parser().parse_args(["versions", "--root", str(tmp_path)])

    assert args.root == [str(tmp_path)]


def test_versions_root_defaults_to_runtime_cwd() -> None:
    args = build_parser().parse_args(["versions"])

    assert args.root is None
