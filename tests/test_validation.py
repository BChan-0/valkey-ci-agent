from __future__ import annotations

import subprocess

from scripts.backport.registry import ValidationRule
from scripts.backport.validation import (
    changed_paths_since_base,
    select_validation_commands,
)


def test_select_validation_commands_appends_matching_rules_once() -> None:
    commands = select_validation_commands(
        ["make"],
        [
            ValidationRule(paths=("src/cluster_legacy.c",), commands=("cluster-smoke",)),
            ValidationRule(paths=("tests/unit/cluster/*.tcl",), commands=("cluster-smoke", "tcl-smoke")),
            ValidationRule(paths=("src/networking.c",), commands=("network-smoke",)),
        ],
        ["tests/unit/cluster/cli.tcl", "README.md"],
    )

    assert commands == ["make", "cluster-smoke", "tcl-smoke"]


def test_changed_paths_since_base_uses_merge_base(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "branch", "base"], cwd=tmp_path, check=True)

    (tmp_path / "changed.txt").write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "changed.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "changed"], cwd=tmp_path, check=True, capture_output=True)

    assert changed_paths_since_base(str(tmp_path), "base") == ("changed.txt",)
