"""Path-based validation command selection for backport branches."""

from __future__ import annotations

import subprocess
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from scripts.backport.registry import ValidationRule


def changed_paths_since_base(repo_dir: str, base_ref: str) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def select_validation_commands(
    base_commands: Iterable[str],
    validation_rules: Iterable["ValidationRule"],
    changed_paths: Iterable[str],
) -> list[str]:
    commands: list[str] = []
    seen: set[str] = set()
    for command in base_commands:
        if command not in seen:
            commands.append(command)
            seen.add(command)

    paths = tuple(changed_paths)
    for rule in validation_rules:
        if not _rule_matches(rule.paths, paths):
            continue
        for command in rule.commands:
            if command not in seen:
                commands.append(command)
                seen.add(command)
    return commands


def _rule_matches(patterns: Iterable[str], changed_paths: Iterable[str]) -> bool:
    return any(fnmatch(path, pattern) for path in changed_paths for pattern in patterns)
