"""Shared build/test command runner for backport validation."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def run_build_commands(repo_dir: str, commands: list[str]) -> tuple[bool, str]:
    """Run validation commands sequentially.

    Returns (success, output). If commands is empty, returns (True, "") —
    build validation is skipped when no commands are configured.
    """
    if not commands:
        return True, ""
    for command in commands:
        logger.info("Running backport validation command: %s", command)
        try:
            # Registry build commands are operator-controlled repo config, not
            # user input from PRs or issues; shell=True is intentional so repos
            # can express normal build pipelines.
            result = subprocess.run(
                command,
                cwd=repo_dir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired as exc:
            output = "\n".join(
                part for part in [
                    _tail_text(exc.stdout),
                    _tail_text(exc.stderr),
                ]
                if part
            ).strip()
            detail = output or f"`{command}` timed out after 1800 seconds"
            return False, detail
        if result.returncode != 0:
            output = "\n".join(
                part for part in [result.stdout[-2000:], result.stderr[-2000:]]
                if part
            ).strip()
            return False, output or f"`{command}` failed with exit code {result.returncode}"
    return True, ""


def _tail_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-2000:]
