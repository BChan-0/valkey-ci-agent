"""Tests for shared build validation."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from scripts.common.build_validator import run_build_commands


def test_run_build_commands_returns_failure_on_timeout(tmp_path) -> None:
    with patch(
        "scripts.common.build_validator.subprocess.run",
        side_effect=subprocess.TimeoutExpired(
            cmd="make",
            timeout=1800,
            output="partial stdout",
            stderr="partial stderr",
        ),
    ):
        ok, output = run_build_commands(str(tmp_path), ["make"])

    assert ok is False
    assert "partial stdout" in output
    assert "partial stderr" in output


def test_run_build_commands_skips_empty_command_list(tmp_path) -> None:
    ok, output = run_build_commands(str(tmp_path), [])

    assert ok is True
    assert output == ""
