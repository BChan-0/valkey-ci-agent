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


def test_run_build_commands_writes_full_log_on_failure(tmp_path) -> None:
    """When log_path is set, the full untruncated stdout/stderr must be persisted."""
    long_stdout = "warning line\n" * 5000  # ~60kB, well past the 2000-char tail
    real_error = "src/foo.c:42: error: undefined reference to 'old_helper'"
    full_stdout = real_error + "\n" + long_stdout

    completed = subprocess.CompletedProcess(
        args="make",
        returncode=1,
        stdout=full_stdout,
        stderr="link failed",
    )
    log_path = tmp_path / "validation.log"
    with patch(
        "scripts.common.build_validator.subprocess.run",
        return_value=completed,
    ):
        ok, summary = run_build_commands(
            str(tmp_path), ["make"], log_path=str(log_path),
        )

    assert ok is False
    # Summary stays short (existing tail behavior)
    assert "warning line" in summary
    # Full log file must contain the actual error that the tail dropped
    log_text = log_path.read_text()
    assert real_error in log_text
    assert "$ make" in log_text
    assert "# exit code: 1" in log_text
    assert "link failed" in log_text


def test_run_build_commands_overwrites_existing_log(tmp_path) -> None:
    """Each call should refresh the log file so stale runs aren't presented as current."""
    log_path = tmp_path / "validation.log"
    log_path.write_text("stale content from previous run")

    completed = subprocess.CompletedProcess(
        args="make", returncode=0, stdout="ok", stderr="",
    )
    with patch(
        "scripts.common.build_validator.subprocess.run",
        return_value=completed,
    ):
        run_build_commands(str(tmp_path), ["make"], log_path=str(log_path))

    text = log_path.read_text()
    assert "stale content" not in text
    assert "$ make" in text
