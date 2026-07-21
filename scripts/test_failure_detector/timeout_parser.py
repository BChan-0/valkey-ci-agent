"""Parse test timeouts from CI console logs.

The Valkey Tcl test runner's ``write_test_failures`` deliberately excludes
timeout failures from the structured JSON artifact (see ``test_helper.tcl``,
line matching ``*[*TIMEOUT*]*``). The timeout information only appears in the
CI console output, printed as::

    [TIMEOUT]: clients state report follows.
    ...
    [TIMEOUT]: <test_name> in <test_file>

and later in the summary::

    *** [TIMEOUT]: <test_name> in <test_file>

This module recovers those failures by scanning the console logs for jobs
whose artifact entries are empty (indicating the test run ended in timeout
without capturing the failure).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from scripts.test_failure_detector.parse_failures import (
    FailureType,
    JobReference,
    UniqueFailure,
)

logger = logging.getLogger(__name__)

# Matches the timeout failure lines printed by the Tcl test runner.
# Both the inline report and the final "*** [TIMEOUT]:" summary use this form.
# ANSI color codes may be stripped or present depending on how the logs are stored.
_TIMEOUT_RE = re.compile(
    r"\[(?:\x1b\[[^m]*m)?TIMEOUT(?:\x1b\[[^m]*m)?\]"
    r":\s*(.+?)\s+in\s+(tests/\S+\.tcl)",
)


def jobs_needing_log_scan(
    all_failures: dict[str, Any],
    failed_job_names: set[str],
) -> set[str]:
    """Identify failed jobs whose artifact entries contain no test failures.

    These are candidates for timeout recovery: the job failed (exit code 1)
    but the structured artifact has only empty lists, which happens when the
    test runner's watchdog killed the run and ``write_test_failures`` skipped
    the timeout entries.

    Jobs that have at least one captured failure in the artifact already have
    their failure represented and don't need log scanning for timeouts.
    """
    needs_scan: set[str] = set()
    for job_name in failed_job_names:
        suites = all_failures.get(job_name)
        if suites is None:
            # Job not in artifact at all (upload step may have been skipped)
            needs_scan.add(job_name)
            continue
        if not isinstance(suites, dict):
            needs_scan.add(job_name)
            continue
        has_entries = any(
            isinstance(entries, list) and len(entries) > 0
            for entries in suites.values()
        )
        if not has_entries:
            needs_scan.add(job_name)
    return needs_scan


def parse_timeouts_from_log(
    log_content: bytes,
    job_name: str,
    job_url: str = "",
) -> list[UniqueFailure]:
    """Extract timeout failures from a single job's console log.

    Returns one UniqueFailure per distinct (test_name, test_file) pair found
    in [TIMEOUT] lines.
    """
    try:
        text = log_content.decode("utf-8", errors="replace")
    except Exception:
        logger.warning("Could not decode log for job %s", job_name)
        return []

    seen: dict[tuple[str, str], UniqueFailure] = {}
    for match in _TIMEOUT_RE.finditer(text):
        test_name = match.group(1).strip()
        test_file = match.group(2).strip()
        if not test_name or not test_file:
            continue

        key = (test_name, test_file)
        if key in seen:
            continue

        seen[key] = UniqueFailure(
            test_name=test_name,
            test_file=test_file,
            failure_type=FailureType.TIMEOUT,
            error="Test timed out (no progress for the configured timeout period)",
            jobs=[JobReference(job=job_name, suite="timeout", url=job_url)],
        )

    if seen:
        logger.info(
            "Recovered %d timeout failure(s) from logs of job %s",
            len(seen), job_name,
        )
    return list(seen.values())


def find_job_log(
    logs: dict[str, bytes],
    job_name: str,
) -> bytes | None:
    """Find the console log for a specific job in the run-logs zip contents.

    GitHub's run-log zip names files as ``<number>_<job_name>.txt``. The job
    name in the filename matches the job name from the API, including any
    spaces and parentheses.
    """
    # Try exact match first (most common)
    for filename, content in logs.items():
        # Strip the leading number prefix: "3_test-valgrind-test.txt" -> "test-valgrind-test"
        if not filename.endswith(".txt"):
            continue
        # Skip system.txt files in subdirectories
        if "/" in filename:
            continue
        name_part = re.sub(r"^\d+_", "", filename)
        name_part = name_part.removesuffix(".txt")
        if name_part == job_name:
            return content

    # Fallback: case-insensitive or partial match
    job_lower = job_name.lower()
    for filename, content in logs.items():
        if "/" in filename:
            continue
        if not filename.endswith(".txt"):
            continue
        name_part = re.sub(r"^\d+_", "", filename).removesuffix(".txt").lower()
        if name_part == job_lower:
            return content

    logger.debug("No log file found for job %s", job_name)
    return None
