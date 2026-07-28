"""Recover timeout failures from CI console logs.

The Tcl test runner excludes timeouts from the structured artifact (and its
watchdog may kill the process before ``write_test_failures`` runs at all), so
a timed-out job's artifact carries no timeout entry regardless of what else it
captured. This module identifies failed jobs without a captured timeout,
downloads their console logs, and extracts [TIMEOUT] failures that would
otherwise be invisible to the detector.

Orchestration is separated from parsing: :mod:`timeout_parser` handles the
regex extraction; this module decides which jobs to scan, downloads their logs,
and returns deduplicated results ready to merge into the main failure list.
"""

from __future__ import annotations

import logging
from typing import Any

from scripts.common.workflow_artifacts import ArtifactClient
from scripts.test_failure_detector.download import JobInfo
from scripts.test_failure_detector.parse_failures import UniqueFailure
from scripts.test_failure_detector.timeout_parser import (
    find_job_log,
    jobs_needing_log_scan,
    parse_timeouts_from_log,
)

logger = logging.getLogger(__name__)


def recover_timeouts(
    all_failures: dict[str, Any],
    job_info: JobInfo,
    artifact_client: ArtifactClient,
    repo_full_name: str,
    run_id: int,
) -> list[UniqueFailure]:
    """Scan console logs for timeout failures missed by the artifact.

    Returns a list of UniqueFailure objects (with FailureType.TIMEOUT) for
    timeouts recovered from logs. These are already deduplicated within each
    job but may overlap with timeouts the runner captured in the artifact;
    callers should merge them with artifact-derived failures and rely on
    parse_and_deduplicate's grouping to collapse duplicates.

    Returns an empty list (rather than raising) if logs are unavailable,
    expired, or contain no timeout markers. This is best-effort recovery:
    a log download failure must not block processing of artifact-derived
    failures.
    """
    needs_scan = jobs_needing_log_scan(all_failures, job_info.failed)
    if not needs_scan:
        return []

    logger.info(
        "Scanning console logs of %d failed job(s) without a captured timeout: %s",
        len(needs_scan), ", ".join(sorted(needs_scan)),
    )

    try:
        run_logs = artifact_client.download_run_logs(repo_full_name, run_id)
    except Exception as exc:
        logger.warning(
            "Could not download run logs for timeout recovery: %s", exc,
        )
        return []

    if not run_logs:
        logger.info("Run logs unavailable or expired; skipping timeout recovery.")
        return []

    recovered: list[UniqueFailure] = []
    for job_name in sorted(needs_scan):
        log_content = find_job_log(run_logs, job_name)
        if log_content is None:
            logger.debug("No log file found for job %s", job_name)
            continue

        job_url = job_info.urls.get(job_name, "")
        timeouts = parse_timeouts_from_log(log_content, job_name, job_url=job_url)
        recovered.extend(timeouts)

    if recovered:
        logger.info("Recovered %d timeout failure(s) from console logs.", len(recovered))
    return recovered
