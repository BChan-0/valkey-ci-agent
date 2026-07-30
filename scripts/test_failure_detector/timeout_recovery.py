"""Recover timeout failures from CI console logs.

The runner's watchdog can kill the process before ``write_test_failures`` runs,
leaving a timed-out job with no timeout entry in the artifact. This module
identifies failed jobs without a captured timeout, downloads their console logs,
and extracts [TIMEOUT] failures that would otherwise be invisible.

It also attaches gtest console output to unit-test failures, whose artifact
entries carry only a verdict, for the same reason: the diagnostic exists only
in the log.

Orchestration is separated from parsing: :mod:`timeout_parser` and
:mod:`gtest_log_parser` handle the regex extraction; this module decides which
jobs to scan, downloads their logs, and returns deduplicated results ready to
merge into the main failure list.
"""

from __future__ import annotations

import logging
from typing import Any

from scripts.common.workflow_artifacts import ArtifactClient
from scripts.test_failure_detector.download import JobInfo
from scripts.test_failure_detector.gtest_log_parser import (
    parse_gtest_failures_from_log,
)
from scripts.test_failure_detector.parse_failures import FailureType, UniqueFailure
from scripts.test_failure_detector.timeout_parser import (
    clients_state_report_from_log,
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

        # Recovered timeouts carry the "timeout" suite, which has no step
        # mapping, so url_for returns the plain job URL here.
        job_url = job_info.url_for(job_name, "timeout")
        timeouts = parse_timeouts_from_log(log_content, job_name, job_url=job_url)
        recovered.extend(timeouts)

    if recovered:
        logger.info("Recovered %d timeout failure(s) from console logs.", len(recovered))
    return recovered


def enrich_log_only_errors(
    failures: list[UniqueFailure],
    job_info: JobInfo,
    artifact_client: ArtifactClient,
    repo_full_name: str,
    run_id: int,
    run_logs: dict[str, bytes] | None = None,
) -> None:
    """Attach console output to failures the artifact records without detail.

    Two types arrive with a placeholder instead of a diagnostic. A gtest failure
    carries only a verdict, because gtest-parallel's JSON dump holds no output.
    A timeout carries only "Test timed out", because the runner has nothing to
    report beyond the watchdog firing. In both cases the detail exists in the
    job log, so it is read back and attached in place.

    Best-effort, and mutates *failures* in place. A log that is expired,
    unavailable, or missing the failure's block leaves the placeholder as it
    was; losing the detail is not worth failing the run over.

    ``run_logs``, when supplied, is reused instead of downloading again. The
    whole run's logs are one zip of tens of megabytes, so the caller fetches it
    once and shares it with timeout recovery.
    """
    gtest_failures = [
        f for f in failures
        if f.failure_type == FailureType.UNITTEST and f.test_name
    ]
    # A timeout recovered from a log already carries its report; only the
    # artifact-derived ones still hold the bare placeholder.
    timeout_failures = [
        f for f in failures
        if f.failure_type == FailureType.TIMEOUT and "\n" not in f.error.strip()
    ]
    if not gtest_failures and not timeout_failures:
        return

    jobs_to_scan = {
        job_ref.job
        for f in (*gtest_failures, *timeout_failures)
        for job_ref in f.jobs
    } & set(job_info.failed)
    if not jobs_to_scan:
        return

    if run_logs is None:
        try:
            run_logs = artifact_client.download_run_logs(repo_full_name, run_id)
        except Exception as exc:
            logger.warning("Could not download run logs for failure detail: %s", exc)
            return

    if not run_logs:
        logger.info("Run logs unavailable or expired; keeping placeholders.")
        return

    # One failure can appear on several jobs. Scan in a stable order so the
    # attached output does not depend on set iteration order.
    gtest_outputs: dict[str, str] = {}
    timeout_reports: dict[str, str] = {}
    for job_name in sorted(jobs_to_scan):
        log_content = find_job_log(run_logs, job_name)
        if log_content is None:
            continue
        for test_name, body in parse_gtest_failures_from_log(log_content).items():
            gtest_outputs.setdefault(test_name, body)
        report = clients_state_report_from_log(log_content)
        if report:
            timeout_reports.setdefault(job_name, report)

    enriched = 0
    for failure in gtest_failures:
        recovered_output = gtest_outputs.get(failure.test_name)
        if recovered_output:
            failure.error = recovered_output
            enriched += 1

    for failure in timeout_failures:
        for job_ref in failure.jobs:
            job_report = timeout_reports.get(job_ref.job)
            if job_report:
                failure.error = f"{failure.error}\n\n{job_report}"
                enriched += 1
                break

    if enriched:
        logger.info("Attached console output to %d failure(s).", enriched)
    else:
        logger.info("No console detail found in logs; keeping placeholders.")
