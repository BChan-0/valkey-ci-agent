"""Gather the evidence for one failure and lay it out for the agent to read.

The console log of a single Valkey CI job runs to about a megabyte, which is far
more than the failure itself needs and enough to dominate the cost of every
analysis. So the search is done here, deterministically: the job's log is
fetched, the block around the reported failure is carved out, and only that
excerpt is written for the agent. In a measured sample this took roughly a
megabyte down to sixty kilobytes.

The excerpt is worth carving rather than summarizing because the Daily workflow
runs its suites with ``--dump-logs``, so the server logs of the failing test are
printed immediately around the verdict line. The window therefore holds the
diagnostic evidence, not just the assertion text.

The Valkey source is cloned at the commit the run was built from, not the branch
tip, so the agent reads the tree that actually failed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.common.git_clone import shallow_clone_at_sha
from scripts.common.github_client import retry_github_call
from scripts.common.text_utils import strip_ansi
from scripts.common.workflow_artifacts import ArtifactClient
from scripts.failure_triage.models import FailureEvidence, JobRef, TriageTarget

logger = logging.getLogger(__name__)

# Directory names inside the agent's working directory.
_EVIDENCE_DIR = "evidence"
_SOURCE_DIR = "valkey"

# Lines kept around the failure marker. The window reaches well back because the
# dumped server logs precede the verdict line, and only a little forward because
# what follows is the next test's setup.
_LINES_BEFORE = 400
_LINES_AFTER = 150

# Upper bound on the rendered excerpt handed to the agent, header included. A
# pathological log line (a single-line memory dump) could otherwise make the
# window arbitrarily large.
_MAX_EXCERPT_CHARS = 80_000

# Room held back from the line budget for the two- or three-line header the
# carver prepends, so the rendered excerpt stays within _MAX_EXCERPT_CHARS.
_HEADER_RESERVE = 300

# The runner stamps every log line with an ISO timestamp. It is a third of the
# bytes and tells the analysis nothing, so it is stripped.
_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s?")

# Verdict markers the Tcl runner and gtest print at a failed test. The runner
# also prints an end-of-run recap of every failure as "*** [err]: ...", so the
# recap lines carry "[err]:" too; the anchor search runs forward and takes the
# first match, which is the failure in place rather than its recap at the tail.
_FAILURE_MARKERS = (
    "[err]:",
    "[exception]:",
    "[TIMEOUT]:",
    "[FAILED]",
)

# The header of the runner's end-of-run recap. Lines at or after it are the recap
# of failures already seen above, not a failure in place, so the anchor search
# skips them when an earlier match exists.
_RECAP_HEADER = "The following tests failed:"

# Signatures of the tools that report without a test name, used when the failure
# has no name to search for.
_TOOL_MARKERS = (
    "ERROR: AddressSanitizer",
    "ERROR: LeakSanitizer",
    "runtime error:",
    "ERROR SUMMARY:",
    "definitely lost",
    "Can't start the server",
)


def collect(
    target: TriageTarget,
    workdir: Path,
    *,
    repo_full_name: str,
    artifact_client: ArtifactClient,
    gh: Any,
) -> FailureEvidence:
    """Lay out the evidence for *target* under *workdir*.

    Always returns evidence. A log that cannot be fetched or a clone that fails
    leaves the corresponding field empty and records why, so the prompt can tell
    the agent what it does not have instead of letting it assume.
    """
    evidence_dir = workdir / _EVIDENCE_DIR
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = FailureEvidence(workdir=workdir)

    _write_failure_facts(target, evidence_dir)
    _attach_log_excerpt(target, evidence, evidence_dir, repo_full_name, artifact_client, gh)
    _attach_source(target, evidence, workdir, repo_full_name, gh)
    return evidence


def _write_failure_facts(target: TriageTarget, evidence_dir: Path) -> None:
    """Record the failure the detector reported, as the agent's starting point."""
    facts = {
        "failure_type": target.failure_type,
        "test_name": target.test_name,
        "test_file": target.test_file,
        "error": target.error,
        "run_id": target.run_id,
        "jobs": [job.name for job in target.jobs],
    }
    (evidence_dir / "failure.json").write_text(
        json.dumps(facts, indent=2, sort_keys=True), encoding="utf-8",
    )


def _attach_log_excerpt(
    target: TriageTarget,
    evidence: FailureEvidence,
    evidence_dir: Path,
    repo_full_name: str,
    artifact_client: ArtifactClient,
    gh: Any,
) -> None:
    """Fetch the failing job's log and write the excerpt around the failure."""
    job = _resolve_job(target, repo_full_name, gh)
    if job is None:
        evidence.log_unavailable_reason = (
            "the failing job could not be identified for this run, so its log "
            "cannot be located"
        )
        return

    try:
        raw = artifact_client.download_job_log(repo_full_name, job.job_id)
    except Exception as exc:
        evidence.log_unavailable_reason = f"the job log could not be downloaded: {exc}"
        logger.warning(
            "Could not download log for job %s (%d): %s",
            job.name, job.job_id, exc, exc_info=True,
        )
        return

    if not raw:
        evidence.log_unavailable_reason = (
            "the job log has expired and is no longer served by GitHub"
        )
        logger.info("Log for job %s (%d) has expired", job.name, job.job_id)
        return

    lines = _clean_log_lines(raw)
    excerpt = carve_excerpt(lines, target)
    if excerpt is None:
        evidence.log_unavailable_reason = (
            "the job log holds no recognizable failure block for this failure"
        )
        return

    path = evidence_dir / "job-log-excerpt.txt"
    path.write_text(excerpt.text, encoding="utf-8")
    evidence.job_name = job.name
    evidence.log_excerpt_path = f"{_EVIDENCE_DIR}/{path.name}"
    evidence.excerpt_lines = excerpt.log_lines
    evidence.log_truncated = excerpt.truncated
    logger.info(
        "Carved a %d-line excerpt from job %s", evidence.excerpt_lines, job.name,
    )


def _resolve_job(target: TriageTarget, repo_full_name: str, gh: Any) -> JobRef | None:
    """The failing job whose log to fetch, resolved against ``target.run_id``.

    ``target.run_id`` is the run the detector last processed, but the job ids in
    the issue body are frozen at first publication and belong to an older run, so
    a body job id would fetch the wrong (often expired) log on a recurrence. The
    run's jobs are listed fresh and matched to a recorded failing job by name,
    since job names are stable across daily runs while ids are not. When no name
    matches, the run's first failed job is used. Only if the run's jobs cannot be
    listed at all does a body job id (with its own run link) serve as a fallback.
    """
    jobs = _run_jobs(gh, repo_full_name, target.run_id)
    if jobs:
        recorded = [job.name for job in target.jobs]
        by_name = {job.name: job for job in jobs}
        for name in recorded:
            if name in by_name:
                return _to_job_ref(by_name[name])
        failed = [job for job in jobs if _job_failed(job)]
        chosen = failed[0] if failed else None
        if chosen is not None:
            return _to_job_ref(chosen)

    # The run's jobs were unavailable. A body link that carries its own job id
    # points at the run that link belongs to, so it is a usable last resort.
    return target.first_job_with_log()


def _run_jobs(gh: Any, repo_full_name: str, run_id: int) -> list[Any]:
    """The jobs of a workflow run, or an empty list when they cannot be read."""
    try:
        run = retry_github_call(
            lambda: gh.get_repo(repo_full_name).get_workflow_run(run_id),
            retries=2, description=f"get run {run_id}",
        )
        return list(retry_github_call(
            lambda: run.jobs(), retries=2, description=f"list jobs for run {run_id}",
        ))
    except Exception:
        logger.warning("Could not list jobs for run %d", run_id, exc_info=True)
        return []


def _job_failed(job: Any) -> bool:
    return str(getattr(job, "conclusion", "") or "") in {"failure", "timed_out"}


def _to_job_ref(job: Any) -> JobRef:
    return JobRef(
        name=str(getattr(job, "name", "") or ""),
        url=str(getattr(job, "html_url", "") or ""),
        job_id=int(getattr(job, "id", 0) or 0),
    )


def _clean_log_lines(raw: bytes) -> list[str]:
    """Decode the log and drop the per-line noise before searching it.

    The runner's timestamp prefix and any ANSI colouring are removed. A leading
    byte-order mark on the first line is stripped so a marker at the start of
    the log still matches.
    """
    text = raw.decode("utf-8", errors="replace").lstrip("﻿")
    return [
        _TIMESTAMP_PREFIX_RE.sub("", strip_ansi(line))
        for line in text.splitlines()
    ]


@dataclass
class Excerpt:
    """A carved log excerpt: its rendered text, the count of log lines it holds
    (not counting the header), and whether the window was truncated to fit."""

    text: str
    log_lines: int
    truncated: bool


def carve_excerpt(lines: list[str], target: TriageTarget) -> Excerpt | None:
    """The excerpt around the failure, or None when none can be located.

    None means the log holds no line that can be tied to this failure, which is
    treated as having no log rather than sending an arbitrary slice of an
    unrelated part of the run.
    """
    anchor = _find_anchor(lines, target)
    if anchor is None:
        return None

    window_start = max(0, anchor - _LINES_BEFORE)
    window_end = min(len(lines), anchor + _LINES_AFTER + 1)

    kept_start, kept_end = _budgeted_span(lines, anchor, window_start, window_end)
    truncated = kept_start > window_start or kept_end < window_end

    # Enforce the hard bound on the rendered text, header included. The budget
    # split reserves room for the header, but rather than trust that reserve to
    # be exactly right, leading lines are dropped until the whole thing fits, so
    # the returned text is never over the cap regardless of the header's length.
    while True:
        kept = lines[kept_start:kept_end]
        header = _excerpt_header(kept_start, kept_end, len(lines), truncated)
        text = "\n".join(header + kept)
        if len(text) <= _MAX_EXCERPT_CHARS or kept_start >= anchor:
            return Excerpt(text=text, log_lines=len(kept), truncated=truncated)
        kept_start += 1
        truncated = True


def _excerpt_header(
    kept_start: int, kept_end: int, total: int, truncated: bool,
) -> list[str]:
    lines = [
        f"Excerpt of the job log around the reported failure "
        f"(lines {kept_start + 1}-{kept_end} of {total})."
    ]
    if truncated:
        lines.append(
            "The excerpt was truncated to fit the size limit; some lines of the "
            "window were dropped."
        )
    lines.append("")
    return lines


def _budgeted_span(
    lines: list[str], anchor: int, window_start: int, window_end: int,
) -> tuple[int, int]:
    """The span of line indices to keep, within the character budget.

    The dumped server logs precede the verdict line and the next test's setup
    follows, so the backward direction carries the evidence. The budget is split
    proportionally to the before/after line counts and the backward reserve is
    spent first, so a run of fat trailing lines cannot starve the preceding
    context. Any budget the forward pass leaves unused is then spent extending
    backward further. The anchor line is always kept, even when it alone exceeds
    the budget, since an excerpt without the failure line is useless.

    A trailing header of a few lines is written by the caller; ``_HEADER_RESERVE``
    holds room for it so the rendered excerpt stays within ``_MAX_EXCERPT_CHARS``.
    """
    budget = _MAX_EXCERPT_CHARS - _HEADER_RESERVE
    forward_budget = budget * _LINES_AFTER // (_LINES_BEFORE + _LINES_AFTER)

    used = len(lines[anchor]) + 1
    kept_end = anchor + 1
    for index in range(anchor + 1, window_end):
        cost = len(lines[index]) + 1
        if used + cost > forward_budget:
            break
        kept_end = index + 1
        used += cost

    kept_start = anchor
    for index in range(anchor - 1, window_start - 1, -1):
        cost = len(lines[index]) + 1
        if used + cost > budget:
            break
        kept_start = index
        used += cost

    return kept_start, kept_end


def _find_anchor(lines: list[str], target: TriageTarget) -> int | None:
    """The index of the line the excerpt is centred on, or None.

    Each search runs forward and takes the first match, so it lands on the
    failure in place rather than on the runner's end-of-run recap of the same
    failures at the tail of the log. A named test on a verdict line is preferred,
    since it pins the excerpt to this failure among several in one job; the
    recorded error text and then any verdict or tool marker are the fallbacks for
    the types that report no test name.
    """
    recap = _recap_start(lines)

    if target.test_name:
        found = _first_match(
            lines, recap,
            lambda line: target.test_name in line and _has_marker(line, _FAILURE_MARKERS),
        )
        if found is not None:
            return found

    error_line = _first_error_line(target.error)
    if error_line:
        found = _first_match(lines, recap, lambda line: error_line in line)
        if found is not None:
            return found

    return _first_match(
        lines, recap, lambda line: _has_marker(line, _FAILURE_MARKERS + _TOOL_MARKERS),
    )


def _first_match(
    lines: list[str], recap: int, predicate: Callable[[str], bool],
) -> int | None:
    """The first line index the predicate accepts, preferring lines before the
    recap. Only if none precede the recap is a match within it accepted, so a
    failure that appears solely in the recap is still found."""
    fallback: int | None = None
    for index, line in enumerate(lines):
        if not predicate(line):
            continue
        if index < recap:
            return index
        if fallback is None:
            fallback = index
    return fallback


def _recap_start(lines: list[str]) -> int:
    """The index of the end-of-run recap header, or ``len(lines)`` if absent."""
    for index, line in enumerate(lines):
        if _RECAP_HEADER in line:
            return index
    return len(lines)


def _has_marker(line: str, markers: tuple[str, ...]) -> bool:
    return any(marker in line for marker in markers)


def _first_error_line(error: str) -> str:
    """The first substantial line of the recorded error, for searching the log.

    Very short lines are rejected: they match too much of a log to locate a
    failure with.
    """
    for line in error.splitlines():
        stripped = line.strip()
        if len(stripped) >= 20:
            return stripped
    return ""


def _attach_source(
    target: TriageTarget,
    evidence: FailureEvidence,
    workdir: Path,
    repo_full_name: str,
    gh: Any,
) -> None:
    """Clone the repository at the commit the run was built from.

    Without the run's commit the clone is skipped rather than falling back to the
    branch tip: reasoning about a different tree than the one that failed
    produces confident references to code that was never involved.
    """
    sha = _run_head_sha(gh, repo_full_name, target.run_id)
    if not sha:
        logger.info(
            "Could not determine the head commit of run %d; skipping the clone",
            target.run_id,
        )
        return

    dest = workdir / _SOURCE_DIR
    if not shallow_clone_at_sha(repo_full_name, dest, sha):
        logger.warning("Could not clone %s at %s", repo_full_name, sha[:12])
        return

    evidence.source_path = _SOURCE_DIR
    evidence.source_sha = sha


def _run_head_sha(gh: Any, repo_full_name: str, run_id: int) -> str:
    """The commit a workflow run was built from, or "" when it cannot be read."""
    try:
        run = retry_github_call(
            lambda: gh.get_repo(repo_full_name).get_workflow_run(run_id),
            retries=2, description=f"get run {run_id}",
        )
    except Exception:
        logger.warning("Could not read run %d", run_id, exc_info=True)
        return ""
    return str(getattr(run, "head_sha", "") or "")
