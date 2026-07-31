"""Find the issues to triage and read the failure facts off their bodies.

The detector records everything triage needs in the issue body it publishes: the
test name and file, the error trace, and a CI link per failing job. The link
carries the run id and the job id, which is what the per-job log endpoint takes,
so an issue is enough to re-fetch the evidence behind it.

Reading the published issue rather than the detector's in-memory failures is
what lets one code path serve both a fresh run and an older issue: the only
difference is which issues the filter selects.

An issue whose body cannot be parsed is skipped with a logged reason. A body
that predates a rendering change, or that a maintainer edited, must not stop the
rest of the batch.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from scripts.common.github_client import retry_github_call
from scripts.failure_triage.models import JobRef, TriageTarget

logger = logging.getLogger(__name__)

# The label the detector puts on every issue it files, whatever the type.
DETECTOR_LABEL = "test-failure"

# Marker namespaces the detector publishes, one per failure type. An issue
# carrying any of them was filed by the detector and records the body shape this
# module parses. Kept as a literal rather than imported from the detector's
# renderer so triage does not depend on the module it reads the output of.
_MARKER_NAMESPACES = {
    "valkey-ci-agent:test-failure": "assertion",
    "valkey-ci-agent:sanitizer-error": "sanitizer",
    "valkey-ci-agent:valgrind-error": "valgrind",
    "valkey-ci-agent:test-timeout": "timeout",
    "valkey-ci-agent:startup-failure": "startup",
    "valkey-ci-agent:test-exception": "exception",
    "valkey-ci-agent:memory-leak": "memory-leak",
    "valkey-ci-agent:unittest-failure": "unittest",
    # A valgrind or sanitizer failure with a cross-tool identity is filed under
    # one shared namespace so the two tools dedup against each other. It is a
    # common shape for that class, so leaving it out drops those issues.
    "valkey-ci-agent:memory-error": "memory-error",
}

# Filler the detector renders for a failure that names no test or file. It is
# not a real identity, so it is read back as an empty field.
_MISSING_FIELD = "[no test]"

# The marker triage leaves on a comment it posted. Versioned so a later change
# to the prompt can re-triage a backlog by raising the version, while a normal
# run never comments on the same issue twice.
TRIAGE_MARKER_VERSION = "v1"
TRIAGE_MARKER_PREFIX = "valkey-ci-agent:ai-triage"

# The detector's per-job CI links. The job anchor is what makes the per-job log
# endpoint reachable, so it is captured; a link without one still yields the run.
_JOB_URL_RE = re.compile(
    r"https://github\.com/[^/\s]+/[^/\s]+/actions/runs/(\d+)/job/(\d+)"
)
_RUN_URL_RE = re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/actions/runs/(\d+)")

# The detector rewrites this marker to the CI run it last processed on every
# update, so it names the freshest run, whereas the body's CI links are frozen at
# first publication. Evidence is fetched from this run.
_LAST_KEY_RE = re.compile(r"last-key:(\d+)")

# The job name the detector renders in front of each CI link.
_CI_LINK_LINE_RE = re.compile(
    r"^\s*-\s+`(?P<job>[^`]+)`:\s*\[[^\]]*\]\((?P<url>[^)]+)\)", re.MULTILINE
)

_TEST_NAME_RE = re.compile(r"^-\s+Test name:\s*`(?P<value>.*)`\s*$", re.MULTILINE)
_TEST_FILE_RE = re.compile(r"^-\s+Test file:\s*`(?P<value>.*)`\s*$", re.MULTILINE)

# The first fenced block under the trace header. The fence length varies (the
# renderer grows it past any backtick run in the trace), so the closing fence is
# matched by backreference on the opener.
_TRACE_RE = re.compile(
    r"\*\*Error stack trace\*\*\s*\n+(?P<fence>`{3,})[^\n]*\n(?P<trace>.*?)\n?(?P=fence)",
    re.DOTALL,
)

# Values that are not a real field: the renderer's own fillers and the trace
# placeholder. Compared lowercased, so the bracketed filler is matched here.
_PLACEHOLDERS = {"", "n/a", "unknown", "none", _MISSING_FIELD.lower()}


def triage_marker(run_id: int) -> str:
    """The marker identifying a triage comment for *run_id*."""
    return f"<!-- {TRIAGE_MARKER_PREFIX}:{TRIAGE_MARKER_VERSION}:{run_id} -->"


def _marker_version_re() -> re.Pattern[str]:
    """Matches a triage marker of the current version for any run."""
    return re.compile(
        rf"<!--\s*{re.escape(TRIAGE_MARKER_PREFIX)}:{re.escape(TRIAGE_MARKER_VERSION)}:\d+\s*-->"
    )


def already_triaged(issue: Any) -> bool:
    """Whether *issue* already carries a triage comment of the current version.

    Listing an issue's comments costs a request per issue, so this is called
    only for issues that passed the cheaper body filters.
    """
    pattern = _marker_version_re()
    try:
        comments = retry_github_call(
            lambda: list(issue.get_comments()),
            retries=2, description=f"list comments on issue #{issue.number}",
        )
    except Exception:
        # Treat an unreadable comment list as already triaged. The alternative
        # is posting a comment that may be a duplicate, and a missed analysis is
        # recoverable on the next run while a duplicate comment is not.
        logger.warning(
            "Could not list comments on issue #%s; skipping it", issue.number,
            exc_info=True,
        )
        return True
    return any(pattern.search(comment.body or "") for comment in comments)


def select_targets(
    gh: Any,
    repo_full_name: str,
    *,
    run_id: int | None = None,
    issue_numbers: Iterable[int] = (),
    limit: int = 10,
    max_age_days: int = 30,
    force: bool = False,
) -> list[TriageTarget]:
    """Choose the issues to triage, newest first.

    ``run_id``, when given, restricts the selection to issues the detector
    stamped with that run as their last source, which is the set a run of the
    detector just created or updated. Without it every open detector issue
    inside ``max_age_days`` is a candidate, which is the backfill case.

    ``issue_numbers`` addresses specific issues and bypasses the age filter, for
    re-running a single issue. ``force`` skips the already-triaged check.
    """
    repo = retry_github_call(
        lambda: gh.get_repo(repo_full_name),
        retries=2, description=f"get repo {repo_full_name}",
    )

    if issue_numbers:
        issues = _issues_by_number(repo, issue_numbers)
    else:
        issues = _open_detector_issues(repo, max_age_days=max_age_days)

    targets: list[TriageTarget] = []
    for issue in issues:
        if len(targets) >= limit:
            logger.info("Reached the limit of %d issue(s); stopping selection", limit)
            break

        # Isolate each issue. A body that cannot be read, or a lazily loaded
        # attribute that raises, must not abort selection of the rest.
        try:
            target = _select_one(issue, run_id=run_id, force=force)
        except Exception:
            number = getattr(issue, "number", "?")
            logger.warning("Could not evaluate issue #%s; skipping it", number, exc_info=True)
            continue
        if target is not None:
            targets.append(target)

    logger.info("Selected %d issue(s) to triage", len(targets))
    return targets


def _select_one(
    issue: Any, *, run_id: int | None, force: bool,
) -> TriageTarget | None:
    """Evaluate one issue against the filters, returning a target or None."""
    body = issue.body or ""
    failure_type = _failure_type_from_body(body)
    if failure_type is None:
        logger.info("Issue #%s carries no detector marker; skipping it", issue.number)
        return None

    if run_id is not None and not _records_run(body, run_id):
        return None

    target = build_target(issue, failure_type)
    if target is None:
        return None

    if not force and already_triaged(issue):
        logger.info("Issue #%s is already triaged; skipping it", issue.number)
        return None

    return target


def _issues_by_number(repo: Any, numbers: Iterable[int]) -> list[Any]:
    """Fetch specific issues, skipping any that cannot be read."""
    issues = []
    for number in numbers:
        def _fetch(number: int = number) -> Any:
            return repo.get_issue(number)

        try:
            issues.append(retry_github_call(
                _fetch, retries=2, description=f"get issue #{number}",
            ))
        except Exception:
            logger.warning("Could not read issue #%s; skipping it", number, exc_info=True)
    return issues


def _open_detector_issues(repo: Any, *, max_age_days: int) -> list[Any]:
    """Open issues carrying the detector's label, newest first.

    The label filter is applied server-side, as the detector's own publisher
    does, so a repository with thousands of open issues is not paged through in
    full. The age cut keeps triage inside the window where the run logs the
    analysis needs still exist.
    """
    issues = retry_github_call(
        lambda: list(repo.get_issues(state="open", labels=[DETECTOR_LABEL])),
        retries=2, description="list open detector issues",
    )
    # The REST issues list returns pull requests too. Read the stored payload
    # rather than the `pull_request` attribute, which would fire a request per
    # issue to complete the object. The attribute is PyGithub-internal, so a
    # missing one is treated as "not a PR" rather than raising.
    issues = [
        issue for issue in issues
        if "pull_request" not in getattr(issue, "_rawData", {})
    ]

    if max_age_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        fresh = []
        for issue in issues:
            created = issue.created_at
            if created is None:
                continue
            # PyGithub returns naive UTC datetimes on older versions.
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created >= cutoff:
                fresh.append(issue)
        dropped = len(issues) - len(fresh)
        if dropped:
            logger.info(
                "Skipped %d issue(s) older than %d day(s); their run logs are "
                "likely expired", dropped, max_age_days,
            )
        issues = fresh

    issues.sort(key=lambda issue: issue.number, reverse=True)
    return issues


def _failure_type_from_body(body: str) -> str | None:
    """The failure type implied by the detector marker in *body*, or None.

    Each namespace is matched with its trailing colon (``<!-- <ns>: ``), and no
    namespace is a prefix of another, so at most one matches a well-formed body
    and the first match is unambiguous.
    """
    for namespace, failure_type in _MARKER_NAMESPACES.items():
        if f"<!-- {namespace}:" in body:
            return failure_type
    return None


def _records_run(body: str, run_id: int) -> bool:
    """Whether *body* names *run_id* as its last recorded source run.

    The detector writes the CI run it last processed into a ``last-key`` marker.
    Matching on it selects the issues that run of the detector created or
    updated, which a recurring failure's accumulated CI links cannot identify on
    their own. ``run_id`` is that CI run id, so a word boundary keeps it from
    matching a longer id that merely starts with the same digits.
    """
    return bool(re.search(rf"last-key:{run_id}\b", body))


def build_target(issue: Any, failure_type: str) -> TriageTarget | None:
    """Read one issue body into a target, or None when it cannot be used.

    Returns None when the body records no run id, which means there is no CI run
    to fetch evidence from and nothing to analyze.
    """
    body = issue.body or ""
    jobs = _parse_jobs(body)
    # Prefer the last-key run: the detector refreshes it to the run it last
    # processed, while the body's CI links stay frozen at first publication, so a
    # recurrence's evidence lives under the last-key run, not the body's links.
    # Fall back to the links only for an issue that predates the last-key marker.
    run_id = _parse_last_key(body) or _parse_run_id(body, jobs)
    if run_id == 0:
        logger.info(
            "Issue #%s records no CI run; skipping it", issue.number,
        )
        return None

    return TriageTarget(
        issue_number=issue.number,
        issue_url=issue.html_url,
        failure_type=failure_type,
        test_name=_parse_field(_TEST_NAME_RE, body),
        test_file=_parse_field(_TEST_FILE_RE, body),
        error=_parse_trace(body),
        run_id=run_id,
        jobs=jobs,
    )


def _parse_jobs(body: str) -> tuple[JobRef, ...]:
    """The failing jobs recorded in the body's CI links, in order.

    A job appearing more than once (the body lists a link per job, and a
    recurrence can re-record one) is kept once, at its first position.
    """
    jobs: list[JobRef] = []
    seen: set[str] = set()
    for match in _CI_LINK_LINE_RE.finditer(body):
        name = match.group("job").strip()
        url = match.group("url").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        job_match = _JOB_URL_RE.search(url)
        job_id = int(job_match.group(2)) if job_match else 0
        jobs.append(JobRef(name=name, url=url, job_id=job_id))
    return tuple(jobs)


def _parse_last_key(body: str) -> int:
    """The run recorded in the last-key marker, or 0 when the body has none."""
    match = _LAST_KEY_RE.search(body)
    return int(match.group(1)) if match else 0


def _parse_run_id(body: str, jobs: tuple[JobRef, ...]) -> int:
    """The run the failure was reported from, or 0 when the body names none.

    A job link is preferred because it is the run the recorded job belongs to.
    Falling back to any run URL in the body covers a link that lost its job
    anchor.
    """
    for job in jobs:
        match = _JOB_URL_RE.search(job.url)
        if match:
            return int(match.group(1))
    match = _RUN_URL_RE.search(body)
    return int(match.group(1)) if match else 0


def _parse_field(pattern: re.Pattern[str], body: str) -> str:
    """A single backticked field from the body, or "" when absent or filler."""
    match = pattern.search(body)
    if not match:
        return ""
    value = match.group("value").strip()
    return "" if value.lower() in _PLACEHOLDERS else value


def _parse_trace(body: str) -> str:
    """The error trace recorded under the trace header, or ""."""
    match = _TRACE_RE.search(body)
    if not match:
        return ""
    trace = match.group("trace").strip()
    return "" if trace.lower() in _PLACEHOLDERS else trace
