"""Download test failure artifacts from a Valkey CI workflow run"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from itertools import islice

from github import Github
from github.WorkflowRun import WorkflowRun

from scripts.common.github_client import retry_github_call
from scripts.common.workflow_artifacts import ArtifactClient

logger = logging.getLogger(__name__)

# Name of the JSON file the Valkey CI workflow uploads inside its artifact zip.
_FAILURES_JSON_NAME = "all-test-failures.json"
_FAILURES_ARTIFACT_NAME = "all-test-failures"

# How far back to look for a usable run. The sweep only ever wants the latest
# one, so a run older than this is stale enough that reporting nothing found is
# more useful than paging through the whole history.
_MAX_RUNS_SCANNED = 50

def get_latest_daily_run(
    gh: Github,
    repo_full_name: str,
    workflow_name: str = "Daily",
    branch: str = "unstable",
) -> WorkflowRun | None:
    """Find the most recent completed (non-cancelled) Daily workflow run."""
    repo = retry_github_call(
        lambda: gh.get_repo(repo_full_name),
        retries=3,
        description=f"get repo {repo_full_name}",
    )

    workflows = retry_github_call(
        lambda: repo.get_workflows(),
        retries=3,
        description="list workflows",
    )

    daily_workflow = None
    for wf in workflows:
        if wf.name == workflow_name:
            daily_workflow = wf
            break

    if daily_workflow is None:
        logger.warning("Workflow %r not found in %s", workflow_name, repo_full_name)
        return None

    # Accept scheduled and manually dispatched runs. Pull-request runs are
    # excluded by their conclusion (action_required/skipped) in the loop below,
    # so we don't need to filter by event at the API level.
    #
    # get_runs() is lazy, so the islice must run inside the retried call for the
    # retries to cover the actual request.
    runs = retry_github_call(
        lambda: list(islice(
            daily_workflow.get_runs(branch=branch, status="completed"),
            _MAX_RUNS_SCANNED,
        )),
        retries=3,
        description=f"list runs for {workflow_name}",
    )

    for run in runs:
        # Skip runs that never actually executed: cancelled/skipped, runs
        # awaiting approval (action_required, e.g. fork PRs), expired (stale),
        # runs that died before any job started (startup_failure, e.g. invalid
        # workflow YAML), and runs with no conclusion yet. These produce no test
        # artifacts and would be mistaken for a clean pass.
        if run.conclusion in (
            "cancelled", "skipped", "action_required", "stale",
            "startup_failure", None,
        ):
            logger.debug(
                "Skipping run #%d (conclusion=%s)", run.run_number, run.conclusion,
            )
            continue
        logger.info(
            "Found daily run #%d (id=%d, conclusion=%s, created=%s)",
            run.run_number, run.id, run.conclusion, run.created_at,
        )
        return run

    logger.warning("No completed non-cancelled run found for %s/%s", workflow_name, branch)
    return None

def download_all_test_failures(
    gh: Github,
    repo_full_name: str,
    run_id: int,
    github_token: str,
    *,
    artifact_client: ArtifactClient | None = None,
) -> bytes | None:
    """Download the 'all-test-failures' artifact from a workflow run.

    Returns the raw JSON content as bytes, or None if the artifact (or the
    JSON file inside it) is not found. Delegates the listing, download, and
    zip extraction to the shared :class:`ArtifactClient`, which handles the
    auth-stripping redirect, transient-failure retries, expired (404)
    artifacts, and a runaway-extraction cap.
    """
    client = artifact_client or ArtifactClient(gh, token=github_token)

    artifacts = client.list_run_artifacts(
        repo_full_name, run_id, name=_FAILURES_ARTIFACT_NAME,
    )
    matches = [a for a in artifacts if a.name == _FAILURES_ARTIFACT_NAME]
    if not matches:
        logger.info(
            "No %r artifact found in run %d", _FAILURES_ARTIFACT_NAME, run_id
        )
        return None

    # Re-running a workflow leaves one artifact per attempt under the same run
    # and name, each expiring on its own clock. Take the newest live one: a
    # stale earlier attempt must not shadow the re-run's usable artifact.
    live = [a for a in matches if not a.expired]
    if not live:
        logger.warning(
            "All %d %r artifact(s) in run %d have expired",
            len(matches), _FAILURES_ARTIFACT_NAME, run_id,
        )
        return None
    target = max(live, key=lambda a: a.artifact_id)

    logger.info("Downloading artifact: %s (id=%d)", target.name, target.artifact_id)
    files = client.download_artifact(repo_full_name, target.artifact_id)

    content = files.get(_FAILURES_JSON_NAME)
    if content is None:
        logger.warning(
            "Artifact zip for run %d does not contain %s; found: %s",
            run_id, _FAILURES_JSON_NAME, sorted(files),
        )
        return None

    logger.info("Extracted %s from artifact zip", _FAILURES_JSON_NAME)
    return content

def get_run_conclusion(
    gh: Github,
    repo_full_name: str,
    run_id: int,
) -> str | None:
    """A workflow run's conclusion, or None if it cannot be determined.

    Used when the run was named explicitly rather than discovered, so the
    caller can still tell a red run apart from a clean one. Returns None on
    any API failure: the conclusion only sharpens an error message, so it must
    not turn a usable run into a hard failure.
    """
    try:
        repo = retry_github_call(
            lambda: gh.get_repo(repo_full_name),
            retries=3,
            description=f"get repo {repo_full_name}",
        )
        run = retry_github_call(
            lambda: repo.get_workflow_run(run_id),
            retries=3,
            description=f"get run {run_id}",
        )
    except Exception:
        logger.warning(
            "Could not fetch conclusion for run %d", run_id, exc_info=True,
        )
        return None
    return run.conclusion


@dataclass(frozen=True)
class JobInfo:
    """URL map and failed-job names derived from a workflow run's job list."""

    urls: dict[str, str]
    failed: set[str]


def get_job_info(
    gh: Github,
    repo_full_name: str,
    run_id: int,
) -> JobInfo:
    """Fetch job metadata for a workflow run in a single API call.

    Returns a :class:`JobInfo` containing:
    - ``urls``: job name -> HTML URL (includes normalized aliases for fuzzy
      matching against artifact names).
    - ``failed``: names of jobs whose conclusion indicates failure.
    """

    repo = retry_github_call(
        lambda: gh.get_repo(repo_full_name),
        retries=3,
        description=f"get repo {repo_full_name}",
    )

    run = retry_github_call(
        lambda: repo.get_workflow_run(run_id),
        retries=3,
        description=f"get run {run_id}",
    )

    # The list() must happen inside the retried call: jobs() returns a lazy
    # PaginatedList that issues no request until iterated, so retrying only the
    # construction would leave the actual HTTP call unprotected.
    job_list = retry_github_call(
        lambda: list(run.jobs()),
        retries=3,
        description=f"list jobs for run {run_id}",
    )

    job_url_map: dict[str, str] = {job.name: job.html_url for job in job_list}
    failed_jobs: set[str] = set()

    for job in job_list:
        if job.conclusion == "failure":
            failed_jobs.add(job.name)

        normalized = re.sub(r"\s*\(([^)]+)\)", r"-\1", job.name)
        normalized = re.sub(r"\s+", "-", normalized)
        if normalized != job.name and normalized not in job_url_map:
            job_url_map[normalized] = job.html_url

    logger.info(
        "Found %d job URL mappings (%d failed) for run %d",
        len(job_url_map), len(failed_jobs), run_id,
    )
    return JobInfo(urls=job_url_map, failed=failed_jobs)


def get_job_urls(
    gh: Github,
    repo_full_name: str,
    run_id: int,
) -> dict[str, str]:
    """Get a mapping of job name -> HTML URL for all jobs in a workflow run.

    Also includes normalized variants (parentheses replaced with dashes,
    spaces replaced with dashes) for fuzzy matching.
    """
    return get_job_info(gh, repo_full_name, run_id).urls
