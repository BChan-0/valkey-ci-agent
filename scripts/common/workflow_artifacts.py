"""GitHub Actions workflow run and artifact retrieval.

Lists recent runs of a target workflow file and downloads their uploaded
artifact bundles into an in-memory ``{path: bytes}`` map.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.common.github_client import (
    RETRYABLE_HTTP_STATUS,
    retry_github_call,
    transient_backoff_delay,
)

if TYPE_CHECKING:
    from github import Github

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowArtifact:
    artifact_id: int
    name: str
    size_in_bytes: int
    expired: bool


class ArtifactClient:
    """Fetches workflow artifacts and logs from GitHub Actions."""

    def __init__(self, github_client: Github, *, token: str, retries: int = 3) -> None:
        if not token:
            raise ValueError("GitHub token is required")
        self._gh = github_client
        self._token = token
        self._retries = retries

    def list_recent_runs(
        self, repo_full_name: str, workflow_file: str,
        *, event: str = "schedule", max_runs: int = 1,
    ) -> list[Any]:
        def _fetch() -> list[Any]:
            repo = self._gh.get_repo(repo_full_name)
            workflow = repo.get_workflow(workflow_file)
            return list(islice(workflow.get_runs(event=event, status="completed"), max_runs))

        return retry_github_call(
            _fetch, retries=self._retries, description=f"list runs {workflow_file}",
        )

    def list_run_artifacts(self, repo_full_name: str, run_id: int) -> list[WorkflowArtifact]:
        repo = self._gh.get_repo(repo_full_name)

        def _fetch() -> Any:
            _, data = repo._requester.requestJsonAndCheck(
                "GET", f"/repos/{repo_full_name}/actions/runs/{run_id}/artifacts",
            )
            return data

        payload = retry_github_call(_fetch, retries=self._retries,
                                    description=f"list artifacts {run_id}")
        if not isinstance(payload, dict):
            return []
        return [
            WorkflowArtifact(
                artifact_id=a["id"], name=a["name"],
                size_in_bytes=a.get("size_in_bytes", 0),
                expired=a.get("expired", False),
            )
            for a in payload.get("artifacts", [])
            if isinstance(a, dict)
            and isinstance(a.get("id"), int)
            and isinstance(a.get("name"), str)
        ]

    def download_artifact(self, repo_full_name: str, artifact_id: int) -> dict[str, bytes]:
        return _extract_zip(self._download(
            f"/repos/{repo_full_name}/actions/artifacts/{artifact_id}/zip"
        ))

    def download_run_logs(self, repo_full_name: str, run_id: int) -> dict[str, bytes]:
        """Download a workflow run's console logs as a ``{path: bytes}`` map.

        GitHub returns the whole run's logs as a zip whose members are the
        per-step text logs (one file per job step, plus per-job rollups).
        Unlike ``download_artifact`` this is the raw CI console output, not a
        user-uploaded artifact bundle. Shares the same token-redirect, retry,
        and uncompressed-size-cap discipline as artifact downloads. Returns an
        empty map if the logs have expired (404) or the zip is unreadable.
        """
        return _extract_zip(self._download(
            f"/repos/{repo_full_name}/actions/runs/{run_id}/logs"
        ))

    def download_job_log(self, repo_full_name: str, job_id: int) -> bytes:
        """Download one job's console log, or ``b""`` when it has expired.

        A run's log zip holds every job and runs to tens of megabytes; a caller
        that already knows which job failed fetches only that job's log here.
        Unlike the run-log endpoint this returns plain text, not a zip. Shares
        the token-redirect and retry discipline of the other downloads. The read
        is capped: this path does not go through ``_extract_zip``, which is where
        the other downloads enforce their size limit, so a pathological log
        cannot pull an unbounded body into memory.
        """
        return self._download(
            f"/repos/{repo_full_name}/actions/jobs/{job_id}/logs",
            max_bytes=_MAX_JOB_LOG_BYTES,
        )

    def _download(self, path: str, *, max_bytes: int | None = None) -> bytes:
        url = f"https://api.github.com{path}"
        req = Request(url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "valkey-ci-agent",
        })
        # Use an unredirected header for the token: urllib forwards normal
        # headers on cross-host redirects, but GitHub redirects to signed S3
        # URLs that must not receive our token. add_unredirected_header keeps
        # the Authorization off the redirected request.
        req.add_unredirected_header("Authorization", f"Bearer {self._token}")
        # Hand-rolled retry rather than retry_github_call: this is a raw urllib
        # call (not a PyGithub operation) and needs HTTP-status-specific
        # handling for the 404/expired case below. Retry classification and
        # backoff are shared with retry_github_call so behavior stays uniform.
        for attempt in range(self._retries + 1):
            try:
                with urlopen(req, timeout=120) as resp:
                    # Read one byte past the cap so an oversized body is detected
                    # rather than silently returned truncated to exactly the cap.
                    if max_bytes is None:
                        return resp.read()
                    data = resp.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        logger.warning(
                            "Response at %s exceeds %d bytes; truncating", path, max_bytes,
                        )
                        return data[:max_bytes]
                    return data
            except HTTPError as exc:
                if exc.code == 404:
                    logger.warning("Log or artifact not found at %s (likely expired)", path)
                    return b""
                if exc.code in RETRYABLE_HTTP_STATUS and attempt < self._retries:
                    time.sleep(transient_backoff_delay(attempt))
                    continue
                raise
            except (URLError, TimeoutError, ConnectionError):
                if attempt < self._retries:
                    time.sleep(transient_backoff_delay(attempt))
                    continue
                raise
        raise AssertionError("unreachable: retry loop must return or raise")


# Defends against a runaway log/artifact dump that would exhaust the runner.
# Real fuzzer artifacts and CI run logs are typically well under this.
_MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024

# Cap for a single job's console log, which the excerpt carver reads whole into
# memory. A real Valkey job log is around a megabyte; this leaves generous room
# while bounding a pathological one.
_MAX_JOB_LOG_BYTES = 50 * 1024 * 1024


def _extract_zip(blob: bytes) -> dict[str, bytes]:
    if not blob:
        return {}
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            total = sum(m.file_size for m in members)
            if total > _MAX_UNCOMPRESSED_BYTES:
                logger.warning(
                    "Artifact uncompressed size %d exceeds cap %d; refusing to extract",
                    total, _MAX_UNCOMPRESSED_BYTES,
                )
                return {}
            return {m.filename: zf.read(m) for m in members}
    except zipfile.BadZipFile:
        logger.warning("Artifact zip is corrupt; returning empty")
        return {}
