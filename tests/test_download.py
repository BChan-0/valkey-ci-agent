"""Tests for artifact download logic (mocked GitHub API)."""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import MagicMock, patch

import pytest

try:
    from scripts.test_failure_detector.download import (
        download_all_test_failures,
        get_job_urls,
        get_latest_daily_run,
    )

    _SKIP_REASON = None
except ImportError as _exc:
    _SKIP_REASON = f"Import failed: {_exc}"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def _make_mock_run(run_number: int, run_id: int, conclusion: str, status: str = "completed"):
    run = MagicMock()
    run.run_number = run_number
    run.id = run_id
    run.conclusion = conclusion
    run.status = status
    run.created_at = "2026-06-01 00:00:00+00:00"
    return run


def _make_artifact_zip(content: dict) -> bytes:
    """Create a zip file in memory containing all-test-failures.json."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("all-test-failures.json", json.dumps(content))
    return buf.getvalue()


class TestGetLatestDailyRun:
    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_skips_cancelled_runs(self, mock_retry) -> None:
        """Cancelled runs should be skipped."""
        cancelled_run = _make_mock_run(10, 100, "cancelled")
        success_run = _make_mock_run(9, 99, "success")

        mock_workflow = MagicMock()
        mock_workflow.name = "Daily"
        mock_workflow.get_runs.return_value = [cancelled_run, success_run]

        mock_repo = MagicMock()
        mock_repo.get_workflows.return_value = [mock_workflow]

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = get_latest_daily_run(mock_gh, "owner/repo")
        assert result == success_run

    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_skips_skipped_runs(self, mock_retry) -> None:
        """Skipped runs should be skipped."""
        skipped_run = _make_mock_run(13, 200, "skipped")
        failure_run = _make_mock_run(12, 199, "failure")

        mock_workflow = MagicMock()
        mock_workflow.name = "Daily"
        mock_workflow.get_runs.return_value = [skipped_run, failure_run]

        mock_repo = MagicMock()
        mock_repo.get_workflows.return_value = [mock_workflow]

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = get_latest_daily_run(mock_gh, "owner/repo")
        assert result == failure_run

    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_returns_first_success_or_failure(self, mock_retry) -> None:
        """Should return the most recent run with conclusion success or failure."""
        runs = [
            _make_mock_run(15, 300, "skipped"),
            _make_mock_run(14, 299, "cancelled"),
            _make_mock_run(13, 298, "success"),
            _make_mock_run(12, 297, "failure"),
        ]

        mock_workflow = MagicMock()
        mock_workflow.name = "Daily"
        mock_workflow.get_runs.return_value = runs

        mock_repo = MagicMock()
        mock_repo.get_workflows.return_value = [mock_workflow]

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = get_latest_daily_run(mock_gh, "owner/repo")
        assert result.id == 298
        assert result.conclusion == "success"

    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_returns_none_when_no_qualifying_run(self, mock_retry) -> None:
        """Should return None if all runs are cancelled/skipped."""
        runs = [
            _make_mock_run(10, 100, "cancelled"),
            _make_mock_run(9, 99, "skipped"),
        ]

        mock_workflow = MagicMock()
        mock_workflow.name = "Daily"
        mock_workflow.get_runs.return_value = runs

        mock_repo = MagicMock()
        mock_repo.get_workflows.return_value = [mock_workflow]

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = get_latest_daily_run(mock_gh, "owner/repo")
        assert result is None

    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_returns_none_when_workflow_not_found(self, mock_retry) -> None:
        """Should return None if the workflow doesn't exist."""
        mock_repo = MagicMock()
        mock_repo.get_workflows.return_value = []

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = get_latest_daily_run(mock_gh, "owner/repo")
        assert result is None


class TestDownloadAllTestFailures:
    @patch("scripts.test_failure_detector.download._download_artifact")
    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_downloads_and_extracts_json(self, mock_retry, mock_dl) -> None:
        """Should download the zip and extract the JSON content."""
        failures_data = {"job-1": {"suite": [{"test_name": "t", "test_file": "f.tcl", "error": "e"}]}}
        mock_dl.return_value = _make_artifact_zip(failures_data)

        mock_artifact = MagicMock()
        mock_artifact.name = "all-test-failures"
        mock_artifact.id = 555
        mock_artifact.archive_download_url = "https://api.github.com/artifacts/555/zip"

        mock_run = MagicMock()
        mock_run.get_artifacts.return_value = [mock_artifact]

        mock_repo = MagicMock()
        mock_repo.get_workflow_run.return_value = mock_run

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = download_all_test_failures(mock_gh, "owner/repo", 123, "fake-token")
        assert result is not None
        assert json.loads(result) == failures_data

    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_returns_none_when_no_artifact(self, mock_retry) -> None:
        """Should return None if no all-test-failures artifact exists."""
        mock_artifact = MagicMock()
        mock_artifact.name = "some-other-artifact"

        mock_run = MagicMock()
        mock_run.get_artifacts.return_value = [mock_artifact]

        mock_repo = MagicMock()
        mock_repo.get_workflow_run.return_value = mock_run

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = download_all_test_failures(mock_gh, "owner/repo", 123, "fake-token")
        assert result is None

    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_returns_none_when_no_artifacts_at_all(self, mock_retry) -> None:
        """Should return None if the run has no artifacts."""
        mock_run = MagicMock()
        mock_run.get_artifacts.return_value = []

        mock_repo = MagicMock()
        mock_repo.get_workflow_run.return_value = mock_run

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = download_all_test_failures(mock_gh, "owner/repo", 123, "fake-token")
        assert result is None


class TestGetJobUrls:
    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_maps_job_names_to_urls(self, mock_retry) -> None:
        """Should return a mapping of job name to HTML URL."""
        job1 = MagicMock()
        job1.name = "test-ubuntu-latest"
        job1.html_url = "https://github.com/owner/repo/actions/runs/1/job/10"

        job2 = MagicMock()
        job2.name = "test-arm64"
        job2.html_url = "https://github.com/owner/repo/actions/runs/1/job/20"

        mock_run = MagicMock()
        mock_run.jobs.return_value = [job1, job2]

        mock_repo = MagicMock()
        mock_repo.get_workflow_run.return_value = mock_run

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = get_job_urls(mock_gh, "owner/repo", 123)
        assert result["test-ubuntu-latest"] == job1.html_url
        assert result["test-arm64"] == job2.html_url

    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_includes_normalized_names(self, mock_retry) -> None:
        """Job names with parens/spaces should also be stored in normalized form."""
        job = MagicMock()
        job.name = "test ubuntu (arm64)"
        job.html_url = "https://example.com/job/1"

        mock_run = MagicMock()
        mock_run.jobs.return_value = [job]

        mock_repo = MagicMock()
        mock_repo.get_workflow_run.return_value = mock_run

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = get_job_urls(mock_gh, "owner/repo", 123)
        assert result["test ubuntu (arm64)"] == job.html_url
        assert result["test-ubuntu-arm64"] == job.html_url

    @patch("scripts.test_failure_detector.download.retry_github_call")
    def test_empty_jobs_returns_empty_dict(self, mock_retry) -> None:
        mock_run = MagicMock()
        mock_run.jobs.return_value = []

        mock_repo = MagicMock()
        mock_repo.get_workflow_run.return_value = mock_run

        mock_retry.side_effect = lambda op, **kwargs: op()

        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo

        result = get_job_urls(mock_gh, "owner/repo", 123)
        assert result == {}
        