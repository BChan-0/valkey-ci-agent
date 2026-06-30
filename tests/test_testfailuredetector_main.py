"""Tests for the test-failure-detector entry point (mocked GitHub + I/O)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# PyGithub requires urllib3 v2 + OpenSSL 1.1.1+. On older dev hosts the import
# fails at collection time. Guard with a skip so the test file is still valid.
try:
    from scripts.test_failure_detector import main as detector_main

    _SKIP_REASON = None
except ImportError as _exc:
    _SKIP_REASON = f"PyGithub import failed: {_exc}"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


class TestRunArtifactJSONGuard:
    """A malformed artifact must be reported, not crash the run."""

    @patch("scripts.test_failure_detector.main.emit_job_summary")
    @patch("scripts.test_failure_detector.main.download_all_test_failures")
    @patch("scripts.test_failure_detector.main.ArtifactClient")
    @patch("scripts.test_failure_detector.main.Github")
    def test_malformed_artifact_returns_nonzero_and_reports(
        self, _mock_gh, _mock_client, mock_download, mock_emit,
    ) -> None:
        # A truncated/invalid artifact body — json.loads would raise.
        mock_download.return_value = b"{not valid json"

        rc = detector_main.run(
            github_token="t", repo_full_name="valkey-io/valkey", run_id=123,
        )

        assert rc == 1
        # The failure is surfaced in the job summary rather than crashing.
        mock_emit.assert_called_once()
        summary = mock_emit.call_args.args[0]
        assert "Could not parse" in summary

    @patch("scripts.test_failure_detector.main.emit_job_summary")
    @patch("scripts.test_failure_detector.main.download_all_test_failures")
    @patch("scripts.test_failure_detector.main.ArtifactClient")
    @patch("scripts.test_failure_detector.main.Github")
    def test_scalar_json_artifact_returns_nonzero_and_reports(
        self, _mock_gh, _mock_client, mock_download, mock_emit,
    ) -> None:
        # A bare scalar: json.loads succeeds, then len() would crash. Reported
        # as malformed rather than propagating a TypeError.
        mock_download.return_value = b"123"

        rc = detector_main.run(
            github_token="t", repo_full_name="valkey-io/valkey", run_id=123,
        )

        assert rc == 1
        mock_emit.assert_called_once()
        assert "unexpected format" in mock_emit.call_args.args[0]

    @patch("scripts.test_failure_detector.main.parse_and_deduplicate")
    @patch("scripts.test_failure_detector.main.get_job_urls")
    @patch("scripts.test_failure_detector.main.emit_job_summary")
    @patch("scripts.test_failure_detector.main.download_all_test_failures")
    @patch("scripts.test_failure_detector.main.ArtifactClient")
    @patch("scripts.test_failure_detector.main.Github")
    def test_list_json_artifact_returns_nonzero_and_reports(
        self, _mock_gh, _mock_client, mock_download, mock_emit,
        mock_job_urls, mock_parse,
    ) -> None:
        # A top-level list parses fine but is the wrong shape: without the guard
        # it slips past parse_and_deduplicate as "no failures" and exits 0.
        mock_download.return_value = b"[1, 2, 3]"

        rc = detector_main.run(
            github_token="t", repo_full_name="valkey-io/valkey", run_id=123,
        )

        assert rc == 1
        # Bailed before parsing the wrong-shaped artifact.
        mock_parse.assert_not_called()
        mock_emit.assert_called_once()
        assert "unexpected format" in mock_emit.call_args.args[0]


class TestRunProcessingErrorsExitCode:
    """Per-failure processing errors must exit non-zero so CI does not stay
    green while issue updates were skipped."""

    @patch("scripts.test_failure_detector.main.process_failures")
    @patch("scripts.test_failure_detector.main.parse_and_deduplicate")
    @patch("scripts.test_failure_detector.main.get_job_urls")
    @patch("scripts.test_failure_detector.main.emit_job_summary")
    @patch("scripts.test_failure_detector.main.download_all_test_failures")
    @patch("scripts.test_failure_detector.main.ArtifactClient")
    @patch("scripts.test_failure_detector.main.Github")
    def test_returns_nonzero_when_process_failures_reports_errors(
        self, _mock_gh, _mock_client, mock_download, mock_emit,
        mock_job_urls, mock_parse, mock_process,
    ) -> None:
        mock_download.return_value = b'{"job": {"suite": []}}'
        mock_parse.return_value = [MagicMock(display_name="t", jobs=[])]
        mock_process.return_value = {
            "created": 1, "updated": 0, "skipped": 0, "errors": 1,
        }

        rc = detector_main.run(
            github_token="t", repo_full_name="valkey-io/valkey", run_id=123,
        )

        assert rc == 1
        # Summary still emitted before exiting non-zero.
        mock_emit.assert_called_once()

    @patch("scripts.test_failure_detector.main.process_failures")
    @patch("scripts.test_failure_detector.main.parse_and_deduplicate")
    @patch("scripts.test_failure_detector.main.get_job_urls")
    @patch("scripts.test_failure_detector.main.emit_job_summary")
    @patch("scripts.test_failure_detector.main.download_all_test_failures")
    @patch("scripts.test_failure_detector.main.ArtifactClient")
    @patch("scripts.test_failure_detector.main.Github")
    def test_returns_zero_when_no_processing_errors(
        self, _mock_gh, _mock_client, mock_download, mock_emit,
        mock_job_urls, mock_parse, mock_process,
    ) -> None:
        mock_download.return_value = b'{"job": {"suite": []}}'
        mock_parse.return_value = [MagicMock(display_name="t", jobs=[])]
        mock_process.return_value = {
            "created": 1, "updated": 1, "skipped": 0, "errors": 0,
        }

        rc = detector_main.run(
            github_token="t", repo_full_name="valkey-io/valkey", run_id=123,
        )

        assert rc == 0
