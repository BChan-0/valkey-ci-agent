"""Unit tests for timeout recovery from CI logs."""

from __future__ import annotations

from scripts.test_failure_detector.main import _merge_timeout_recoveries
from scripts.test_failure_detector.parse_failures import (
    FailureType,
    JobReference,
    UniqueFailure,
)
from scripts.test_failure_detector.timeout_parser import (
    find_job_log,
    jobs_needing_log_scan,
    parse_timeouts_from_log,
)

# --- jobs_needing_log_scan ---

class TestJobsNeedingLogScan:
    def test_failed_job_with_empty_artifact_needs_scan(self) -> None:
        all_failures = {
            "test-ubuntu-jemalloc": {"valkey": [], "sentinel": []},
        }
        failed = {"test-ubuntu-jemalloc"}
        assert jobs_needing_log_scan(all_failures, failed) == {"test-ubuntu-jemalloc"}

    def test_failed_job_with_captured_failures_skipped(self) -> None:
        all_failures = {
            "test-ubuntu-jemalloc": {
                "valkey": [{"test_name": "t", "test_file": "f.tcl", "error": "e"}],
                "sentinel": [],
            },
        }
        failed = {"test-ubuntu-jemalloc"}
        assert jobs_needing_log_scan(all_failures, failed) == set()

    def test_failed_job_not_in_artifact_needs_scan(self) -> None:
        all_failures = {"test-ubuntu-jemalloc": {"valkey": [], "sentinel": []}}
        failed = {"test-valgrind-test"}
        assert jobs_needing_log_scan(all_failures, failed) == {"test-valgrind-test"}

    def test_successful_job_never_needs_scan(self) -> None:
        all_failures = {"test-ubuntu-jemalloc": {"valkey": []}}
        failed: set[str] = set()
        assert jobs_needing_log_scan(all_failures, failed) == set()

    def test_multiple_failed_jobs_mixed(self) -> None:
        all_failures = {
            "job-a": {"valkey": [{"test_name": "t", "test_file": "f.tcl", "error": ""}]},
            "job-b": {"valkey": [], "sentinel": []},
        }
        failed = {"job-a", "job-b", "job-c"}
        result = jobs_needing_log_scan(all_failures, failed)
        assert result == {"job-b", "job-c"}


# --- parse_timeouts_from_log ---

# Simulated CI log output for a timeout event
SAMPLE_TIMEOUT_LOG = b"""\
2026-07-05T02:33:45.2195107Z ./runtest --valgrind --failures-output test-failures/valkey.json --verbose --clients 1 --timeout 2400
2026-07-05T04:13:45.0000000Z [ok]: Some passing test (1234 ms)
2026-07-05T04:13:50.0000000Z [TIMEOUT]: clients state report follows.
2026-07-05T04:13:50.0000000Z 5 => (IN PROGRESS) PSYNC2 test (pid 12345)
2026-07-05T04:13:50.0000000Z [TIMEOUT]: PSYNC2 test in tests/integration/replication-psync.tcl
2026-07-05T04:13:50.0000000Z 7 => (IN PROGRESS) Cluster slot migration (pid 12346)
2026-07-05T04:13:50.0000000Z [TIMEOUT]: Cluster slot migration in tests/unit/cluster.tcl
2026-07-05T04:13:55.0000000Z === Server log (pid 12345): ./tests/tmp/server.6510.2310/stdout ===
2026-07-05T04:14:00.0000000Z                    The End
2026-07-05T04:14:00.0000000Z !!! WARNING The following tests failed:
2026-07-05T04:14:00.0000000Z *** [TIMEOUT]: PSYNC2 test in tests/integration/replication-psync.tcl
2026-07-05T04:14:00.0000000Z *** [TIMEOUT]: Cluster slot migration in tests/unit/cluster.tcl
"""


class TestParseTimeoutsFromLog:
    def test_extracts_timeout_failures(self) -> None:
        results = parse_timeouts_from_log(
            SAMPLE_TIMEOUT_LOG, "test-valgrind-test", job_url="http://example.com/job/1",
        )
        assert len(results) == 2
        names = {f.test_name for f in results}
        assert names == {"PSYNC2 test", "Cluster slot migration"}

    def test_deduplicates_repeated_timeout_lines(self) -> None:
        results = parse_timeouts_from_log(SAMPLE_TIMEOUT_LOG, "job")
        # Each test appears twice in the sample (inline + summary), but only one UniqueFailure each
        assert len(results) == 2

    def test_populates_job_reference(self) -> None:
        results = parse_timeouts_from_log(
            SAMPLE_TIMEOUT_LOG, "test-valgrind-test", job_url="http://ci/job/42",
        )
        for f in results:
            assert len(f.jobs) == 1
            assert f.jobs[0].job == "test-valgrind-test"
            assert f.jobs[0].suite == "timeout"
            assert f.jobs[0].url == "http://ci/job/42"

    def test_sets_timeout_failure_type(self) -> None:
        results = parse_timeouts_from_log(SAMPLE_TIMEOUT_LOG, "job")
        for f in results:
            assert f.failure_type == FailureType.TIMEOUT

    def test_sets_timeout_error_message(self) -> None:
        results = parse_timeouts_from_log(SAMPLE_TIMEOUT_LOG, "job")
        for f in results:
            assert "timed out" in f.error.lower()

    def test_no_timeout_returns_empty(self) -> None:
        log = b"2026-07-05T02:33:45Z [ok]: Some test (100 ms)\nThe End\n"
        results = parse_timeouts_from_log(log, "job")
        assert results == []

    def test_handles_ansi_color_codes(self) -> None:
        log = (
            b"2026-07-05T04:13:50Z [\x1b[31mTIMEOUT\x1b[0m]: "
            b"My test in tests/unit/foo.tcl\n"
        )
        results = parse_timeouts_from_log(log, "job")
        assert len(results) == 1
        assert results[0].test_name == "My test"
        assert results[0].test_file == "tests/unit/foo.tcl"

    def test_empty_log_returns_empty(self) -> None:
        assert parse_timeouts_from_log(b"", "job") == []

    def test_binary_garbage_handled_gracefully(self) -> None:
        results = parse_timeouts_from_log(b"\x00\xff\xfe" * 100, "job")
        assert results == []


# --- find_job_log ---

class TestFindJobLog:
    def test_matches_numbered_prefix(self) -> None:
        logs = {
            "3_test-valgrind-test.txt": b"log content",
            "test-valgrind-test/system.txt": b"system",
        }
        assert find_job_log(logs, "test-valgrind-test") == b"log content"

    def test_matches_with_parentheses(self) -> None:
        logs = {
            "14_test-sanitizer-address (gcc).txt": b"gcc log",
            "test-sanitizer-address (gcc)/system.txt": b"system",
        }
        assert find_job_log(logs, "test-sanitizer-address (gcc)") == b"gcc log"

    def test_skips_subdirectory_files(self) -> None:
        logs = {
            "test-ubuntu-jemalloc/system.txt": b"system info",
            "5_test-ubuntu-jemalloc.txt": b"real log",
        }
        assert find_job_log(logs, "test-ubuntu-jemalloc") == b"real log"

    def test_returns_none_when_not_found(self) -> None:
        logs = {"3_other-job.txt": b"other"}
        assert find_job_log(logs, "test-missing-job") is None

    def test_case_insensitive_fallback(self) -> None:
        logs = {"3_Test-Ubuntu-Jemalloc.txt": b"content"}
        assert find_job_log(logs, "test-ubuntu-jemalloc") == b"content"


# --- _merge_timeout_recoveries ---


class TestMergeTimeoutRecoveries:
    def test_adds_new_timeout_to_list(self) -> None:
        existing = [
            UniqueFailure(
                test_name="assertion test",
                test_file="tests/unit/foo.tcl",
                failure_type=FailureType.ASSERTION,
                error="oops",
                jobs=[JobReference(job="job-a", suite="valkey", url="u")],
            )
        ]
        recovered = [
            UniqueFailure(
                test_name="PSYNC2 test",
                test_file="tests/integration/replication-psync.tcl",
                failure_type=FailureType.TIMEOUT,
                error="Test timed out",
                jobs=[JobReference(job="job-b", suite="timeout", url="u2")],
            )
        ]
        result = _merge_timeout_recoveries(existing, recovered)
        assert len(result) == 2
        timeout = [f for f in result if f.failure_type == FailureType.TIMEOUT]
        assert len(timeout) == 1
        assert timeout[0].test_name == "PSYNC2 test"

    def test_folds_job_into_existing_timeout(self) -> None:
        """If the same timeout already came from the artifact, fold the new
        job reference into it instead of creating a duplicate."""
        existing = [
            UniqueFailure(
                test_name="PSYNC2 test",
                test_file="tests/integration/replication-psync.tcl",
                failure_type=FailureType.TIMEOUT,
                error="Test timed out",
                jobs=[JobReference(job="job-a", suite="valkey", url="u1")],
            )
        ]
        recovered = [
            UniqueFailure(
                test_name="PSYNC2 test",
                test_file="tests/integration/replication-psync.tcl",
                failure_type=FailureType.TIMEOUT,
                error="Test timed out (no progress for the configured timeout period)",
                jobs=[JobReference(job="job-b", suite="timeout", url="u2")],
            )
        ]
        result = _merge_timeout_recoveries(existing, recovered)
        assert len(result) == 1
        assert len(result[0].jobs) == 2
        job_names = {j.job for j in result[0].jobs}
        assert job_names == {"job-a", "job-b"}

    def test_does_not_duplicate_same_job(self) -> None:
        """If the recovered job is already recorded, don't add it again."""
        existing = [
            UniqueFailure(
                test_name="PSYNC2 test",
                test_file="tests/integration/replication-psync.tcl",
                failure_type=FailureType.TIMEOUT,
                error="Test timed out",
                jobs=[JobReference(job="job-a", suite="valkey", url="u1")],
            )
        ]
        recovered = [
            UniqueFailure(
                test_name="PSYNC2 test",
                test_file="tests/integration/replication-psync.tcl",
                failure_type=FailureType.TIMEOUT,
                error="Test timed out",
                jobs=[JobReference(job="job-a", suite="timeout", url="u1")],
            )
        ]
        result = _merge_timeout_recoveries(existing, recovered)
        assert len(result) == 1
        assert len(result[0].jobs) == 1

    def test_empty_recoveries_returns_original(self) -> None:
        existing = [
            UniqueFailure(
                test_name="test", test_file="f.tcl",
                failure_type=FailureType.ASSERTION, error="e",
            )
        ]
        result = _merge_timeout_recoveries(existing, [])
        assert result is existing
