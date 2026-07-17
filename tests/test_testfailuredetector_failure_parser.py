"""Unit tests for test failure parse/dedup logic."""

from __future__ import annotations

import pytest

from scripts.test_failure_detector.parse_failures import (
    FailureType,
    JobReference,
    UniqueFailure,
    normalize_error_identity,
    parse_and_deduplicate,
)

# --- Fixture data mimicking real all-test-failures.json ---

SAMPLE_ALL_FAILURES = {
    "test-ubuntu-latest": {
        "integration": [
            {
                "test_name": "PSYNC2 test",
                "test_file": "tests/integration/replication-psync.tcl",
                "error": "Expected replica to be in sync within 5000ms",
            },
            {
                "test_name": "Lazy free of stream",
                "test_file": "tests/unit/lazyfree.tcl",
                "error": "assertion:Expected 0 == 1",
            },
        ],
        "sentinel": [
            {
                "test_name": "PSYNC2 test",
                "test_file": "tests/integration/replication-psync.tcl",
                "error": "Expected replica to be in sync within 5000ms",
            },
        ],
    },
    "test-ubuntu-latest-cluster": {
        "integration": [
            {
                "test_name": "PSYNC2 test",
                "test_file": "tests/integration/replication-psync.tcl",
                "error": "Expected replica to be in sync within 5000ms",
            },
            {
                "test_name": "Cluster slot migration",
                "test_file": "tests/unit/cluster.tcl",
                "error": "timeout waiting for cluster to be stable",
            },
        ],
    },
}

SAMPLE_JOB_URLS = {
    "test-ubuntu-latest": "https://github.com/valkey-io/valkey/actions/runs/123/job/456",
    "test-ubuntu-latest-cluster": "https://github.com/valkey-io/valkey/actions/runs/123/job/789",
}


class TestParseAndDeduplicate:
    def test_deduplicates_same_test_across_jobs(self) -> None:
        """Same test failing in multiple jobs should produce one UniqueFailure."""
        results = parse_and_deduplicate(SAMPLE_ALL_FAILURES, SAMPLE_JOB_URLS)

        psync_failures = [f for f in results if f.test_name == "PSYNC2 test"]
        assert len(psync_failures) == 1

        psync = psync_failures[0]
        # Should appear in both jobs (but deduplicated within test-ubuntu-latest)
        job_names = [j.job for j in psync.jobs]
        assert "test-ubuntu-latest" in job_names
        assert "test-ubuntu-latest-cluster" in job_names
        assert len(psync.jobs) == 2

    def test_deduplicates_same_test_across_suites_within_job(self) -> None:
        """Same test in multiple suites of the same job should only record the job once."""
        results = parse_and_deduplicate(SAMPLE_ALL_FAILURES, SAMPLE_JOB_URLS)

        psync_failures = [f for f in results if f.test_name == "PSYNC2 test"]
        assert len(psync_failures) == 1

        psync = psync_failures[0]
        # test-ubuntu-latest appears in both integration and sentinel suites,
        # but should only be recorded once
        ubuntu_refs = [j for j in psync.jobs if j.job == "test-ubuntu-latest"]
        assert len(ubuntu_refs) == 1

    def test_unique_failures_count(self) -> None:
        """Should produce 3 unique failures from the sample data."""
        results = parse_and_deduplicate(SAMPLE_ALL_FAILURES, SAMPLE_JOB_URLS)
        assert len(results) == 3

        names = {f.test_name for f in results}
        assert names == {"PSYNC2 test", "Lazy free of stream", "Cluster slot migration"}

    def test_job_urls_are_attached(self) -> None:
        """Job references should include the URL from job_urls mapping."""
        results = parse_and_deduplicate(SAMPLE_ALL_FAILURES, SAMPLE_JOB_URLS)

        cluster_failures = [f for f in results if f.test_name == "Cluster slot migration"]
        assert len(cluster_failures) == 1
        assert cluster_failures[0].jobs[0].url == SAMPLE_JOB_URLS["test-ubuntu-latest-cluster"]

    def test_missing_job_url_gives_empty_string(self) -> None:
        """If a job name isn't in job_urls, the URL should be empty."""
        results = parse_and_deduplicate(SAMPLE_ALL_FAILURES, {})

        for failure in results:
            for job_ref in failure.jobs:
                assert job_ref.url == ""

    def test_empty_failures_returns_empty_list(self) -> None:
        results = parse_and_deduplicate({}, {})
        assert results == []

    def test_no_failures_in_suites_returns_empty(self) -> None:
        """Jobs with empty failure lists should produce no results."""
        data = {"job-1": {"suite-a": [], "suite-b": []}}
        results = parse_and_deduplicate(data, {})
        assert results == []

    def test_entries_missing_test_name_grouped_by_error(self) -> None:
        """Entries without test_name but with error are kept (grouped by error
        identity). This supports sanitizer/valgrind nameless failures."""
        data = {
            "job-1": {
                "suite": [
                    {"test_file": "foo.tcl", "error": "oops"},  # no test_name
                    {"test_name": "real test", "test_file": "bar.tcl", "error": "err"},
                ]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 2
        named = [f for f in results if f.test_name == "real test"]
        nameless = [f for f in results if not f.test_name]
        assert len(named) == 1
        assert len(nameless) == 1
        assert nameless[0].error == "oops"

    def test_entries_missing_test_name_and_error_are_skipped(self) -> None:
        """Entries with neither test_name nor error cannot be fingerprinted."""
        data = {
            "job-1": {
                "suite": [
                    {"test_file": "foo.tcl", "error": ""},  # no name, no error
                    {"test_name": "real test", "test_file": "bar.tcl", "error": "err"},
                ]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].test_name == "real test"

    def test_entries_missing_test_file_with_error_are_kept(self) -> None:
        """Entries with test_name but no test_file are kept when they have
        a test_name (the name itself provides identity)."""
        data = {
            "job-1": {
                "suite": [
                    {"test_name": "orphan", "error": "oops"},  # no test_file
                ]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].test_name == "orphan"
        assert results[0].test_file == ""

    def test_preserves_error_from_first_occurrence(self) -> None:
        """The error message should come from the first occurrence."""
        data = {
            "job-1": {"suite": [{"test_name": "t", "test_file": "f.tcl", "error": "first error"}]},
            "job-2": {"suite": [{"test_name": "t", "test_file": "f.tcl", "error": "second error"}]},
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].error == "first error"

    def test_display_name(self) -> None:
        f = UniqueFailure(test_name="my test", test_file="tests/foo.tcl")
        assert f.display_name == "my test in tests/foo.tcl"

    def test_separator_in_test_name_does_not_collide(self) -> None:
        """Two distinct (name, file) pairs that would join to the same string
        under a ' in ' separator must stay separate. The grouping key is a
        tuple, so a separator appearing inside a test name can't cause a
        collision."""
        data = {
            "job-1": {
                "suite": [
                    # Under f"{name} in {file}" both collapse to
                    # "foo in bar.tcl in baz.tcl" — but they are different tests.
                    {"test_name": "foo in bar.tcl", "test_file": "baz.tcl", "error": "a"},
                    {"test_name": "foo", "test_file": "bar.tcl in baz.tcl", "error": "b"},
                ]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 2
        assert {(f.test_name, f.test_file) for f in results} == {
            ("foo in bar.tcl", "baz.tcl"),
            ("foo", "bar.tcl in baz.tcl"),
        }


# --- Tests for FailureType and typed parsing ---


class TestNormalizeErrorIdentity:
    """Test that error normalization strips volatile tokens and keeps
    the meaningful error type/location for stable fingerprinting."""

    def test_strips_valgrind_pids(self) -> None:
        error = "==12345== Invalid read of size 4\n==12345==    at 0xABCDEF: someFunc (file.c:123)"
        result = normalize_error_identity(error)
        assert "12345" not in result

    def test_strips_hex_addresses(self) -> None:
        error = "==999== Invalid read of size 4\n==999==    at 0xDEADBEEF: func (file.c:10)"
        result = normalize_error_identity(error)
        assert "DEADBEEF" not in result.upper()
        assert "deadbeef" not in result.lower()

    def test_same_error_different_pids_normalizes_equal(self) -> None:
        e1 = "==100== Invalid read of size 4\n==100==    at 0xAAA: dictResize (dict.c:55)"
        e2 = "==200== Invalid read of size 4\n==200==    at 0xBBB: dictResize (dict.c:55)"
        assert normalize_error_identity(e1) == normalize_error_identity(e2)

    def test_different_errors_normalize_different(self) -> None:
        e1 = "==1== Invalid read of size 4\n==1==    at 0xA: dictResize (dict.c:55)"
        e2 = "==1== Invalid write of size 8\n==1==    at 0xA: hashExpand (hash.c:99)"
        assert normalize_error_identity(e1) != normalize_error_identity(e2)

    def test_strips_temp_paths(self) -> None:
        error = "ERROR: can't open /tmp/valkey-test-abc123/config\nSanitizer error"
        result = normalize_error_identity(error)
        assert "/tmp/" not in result

    def test_strips_ansi_codes(self) -> None:
        error = "\033[31m[  TIMEOUT ]\033[0m test timed out"
        result = normalize_error_identity(error)
        assert "\033" not in result

    def test_sanitizer_output_keeps_error_type(self) -> None:
        error = (
            "==555== ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1234\n"
            "==555==    at 0xABC: serverCron (server.c:300)\n"
            "==555== SUMMARY: AddressSanitizer: heap-buffer-overflow server.c:300 in serverCron"
        )
        result = normalize_error_identity(error)
        assert "heap-buffer-overflow" in result

    def test_empty_error_returns_empty(self) -> None:
        assert normalize_error_identity("") == ""

    def test_strips_port_and_pid_annotations(self) -> None:
        error = "Error connecting on port=6379 pid 12345\nInvalid operation"
        result = normalize_error_identity(error)
        assert "6379" not in result
        assert "12345" not in result


class TestParseWithTypes:
    """Test the extended parsing that handles the 'type' field."""

    def test_backward_compat_no_type_field_defaults_to_assertion(self) -> None:
        data = {"job-1": {"suite": [{"test_name": "t", "test_file": "f.tcl", "error": "err"}]}}
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].failure_type == FailureType.ASSERTION

    def test_assertion_type_explicit(self) -> None:
        data = {"job-1": {"suite": [{"test_name": "t", "test_file": "f.tcl", "type": "assertion", "error": "err"}]}}
        results = parse_and_deduplicate(data, {})
        assert results[0].failure_type == FailureType.ASSERTION

    def test_sanitizer_entry_without_test_name(self) -> None:
        data = {
            "test-sanitizer-address-gcc": {
                "valkey": [{
                    "test_name": "",
                    "test_file": "tests/unit/expire.tcl",
                    "type": "sanitizer",
                    "error": "Sanitizer error: ==123== ERROR: AddressSanitizer: heap-buffer-overflow"
                }]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].failure_type == FailureType.SANITIZER
        assert results[0].test_file == "tests/unit/expire.tcl"
        assert not results[0].has_test_identity

    def test_same_sanitizer_error_across_jobs_deduplicates(self) -> None:
        """Same bug with different PIDs/addresses from different jobs collapses."""
        data = {
            "test-sanitizer-address-gcc": {
                "valkey": [{
                    "test_name": "",
                    "test_file": "tests/unit/expire.tcl",
                    "type": "sanitizer",
                    "error": "==111== ERROR: AddressSanitizer: heap-buffer-overflow\n==111==    at 0xAAA: dictResize (dict.c:100)"
                }]
            },
            "test-sanitizer-address-clang": {
                "valkey": [{
                    "test_name": "",
                    "test_file": "tests/unit/expire.tcl",
                    "type": "sanitizer",
                    "error": "==222== ERROR: AddressSanitizer: heap-buffer-overflow\n==222==    at 0xBBB: dictResize (dict.c:100)"
                }]
            },
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert len(results[0].jobs) == 2

    def test_same_valgrind_error_different_test_files_deduplicates(self) -> None:
        """Same leak detected after different test files produces one failure."""
        data = {
            "test-valgrind-test": {
                "valkey": [{
                    "test_name": "",
                    "test_file": "tests/unit/expire.tcl",
                    "type": "valgrind",
                    "error": "==1== Invalid read of size 4\n==1==    at 0xA: dictResize (dict.c:100)"
                }]
            },
            "test-valgrind-misc": {
                "valkey": [{
                    "test_name": "",
                    "test_file": "tests/unit/cluster.tcl",
                    "type": "valgrind",
                    "error": "==2== Invalid read of size 4\n==2==    at 0xB: dictResize (dict.c:100)"
                }]
            },
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert len(results[0].jobs) == 2

    def test_different_valgrind_errors_stay_separate(self) -> None:
        """Different bugs produce separate failures even from the same job."""
        data = {
            "test-valgrind-test": {
                "valkey": [
                    {
                        "test_name": "",
                        "test_file": "tests/unit/expire.tcl",
                        "type": "valgrind",
                        "error": "==1== Invalid read of size 4\n==1==    at 0xA: dictResize (dict.c:100)"
                    },
                    {
                        "test_name": "",
                        "test_file": "tests/unit/expire.tcl",
                        "type": "valgrind",
                        "error": "==1== Invalid write of size 8\n==1==    at 0xA: hashExpand (hash.c:200)"
                    },
                ]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 2

    def test_timeout_with_test_name(self) -> None:
        data = {
            "test-ubuntu-jemalloc": {
                "valkey": [{
                    "test_name": "PSYNC2 partial sync",
                    "test_file": "tests/integration/replication-psync.tcl",
                    "type": "timeout",
                    "error": "Test timed out"
                }]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].failure_type == FailureType.TIMEOUT
        assert results[0].test_name == "PSYNC2 partial sync"
        assert results[0].has_test_identity

    def test_unittest_failure(self) -> None:
        data = {
            "test-ubuntu-jemalloc": {
                "unittest": [{
                    "test_name": "DictTest.BasicOperations",
                    "test_file": "src/unit/valkey-unit-gtests",
                    "type": "unittest",
                    "error": "gtest FAIL"
                }]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].failure_type == FailureType.UNITTEST
        assert results[0].test_name == "DictTest.BasicOperations"

    def test_startup_failure(self) -> None:
        data = {
            "test-ubuntu-jemalloc": {
                "valkey": [{
                    "test_name": "",
                    "test_file": "tests/unit/cluster.tcl",
                    "type": "startup",
                    "error": "Can't start /path/to/valkey-server\nCONFIGURATION:\n...\nERROR:\nFailed listening on port 6379"
                }]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].failure_type == FailureType.STARTUP
        assert not results[0].has_test_identity

    def test_exception_failure(self) -> None:
        data = {
            "test-ubuntu-jemalloc": {
                "valkey": [{
                    "test_name": "",
                    "test_file": "",
                    "type": "exception",
                    "error": "can't read \"fd\": no such variable"
                }]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 1
        assert results[0].failure_type == FailureType.EXCEPTION

    def test_unknown_type_defaults_to_assertion(self) -> None:
        data = {"job": {"s": [{"test_name": "t", "test_file": "f.tcl", "type": "bogus", "error": "x"}]}}
        results = parse_and_deduplicate(data, {})
        assert results[0].failure_type == FailureType.ASSERTION

    def test_entry_with_no_test_name_and_no_error_is_skipped(self) -> None:
        data = {"job": {"s": [{"test_name": "", "test_file": "f.tcl", "type": "sanitizer", "error": ""}]}}
        results = parse_and_deduplicate(data, {})
        assert results == []

    def test_display_name_for_nameless_failure(self) -> None:
        f = UniqueFailure(test_name="", test_file="tests/unit/expire.tcl", failure_type=FailureType.SANITIZER)
        assert "[sanitizer]" in f.display_name
        assert "expire.tcl" in f.display_name

    def test_display_name_for_nameless_failure_no_file(self) -> None:
        f = UniqueFailure(test_name="", test_file="", failure_type=FailureType.EXCEPTION)
        assert "[exception]" in f.display_name
        assert "unknown" in f.display_name

    def test_mixed_types_in_single_run(self) -> None:
        """A run can produce failures of multiple types from the same job."""
        data = {
            "test-valgrind-test": {
                "valkey": [
                    {"test_name": "PSYNC2 test", "test_file": "tests/integration/replication-psync.tcl", "type": "assertion", "error": "Expected sync"},
                    {"test_name": "", "test_file": "tests/integration/replication-psync.tcl", "type": "valgrind", "error": "==1== Invalid read\n==1==    at 0xA: func (x.c:1)"},
                    {"test_name": "PSYNC2 test", "test_file": "tests/integration/replication-psync.tcl", "type": "timeout", "error": "Test timed out"},
                ]
            }
        }
        results = parse_and_deduplicate(data, {})
        assert len(results) == 3
        types = {f.failure_type for f in results}
        assert types == {FailureType.ASSERTION, FailureType.VALGRIND, FailureType.TIMEOUT}
