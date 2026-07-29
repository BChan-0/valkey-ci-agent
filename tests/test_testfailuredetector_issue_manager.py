"""Tests for test-failure issue creation/update (mocked GitHub API)."""

from __future__ import annotations

import re
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

# PyGithub requires urllib3 v2 + OpenSSL 1.1.1+. On older dev hosts the import
# fails at collection time. Guard with a skip so the test file is still valid.
try:
    from scripts.test_failure_detector.issue_renderer import (
        MARKER_NAMESPACE,
        _build_body,
        _build_title,
        _extract_environments_from_body,
        _extract_error_from_body,
        _update_environments_in_body,
        fingerprint_for,
        label_for,
        marker_namespace_for,
        renderer_for,
        title_for,
    )
    from scripts.test_failure_detector.manage_issues import (
        CLOSED_ISSUE_LOOKBACK,
        _merge_same_fingerprint_failures,
        process_failures,
    )
    from scripts.test_failure_detector.parse_failures import (
        FailureType,
        JobReference,
        UniqueFailure,
        normalize_error_identity,
    )

    _SKIP_REASON = None
except ImportError as _exc:
    _SKIP_REASON = f"PyGithub import failed: {_exc}"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# --- Helper fixtures ---


def _make_failure(
    test_name: str = "PSYNC2 test",
    test_file: str = "tests/integration/replication-psync.tcl",
    error: str = "Expected replica to be in sync",
    jobs: list[tuple[str, str, str]] | None = None,
) -> UniqueFailure:
    if jobs is None:
        jobs = [("test-ubuntu-latest", "integration", "https://example.com/job/1")]
    return UniqueFailure(
        test_name=test_name,
        test_file=test_file,
        error=error,
        jobs=[JobReference(job=j, suite=s, url=u) for j, s, u in jobs],
    )


# --- Unit tests for the renderer ---


class TestBuildIssueTitle:
    def test_format(self) -> None:
        title = _build_title(_make_failure())
        assert title == "[TEST-FAILURE] PSYNC2 test in tests/integration/replication-psync.tcl"


class TestFingerprint:
    def test_is_stable_hex_token(self) -> None:
        """Hashed, not raw: a fixed-shape lowercase-hex token safe to embed in
        an HTML comment marker and a search query."""
        fp = fingerprint_for(_make_failure())
        assert re.fullmatch(r"[0-9a-f]{20}", fp)

    def test_deterministic(self) -> None:
        assert fingerprint_for(_make_failure()) == fingerprint_for(_make_failure())

    def test_distinguishes_name_and_file(self) -> None:
        base = fingerprint_for(_make_failure())
        assert fingerprint_for(_make_failure(test_name="other")) != base
        assert fingerprint_for(_make_failure(test_file="other.tcl")) != base

    def test_digits_are_significant(self) -> None:
        """PSYNC2 vs PSYNC3 must not collapse; the identity is not normalized."""
        assert (
            fingerprint_for(_make_failure(test_name="PSYNC2"))
            != fingerprint_for(_make_failure(test_name="PSYNC3"))
        )

    def test_unsafe_characters_do_not_leak(self) -> None:
        """Quotes, newlines, and comment-breaking text are hashed away, so the
        marker/query embedding can't be broken by hostile test names."""
        fp = fingerprint_for(_make_failure(
            test_name='evil "--> <!-- ' + "\n" + 'x', test_file="a\"b\nc",
        ))
        assert re.fullmatch(r"[0-9a-f]{20}", fp)


class TestBuildIssueBody:
    def _body(self, failure: UniqueFailure) -> str:
        return _build_body(failure, marker="<!-- m -->", occurrences=1)

    def test_contains_marker_and_occurrences(self) -> None:
        body = self._body(_make_failure())
        assert "<!-- m -->" in body
        assert f"<!-- {MARKER_NAMESPACE}:occurrences:1 -->" in body

    def test_contains_test_name(self) -> None:
        assert "`PSYNC2 test`" in self._body(_make_failure())

    def test_contains_test_file(self) -> None:
        assert "`tests/integration/replication-psync.tcl`" in self._body(_make_failure())

    def test_contains_error_trace(self) -> None:
        assert "assertion failed at line 42" in self._body(
            _make_failure(error="assertion failed at line 42")
        )

    def test_contains_environments_and_links(self) -> None:
        body = self._body(_make_failure(jobs=[
            ("job-a", "suite", "https://example.com/run"),
            ("job-b", "suite", "url2"),
        ]))
        assert "`job-a`" in body
        assert "`job-b`" in body
        assert "[CI link](https://example.com/run)" in body

    def test_contains_auto_created_footer(self) -> None:
        assert "Auto-created by Test Failure Detector" in self._body(_make_failure())


class TestExtractEnvironments:
    def test_extracts_backtick_envs(self) -> None:
        body = "**Environments:** `job-a`, `job-b`, `job-c`"
        assert _extract_environments_from_body(body) == ["job-a", "job-b", "job-c"]

    def test_returns_empty_when_no_match(self) -> None:
        assert _extract_environments_from_body("No environments line here") == []


class TestUpdateEnvironments:
    def test_replaces_environments_line(self) -> None:
        body = "Some text\n**Environments:** `old-job`\nMore text"
        updated = _update_environments_in_body(body, ["old-job", "new-job"])
        assert "**Environments:** `old-job`, `new-job`" in updated
        assert "Some text" in updated
        assert "More text" in updated


class TestMergeEnvironments:
    """The body_transform hook that carries the running env list forward."""

    def test_adds_new_environment(self) -> None:
        renderer = renderer_for(_make_failure(jobs=[("new-job", "suite", "url")]))
        result = renderer.merge_environments("**Environments:** `old-job`")
        assert "`old-job`" in result
        assert "`new-job`" in result

    def test_no_change_when_env_already_present(self) -> None:
        body = "**Environments:** `test-ubuntu-latest`"
        renderer = renderer_for(_make_failure())  # job is test-ubuntu-latest
        assert renderer.merge_environments(body) == body


# --- Integration tests with a mocked publisher ---


class TestProcessFailures:
    @patch("scripts.test_failure_detector.manage_issues.IssueDedupPublisher")
    def test_tallies_actions(self, mock_publisher_cls) -> None:
        publisher = mock_publisher_cls.return_value
        publisher.upsert.side_effect = [
            ("created", "https://x/issues/1"),
            ("updated", "https://x/issues/2"),
            ("skipped-duplicate", "https://x/issues/3"),
            ("skipped-recently-closed", "https://x/issues/4"),
        ]

        failures = [
            _make_failure(test_name="a"),
            _make_failure(test_name="b"),
            _make_failure(test_name="c"),
            _make_failure(test_name="d"),
        ]
        result = process_failures(MagicMock(), "valkey-io/valkey", failures)

        assert result == {
            "created": 1, "updated": 1, "skipped": 1, "skipped_closed": 1, "errors": 0,
        }

    @patch("scripts.test_failure_detector.manage_issues.IssueDedupPublisher")
    def test_one_failing_upsert_does_not_abort_the_batch(self, mock_publisher_cls) -> None:
        """A raised exception on one failure is counted as an error and skipped;
        the failures after it are still processed."""
        publisher = mock_publisher_cls.return_value
        publisher.upsert.side_effect = [
            ("created", "https://x/issues/1"),
            RuntimeError("boom"),  # failure b must not kill the loop
            ("updated", "https://x/issues/3"),
        ]

        failures = [
            _make_failure(test_name="a"),
            _make_failure(test_name="b"),
            _make_failure(test_name="c"),
        ]
        result = process_failures(MagicMock(), "valkey-io/valkey", failures)

        assert result == {
            "created": 1, "updated": 1, "skipped": 0, "skipped_closed": 0, "errors": 1,
        }
        # All three were attempted despite the middle one raising.
        assert publisher.upsert.call_count == 3

    @patch("scripts.test_failure_detector.manage_issues.IssueDedupPublisher")
    def test_unexpected_action_is_isolated_as_error(self, mock_publisher_cls) -> None:
        """An unexpected upsert action is contained as a single errored failure
        rather than propagating and aborting the run."""
        publisher = mock_publisher_cls.return_value
        publisher.upsert.side_effect = [
            ("bogus-action", "https://x/issues/1"),
            ("created", "https://x/issues/2"),
        ]

        result = process_failures(
            MagicMock(), "valkey-io/valkey",
            [_make_failure(test_name="a"), _make_failure(test_name="b")],
        )

        assert result == {
            "created": 1, "updated": 0, "skipped": 0, "skipped_closed": 0, "errors": 1,
        }

    @patch("scripts.test_failure_detector.manage_issues.IssueDedupPublisher")
    def test_passes_run_id_as_idempotency_key(self, mock_publisher_cls) -> None:
        publisher = mock_publisher_cls.return_value
        publisher.upsert.return_value = ("created", "https://x/issues/1")

        process_failures(MagicMock(), "valkey-io/valkey", [_make_failure()], run_id=12345)

        kwargs = publisher.upsert.call_args.kwargs
        assert kwargs["idempotency_key"] == "12345"
        assert kwargs["fingerprint"] == fingerprint_for(_make_failure())
        assert callable(kwargs["body_transform"])
        # The migration fallback title matches what render produces.
        assert kwargs["title_fallback"] == title_for(_make_failure())
        assert kwargs["title_fallback"] == _build_title(_make_failure())

    @patch("scripts.test_failure_detector.manage_issues.IssueDedupPublisher")
    def test_no_run_id_means_no_idempotency_key(self, mock_publisher_cls) -> None:
        publisher = mock_publisher_cls.return_value
        publisher.upsert.return_value = ("created", "https://x/issues/1")

        process_failures(MagicMock(), "valkey-io/valkey", [_make_failure()])

        assert publisher.upsert.call_args.kwargs["idempotency_key"] is None

    @patch("scripts.test_failure_detector.manage_issues.IssueDedupPublisher")
    def test_detector_opts_in_to_closed_lookback(self, mock_publisher_cls) -> None:
        """The recently-closed check is off by default on the shared publisher;
        the detector must enable it explicitly with its 1-day window."""
        publisher = mock_publisher_cls.return_value
        publisher.upsert.return_value = ("created", "https://x/issues/1")

        process_failures(MagicMock(), "valkey-io/valkey", [_make_failure()])

        kwargs = mock_publisher_cls.call_args.kwargs
        assert kwargs["closed_lookback"] == CLOSED_ISSUE_LOOKBACK
        assert CLOSED_ISSUE_LOOKBACK == timedelta(days=1)

    def test_render_callable_produces_labelled_content(self) -> None:
        content = renderer_for(_make_failure()).render("<!-- m -->", 1)
        assert content.labels == ("test-failure",)
        assert content.title.startswith("[TEST-FAILURE]")


def _make_valgrind_failure(size: str, job: str) -> UniqueFailure:
    """A nameless valgrind leak whose byte count varies run to run.

    The parser keeps the two size variants as distinct UniqueFailures while
    the fingerprint normalizes digits away, so the pair collides on one
    fingerprint. `Invalid read of size N` is used because the count scrubber
    only strips bytes/blocks phrases, leaving the digit for the fingerprint
    normalizer to collapse.
    """
    return UniqueFailure(
        test_name="", test_file="tests/unit/dummy.tcl",
        failure_type=FailureType.VALGRIND,
        error=(
            f"==1== Invalid read of size {size}\n"
            "==1==    at 0xA: dictResize (dict.c:100)"
        ),
        jobs=[JobReference(job=job, suite="s", url=f"https://ci/{job}")],
    )


class TestMergeSameFingerprintFailures:
    """Same-run failures that hash to one fingerprint must publish as one
    issue carrying every job, not race for it (the run-id idempotency key
    rejects the loser and its environments/CI links silently vanish, which is
    why issue #91 listed one environment though both valgrind jobs failed)."""

    def test_same_fingerprint_failures_merge_jobs(self) -> None:
        f1 = _make_valgrind_failure("4", "valgrind-ubuntu")
        f2 = _make_valgrind_failure("8", "valgrind-arm64")
        assert fingerprint_for(f1) == fingerprint_for(f2)

        merged = _merge_same_fingerprint_failures([f1, f2])

        assert len(merged) == 1
        assert {j.job for j in merged[0].jobs} == {"valgrind-ubuntu", "valgrind-arm64"}

    def test_merge_does_not_duplicate_shared_job(self) -> None:
        f1 = _make_valgrind_failure("4", "valgrind-ubuntu")
        f2 = _make_valgrind_failure("8", "valgrind-ubuntu")

        merged = _merge_same_fingerprint_failures([f1, f2])

        assert len(merged) == 1
        assert [j.job for j in merged[0].jobs] == ["valgrind-ubuntu"]

    def test_distinct_fingerprints_stay_separate(self) -> None:
        failures = [
            _make_failure(test_name="a"),
            _make_failure(test_name="b"),
        ]
        assert _merge_same_fingerprint_failures(failures) == failures

    @patch("scripts.test_failure_detector.manage_issues.IssueDedupPublisher")
    def test_process_failures_publishes_colliding_pair_once(self, mock_publisher_cls) -> None:
        """End to end: the colliding pair reaches upsert as one failure whose
        render carries both environments, instead of a second upsert that the
        idempotency key would reject."""
        publisher = mock_publisher_cls.return_value
        publisher.upsert.return_value = ("created", "https://x/issues/91")

        f1 = _make_valgrind_failure("4", "valgrind-ubuntu")
        f2 = _make_valgrind_failure("8", "valgrind-arm64")
        result = process_failures(
            MagicMock(), "valkey-io/valkey", [f1, f2], run_id=29944432899,
        )

        assert result == {
            "created": 1, "updated": 0, "skipped": 0, "skipped_closed": 0, "errors": 0,
        }
        assert publisher.upsert.call_count == 1
        body = publisher.upsert.call_args.kwargs["render"]("<!-- m -->", 1).body
        assert "`valgrind-ubuntu`" in body
        assert "`valgrind-arm64`" in body


class TestRecurrenceCommentNewlyFailing:
    """The recurrence comment calls out environments failing for the first time
    on this run (PR #24 review r3431750542)."""

    def test_names_newly_failing_environments(self) -> None:
        # New job 'test-arm64' is not in the prior body; the body_transform
        # records it, then render names it in the recurrence comment.
        renderer = renderer_for(_make_failure(jobs=[("test-arm64", "suite", "url")]))
        renderer.merge_environments("**Environments:** `test-ubuntu-latest`")
        comment = renderer.render("<!-- m -->", 2).comment
        assert "**Newly failing in:** `test-arm64`" in comment
        assert "Test failed again on" in comment

    def test_omits_newly_failing_line_when_no_new_environment(self) -> None:
        # The only job is already recorded, so there is nothing new to call out.
        renderer = renderer_for(_make_failure())  # job is test-ubuntu-latest
        renderer.merge_environments("**Environments:** `test-ubuntu-latest`")
        comment = renderer.render("<!-- m -->", 2).comment
        assert "Newly failing in" not in comment

    def test_no_newly_failing_line_without_body_transform(self) -> None:
        # On the create path body_transform never runs, so the comment (unused
        # there) carries no newly-failing line rather than a spurious one.
        comment = renderer_for(_make_failure()).render("<!-- m -->", 1).comment
        assert "Newly failing in" not in comment


class TestExtractErrorFromBody:
    """Round-trips the Error stack trace section written by _build_body."""

    def test_extracts_trace_written_by_build_body(self) -> None:
        body = _build_body(
            _make_failure(error="assertion failed at line 42"),
            marker="<!-- m -->", occurrences=1,
        )
        assert _extract_error_from_body(body) == "assertion failed at line 42"

    def test_returns_empty_when_no_error_section(self) -> None:
        # Issues created before the Error stack trace section existed.
        assert _extract_error_from_body("**Environments:** `job-a`") == ""

    def test_round_trips_error_containing_backtick_fence(self) -> None:
        """An error that itself contains ``` must survive the body round-trip
        intact; a truncated read-back would make _detect_new_error flag a
        spurious "new error" on every recurrence."""
        error = "assertion failed\n```\nembedded block\n```\ntrailing context"
        body = _build_body(
            _make_failure(error=error), marker="<!-- m -->", occurrences=1,
        )
        assert _extract_error_from_body(body) == error


class TestRecurrenceCommentNewError:
    """The recurrence comment surfaces a changed error trace so a triager can
    notice the failure mode shifted without diffing the issue body."""

    def _body_with_error(self, error: str) -> str:
        return _build_body(
            _make_failure(error=error), marker="<!-- m -->", occurrences=1,
        )

    def test_calls_out_changed_trace(self) -> None:
        # The issue recorded one trace; this run failed with a different one.
        renderer = renderer_for(_make_failure(error="NEW: segfault in dictResize"))
        renderer.merge_environments(self._body_with_error("OLD: timeout waiting for sync"))
        comment = renderer.render("<!-- m -->", 2).comment
        assert "**New error stack trace**" in comment
        assert "NEW: segfault in dictResize" in comment

    def test_stays_quiet_when_trace_unchanged(self) -> None:
        renderer = renderer_for(_make_failure(error="same error every time"))
        renderer.merge_environments(self._body_with_error("same error every time"))
        comment = renderer.render("<!-- m -->", 2).comment
        assert "New error stack trace" not in comment

    def test_normalized_equal_trace_stays_quiet(self) -> None:
        # Differs only in run-specific noise (port, timestamp, hex address);
        # normalization treats these as the same trace.
        old = "conn failed 2026-06-26 10:00:00 port=6379 at 0xdead"
        new = "conn failed 2026-06-27 11:22:33 port=7000 at 0xbeef"
        renderer = renderer_for(_make_failure(error=new))
        renderer.merge_environments(self._body_with_error(old))
        comment = renderer.render("<!-- m -->", 2).comment
        assert "New error stack trace" not in comment

    def test_empty_new_error_stays_quiet(self) -> None:
        renderer = renderer_for(_make_failure(error=""))
        renderer.merge_environments(self._body_with_error("OLD: some trace"))
        comment = renderer.render("<!-- m -->", 2).comment
        assert "New error stack trace" not in comment

    def test_legacy_issue_without_recorded_trace_stays_quiet(self) -> None:
        # A legacy issue with no Error stack trace section has no baseline, so
        # the trace must not be called out. Otherwise it would diff against ""
        # and re-post the same "new" trace on every recurrence.
        legacy_body = "**Environments:** `test-ubuntu-latest`"
        renderer = renderer_for(_make_failure(error="some real trace"))
        renderer.merge_environments(legacy_body)
        comment = renderer.render("<!-- m -->", 2).comment
        assert "New error stack trace" not in comment

    def test_no_new_error_line_without_body_transform(self) -> None:
        # Create path: body_transform never runs, so no spurious callout.
        comment = renderer_for(_make_failure()).render("<!-- m -->", 1).comment
        assert "New error stack trace" not in comment


# --- Tests for type-specific fingerprinting and rendering ---


class TestTypeSpecificFingerprint:
    """Fingerprints are scoped by failure type so different categories
    cannot collide, and nameless errors are fingerprinted by error identity."""

    def test_different_types_different_fingerprints(self) -> None:
        """Same test name + file, different type => different fingerprint."""
        assertion = _make_failure()
        timeout = UniqueFailure(
            test_name="PSYNC2 test",
            test_file="tests/integration/replication-psync.tcl",
            failure_type=FailureType.TIMEOUT,
            error="Test timed out",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        assert fingerprint_for(assertion) != fingerprint_for(timeout)

    def test_sanitizer_fingerprint_ignores_pid(self) -> None:
        """Same sanitizer error with different PIDs => same fingerprint."""
        f1 = UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.SANITIZER,
            error="==111== ERROR: AddressSanitizer: heap-buffer-overflow\n==111==    at 0xAAA: dictResize (dict.c:100)",
        )
        f2 = UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.SANITIZER,
            error="==222== ERROR: AddressSanitizer: heap-buffer-overflow\n==222==    at 0xBBB: dictResize (dict.c:100)",
        )
        assert fingerprint_for(f1) == fingerprint_for(f2)

    def test_valgrind_cross_file_same_fingerprint(self) -> None:
        """Same valgrind error in different test files => same fingerprint.
        The test_file is intentionally excluded from the nameless fingerprint."""
        f1 = UniqueFailure(
            test_name="", test_file="tests/unit/expire.tcl",
            failure_type=FailureType.VALGRIND,
            error="==1== Invalid read of size 4\n==1==    at 0xA: dictResize (dict.c:100)",
        )
        f2 = UniqueFailure(
            test_name="", test_file="tests/unit/cluster.tcl",
            failure_type=FailureType.VALGRIND,
            error="==2== Invalid read of size 4\n==2==    at 0xB: dictResize (dict.c:100)",
        )
        assert fingerprint_for(f1) == fingerprint_for(f2)

    def test_valgrind_same_leak_two_jobs_one_fingerprint(self) -> None:
        """The real #114/#115 case: one leak from debugCommand, reported by two
        valgrind jobs, differs only in the leaked size and whether the trailing
        "ERROR SUMMARY" line falls inside the identity window. Both must produce
        one fingerprint so the pair collapses into a single issue."""
        def report(size: int, extra_tail: str) -> UniqueFailure:
            error = (
                " Valgrind error: ==1== Memcheck, a memory error detector\n"
                "==1== HEAP SUMMARY:\n"
                f"==1== {size} bytes in 1 blocks are definitely lost in loss record 900 of 1,111\n"
                "==1==    at 0x4846828: malloc (vgpreload_memcheck.so)\n"
                "==1==    by 0x318A40: ztrymalloc_usable_internal (zmalloc.c:172)\n"
                "==1==    by 0x29078A: sdsdup (sds.c:190)\n"
                "==1==    by 0x1E80D6: debugCommand (debug.c:569)\n"
                f"{extra_tail}"
            )
            return UniqueFailure(
                test_name="", test_file="tests/unit/dummy-memory.tcl",
                failure_type=FailureType.VALGRIND, error=error,
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
        f114 = report(49, "==1== ERROR SUMMARY: 36 errors from 36 contexts (suppressed: 0 from 0)")
        f115 = report(41, "==1== still reachable: 931,751 bytes in 12,710 blocks\n==1== suppressed: 0 bytes")
        assert fingerprint_for(f114) == fingerprint_for(f115)

    def test_different_sanitizer_bugs_different_fingerprints(self) -> None:
        f1 = UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.SANITIZER,
            error="==1== ERROR: AddressSanitizer: heap-buffer-overflow\n==1==    at 0xA: dictResize (dict.c:100)",
        )
        f2 = UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.SANITIZER,
            error="==1== ERROR: AddressSanitizer: use-after-free\n==1==    at 0xA: listRelease (adlist.c:50)",
        )
        assert fingerprint_for(f1) != fingerprint_for(f2)

    def test_unittest_has_named_fingerprint(self) -> None:
        """gtest failures have test_name, so they use the named path."""
        f = UniqueFailure(
            test_name="DictTest.BasicOps",
            test_file="src/unit/valkey-unit-gtests",
            failure_type=FailureType.UNITTEST,
        )
        assert f.has_test_identity
        fp = fingerprint_for(f)
        assert re.fullmatch(r"[0-9a-f]{20}", fp)

    def test_assertion_fingerprint_unchanged_from_legacy(self) -> None:
        """Assertion-type fingerprint uses the same namespace as before
        so existing issues are still matched."""
        f = _make_failure()
        ns = marker_namespace_for(f)
        assert ns == MARKER_NAMESPACE


class TestTypeSpecificRendering:
    """Type-specific title prefixes, labels, and body format."""

    def test_sanitizer_title_prefix(self) -> None:
        f = UniqueFailure(
            test_name="", test_file="tests/unit/expire.tcl",
            failure_type=FailureType.SANITIZER,
            error="Sanitizer error: heap-buffer-overflow in dictResize",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert title.startswith("[SANITIZER]")

    def test_valgrind_title_prefix(self) -> None:
        f = UniqueFailure(
            test_name="", test_file="tests/unit/expire.tcl",
            failure_type=FailureType.VALGRIND,
            error="Valgrind error: Invalid read of size 4",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        assert title_for(f).startswith("[VALGRIND]")

    def test_valgrind_leak_title_format(self) -> None:
        # Issue #91/#93: the Memcheck banner is the first line of every
        # valgrind report, so a first-line title gives all valgrind issues
        # the same name. A leak title leads with the leak kind, then the
        # size and the first non-plumbing source frame:
        # "Definitely lost: 49 bytes in debugCommand (debug.c:569)".
        error = (
            " Valgrind error: ==6554== Memcheck, a memory error detector\n"
            "==6554== Copyright (C) 2002-2022, and GNU GPL'd, by Julian Seward et al.\n"
            "==6554== HEAP SUMMARY:\n"
            "==6554== 49 bytes in 1 blocks are definitely lost in loss record 900 of 1,109\n"
            "==6554==    at 0x4846828: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)\n"
            "==6554==    by 0x3189FB: ztrymalloc_usable_internal (zmalloc.c:172)\n"
            "==6554==    by 0x2902DE: _sdsnewlen (sds.c:102)\n"
            "==6554==    by 0x1E8076: debugCommand (debug.c:569)\n"
        )
        f = UniqueFailure(
            test_name="", test_file="tests/unit/dummy-memory.tcl",
            failure_type=FailureType.VALGRIND,
            error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        # The test file is the volatile detection context (valgrind keys its
        # fingerprint on the error, not the file), so it is not in the title.
        title = title_for(f)
        assert title == (
            "[VALGRIND] Definitely lost: 49 bytes in debugCommand (debug.c:569)"
        )

    def test_valgrind_leak_title_ignores_loss_record_and_pid_drift(self) -> None:
        # Loss-record coordinates and PIDs drift between runs of the same
        # leak and must not affect the title. The size is shown as-is (the
        # publisher refreshes the title on recurrence; dedup is owned by the
        # fingerprint, which scrubs sizes).
        def leak(record: str, pid: str) -> UniqueFailure:
            error = (
                f" Valgrind error: =={pid}== Memcheck, a memory error detector\n"
                f"=={pid}== 49 bytes in 1 blocks are definitely lost in {record}\n"
                f"=={pid}==    by 0x1E8076: debugCommand (debug.c:569)\n"
            )
            return UniqueFailure(
                test_name="", test_file="tests/unit/dummy-memory.tcl",
                failure_type=FailureType.VALGRIND, error=error,
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
        t1 = title_for(leak("loss record 900 of 1,109", "6554"))
        t2 = title_for(leak("loss record 903 of 1,214", "7801"))
        assert t1 == t2
        assert "loss record" not in t1
        assert "6554" not in t1

    def test_valgrind_titles_distinguish_different_leak_sites(self) -> None:
        def leak(site: str) -> UniqueFailure:
            error = (
                " Valgrind error: ==1== Memcheck, a memory error detector\n"
                "==1== 49 bytes in 1 blocks are definitely lost in loss record 900 of 1,109\n"
                f"==1==    by 0x1E8076: {site}\n"
            )
            return UniqueFailure(
                test_name="", test_file="tests/unit/dummy-memory.tcl",
                failure_type=FailureType.VALGRIND, error=error,
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
        t_debug = title_for(leak("debugCommand (debug.c:569)"))
        t_cluster = title_for(leak("clusterCommand (cluster.c:123)"))
        assert t_debug != t_cluster

    def test_timeout_title_with_test_name(self) -> None:
        f = UniqueFailure(
            test_name="PSYNC2 test",
            test_file="tests/integration/replication-psync.tcl",
            failure_type=FailureType.TIMEOUT,
            error="Test timed out",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert title.startswith("[TIMEOUT]")
        assert "PSYNC2 test" in title

    def test_unittest_title(self) -> None:
        f = UniqueFailure(
            test_name="DictTest.BasicOps",
            test_file="src/unit/valkey-unit-gtests",
            failure_type=FailureType.UNITTEST,
            error="gtest FAIL",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert title.startswith("[UNITTEST]")
        assert "DictTest.BasicOps" in title

    def test_startup_title_without_test_name(self) -> None:
        # Startup keys its fingerprint on the error, not the file, so the file
        # is left out of the title and the same failure keeps one title across
        # the different files it is detected under.
        def startup(test_file: str) -> UniqueFailure:
            return UniqueFailure(
                test_name="", test_file=test_file,
                failure_type=FailureType.STARTUP,
                error="Can't start /path/to/valkey-server",
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
        title = title_for(startup("tests/unit/cluster.tcl"))
        assert title.startswith("[STARTUP-FAILURE]")
        assert "cluster.tcl" not in title
        assert title == title_for(startup("tests/unit/expire.tcl"))

    def test_all_types_use_test_failure_label(self) -> None:
        for ftype in FailureType:
            f = UniqueFailure(
                test_name="t" if ftype in (FailureType.ASSERTION, FailureType.TIMEOUT, FailureType.UNITTEST) else "",
                test_file="f.tcl",
                failure_type=ftype,
                error="some error",
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
            assert label_for(f) == "test-failure"

    def test_renderer_uses_test_failure_label(self) -> None:
        f = UniqueFailure(
            test_name="DictTest.Ops",
            test_file="src/unit/valkey-unit-gtests",
            failure_type=FailureType.UNITTEST,
            error="gtest FAIL",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        content = renderer_for(f).render("<!-- m -->", 1)
        assert content.labels == ("test-failure",)

    def test_nameless_body_has_error_details_section(self) -> None:
        """Nameless failures get 'Error details' instead of 'Failing test(s)'."""
        f = UniqueFailure(
            test_name="", test_file="tests/unit/expire.tcl",
            failure_type=FailureType.SANITIZER,
            error="Sanitizer error: heap-buffer-overflow",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        body = _build_body(f, marker="<!-- m -->", occurrences=1)
        assert "**Error details**" in body
        assert "Sanitizer" in body
        assert "expire.tcl" in body

    def test_named_body_has_failure_type_field(self) -> None:
        """Named failures include a Failure type line in the body."""
        f = UniqueFailure(
            test_name="PSYNC2 test",
            test_file="tests/integration/replication-psync.tcl",
            failure_type=FailureType.TIMEOUT,
            error="Test timed out",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        body = _build_body(f, marker="<!-- m -->", occurrences=1)
        assert "Failure type: `Timeout`" in body

    def test_type_specific_namespace_in_body(self) -> None:
        """Body uses type-specific namespace for the occurrences marker."""
        f = UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.VALGRIND,
            error="Valgrind error: Invalid read",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        body = _build_body(f, marker="<!-- m -->", occurrences=3)
        assert "valkey-ci-agent:valgrind-error:occurrences:3" in body


class TestVolatileTimeoutFingerprint:
    """Nameless timeouts (volatile PID demoted) must have a stable fingerprint
    keyed by file, not by the generic error text (#82, #86)."""

    def test_nameless_timeout_fingerprint_stable_across_pids(self) -> None:
        f1 = UniqueFailure(
            test_name="", test_file="tests/integration/replication.tcl",
            failure_type=FailureType.TIMEOUT, error="Test timed out",
        )
        f2 = UniqueFailure(
            test_name="", test_file="tests/integration/replication.tcl",
            failure_type=FailureType.TIMEOUT, error="Test timed out",
        )
        assert fingerprint_for(f1) == fingerprint_for(f2)

    def test_nameless_timeouts_in_different_files_differ(self) -> None:
        f1 = UniqueFailure(
            test_name="", test_file="tests/integration/replication.tcl",
            failure_type=FailureType.TIMEOUT, error="Test timed out",
        )
        f2 = UniqueFailure(
            test_name="", test_file="tests/unit/cluster.tcl",
            failure_type=FailureType.TIMEOUT, error="Test timed out",
        )
        assert fingerprint_for(f1) != fingerprint_for(f2)

    def test_nameless_timeout_title_uses_file_not_pid(self) -> None:
        f = UniqueFailure(
            test_name="", test_file="tests/integration/replication.tcl",
            failure_type=FailureType.TIMEOUT, error="Test timed out",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert "pid" not in title.lower()
        assert "replication.tcl" in title


class TestValgrindBannerTitle:
    """The valgrind runner prepends a banner ('Valgrind error: Memcheck, a
    memory error detector') that every valgrind issue would share. The title
    must surface the real diagnostic line instead (#91)."""

    def test_title_uses_diagnostic_not_banner(self) -> None:
        error = (
            " Valgrind error: ==6554== Memcheck, a memory error detector\n"
            "==6554== Copyright (C) 2002-2022\n"
            "==6554== \n"
            "==6554== HEAP SUMMARY:\n"
            "==6554== 49 bytes in 1 blocks are definitely lost in loss record 900\n"
            "==6554==    at 0x4846828: malloc (...)\n"
        )
        f = UniqueFailure(
            test_name="", test_file="tests/unit/dummy-memory.tcl",
            failure_type=FailureType.VALGRIND, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert "Memcheck" not in title
        # No source frame in this trace (only the malloc interceptor), so the
        # title is kind + size without a site.
        assert "Definitely lost: 49 bytes" in title

    def test_title_names_leaking_code_path_not_the_allocator(self) -> None:
        """Valgrind resolves its malloc interceptor to a source file inside the
        preload library, so a file-only plumbing check lets it through and every
        leak title reads "in malloc". The title must name the first frame that
        identifies the leaking code path, matching the identity's anchor.
        """
        error = (
            " Valgrind error: ==6554== Memcheck, a memory error detector\n"
            "==6554== 41 bytes in 1 blocks are definitely lost in loss record 900 of 1,109\n"
            "==6554==    at 0x4846828: malloc (vg_replace_malloc.c:307)\n"
            "==6554==    by 0x3189FB: ztrymalloc_usable_internal (zmalloc.c:172)\n"
            "==6554==    by 0x29072A: sdsdup (sds.c:190)\n"
            "==6554==    by 0x1E8076: debugCommand (debug.c:569)\n"
        )
        f = UniqueFailure(
            test_name="", test_file="tests/unit/dummy-memory.tcl",
            failure_type=FailureType.VALGRIND, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert "debugCommand (debug.c:569)" in title
        assert "malloc" not in title
        assert "zmalloc.c" not in title

    def test_two_leaks_sharing_an_allocator_get_distinct_titles(self) -> None:
        """Two leaks whose stacks differ only past the shared allocator frames
        must not collapse to one title, or an issue list cannot tell them apart.
        """
        def leak(site_func: str, site_loc: str) -> UniqueFailure:
            error = (
                "==6554== 41 bytes in 1 blocks are definitely lost in loss record 9 of 99\n"
                "==6554==    at 0x4846828: malloc (vg_replace_malloc.c:307)\n"
                "==6554==    by 0x3189FB: ztrymalloc_usable_internal (zmalloc.c:172)\n"
                f"==6554==    by 0x1E8076: {site_func} ({site_loc})\n"
            )
            return UniqueFailure(
                test_name="", test_file="tests/unit/dummy-memory.tcl",
                failure_type=FailureType.VALGRIND, error=error,
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
        first = leak("debugCommand", "debug.c:569")
        second = leak("clusterCommand", "cluster.c:123")
        assert title_for(first) != title_for(second)
        assert fingerprint_for(first) != fingerprint_for(second)

    def test_sanitizer_banner_stripped(self) -> None:
        error = (
            " Sanitizer error: \n"
            "==12617==ERROR: LeakSanitizer: detected memory leaks\n"
            "Direct leak of 41 byte(s)\n"
        )
        f = UniqueFailure(
            test_name="", test_file="tests/unit/foo.tcl",
            failure_type=FailureType.SANITIZER, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert "Sanitizer error:" not in title
        assert "detected memory leaks" in title

    def test_sanitizer_title_stable_across_test_file(self) -> None:
        """One sanitizer bug detected under different test files across runs
        keeps one title, matching its file-independent fingerprint."""
        error = (
            "==1==ERROR: AddressSanitizer: heap-use-after-free\n"
            "    #1 0x55 in freeStringObject object.c:400\n"
        )
        def san(test_file: str) -> UniqueFailure:
            return UniqueFailure(
                test_name="", test_file=test_file,
                failure_type=FailureType.SANITIZER, error=error,
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
        first = san("tests/unit/type/string.tcl")
        second = san("tests/unit/expire.tcl")
        assert fingerprint_for(first) == fingerprint_for(second)
        assert title_for(first) == title_for(second)

    def test_sanitizer_title_scrubs_volatile_address(self) -> None:
        """The address and registers in an AddressSanitizer diagnostic line
        drift every run; the title drops them so it does not change while the
        fingerprint stays stable."""
        def san(address: str, pc: str) -> UniqueFailure:
            error = (
                f"==1==ERROR: AddressSanitizer: heap-use-after-free on address "
                f"{address} at pc {pc} bp 0x7ffd sp 0x7ffd\n"
                "    #1 0x55 in freeStringObject object.c:400\n"
            )
            return UniqueFailure(
                test_name="", test_file="tests/unit/expire.tcl",
                failure_type=FailureType.SANITIZER, error=error,
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
        title = title_for(san("0x60200000eff0", "0x000000abcdef"))
        assert "0x" not in title
        assert "heap-use-after-free" in title
        assert title == title_for(san("0x602000001234", "0x000000fedcba"))

    def test_error_severity_tag_stripped_tool_name_kept(self) -> None:
        """The 'ERROR:' tag is dropped from titles (it says nothing) but the
        tool name after it stays so maintainers see which detector fired."""
        error = (
            "==107611==ERROR: LeakSanitizer: detected memory leaks\n"
            "Direct leak of 128 byte(s) in 4 object(s) allocated from:\n"
        )
        f = UniqueFailure(
            test_name="", test_file="tests/unit/fuzzer.tcl",
            failure_type=FailureType.SANITIZER, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert "ERROR:" not in title
        assert "LeakSanitizer: detected memory leaks" in title

    def test_sanitizer_leak_title_shows_size_and_site(self) -> None:
        """When the report has the "SUMMARY: AddressSanitizer: N byte(s) leaked"
        line, the title leads with the size and names the leaking code path
        instead of the magnitude-free "detected memory leaks" banner."""
        error = (
            " Sanitizer error: \n"
            "==6366==ERROR: LeakSanitizer: detected memory leaks\n"
            "Direct leak of 41 byte(s) in 1 object(s) allocated from:\n"
            "    #0 0x55ba0d2fbe33 in malloc (src/valkey-server+0x20de33)\n"
            "    #1 0x55ba0d702bb3 in ztrymalloc_usable_internal src/zmalloc.c:172:17\n"
            "    #2 0x55ba0d5f5c07 in _sdsnewlen src/sds.c:102:22\n"
            "    #3 0x55ba0d435e7f in debugCommand src/debug.c:569:9\n"
            "SUMMARY: AddressSanitizer: 41 byte(s) leaked in 1 allocation(s).\n"
        )
        f = UniqueFailure(
            test_name="", test_file="tests/unit/dummy-memory.tcl",
            failure_type=FailureType.SANITIZER, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert "detected memory leaks" not in title
        assert "Leaked 41 byte(s)" in title
        # The site skips the allocator wrappers (zmalloc.c/sds.c) and names the
        # code path that leaked.
        assert "debugCommand (debug.c:569)" in title

    def test_sanitizer_leak_title_survives_missing_frames(self) -> None:
        """A summary line with no source frames still yields a size-based
        title rather than falling back to the banner."""
        error = (
            "==1==ERROR: LeakSanitizer: detected memory leaks\n"
            "Direct leak of 96 byte(s) in 2 object(s) allocated from:\n"
            "    #0 0x1 in malloc (src/valkey-server+0x1)\n"
            "SUMMARY: AddressSanitizer: 96 byte(s) leaked in 2 allocation(s).\n"
        )
        f = UniqueFailure(
            test_name="", test_file="tests/unit/foo.tcl",
            failure_type=FailureType.SANITIZER, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        assert "Leaked 96 byte(s)" in title_for(f)

    def test_sanitizer_leak_site_skips_interceptor_frame(self) -> None:
        """GCC's ASan interceptor frame carries a real file:line into the
        sanitizer's own sources (asan_malloc_linux.cpp), so it passes the
        source-frame regex; the site must skip it like the allocator
        wrappers, or every leak titles as 'in malloc'."""
        error = (
            " Sanitizer error: \n"
            "==3021==ERROR: LeakSanitizer: detected memory leaks\n"
            "Direct leak of 49 byte(s) in 1 object(s) allocated from:\n"
            "    #0 0x7f8a4a2b476f in malloc ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:69\n"
            "    #1 0x55c908b21a02 in ztrymalloc_usable_internal /home/runner/work/valkey/valkey/src/zmalloc.c:172\n"
            "    #2 0x55c908d1e222 in debugCommand /home/runner/work/valkey/valkey/src/debug.c:569\n"
            "SUMMARY: AddressSanitizer: 49 byte(s) leaked in 1 allocation(s).\n"
        )
        f = UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.SANITIZER, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert "asan_malloc_linux" not in title
        assert "debugCommand (debug.c:569)" in title


class TestStartupFailureTitle:
    """A startup blob's first line names only the executable, which is the
    same for every startup failure; the title must carry the reason after the
    ERROR: header so two causes are tellable apart in an issue list."""

    def _failure(self, reason: str) -> UniqueFailure:
        config = "\n".join(f"directive-{i} value-{i}" for i in range(40))
        error = (
            "Can't start /path/to/valkey-server\n"
            f"CONFIGURATION:\n{config}\nERROR:\n{reason}"
        )
        return UniqueFailure(
            test_name="", test_file="tests/unit/dummy-startup.tcl",
            failure_type=FailureType.STARTUP, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )

    def test_title_names_the_reason(self) -> None:
        title = title_for(self._failure("Unable to bind unix socket: Permission denied"))
        assert "Unable to bind unix socket" in title

    def test_title_skips_fatal_banner(self) -> None:
        title = title_for(self._failure(
            "*** FATAL CONFIG FILE ERROR (Version 9.0.0) ***\n"
            "Bad directive or wrong number of arguments"
        ))
        assert "***" not in title
        assert "Bad directive" in title

    def test_different_causes_get_different_titles(self) -> None:
        t1 = title_for(self._failure("Unable to bind unix socket: Permission denied"))
        t2 = title_for(self._failure("Bad directive or wrong number of arguments"))
        assert t1 != t2

    def test_blob_without_error_section_falls_back(self) -> None:
        f = UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.STARTUP,
            error="Can't start /path/to/valkey-server",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        assert "Can't start" in title_for(f)

    def test_title_survives_runner_status_tag_whitespace(self) -> None:
        """The runner's "[err]: " tag is stripped upstream and leaves a leading
        space. Without tolerating it the startup branch never fires and the
        title becomes the executable path truncated mid-word."""
        f = self._failure("Bad directive or wrong number of arguments")
        f.error = f" {f.error}"
        title = title_for(f)
        assert "Bad directive" in title
        assert "valkey-serve" not in title

    def test_title_skips_progress_and_position_lines(self) -> None:
        """The harness's "###" marker and the config loader's position/echo
        lines precede the reason but name no cause."""
        title = title_for(self._failure(
            "### Starting server for test \n\n"
            "*** FATAL CONFIG FILE ERROR (Version 9.0.0) ***\n"
            "Reading the configuration file, at line 30\n"
            ">>> 'invalid-config-key-that-does-not-exist bogus'\n"
            "Bad directive or wrong number of arguments"
        ))
        assert "Bad directive or wrong number of arguments" in title
        assert "###" not in title
        assert "at line" not in title

    def test_valgrind_wrapped_startup_matches_plain_startup(self) -> None:
        """Under valgrind the capture opens with the tool's own banner and
        interleaves ==PID== markers. It is the same config error, so it must
        not mint a second issue."""
        plain = self._failure(
            "*** FATAL CONFIG FILE ERROR (Version 9.0.0) ***\n"
            "Reading the configuration file, at line 30\n"
            "Bad directive or wrong number of arguments"
        )
        under_valgrind = self._failure(
            "### Starting server for test \n"
            "==6688== Memcheck, a memory error detector\n"
            "==6688== Using Valgrind-3.22.0 and LibVEX\n\n"
            "*** FATAL CONFIG FILE ERROR (Version 9.0.0) ***\n"
            "Reading the configuration file, at line 30\n"
            "Bad directive or wrong number of arguments\n"
            "==6688== HEAP SUMMARY:\n"
        )
        assert title_for(plain) == title_for(under_valgrind)
        assert normalize_error_identity(plain.error) == normalize_error_identity(
            under_valgrind.error
        )


class TestValgrindRecurrenceStaysQuiet:
    """A recurrence of the same valgrind leak differs only in ==PID== markers,
    sizes, and loss-record coordinates. The fingerprint calls it the same bug,
    so the recurrence comment must not flag it as a new error."""

    _RUN1 = (
        "Valgrind error: ==12345== Memcheck, a memory error detector\n"
        "==12345== 49 bytes in 1 blocks are definitely lost in loss record 900 of 1,109\n"
        "==12345==    by 0x1E8076: ztrymalloc_usable_internal (zmalloc.c:172)\n"
        "==12345==    by 0x3CD456: debugCommand (debug.c:569)\n"
    )

    def _failure(self, error: str) -> UniqueFailure:
        return UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.VALGRIND, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )

    def _recur(self, old_error: str, new_error: str) -> str:
        body = _build_body(self._failure(old_error), marker="<!-- m -->", occurrences=1)
        renderer = renderer_for(self._failure(new_error))
        renderer.merge_environments(body)
        return renderer.render("<!-- m -->", 2).comment

    def test_same_leak_new_pid_and_size_stays_quiet(self) -> None:
        rerun = (
            self._RUN1.replace("12345", "999")
            .replace("49 bytes", "41 bytes")
            .replace("900 of 1,109", "850 of 1,050")
        )
        assert "New error stack trace" not in self._recur(self._RUN1, rerun)

    def test_different_allocation_site_is_flagged(self) -> None:
        other = self._RUN1.replace(
            "debugCommand (debug.c:569)", "clusterCommand (cluster.c:1201)",
        )
        assert "New error stack trace" in self._recur(self._RUN1, other)

    def test_heap_usage_totals_stay_quiet(self) -> None:
        """Valgrind's heap totals count every allocation the server made, so
        they drift on every run of one leak. The comparison must scrub them or
        each recurrence reposts the whole trace.
        """
        with_totals = self._RUN1 + (
            "==12345==   total heap usage: 18,053 allocs, 4,513 frees, "
            "1,383,828 bytes allocated\n"
        )
        rerun = with_totals.replace("18,053 allocs, 4,513 frees", "18,066 allocs, 4,517 frees")
        assert "New error stack trace" not in self._recur(with_totals, rerun)

    def test_sanitizer_build_id_stays_quiet(self) -> None:
        """A sanitizer frame's BuildId changes whenever the binary is rebuilt,
        which is every CI run, so it is not evidence of a different bug.
        """
        def report(build_id: str) -> str:
            return (
                "==1==ERROR: LeakSanitizer: detected memory leaks\n"
                "Direct leak of 41 byte(s) in 1 object(s) allocated from:\n"
                f"    #0 0x55 in malloc (/src/valkey-server+0x20de33) (BuildId: {build_id})\n"
                "    #4 0x55 in debugCommand /src/debug.c:569:9\n"
            )
        assert "New error stack trace" not in self._recur(
            report("eaa319adda1cfec4d818"), report("2bf960fbb10e52be190f"),
        )

    def test_macos_leaks_footprint_stays_quiet(self) -> None:
        """The leaks report's footprint measures the live server rather than
        the leak, so it varies run to run for one leak.
        """
        def report(footprint: str) -> str:
            return (
                "Check for memory leaks in tests/unit/dummy-memory.tcl\n"
                f"Physical footprint:         {footprint}\n"
                f"Physical footprint (peak):  {footprint}\n"
                "Process 9761: 1 leak for 48 total leaked bytes.\n"
                "    1 (48 bytes) ROOT LEAK: <malloc in sdsnewlen>\n"
            )
        assert "New error stack trace" not in self._recur(
            report("2865K"), report("2801K"),
        )


class TestTraceTruncation:
    """GitHub rejects bodies over 65536 chars; oversized traces are capped
    keeping head (names the error) and tail (holds the summary totals)."""

    def _big_failure(self) -> UniqueFailure:
        error = "HEAD: first line names the error\n" + ("x" * 100 + "\n") * 1000 + "TAIL: summary totals"
        return UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.VALGRIND, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )

    def test_body_stays_under_github_limit(self) -> None:
        body = _build_body(self._big_failure(), marker="<!-- m -->", occurrences=1)
        assert len(body) < 65536

    def test_truncation_keeps_head_and_tail_and_says_so(self) -> None:
        body = _build_body(self._big_failure(), marker="<!-- m -->", occurrences=1)
        assert "HEAD: first line names the error" in body
        assert "TAIL: summary totals" in body
        assert "trace truncated" in body

    def test_short_trace_untouched(self) -> None:
        f = UniqueFailure(
            test_name="", test_file="",
            failure_type=FailureType.VALGRIND, error="short trace",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        body = _build_body(f, marker="<!-- m -->", occurrences=1)
        assert "trace truncated" not in body
        assert "short trace" in body

    def test_truncated_recurrence_stays_quiet(self) -> None:
        """The stored trace is the truncated form; the fresh full-length trace
        must compare equal to it or every recurrence posts a new-error comment."""
        f = self._big_failure()
        body = _build_body(f, marker="<!-- m -->", occurrences=1)
        renderer = renderer_for(self._big_failure())
        renderer.merge_environments(body)
        comment = renderer.render("<!-- m -->", 2).comment
        assert "New error stack trace" not in comment


class TestExceptionTitle:
    """Uncaught test-client exceptions arrive wrapped in the runner's
    "Executing test client: <message>" prefix. The title must surface the
    message, not the wrapper (#116)."""

    def _failure(self, error: str) -> UniqueFailure:
        return UniqueFailure(
            test_name="", test_file="tests/unit/dummy-exception.tcl",
            failure_type=FailureType.EXCEPTION, error=error,
            jobs=[JobReference(job="j", suite="s", url="u")],
        )

    def test_strips_executing_test_client_prefix(self) -> None:
        error = (
            " Executing test client: Intentional runtime exception for detector testing.\n"
            " in error at tests/unit/dummy-exception.tcl:12\n"
            " in test at tests/support/test.tcl:262\n"
        )
        title = title_for(self._failure(error))
        assert "Executing test client:" not in title
        assert "Intentional runtime exception for detector testing." in title
        assert title.startswith("[EXCEPTION] ")

    def test_title_stable_across_volatile_ports_and_pids(self) -> None:
        """The fingerprint scrubs ports/PIDs, so one recurring exception keeps
        one issue; the title must scrub them too or the publisher rewrites it
        with the new port on every recurrence."""
        template = " Executing test client: couldn't open socket: connection refused, port {port}\n"
        t1 = title_for(self._failure(template.format(port=21079)))
        t2 = title_for(self._failure(template.format(port=21987)))
        assert t1 == t2
        assert "21079" not in t1


def _macos_leaks_error(pid: int, leaks: int, leaked_bytes: int, address: str) -> str:
    """A macOS /usr/bin/leaks failure blob as recorded in the artifact
    (real shape from valkey-io Daily run 29461435670, test-macos-latest)."""
    return (
        f" Check for memory leaks (pid {pid}) in tests/unit/multi.tcl\n"
        f"Expected '*0 leaks*' to equal or match 'Process:         valkey-server [{pid}]\n"
        "Path:            /Users/USER/*/valkey-server\n"
        "Load Address:    0x102610000\n"
        "Platform:        macOS\n"
        "Analysis Tool:   /usr/bin/leaks\n"
        "----\n"
        "leaks Report Version: 4.0\n"
        f"Process {pid}: 14810 nodes malloced for 1403 KB\n"
        f"Process {pid}: {leaks} leak for {leaked_bytes} total leaked bytes.\n"
        "\n"
        f"    {leaks} ({leaked_bytes} bytes) ROOT LEAK: {address} [{leaked_bytes}]\n"
        "\n"
        "child process exited abnormally'\n"
    )


class TestMacosLeaksTitle:
    """macOS /usr/bin/leaks failures (the memory-leak type). Their first line
    is the Tcl test name with a volatile PID; the title must surface the
    report's totals line instead (#92)."""

    def _failure(self, error: str) -> UniqueFailure:
        return UniqueFailure(
            test_name="", test_file="tests/unit/multi.tcl",
            failure_type=FailureType.MEMORY_LEAK, error=error,
            jobs=[JobReference(job="test-macos-latest", suite="valkey", url="u")],
        )

    def test_title_uses_leaks_totals_line(self) -> None:
        f = self._failure(_macos_leaks_error(9443, 1, 48, "0x953074d20"))
        title = title_for(f)
        assert title == "[MEMORY-LEAK] 1 leak for 48 total leaked bytes"

    def test_title_omits_the_test_file(self) -> None:
        """A leak in shared code is reported after whichever test file exposed
        it, and the fingerprint keys on the leak site rather than the file. If
        the title carried the file, one issue's title would be rewritten each
        time the same leak surfaced elsewhere."""
        site = "<malloc in sdsnewlen 0x953074d20>"
        under_multi = UniqueFailure(
            test_name="", test_file="tests/unit/multi.tcl",
            failure_type=FailureType.MEMORY_LEAK,
            error=_macos_leaks_error(9443, 1, 48, site),
            jobs=[JobReference(job="test-macos-latest", suite="valkey", url="u")],
        )
        under_expire = UniqueFailure(
            test_name="", test_file="tests/unit/expire.tcl",
            failure_type=FailureType.MEMORY_LEAK,
            error=_macos_leaks_error(7211, 1, 48, site),
            jobs=[JobReference(job="test-macos-latest", suite="valkey", url="u")],
        )
        # One fingerprint, so one issue; therefore one stable title.
        assert fingerprint_for(under_multi) == fingerprint_for(under_expire)
        assert title_for(under_multi) == title_for(under_expire)
        assert "multi.tcl" not in title_for(under_multi)

    def test_title_stable_across_pids_and_addresses(self) -> None:
        t1 = title_for(self._failure(_macos_leaks_error(9443, 1, 48, "0x953074d20")))
        t2 = title_for(self._failure(_macos_leaks_error(7211, 1, 48, "0x9dd024100")))
        assert t1 == t2
        assert "9443" not in t1
        assert "pid" not in t1.lower()

    def test_titles_distinguish_leak_magnitudes(self) -> None:
        t1 = title_for(self._failure(_macos_leaks_error(9443, 1, 48, "0x953074d20")))
        t2 = title_for(self._failure(_macos_leaks_error(9443, 12, 4096, "0x953074d20")))
        assert t1 != t2

    def test_unsymbolicated_leaks_in_one_file_share_a_fingerprint(self) -> None:
        """Documents a known granularity limit: with bare-address ROOT LEAK
        lines (no symbol names), the blob carries no allocation-site signal,
        so two different leaks in the same test file collapse into one issue
        (see fingerprint_for). Symbolicated roots stay distinct via the
        root-site anchor in normalize_error_identity."""
        f1 = self._failure(_macos_leaks_error(9443, 1, 48, "0x953074d20"))
        f2 = self._failure(_macos_leaks_error(9443, 12, 4096, "0x9dd024100"))
        assert fingerprint_for(f1) == fingerprint_for(f2)

    def test_symbolicated_leaks_in_one_file_stay_distinct(self) -> None:
        f1 = self._failure(
            _macos_leaks_error(9443, 1, 48, "<malloc in sdsnewlen 0x953074d20>")
        )
        f2 = self._failure(
            _macos_leaks_error(9443, 1, 48, "<malloc in clusterInit 0x9dd024100>")
        )
        assert fingerprint_for(f1) != fingerprint_for(f2)
