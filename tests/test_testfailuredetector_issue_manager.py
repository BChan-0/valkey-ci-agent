"""Tests for test-failure issue creation/update (mocked GitHub API)."""

from __future__ import annotations

import re
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
    from scripts.test_failure_detector.manage_issues import process_failures
    from scripts.test_failure_detector.parse_failures import (
        FailureType,
        JobReference,
        UniqueFailure,
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
        ]

        failures = [
            _make_failure(test_name="a"),
            _make_failure(test_name="b"),
            _make_failure(test_name="c"),
        ]
        result = process_failures(MagicMock(), "valkey-io/valkey", failures)

        assert result == {"created": 1, "updated": 1, "skipped": 1, "errors": 0}

    @patch("scripts.test_failure_detector.manage_issues.IssueDedupPublisher")
    def test_one_failing_upsert_does_not_abort_the_batch(self, mock_publisher_cls) -> None:
        """A raised exception on one failure is counted as an error and skipped;
        the failures after it are still processed."""
        publisher = mock_publisher_cls.return_value
        publisher.upsert.side_effect = [
            ("created", "https://x/issues/1"),
            RuntimeError("boom"),  # failure b — must NOT kill the loop
            ("updated", "https://x/issues/3"),
        ]

        failures = [
            _make_failure(test_name="a"),
            _make_failure(test_name="b"),
            _make_failure(test_name="c"),
        ]
        result = process_failures(MagicMock(), "valkey-io/valkey", failures)

        assert result == {"created": 1, "updated": 1, "skipped": 0, "errors": 1}
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

        assert result == {"created": 1, "updated": 0, "skipped": 0, "errors": 1}

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

    def test_render_callable_produces_labelled_content(self) -> None:
        content = renderer_for(_make_failure()).render("<!-- m -->", 1)
        assert content.labels == ("test-failure",)
        assert content.title.startswith("[TEST-FAILURE]")


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
        f = UniqueFailure(
            test_name="", test_file="tests/unit/cluster.tcl",
            failure_type=FailureType.STARTUP,
            error="Can't start /path/to/valkey-server",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        title = title_for(f)
        assert title.startswith("[STARTUP-FAILURE]")
        assert "cluster.tcl" in title

    def test_type_specific_labels(self) -> None:
        cases = [
            (FailureType.ASSERTION, "test-failure"),
            (FailureType.SANITIZER, "sanitizer-error"),
            (FailureType.VALGRIND, "valgrind-error"),
            (FailureType.TIMEOUT, "test-timeout"),
            (FailureType.STARTUP, "startup-failure"),
            (FailureType.EXCEPTION, "test-exception"),
            (FailureType.MEMORY_LEAK, "memory-leak"),
            (FailureType.UNITTEST, "unittest-failure"),
        ]
        for ftype, expected_label in cases:
            f = UniqueFailure(
                test_name="t" if ftype in (FailureType.ASSERTION, FailureType.TIMEOUT, FailureType.UNITTEST) else "",
                test_file="f.tcl",
                failure_type=ftype,
                error="some error",
                jobs=[JobReference(job="j", suite="s", url="u")],
            )
            assert label_for(f) == expected_label

    def test_renderer_uses_type_specific_label(self) -> None:
        f = UniqueFailure(
            test_name="DictTest.Ops",
            test_file="src/unit/valkey-unit-gtests",
            failure_type=FailureType.UNITTEST,
            error="gtest FAIL",
            jobs=[JobReference(job="j", suite="s", url="u")],
        )
        content = renderer_for(f).render("<!-- m -->", 1)
        assert content.labels == ("unittest-failure",)

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
