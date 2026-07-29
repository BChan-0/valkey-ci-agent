"""Render detected test failures into GitHub issue title, body, and comment text.

Supports multiple failure types (assertion, sanitizer, valgrind, timeout,
exception, startup, memory-leak, unittest) with type-specific titles, labels,
and fingerprint namespaces. The create-or-update machinery lives in
:mod:`scripts.common.issue_dedup`.

For failures WITH a test_name (assertions, timeouts, gtest), the identity is
the (type, test_name, test_file) triple. For failures WITHOUT a test_name
(sanitizer/valgrind/startup), the identity is (type, normalized_error), so the
same underlying bug produces one issue regardless of which test file triggered
the detection. Titles follow the identity: the test file appears only for types
whose fingerprint keys on it, so a title is not rewritten when the same bug is
detected under a different file.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from scripts.common.incidents import compute_fingerprint
from scripts.common.issue_dedup import IssueContent
from scripts.test_failure_detector.parse_failures import (
    FailureType,
    UniqueFailure,
    is_plumbing_frame,
    normalize_error_identity,
    scrub_volatile_tokens,
    startup_reason_from_lines,
)

MARKER_NAMESPACE = "valkey-ci-agent:test-failure"

LABEL_NAME = "test-failure"

# Type-specific marker namespaces for fingerprinting and issue search.
_TYPE_NAMESPACE: dict[FailureType, str] = {
    FailureType.ASSERTION: "valkey-ci-agent:test-failure",
    FailureType.SANITIZER: "valkey-ci-agent:sanitizer-error",
    FailureType.VALGRIND: "valkey-ci-agent:valgrind-error",
    FailureType.TIMEOUT: "valkey-ci-agent:test-timeout",
    FailureType.STARTUP: "valkey-ci-agent:startup-failure",
    FailureType.EXCEPTION: "valkey-ci-agent:test-exception",
    FailureType.MEMORY_LEAK: "valkey-ci-agent:memory-leak",
    FailureType.UNITTEST: "valkey-ci-agent:unittest-failure",
}

_LABEL_NAME = "test-failure"

# Title prefix per failure type.
_TYPE_TITLE_PREFIX: dict[FailureType, str] = {
    FailureType.ASSERTION: "[TEST-FAILURE]",
    FailureType.SANITIZER: "[SANITIZER]",
    FailureType.VALGRIND: "[VALGRIND]",
    FailureType.TIMEOUT: "[TIMEOUT]",
    FailureType.STARTUP: "[STARTUP-FAILURE]",
    FailureType.EXCEPTION: "[EXCEPTION]",
    FailureType.MEMORY_LEAK: "[MEMORY-LEAK]",
    FailureType.UNITTEST: "[UNITTEST]",
}


def marker_namespace_for(failure: UniqueFailure) -> str:
    """Return the marker namespace for a failure's type."""
    return _TYPE_NAMESPACE.get(failure.failure_type, MARKER_NAMESPACE)


def label_for(failure: UniqueFailure) -> str:
    """Return the issue label. All failure types use the same label."""
    return _LABEL_NAME


def fingerprint_for(failure: UniqueFailure) -> str:
    """Stable dedup key for a failure.

    For failures with a test_name: hash of (type_namespace, test_name, test_file).
    The pair goes in ``namespace`` (joined in order, never normalized) rather
    than ``shapes``, which keeps digits significant so PSYNC2 and PSYNC3 stay
    distinct and preserves order so a name/file swap cannot collide.

    For nameless timeouts: hash of (type_namespace, test_file). The error text
    is always a generic "Test timed out" shared by all timeouts, so including
    it would collapse unrelated timeouts in different files into one issue.

    For other nameless failures (sanitizer/valgrind/startup): hash of
    (type_namespace,) with the normalized error as shapes input. This means the
    same bug detected after different test files produces the same fingerprint.

    Known granularity limit: a macOS /usr/bin/leaks blob whose root-leak
    lines are unsymbolicated (bare addresses, no site names) normalizes to
    the boilerplate shared by every leak in that test file, so two such
    distinct leaks collapse into one issue. Symbolicated roots stay distinct
    via the root-site anchor in normalize_error_identity.
    """
    ns = marker_namespace_for(failure)

    if failure.has_test_identity:
        return compute_fingerprint(
            namespace=(ns, failure.test_name, failure.test_file),
            shapes=(),
        )
    elif failure.failure_type == FailureType.TIMEOUT and failure.test_file:
        return compute_fingerprint(
            namespace=(ns, failure.test_file),
            shapes=(),
        )
    else:
        error_identity = normalize_error_identity(failure.error)
        return compute_fingerprint(
            namespace=(ns,),
            shapes=(error_identity,),
        )


def title_for(failure: UniqueFailure) -> str:
    """Issue title for a failure."""
    return _build_title(failure)


def renderer_for(failure: UniqueFailure) -> _FailureRenderer:
    """Return a renderer supplying the ``render`` and ``body_transform`` hooks
    that :class:`IssueDedupPublisher.upsert` expects for one failure.
    """
    return _FailureRenderer(failure)


class _FailureRenderer:
    """Per-failure render/body_transform pair. Created via :func:`renderer_for`."""

    def __init__(self, failure: UniqueFailure) -> None:
        self._failure = failure
        self._newly_failing: list[str] = []
        self._new_error: str | None = None

    def render(self, marker: str, occurrences: int) -> IssueContent:
        """The ``render`` callback: title/body/comment/labels for the issue."""
        return IssueContent(
            title=title_for(self._failure),
            body=_build_body(self._failure, marker, occurrences=occurrences),
            comment=_build_comment(
                self._failure,
                newly_failing=self._newly_failing,
                new_error=self._new_error,
            ),
            labels=(label_for(self._failure),),
        )

    def merge_environments(self, existing_body: str) -> str:
        """The ``body_transform`` callback: fold this failure's environments
        into the existing issue body, preserving environments recorded by
        earlier runs and recording which ones are newly failing.
        """
        self._new_error = self._detect_new_error(existing_body)
        existing_envs = _extract_environments_from_body(existing_body)
        self._newly_failing = [
            j.job for j in self._failure.jobs if j.job not in existing_envs
        ]
        if not self._newly_failing:
            return existing_body
        return _update_environments_in_body(
            existing_body, existing_envs + self._newly_failing,
        )

    def _detect_new_error(self, existing_body: str) -> str | None:
        """Return the failure's error trace when it meaningfully differs from
        what is stored on the issue, else None.
        """
        # The stored trace was truncated when published, so the fresh trace
        # must be compared in its truncated form too, or every recurrence of
        # an oversized trace would register as a new error.
        new_error = _truncate_trace(self._failure.error)
        if not new_error.strip():
            return None
        stored = _extract_error_from_body(existing_body)
        if not stored.strip():
            return None
        if _normalize_trace(stored) == _normalize_trace(new_error):
            return None
        return new_error


# Keywords that mark a line as carrying diagnostic content rather than
# boilerplate. Used by _error_summary_line to prefer the real payload over
# the generic runner prefix/banner.
_TITLE_KEYWORDS = (
    "Invalid", "definitely lost", "indirectly lost",
    "heap-buffer-overflow", "heap-use-after-free",
    "stack-buffer-overflow", "use-after-poison",
    "uninitialized", "runtime error",
    "detected memory leaks", "LEAK SUMMARY",
    "fishy", "overlap", "Mismatched",
)


# Heap-layout coordinates in a diagnostic line ("in loss record 900 of
# 1,109") shift between runs of the same bug. The fingerprint already
# scrubs them; the title must too, or each recurrence rewrites the title
# of the same issue.
_LOSS_RECORD_RE = re.compile(r"\s*\bin loss record \d[\d,]* of \d[\d,]*")

# Allocation sizes drift run to run for the same leak ("49 bytes" vs
# "52 bytes"), so titles show them as N: "N bytes in N blocks are
# definitely lost".
_COUNT_RE = re.compile(r"\b\d[\d,]*(\s+(?:bytes?|blocks?|byte\(s\)|object\(s\)))\b")

# An AddressSanitizer diagnostic line ends in a volatile address dump
# ("heap-use-after-free on address 0x60... at pc 0x... bp 0x... sp 0x..."). The
# address and registers change every run of the same bug; the fingerprint
# scrubs them, so the title must too or it is rewritten on each recurrence.
_ASAN_ADDR_NOISE_RE = re.compile(r"\s+on address 0x[0-9a-fA-F]+.*$")

# Volatile run-specific tokens in generic title candidates: ports, PIDs, hex
# addresses, temp paths, and long bare numbers. The fingerprint scrubs these
# for identity, so one recurring failure keeps one issue; the title must scrub
# them too, or the publisher rewrites the title with the new port/PID on every
# recurrence. Leak titles are exempt: their sizes are triage signal and
# refreshing them is intentional.
_TITLE_VOLATILE_SUBS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"0x[0-9a-fA-F]+"), "0xN"),
    (re.compile(r"/tmp/[^\s:]+"), "/tmp/..."),
    (re.compile(r"\b(pid|port)([=\s]+)\d+", re.IGNORECASE), r"\1\2N"),
    (re.compile(r"\b\d{4,}\b"), "N"),
)


def _scrub_volatile_title_tokens(line: str) -> str:
    for pattern, repl in _TITLE_VOLATILE_SUBS:
        line = pattern.sub(repl, line)
    return line

# Valgrind stack frame naming a source location: "by 0x1E8076: debugCommand
# (debug.c:569)". Frames in tool preload libraries (malloc interceptors)
# name no source file:line, so they never match.
_SOURCE_FRAME_RE = re.compile(
    r"^\s*(?:at|by)\s+0x[0-9a-fA-F]+:\s*(?P<func>\S+)\s+\((?P<file>[^():]+):(?P<line>\d+)\)"
)

# Sanitizer stack frame naming a source location: "#4 0x55ba... in debugCommand
# /home/runner/.../src/debug.c:569:9". The malloc interceptor frame carries a
# parenthesized binary offset ("in malloc (.../valkey-server+0x20de33)") rather
# than a file:line, so it never matches.
_SAN_SOURCE_FRAME_RE = re.compile(
    r"^#\d+\s+0x[0-9a-fA-F]+\s+in\s+(?P<func>\S+)\s+(?P<file>\S+?):(?P<line>\d+)(?::\d+)?\b"
)

# A valgrind leak record: "49 bytes in 1 blocks are definitely lost ...".
# The title reformats it as "Definitely lost: 49 bytes in <site>". Leak titles
# keep their real byte counts even though the fingerprint scrubs them as
# volatile: maintainers triage leaks by magnitude, and the publisher refreshes
# the title on each recurrence, so drift keeps it current.
_LEAK_RECORD_RE = re.compile(
    r"(?P<size>\d[\d,]*\s+bytes?)\s+in\s+\d[\d,]*\s+blocks?\s+are\s+"
    r"(?P<kind>definitely|indirectly|possibly)\s+lost"
)

# The AddressSanitizer/LeakSanitizer summary line: "SUMMARY: AddressSanitizer:
# 41 byte(s) leaked in 1 allocation(s)." The banner ("detected memory leaks")
# names no magnitude; this line does.
_SANITIZER_LEAK_RE = re.compile(
    r"(?P<size>\d[\d,]*\s+byte\(s\))\s+leaked\s+in\s+\d[\d,]*\s+allocation\(s\)"
)

# The totals line of a macOS /usr/bin/leaks report: "Process 9443: 1 leak for
# 48 total leaked bytes." This is the memory-leak type's payload, and the type
# does not overlap valgrind/sanitizer: it is the only leak detector on the
# macos jobs (valgrind has no Apple Silicon port; the CI matrix builds ASan
# only on Linux), and it inspects the live server after each test file rather
# than at exit, so it catches leaks the Linux leak jobs miss.
_LEAKS_TOTAL_RE = re.compile(
    r"Process\s+\d+:\s*"
    r"(?P<phrase>\d[\d,]*\s+leaks?\s+for\s+\d[\d,]*\s+total\s+leaked\s+bytes)"
)

def _leak_site(error: str) -> str:
    """Distinctive "func (file:line)" in the report's first stack, or "".

    Skips shared allocation and sanitizer-runtime plumbing via the same
    predicate the identity uses, so the site names the code path that leaked
    (debugCommand (debug.c:569)) rather than the allocator every leak passes
    through. Falls back to the first source frame when the whole stack is
    plumbing, so a title is never left empty.
    """
    first_source_site = ""
    for line in error.split("\n"):
        line = re.sub(r"==\d+==\s*", "", line).strip()
        match = _SOURCE_FRAME_RE.match(line) or _SAN_SOURCE_FRAME_RE.match(line)
        if not match:
            continue
        source_path = match.group("file")
        source_file = source_path.rsplit("/", 1)[-1]
        func = match.group("func")
        site = f"{func} ({source_file}:{match.group('line')})"
        if not first_source_site:
            first_source_site = site
        if is_plumbing_frame(func, source_path):
            continue
        return site
    return first_source_site


# A startup blob: "Can't start <exe>\nCONFIGURATION:\n<config>\nERROR:\n
# <reason>". The exe path is identical for every startup failure; the reason
# is what tells two causes apart, so the title must carry it. Progress and
# banner lines before the reason are skipped, mirroring the identity
# extraction in normalize_error_identity.
_STARTUP_REASON_RE = re.compile(r"\nERROR:\n(?P<tail>.+)", re.DOTALL)


def _startup_reason(error: str) -> str:
    """The fatal reason after the startup blob's ERROR: header, or ""."""
    match = _STARTUP_REASON_RE.search(error)
    if not match:
        return ""
    tail = _scrub_volatile_title_tokens(match.group("tail"))
    return startup_reason_from_lines([line.strip() for line in tail.split("\n")])


def _error_summary_line(error: str) -> str:
    """Extract a short summary from an error for the title.

    Strips ANSI codes and valgrind PID annotations, then drops the runner's
    wrapper prefix ("Valgrind error: ...", "Sanitizer error: ...") which is
    always the same across issues. Prefers lines containing diagnostic
    keywords over generic banners so different bugs get distinct titles,
    and names the first user-code stack frame so two bugs with the same
    diagnostic line stay tellable apart in an issue list.
    """
    # The runner emits the message behind a "[err]: " status tag whose removal
    # leaves a leading space, so match on the stripped text rather than the raw
    # field; otherwise a startup blob falls through to generic truncation and
    # the title becomes the runner's absolute exe path cut mid-word.
    error = error.strip()

    if error.startswith("Can't start"):
        reason = _startup_reason(error)
        if reason:
            return f"Can't start server: {reason}"[:80]

    # A macOS leaks report's first line is the Tcl test name with a volatile
    # PID ("Check for memory leaks (pid 9443) in ..."); the payload is the
    # totals line.
    leaks_total = _LEAKS_TOTAL_RE.search(error)
    if leaks_total:
        return leaks_total.group("phrase")[:60]

    # Lead with the size, then the leaking code path, so two sanitizer leaks
    # stay distinct in a title list.
    sanitizer_leak = _SANITIZER_LEAK_RE.search(error)
    if sanitizer_leak:
        summary = f"Leaked {sanitizer_leak.group('size')}"
        site = _leak_site(error)
        if site:
            summary = f"{summary} in {site}"
        return summary[:80]

    clean = re.sub(r"\033\[[0-9;]*m", "", error)
    clean = re.sub(r"==\d+==\s*", "", clean)
    # The test runner prepends a wrapper on the first line: "Valgrind error: ...",
    # "Sanitizer error: ...", or "Executing test client: <message>" for an
    # uncaught exception. Strip it so the summary comes from the actual message,
    # not the wrapper.
    clean = re.sub(
        r"^\s*(?:Valgrind\s+error:|Sanitizer\s+error:|Executing\s+test\s+client:)\s*",
        "", clean, count=1,
    )

    candidates = []
    for line in clean.split("\n"):
        line = line.strip()
        # Drop a leading "ERROR:" severity tag so the tool name behind it
        # ("LeakSanitizer: ...") leads the title instead of the empty tag.
        line = re.sub(r"^ERROR:\s*", "", line)
        if line and not line.startswith("at ") and len(line) > 5:
            candidates.append(line)

    # Prefer a line carrying a diagnostic keyword over the first non-empty
    # line (which is often a generic banner like "Memcheck, a memory error
    # detector" that every valgrind issue would share).
    for line in candidates:
        if any(kw in line for kw in _TITLE_KEYWORDS):
            leak = _LEAK_RECORD_RE.search(line)
            if leak:
                # Lead with the leak kind, then size and site: "Definitely
                # lost: 49 bytes in debugCommand (debug.c:569)".
                summary = f"{leak.group('kind').capitalize()} lost: {leak.group('size')}"
                site = _leak_site(error)
                if site:
                    summary = f"{summary} in {site}"
                # 80 instead of the generic 60: the site is the
                # distinguishing token and must survive truncation.
                return summary[:80]
            summary = _LOSS_RECORD_RE.sub("", line)
            summary = _COUNT_RE.sub(r"N\1", summary)
            summary = _ASAN_ADDR_NOISE_RE.sub("", summary)
            summary = _scrub_volatile_title_tokens(summary)
            site = _leak_site(error)
            if site and site not in summary:
                summary = f"{summary} in {site}"
            return summary[:60]

    if candidates:
        return _scrub_volatile_title_tokens(candidates[0])[:60]
    stripped = clean.strip()
    return _scrub_volatile_title_tokens(stripped)[:60] if stripped else "unknown error"


# Nameless failure types whose fingerprint keys on test_file, so the file is
# stable identity and belongs in the title. Only TIMEOUT qualifies: it keys on
# the file directly (see fingerprint_for). Every other nameless type keys on the
# normalized error, which discards the file, so putting the file in the title
# would rewrite one issue's title whenever the same bug surfaced under a
# different file. Memory leaks are the case that makes this concrete: one leak in
# shared code is reported after whichever test file happened to expose it.
_TITLE_SHOWS_FILE = frozenset({FailureType.TIMEOUT})


def _build_title(failure: UniqueFailure) -> str:
    prefix = _TYPE_TITLE_PREFIX.get(failure.failure_type, "[TEST-FAILURE]")
    if failure.has_test_identity:
        return f"{prefix} {failure.test_name} in {failure.test_file}"
    summary = _error_summary_line(failure.error)
    if failure.test_file and failure.failure_type in _TITLE_SHOWS_FILE:
        return f"{prefix} {summary} in {failure.test_file}"
    return f"{prefix} {summary}"


def _build_body(failure: UniqueFailure, marker: str, *, occurrences: int) -> str:
    """Build the issue body for a test failure."""
    ns = marker_namespace_for(failure)
    ci_links = "\n".join(
        f"- `{j.job}`: [CI link]({j.url})" for j in failure.jobs
    )
    env_list = ", ".join(f"`{j.job}`" for j in failure.jobs)
    type_label = failure.failure_type.value.replace("-", " ").title()

    lines = [
        marker,
        f"<!-- {ns}:occurrences:{occurrences} -->",
        "",
        "**Summary**",
        "",
    ]

    if failure.has_test_identity:
        lines.append(
            f"`{failure.test_name}` in `{failure.test_file}` is failing in CI."
        )
        lines.extend([
            "",
            "**Failing test(s)**",
            "",
            f"- Test name: `{failure.test_name}`",
            f"- Test file: `{failure.test_file}`",
            f"- Failure type: `{type_label}`",
            "- CI link(s):",
            ci_links,
        ])
    else:
        lines.append(f"A **{type_label}** error was detected in CI.")
        if failure.test_file:
            lines.append(f"Context: running `{failure.test_file}`")
        lines.extend([
            "",
            "**Error details**",
            "",
            f"- Failure type: `{type_label}`",
        ])
        if failure.test_file:
            lines.append(f"- Test file context: `{failure.test_file}`")
        lines.extend([
            "- CI link(s):",
            ci_links,
        ])

    error_text = _truncate_trace(failure.error) or "N/A"
    fence = _fence_for(error_text)
    lines.extend([
        "",
        "**Error stack trace**",
        "",
        fence,
        error_text,
        fence,
        "",
        f"**Environments:** {env_list}",
        "",
        "---",
        "*Auto-created by Test Failure Detector*",
    ])
    return "\n".join(lines)


def _fence_for(text: str) -> str:
    """A code fence longer than any backtick run in *text*.

    A fence closes only on a run at least as long as its opener, so an error
    that itself contains ``` must be wrapped in a longer fence or it would
    close the block early (and the round-trip in _extract_error_from_body
    would return a truncated trace, triggering a spurious new-error comment
    on every recurrence).
    """
    longest = max((len(m.group()) for m in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


# GitHub rejects issue bodies and comments over 65536 characters. A full
# valgrind or sanitizer log can be several times that; the create call would
# 422 and the failure would never get an issue. The cap leaves ample room for
# the surrounding body (markers, links, environments). Head and tail are both
# kept: the head names the error, the tail holds the summary totals.
_MAX_TRACE_CHARS = 40_000

_TRUNCATION_NOTICE = "\n... [trace truncated by Test Failure Detector] ...\n"


def _truncate_trace(trace: str) -> str:
    """Cap a trace to fit a GitHub issue body, keeping its head and tail."""
    if len(trace) <= _MAX_TRACE_CHARS:
        return trace
    keep = (_MAX_TRACE_CHARS - len(_TRUNCATION_NOTICE)) // 2
    return f"{trace[:keep]}{_TRUNCATION_NOTICE}{trace[-keep:]}"


def _build_comment(
    failure: UniqueFailure,
    *,
    newly_failing: list[str],
    new_error: str | None = None,
) -> str:
    """Build a comment for an existing issue that failed again."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ci_links = "\n".join(
        f"- `{j.job}`: [CI link]({j.url})" for j in failure.jobs
    )
    lines = [f"Test failed again on {today}."]
    if newly_failing:
        new_envs = ", ".join(f"`{e}`" for e in newly_failing)
        lines.append(f"\n**Newly failing in:** {new_envs}")
    if new_error:
        new_error = _truncate_trace(new_error)
        fence = _fence_for(new_error)
        lines.append(f"\n**New error stack trace**\n\n{fence}\n{new_error}\n{fence}")
    lines.append(f"\n**Failed in:**\n{ci_links}")
    return "\n".join(lines)


def _extract_environments_from_body(body: str) -> list[str]:
    """Extract existing environment names from an issue body."""
    env_match = re.search(r"\*\*Environments:\*\*\s*(.+)", body)
    if not env_match:
        return []
    return re.findall(r"`([^`]+)`", env_match.group(1))


# The fence length varies (see _fence_for); the backreference requires the
# closer to be the same run that opened the block, so an embedded shorter
# backtick run inside the error does not end the match early.
_ERROR_BLOCK_RE = re.compile(
    r"\*\*Error stack trace\*\*\s*(`{3,})\n(.*?)\n\1",
    re.DOTALL,
)


def _extract_error_from_body(body: str) -> str:
    """Extract the error trace recorded under the Error stack trace header."""
    match = _ERROR_BLOCK_RE.search(body)
    if not match:
        return ""
    return match.group(2).strip()


# Must scrub at least everything the fingerprint scrubs: two traces the
# fingerprint calls the same bug must compare equal here, or every recurrence
# posts a spurious "new error stack trace" comment. The shared scrub covers
# the identity's tokens (PID markers, addresses, loss records, byte/block
# counts); these add the timestamps a trace can carry that an identity's few
# significant lines never reach.
_TRACE_NOISE_RES = (
    re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),
    # A sanitizer frame's build hash ("(BuildId: 2bf960fb...)"). It changes
    # whenever the binary is recompiled, so it differs between two runs of one
    # bug. Only the trace carries it; a frame reaches the identity as a
    # function name, without this suffix.
    re.compile(r"\s*\(BuildId:\s*[0-9a-fA-F]+\)"),
    # The macOS leaks report's process footprint ("Physical footprint: 2801K").
    # It measures the live server when the report was taken, not the leak.
    re.compile(r"(Physical footprint(?:\s*\(peak\))?:\s*)\d+K"),
)


def _normalize_trace(text: str) -> str:
    """Normalize a trace for comparison by scrubbing run-specific noise."""
    for noise in _TRACE_NOISE_RES:
        text = noise.sub("", text)
    return " ".join(scrub_volatile_tokens(text).split())


def _update_environments_in_body(body: str, all_envs: list[str]) -> str:
    """Replace the Environments line in the issue body with an updated list."""
    new_env_line = f"**Environments:** {', '.join(f'`{e}`' for e in all_envs)}"
    return re.sub(r"\*\*Environments:\*\*\s*.+", new_env_line, body)
