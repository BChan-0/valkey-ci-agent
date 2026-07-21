"""Render detected test failures into GitHub issue title, body, and comment text.

Supports multiple failure types (assertion, sanitizer, valgrind, timeout,
exception, startup, memory-leak, unittest) with type-specific titles, labels,
and fingerprint namespaces. The create-or-update machinery lives in
:mod:`scripts.common.issue_dedup`.

For failures WITH a test_name (assertions, timeouts, gtest), the identity is
the (type, test_name, test_file) triple. For failures WITHOUT a test_name
(sanitizer/valgrind/startup), the identity is (type, normalized_error), so the
same underlying bug produces one issue regardless of which test file triggered
the detection.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from scripts.common.incidents import compute_fingerprint
from scripts.common.issue_dedup import IssueContent
from scripts.test_failure_detector.parse_failures import (
    FailureType,
    UniqueFailure,
    normalize_error_identity,
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

    For failures without a test_name (sanitizer/valgrind/startup): hash of
    (type_namespace,) with the normalized error as shapes input. This means the
    same bug detected after different test files produces the same fingerprint.
    """
    ns = marker_namespace_for(failure)

    if failure.has_test_identity:
        return compute_fingerprint(
            namespace=(ns, failure.test_name, failure.test_file),
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
        new_error = self._failure.error
        if not new_error.strip():
            return None
        stored = _extract_error_from_body(existing_body)
        if not stored.strip():
            return None
        if _normalize_trace(stored) == _normalize_trace(new_error):
            return None
        return new_error


def _error_summary_line(error: str) -> str:
    """Extract a short (<=60 char) summary from an error for the title."""
    clean = re.sub(r"\033\[[0-9;]*m", "", error)
    clean = re.sub(r"==\d+==\s*", "", clean)
    for line in clean.split("\n"):
        line = line.strip()
        if line and not line.startswith("at ") and len(line) > 5:
            return line[:60]
    return clean[:60] if clean.strip() else "unknown error"


def _build_title(failure: UniqueFailure) -> str:
    prefix = _TYPE_TITLE_PREFIX.get(failure.failure_type, "[TEST-FAILURE]")
    if failure.has_test_identity:
        return f"{prefix} {failure.test_name} in {failure.test_file}"
    elif failure.test_file:
        summary = _error_summary_line(failure.error)
        return f"{prefix} {summary} ({failure.test_file})"
    else:
        summary = _error_summary_line(failure.error)
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

    lines.extend([
        "",
        "**Error stack trace**",
        "",
        "```",
        failure.error or "N/A",
        "```",
        "",
        f"**Environments:** {env_list}",
        "",
        "---",
        "*Auto-created by Test Failure Detector*",
    ])
    return "\n".join(lines)


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
        lines.append(f"\n**New error stack trace**\n\n```\n{new_error}\n```")
    lines.append(f"\n**Failed in:**\n{ci_links}")
    return "\n".join(lines)


def _extract_environments_from_body(body: str) -> list[str]:
    """Extract existing environment names from an issue body."""
    env_match = re.search(r"\*\*Environments:\*\*\s*(.+)", body)
    if not env_match:
        return []
    return re.findall(r"`([^`]+)`", env_match.group(1))


_ERROR_BLOCK_RE = re.compile(
    r"\*\*Error stack trace\*\*\s*```\n(.*?)\n```",
    re.DOTALL,
)


def _extract_error_from_body(body: str) -> str:
    """Extract the error trace recorded under the Error stack trace header."""
    match = _ERROR_BLOCK_RE.search(body)
    if not match:
        return ""
    return match.group(1).strip()


_TRACE_NOISE_RES = (
    re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),
    re.compile(r"0x[0-9a-fA-F]+"),
    re.compile(r"/tmp/[^\s:]+"),
    re.compile(r"\b(pid|port)[=\s]+\d+", re.IGNORECASE),
)


def _normalize_trace(text: str) -> str:
    """Normalize a trace for comparison by scrubbing run-specific noise."""
    for noise in _TRACE_NOISE_RES:
        text = noise.sub("", text)
    return " ".join(text.split())


def _update_environments_in_body(body: str, all_envs: list[str]) -> str:
    """Replace the Environments line in the issue body with an updated list."""
    new_env_line = f"**Environments:** {', '.join(f'`{e}`' for e in all_envs)}"
    return re.sub(r"\*\*Environments:\*\*\s*.+", new_env_line, body)
