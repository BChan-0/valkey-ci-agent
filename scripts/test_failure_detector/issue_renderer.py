"""Render detected test failures into GitHub issue title, body, and comment text.

The rendering is test-failure-specific (test name/file, error trace, the list
of CI jobs the failure appeared in); the create-or-update machinery lives in
:mod:`scripts.common.issue_dedup`.

A test failure's identity is the ``test_name`` + ``test_file`` pair, which is
the dedup fingerprint. Across recurrences we accumulate the set of failing
environments (CI jobs) into the issue body; because the dedup publisher's
``render`` callback can't see the previously published body, that merge is done
via the publisher's ``body_transform`` hook (see
:meth:`_FailureRenderer.merge_environments`).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from scripts.common.incidents import compute_fingerprint
from scripts.common.issue_dedup import IssueContent
from scripts.test_failure_detector.parse_failures import UniqueFailure

MARKER_NAMESPACE = "valkey-ci-agent:test-failure"

_LABEL_NAME = "test-failure"


def fingerprint_for(failure: UniqueFailure) -> str:
    """Stable dedup key for a failure: a hash of test name + file.

    The identity is ``test_name`` + ``test_file``, hashed via
    :func:`scripts.common.incidents.compute_fingerprint` like the fuzzer
    pipeline, into a fixed-shape hex token that is safe to embed in the marker
    and the search query.

    The pair is the identity, so it goes in ``namespace`` (joined in order,
    never normalized) rather than ``shapes``. That keeps digits significant so
    PSYNC2 and PSYNC3 stay distinct, and preserves order so a name/file swap
    cannot collide.
    """
    return compute_fingerprint(
        namespace=(MARKER_NAMESPACE, failure.test_name, failure.test_file),
        shapes=(),
    )


def title_for(failure: UniqueFailure) -> str:
    """Issue title for a failure.

    Exposed rather than inlined in the renderer so callers can pass the same
    title to ``IssueDedupPublisher.upsert`` as ``title_fallback`` when
    migrating issues off the old raw-fingerprint marker.
    """
    return _build_title(failure)


def renderer_for(failure: UniqueFailure) -> _FailureRenderer:
    """Return a renderer supplying the ``render`` and ``body_transform`` hooks
    that :class:`IssueDedupPublisher.upsert` expects for one failure.

    The two hooks are coupled so the recurrence comment can name the *newly*
    failing environments. ``upsert`` runs ``body_transform`` (which diffs the
    failure's environments against the previously published body) before
    ``render`` (which builds the comment), so by the time the comment is
    rendered the renderer already knows which environments were not recorded
    before. See :meth:`_FailureRenderer.merge_environments`.
    """
    return _FailureRenderer(failure)


class _FailureRenderer:
    """Per-failure ``render``/``body_transform`` pair sharing the set of newly
    failing environments. Created via :func:`renderer_for`."""

    def __init__(self, failure: UniqueFailure) -> None:
        self._failure = failure
        # Environments failing for the first time on this run, populated by
        # ``merge_environments`` on the update path. Empty on the create path
        # (no prior body to diff, and no comment is posted there anyway).
        self._newly_failing: list[str] = []

    def render(self, marker: str, occurrences: int) -> IssueContent:
        """The ``render`` callback: title/body/comment/labels for the issue."""
        return IssueContent(
            title=title_for(self._failure),
            body=_build_body(self._failure, marker, occurrences=occurrences),
            comment=_build_comment(self._failure, newly_failing=self._newly_failing),
            labels=(_LABEL_NAME,),
        )

    def merge_environments(self, existing_body: str) -> str:
        """The ``body_transform`` callback: fold this failure's environments
        into the existing issue body, preserving environments recorded by
        earlier runs and recording which ones are newly failing so
        :meth:`render` can call them out in the recurrence comment.
        """
        existing_envs = _extract_environments_from_body(existing_body)
        self._newly_failing = [
            j.job for j in self._failure.jobs if j.job not in existing_envs
        ]
        if not self._newly_failing:
            return existing_body
        return _update_environments_in_body(
            existing_body, existing_envs + self._newly_failing,
        )


def _build_title(failure: UniqueFailure) -> str:
    return f"[TEST-FAILURE] {failure.test_name} in {failure.test_file}"


def _build_body(failure: UniqueFailure, marker: str, *, occurrences: int) -> str:
    """Build the issue body for a test failure."""
    ci_links = "\n".join(
        f"- `{j.job}`: [CI link]({j.url})" for j in failure.jobs
    )
    env_list = ", ".join(f"`{j.job}`" for j in failure.jobs)

    return "\n".join([
        marker,
        f"<!-- {MARKER_NAMESPACE}:occurrences:{occurrences} -->",
        "",
        "**Summary**",
        "",
        f"`{failure.test_name}` in `{failure.test_file}` is failing in CI.",
        "",
        "**Failing test(s)**",
        "",
        f"- Test name: `{failure.test_name}`",
        f"- Test file: `{failure.test_file}`",
        "- CI link(s):",
        ci_links,
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


def _build_comment(failure: UniqueFailure, *, newly_failing: list[str]) -> str:
    """Build a comment for an existing issue that failed again.

    When ``newly_failing`` names environments not recorded on the issue before,
    the comment calls them out so a triager can spot a regression spreading to
    new platforms without diffing the body's Environments line.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ci_links = "\n".join(
        f"- `{j.job}`: [CI link]({j.url})" for j in failure.jobs
    )
    lines = [f"Test failed again on {today}."]
    if newly_failing:
        new_envs = ", ".join(f"`{e}`" for e in newly_failing)
        lines.append(f"\n**Newly failing in:** {new_envs}")
    lines.append(f"\n**Failed in:**\n{ci_links}")
    return "\n".join(lines)


def _extract_environments_from_body(body: str) -> list[str]:
    """Extract existing environment names from an issue body."""
    env_match = re.search(r"\*\*Environments:\*\*\s*(.+)", body)
    if not env_match:
        return []
    return re.findall(r"`([^`]+)`", env_match.group(1))


def _update_environments_in_body(body: str, all_envs: list[str]) -> str:
    """Replace the Environments line in the issue body with an updated list."""
    new_env_line = f"**Environments:** {', '.join(f'`{e}`' for e in all_envs)}"
    return re.sub(r"\*\*Environments:\*\*\s*.+", new_env_line, body)
