"""Render the analysis into an issue comment and post it.

The comment is the only thing this workflow writes. Everything the model
produced passes through here, so this module owns the two safety properties that
matter for text going onto a public issue.

Marker-shaped text is defused. The detector deduplicates its issues on HTML
comment markers, so a literal marker reaching an issue would corrupt the dedup
for that issue permanently. Every marker opener in model output and in log text
is made inert.

Fences are sized to their content. A log excerpt can itself contain backtick
runs, and a fence closes on the first run at least as long as its opener, so the
fence is grown past the longest run inside it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from scripts.common.github_client import retry_github_call
from scripts.failure_triage.discover import triage_marker
from scripts.failure_triage.models import TriageAnalysis, TriageTarget

logger = logging.getLogger(__name__)

# GitHub rejects a comment over 65536 characters. The per-field caps in the
# analyzer keep a normal comment well under this, so _clamp is a backstop against
# an unforeseen large field rather than an expected path.
_MAX_COMMENT_CHARS = 60_000

# Longest fence written. A fence is emitted twice per block, so an unbounded one
# could outgrow the comment on its own. When content holds a backtick run that
# would need a longer fence, _fenced replaces the runs instead of clamping the
# fence short, so the block always closes.
_MAX_FENCE_CHARS = 64

_HTML_COMMENT_OPEN_RE = re.compile(r"<!--")

# How the failure class reads in the comment.
_CLASS_LABELS = {
    "product-bug": "Likely a defect in Valkey",
    "test-bug": "Likely a problem with the test",
    "flaky-test": "Likely flaky (timing or ordering dependent)",
    "infrastructure": "Likely an infrastructure or environment problem",
    "undetermined": "Undetermined",
}


def post(
    gh: Any,
    repo_full_name: str,
    target: TriageTarget,
    analysis: TriageAnalysis,
    *,
    triage_run_url: str = "",
) -> str:
    """Post the analysis on the target's issue and return the comment URL.

    Raises on a failure to post so the caller records the issue as errored; the
    caller isolates each target, so one failed post does not stop the batch.
    """
    body = render(target, analysis, triage_run_url=triage_run_url)
    issue = retry_github_call(
        lambda: gh.get_repo(repo_full_name).get_issue(target.issue_number),
        retries=2, description=f"get issue #{target.issue_number}",
    )
    comment = retry_github_call(
        lambda: issue.create_comment(body=body),
        retries=2, description=f"comment on issue #{target.issue_number}",
    )
    logger.info(
        "Posted the analysis on issue #%s: %s", target.issue_number, comment.html_url,
    )
    return str(comment.html_url)


def render(
    target: TriageTarget,
    analysis: TriageAnalysis,
    *,
    triage_run_url: str = "",
) -> str:
    """Build the comment body for one analysis."""
    lines = [triage_marker(target.run_id), ""]

    if analysis.failed:
        lines.extend(_render_unavailable(analysis))
    else:
        lines.extend(_render_analysis(analysis))

    lines.extend(_render_footer(triage_run_url))
    return _clamp("\n".join(lines))


def _render_analysis(analysis: TriageAnalysis) -> list[str]:
    """The analysis sections, in the order a maintainer reads them."""
    lines = [
        "## AI failure analysis",
        "",
        _block(analysis.summary) if analysis.summary else "_No summary was produced._",
        "",
        f"**Assessment:** {_CLASS_LABELS.get(analysis.failure_class, 'Undetermined')} "
        f"(confidence: {analysis.confidence})",
    ]
    if analysis.category:
        lines.append(f"**Category:** `{_inline(analysis.category)}`")
    lines.append("")

    if analysis.root_cause:
        lines.extend([
            "### Hypothesized root cause",
            "",
            _block(analysis.root_cause),
            "",
        ])

    if analysis.evidence:
        lines.extend(["### Supporting evidence", ""])
        # Each item is flattened to one line so a newline in it cannot escape the
        # list and render as the comment's own markdown.
        lines.extend(f"- {_inline(item)}" for item in analysis.evidence)
        lines.append("")

    if analysis.suspected_area:
        lines.extend([
            f"**Suspected area:** `{_inline(analysis.suspected_area)}`",
            "",
        ])

    if analysis.has_fix_suggestion:
        lines.extend([
            "### Suggested fix",
            "",
            _block(analysis.fix_suggestion),
            "",
            f"_Fix confidence: {analysis.fix_confidence}._",
            "",
        ])

    if analysis.reproduction_hint:
        lines.extend([
            "### Reproduce",
            "",
            _fenced(analysis.reproduction_hint),
            "",
        ])

    return lines


def _render_unavailable(analysis: TriageAnalysis) -> list[str]:
    """The note posted when no analysis could be produced.

    Posted rather than staying silent so the issue records that triage ran and
    got nowhere, which is otherwise indistinguishable from triage never running.
    """
    reason = _inline(analysis.error) if analysis.error else "the reason was not recorded"
    return [
        "## AI failure analysis",
        "",
        f"No analysis could be produced for this failure: {reason}.",
        "",
    ]


def _render_footer(triage_run_url: str) -> list[str]:
    """The provenance and caveat closing every comment.

    The caveat is not decoration. The comment sits among human triage on a public
    issue, so it has to say plainly that it is machine-generated and unverified.
    """
    note = (
        "Generated by valkey-ci-agent. This analysis is AI-generated and may be "
        "incorrect. Verify it against the CI logs before acting on it; it is not "
        "a maintainer's assessment."
    )
    if triage_run_url:
        note += f" [Triage run]({triage_run_url})."
    return ["---", f"_{note}_"]


def _defuse(text: str) -> str:
    """Make marker-shaped comments in *text* inert.

    The detector's dedup reads HTML comment markers out of issue bodies, so text
    that reaches an issue must not be able to introduce one.
    """
    return _HTML_COMMENT_OPEN_RE.sub("<! --", text)


# A line that opens a fenced block or a block-level construct: an indented code
# fence, an ATX heading, or a blockquote. Prefixing the run with a backslash
# keeps the model's free text from opening an unterminated fence that swallows
# the rest of the comment, or injecting a heading that reads as the tool's own.
_BLOCK_OPENER_RE = re.compile(r"(?m)^(\s{0,3})([`~]{3,}|#{1,6}\s|>)")


def _block(text: str) -> str:
    """Neutralize a multi-line free-text field for insertion into the comment.

    Defuses markers and escapes block-level markdown at the start of any line, so
    a field that contains a fence, a heading, or a blockquote cannot break the
    comment's structure or impersonate one of its sections.
    """
    return _BLOCK_OPENER_RE.sub(r"\1\\\2", _defuse(text))


def _inline(text: str) -> str:
    """Flatten text for a backtick span and neutralize any backtick in it.

    A newline would break out of the span, and a backtick would close it early.
    """
    return " ".join(_defuse(text).split()).replace("`", "'")


def _fenced(text: str) -> str:
    """Wrap *text* in a fence longer than any backtick run inside it.

    The fence has a maximum length, so a backtick run that would need a longer
    fence has its runs replaced with single quotes rather than the fence being
    clamped short and left unable to close the block. Reproduction hints are
    short, so this only fires on pathological input.
    """
    body = _defuse(text)
    longest = max((len(match.group()) for match in re.finditer(r"`+", body)), default=0)
    if longest + 1 > _MAX_FENCE_CHARS:
        body = re.sub(r"`+", lambda match: "'" * len(match.group()), body)
        longest = 0
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{body}\n{fence}"


def _clamp(body: str) -> str:
    """Hold the comment under the size GitHub accepts.

    Truncating loses the footer, so the caveat is re-appended: a comment that
    dropped its own disclaimer is worse than one that is visibly cut short. The
    cut lands on a line boundary and any fence the cut left open is closed, so
    the re-appended caveat renders as text rather than being captured inside a
    dangling code block.
    """
    if len(body) <= _MAX_COMMENT_CHARS:
        return body
    notice = (
        "_This analysis was truncated because it exceeded the comment size "
        "limit. AI-generated and may be incorrect._"
    )
    cut = _MAX_COMMENT_CHARS - len(notice) - len("\n\n\n```\n")
    kept = body[:cut]
    # Prefer a line boundary so the cut does not land mid-token (including inside
    # a defused marker), falling back to the raw cut if there is no newline.
    newline = kept.rfind("\n")
    if newline > 0:
        kept = kept[:newline]
    if _open_fence(kept):
        kept += "\n```"
    return f"{kept}\n\n{notice}"


def _open_fence(text: str) -> bool:
    """Whether *text* ends inside an unclosed ``` code fence.

    Only backtick fences are counted, which is all this module emits. An odd
    number of fence lines means the last one was never closed.
    """
    fences = sum(1 for line in text.splitlines() if line.lstrip().startswith("```"))
    return fences % 2 == 1
