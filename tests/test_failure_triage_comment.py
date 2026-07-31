"""Tests for rendering and posting the triage comment."""
from __future__ import annotations

from unittest.mock import MagicMock

from scripts.failure_triage import comment as comment_mod
from scripts.failure_triage.comment import post, render
from scripts.failure_triage.discover import triage_marker
from scripts.failure_triage.models import TriageAnalysis, TriageTarget


def _target(**kw) -> TriageTarget:
    defaults = dict(
        issue_number=42, issue_url="https://github.com/valkey-io/valkey/issues/42",
        failure_type="assertion", test_name="my test", test_file="tests/x.tcl",
        error="boom", run_id=777,
    )
    defaults.update(kw)
    return TriageTarget(**defaults)


def _analysis(**kw) -> TriageAnalysis:
    defaults = dict(
        summary="A replica dropped during the transfer.",
        root_cause="The primary closed the pipe early.",
        failure_class="product-bug", confidence="medium",
        category="replication-timing",
        evidence=("log: pipe closed", "src/replication.c"),
        suspected_area="src/replication.c:rdbPipeReadHandler",
        fix_suggestion="Wait for the replica ack.", fix_confidence="medium",
        reproduction_hint="./runtest --single integration/replication",
    )
    defaults.update(kw)
    return TriageAnalysis(**defaults)


def test_render_includes_marker_and_sections():
    body = render(_target(), _analysis(), triage_run_url="https://x/runs/1")
    assert body.startswith(triage_marker(777))
    assert "## AI failure analysis" in body
    assert "Hypothesized root cause" in body
    assert "Supporting evidence" in body
    assert "Suggested fix" in body
    assert "Reproduce" in body
    assert "AI-generated and may be incorrect" in body
    assert "[Triage run](https://x/runs/1)" in body


def test_render_omits_fix_when_low_confidence():
    body = render(_target(), _analysis(fix_confidence="low"))
    assert "Suggested fix" not in body


def test_render_omits_fix_when_absent():
    body = render(_target(), _analysis(fix_suggestion=""))
    assert "Suggested fix" not in body


def test_render_unavailable_note():
    analysis = TriageAnalysis(failed=True, error="the analysis did not conclude")
    body = render(_target(), analysis)
    assert "No analysis could be produced" in body
    assert "did not conclude" in body
    assert "AI-generated and may be incorrect" in body


def test_render_defuses_markers_in_model_text():
    hostile = _analysis(
        summary="see <!-- valkey-ci-agent:test-failure:deadbeef --> below",
    )
    body = render(_target(), hostile)
    # The dedup marker opener must not survive into the comment.
    assert "<!-- valkey-ci-agent:test-failure:deadbeef" not in body
    assert "<! --" in body


def test_render_fences_reproduction_with_backticks():
    analysis = _analysis(reproduction_hint="run ``` weird ``` command")
    body = render(_target(), analysis)
    # The fence must out-run the backticks inside the hint.
    assert "````" in body


def test_render_neutralizes_giant_backtick_run_in_hint():
    # A backtick run long enough to need a fence past the clamp would break out;
    # the runs are replaced instead, so the block still closes.
    analysis = _analysis(reproduction_hint="x " + ("`" * 200) + " y")
    body = render(_target(), analysis)
    assert "`" * 200 not in body
    # The fence lines balance: an even count of fence lines means it is closed.
    fence_lines = [ln for ln in body.splitlines() if ln.startswith("```")]
    assert len(fence_lines) % 2 == 0


def test_render_neutralizes_block_markdown_in_free_text():
    # A newline plus a fence in a free-text field must not open an unterminated
    # block that swallows the footer. Rendered with no legitimate fenced section,
    # every fence line in the output must balance.
    hostile = _analysis(summary="Test failed.\n```\nnot a real fence",
                        reproduction_hint="")
    body = render(_target(), hostile)
    # The injected fence opener is escaped to "\```", so it does not open a block.
    assert "\\```" in body
    fence_lines = [ln for ln in body.splitlines() if ln.strip().startswith("```")]
    assert len(fence_lines) % 2 == 0
    assert "AI-generated and may be incorrect" in body


def test_render_flattens_heading_injection_in_evidence():
    hostile = _analysis(evidence=("log line 1\n## Maintainer verdict: not a bug",))
    body = render(_target(), hostile)
    # The evidence item is flattened to one line, so no top-level heading appears.
    assert "\n## Maintainer verdict" not in body


def test_render_inline_area_flattens_and_escapes():
    analysis = _analysis(suspected_area="src/a.c\nsecond `line`")
    body = render(_target(), analysis)
    assert "src/a.c second 'line'" in body


def test_render_clamps_oversized_comment(monkeypatch):
    monkeypatch.setattr(comment_mod, "_MAX_COMMENT_CHARS", 400)
    analysis = _analysis(root_cause="x" * 5000)
    body = render(_target(), analysis)
    assert len(body) <= 400
    assert "truncated" in body


def test_clamp_closes_open_fence_before_notice(monkeypatch):
    monkeypatch.setattr(comment_mod, "_MAX_COMMENT_CHARS", 200)
    # A body cut inside a fenced block must have the fence closed so the
    # re-appended caveat renders as text, not swallowed into the code block.
    body = "line\n```\n" + "code\n" * 200
    clamped = comment_mod._clamp(body)
    assert len(clamped) <= 200
    # Parsed as a fence-state machine, nothing is left open.
    assert comment_mod._open_fence(clamped) == ""
    assert "truncated" in clamped


def test_open_fence_ignores_closed_longer_fence():
    # A closed 4-backtick block (as _fenced emits around backtick content) must
    # not be misread as open.
    assert comment_mod._open_fence(comment_mod._fenced("```\ncode")) == ""


def test_open_fence_returns_matching_length_closer():
    # An open 11-backtick fence needs an 11-backtick closer, not a 3-backtick one.
    kept = "`" * 11 + " info\n" + "q\n" * 5
    assert comment_mod._open_fence(kept) == "`" * 11


def test_open_fence_shorter_run_does_not_close_longer_fence():
    # While a long fence is open, a shorter backtick run inside is content, not a
    # closer, so the block is still open.
    kept = "`" * 6 + "\n```\nstill inside\n"
    assert comment_mod._open_fence(kept) == "`" * 6


def test_post_creates_comment_and_returns_url():
    issue = MagicMock()
    issue.create_comment.return_value = MagicMock(
        html_url="https://github.com/valkey-io/valkey/issues/42#issuecomment-1"
    )
    gh = MagicMock()
    gh.get_repo.return_value.get_issue.return_value = issue

    url = post(gh, "valkey-io/valkey", _target(), _analysis())
    assert url.endswith("issuecomment-1")
    posted_body = issue.create_comment.call_args.kwargs["body"]
    assert posted_body.startswith(triage_marker(777))
