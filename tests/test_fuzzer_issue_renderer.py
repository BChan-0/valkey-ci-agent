"""Tests for fuzzer-specific issue rendering."""
from __future__ import annotations

from scripts.fuzzer.issue_renderer import (
    MARKER_NAMESPACE,
    _build_title,
    _render_body,
    _render_comment,
    render_for,
)
from scripts.fuzzer.models import FuzzerRunAnalysis, FuzzerSignal


def _analysis(**kw) -> FuzzerRunAnalysis:
    defaults = dict(
        repo="valkey-io/valkey-fuzzer", workflow_file="fuzzer-run.yml",
        run_id=100, run_url="https://github.com/r/actions/runs/100",
        conclusion="failure", head_sha="abc", overall_status="anomalous",
        triage_verdict="likely-core-valkey-bug", summary="crash found",
        anomalies=[FuzzerSignal("Node crash", "critical", "segfault")],
        incident_fingerprint="fp_test_12345678901",
        suggested_labels=["possible-valkey-bug"],
    )
    defaults.update(kw)
    return FuzzerRunAnalysis(**defaults)


def test_build_title_from_root_cause():
    assert _build_title(_analysis(root_cause_category="split-brain")) == "[fuzzer-run] Split Brain"


def test_build_title_from_anomaly():
    assert _build_title(_analysis()) == "[fuzzer-run] Node crash"


def test_build_title_for_analyzer_failure():
    """A run the analyzer could not verdict gets an honest title, not
    "Anomalous behavior detected"."""
    analysis = _analysis(
        overall_status="warning", triage_verdict="needs-human-triage",
        summary="Run 100: the analyzer could not complete. ... AI analysis failed: timeout after 1800s",
        anomalies=[], root_cause_category=None, analyzer_incomplete=True,
    )
    assert _build_title(analysis) == "[fuzzer-run] Analyzer could not complete"


def test_build_title_does_not_claim_failure_on_completed_triage_verdict():
    """A completed analysis whose verdict is "needs-human-triage" must NOT be
    mislabeled as an analyzer failure. This is the #104 conflation guard."""
    analysis = _analysis(
        overall_status="warning", triage_verdict="needs-human-triage",
        summary="Cluster recovered but consensus was slow; a human should look.",
        anomalies=[], root_cause_category=None, analyzer_incomplete=False,
    )
    assert _build_title(analysis) == "[fuzzer-run] Anomalous behavior detected"


def test_build_title_for_missing_artifact_error():
    """A run that could not be analyzed because artifacts were missing gets the
    same honest title as an AI-analysis failure, not "Anomalous behavior"."""
    from scripts.fuzzer.analyzer import _build_error_analysis
    from scripts.fuzzer.models import FuzzerRunContext
    ctx = FuzzerRunContext(repo="r", workflow_file="w", run_id=1, run_url="u",
                           conclusion="failure", head_sha="h")
    analysis = _build_error_analysis(ctx, "no fuzzer artifact bundle found")
    assert _build_title(analysis) == "[fuzzer-run] Analyzer could not complete"


def test_render_body_surfaces_analyzer_error():
    analysis = _analysis(
        overall_status="warning", triage_verdict="needs-human-triage",
        summary="Run 100: the analyzer could not complete. ... AI analysis failed: timeout after 1800s",
        anomalies=[], analyzer_incomplete=True,
    )
    body = _render_body(analysis, "<!-- marker -->", occurrences=1)
    assert "timeout after 1800s" in body
    assert "Anomalous behavior detected" not in body


def test_render_body_contains_essentials():
    body = _render_body(_analysis(), "<!-- marker -->", occurrences=1)
    assert "<!-- marker -->" in body
    assert f"<!-- {MARKER_NAMESPACE}:occurrences:1 -->" in body
    assert "Node crash" in body
    assert "crash found" in body


def test_render_for_returns_callable_with_labels():
    """The factory hands IssueDedupPublisher.upsert a callable that produces
    fully-populated IssueContent including labels."""
    cb = render_for(_analysis())
    content = cb("<!-- marker -->", 3)
    assert content.title == "[fuzzer-run] Node crash"
    assert "<!-- marker -->" in content.body
    assert f"<!-- {MARKER_NAMESPACE}:occurrences:3 -->" in content.body
    assert "Occurrence #3" in content.comment
    assert content.labels == ("possible-valkey-bug",)
