"""Tests for fuzzer analyzer."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.fuzzer.analyzer import (
    _dedupe_signals,
    _load_artifacts,
    _parse_claude_response,
    _scan_logs,
    _triage,
)
from scripts.fuzzer.models import FuzzerRunContext, FuzzerSignal


def _ctx(**kw) -> FuzzerRunContext:
    defaults = dict(repo="r", workflow_file="w", run_id=1, run_url="u",
                    conclusion="failure", head_sha="h")
    defaults.update(kw)
    return FuzzerRunContext(**defaults)


def test_scan_logs_detects_crash():
    ctx = _ctx()
    ctx.node_logs = {"node-1.log": "ASSERTION FAILED at server.c:123"}
    anomalies = _scan_logs(ctx)
    assert any("crash" in a.title.lower() or "assertion" in a.title.lower() for a in anomalies)


def test_scan_logs_validation_failure():
    ctx = _ctx()
    ctx.results = {
        "success": False, "error_message": "failed",
        "final_validation": {"checks": {"slot_coverage": {"success": False, "error": "lost slots"}}},
    }
    anomalies = _scan_logs(ctx)
    assert any("slot" in a.title.lower() for a in anomalies)


def test_load_artifacts_reads_manifest_and_results():
    ctx = _ctx()
    _load_artifacts(ctx, {
        "manifest.json": b'{"scenario_id": "chaos-1", "seed": 42, "valkey_sha": "deadbeef1234567"}',
        "results.json": b'{"results": [{"success": false}]}',
        "node-1.log": b"log output",
    })
    assert ctx.scenario_id == "chaos-1"
    assert ctx.seed == "42"
    assert ctx.tested_valkey_sha == "deadbeef1234567"
    assert "node-1.log" in ctx.node_logs


def test_triage_normal():
    assert _triage([]) == ("normal", "expected-chaos-noise")


def test_triage_critical_bug_indicator():
    status, verdict = _triage([FuzzerSignal("Node crash or assertion", "critical", "x")])
    assert status == "anomalous"
    assert verdict == "likely-core-valkey-bug"


def test_triage_critical_non_bug_indicator():
    # OOM is critical but not in the bug-indicator subset.
    status, verdict = _triage([FuzzerSignal("OOM", "critical", "x")])
    assert (status, verdict) == ("anomalous", "possible-core-valkey-bug")


def test_triage_warning():
    assert _triage([FuzzerSignal("X", "warning", "y")]) == ("warning", "possible-core-valkey-bug")


def test_dedupe_signals():
    signals = [
        FuzzerSignal("a", "critical", "x"),
        FuzzerSignal("a", "critical", "x"),
        FuzzerSignal("b", "warning", "y"),
    ]
    assert len(_dedupe_signals(signals)) == 2


def test_parse_claude_response_plain_json():
    assert _parse_claude_response('{"overall_status": "normal"}')["overall_status"] == "normal"


def test_parse_claude_response_stream_json():
    stream = "\n".join([
        '{"type": "progress", "data": "thinking"}',
        '{"type": "result", "result": "{\\"overall_status\\": \\"warning\\"}"}',
    ])
    assert _parse_claude_response(stream)["overall_status"] == "warning"


def test_parse_claude_response_requires_overall_status():
    """Progress events without overall_status must not be returned as the verdict."""
    with pytest.raises(ValueError):
        _parse_claude_response('{"type": "progress", "data": "x"}')


def test_parse_claude_response_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_claude_response("no json here at all")


def test_format_source_note_when_clones_succeed():
    from scripts.fuzzer.analyzer import _format_source_note
    ctx = _ctx(tested_valkey_sha="deadbeef")
    note = _format_source_note(ctx, valkey_ok=True, fuzzer_ok=True)
    assert "valkey/" in note and "deadbeef" in note
    assert "valkey-fuzzer/" in note
    assert "NOT AVAILABLE" not in note


def test_format_source_note_when_clones_fail():
    """A failed clone must be called out so Claude doesn't cite line numbers."""
    from scripts.fuzzer.analyzer import _format_source_note
    note = _format_source_note(
        _ctx(tested_valkey_sha="deadbeef"),
        valkey_ok=False,
        fuzzer_ok=False,
    )
    assert "NOT AVAILABLE" in note
    assert "clone failed" in note
    assert "Do not cite source line numbers" in note


def test_format_source_note_when_sha_unrecorded():
    """Missing SHA must produce a different note than a failed clone — we
    never want to silently fall back to the default branch.
    """
    from scripts.fuzzer.analyzer import _format_source_note
    note = _format_source_note(_ctx(), valkey_ok=False, fuzzer_ok=True)
    assert "NOT AVAILABLE" in note
    assert "manifest did not record the tested commit" in note
    assert "Do not cite source line numbers" in note


def test_invoke_claude_skips_valkey_clone_when_sha_missing(monkeypatch, tmp_path):
    """If the manifest didn't record valkey_sha, the analyzer must NOT clone
    the default branch — that would have Claude triage a different tree.
    """
    from scripts.fuzzer import analyzer as analyzer_mod

    clone_calls: list[tuple] = []

    def fake_clone(repo, dest, sha=None):
        clone_calls.append((repo, sha))
        return True

    fake_result = MagicMock(returncode=0, stdout='{"overall_status": "normal"}', stderr="")
    monkeypatch.setattr(analyzer_mod, "shallow_clone_at_sha", fake_clone)
    monkeypatch.setattr(analyzer_mod, "run_agent", lambda *a, **kw: fake_result)

    ctx = _ctx(head_sha="abc123")  # tested_valkey_sha left as None
    analyzer_mod._invoke_claude(ctx, [], tmp_path)

    cloned_repos = [r for r, _ in clone_calls]
    assert "valkey-io/valkey" not in cloned_repos
    assert "r" in cloned_repos  # fuzzer repo (the test ctx repo) still cloned


def test_build_error_analysis_has_distinct_fingerprint_per_reason():
    """Different infra-failure reasons must NOT collide on a single issue."""
    from scripts.fuzzer.analyzer import _build_error_analysis
    a = _build_error_analysis(_ctx(), "no fuzzer artifact bundle found")
    b = _build_error_analysis(_ctx(), "fuzzer artifact bundle was empty or unreadable")
    assert a.incident_fingerprint
    assert b.incident_fingerprint
    assert a.incident_fingerprint != b.incident_fingerprint


def test_build_error_analysis_fingerprint_is_stable():
    """Same context + reason produces the same fingerprint across calls."""
    from scripts.fuzzer.analyzer import _build_error_analysis
    a = _build_error_analysis(_ctx(), "no fuzzer artifact bundle found")
    b = _build_error_analysis(_ctx(), "no fuzzer artifact bundle found")
    assert a.incident_fingerprint == b.incident_fingerprint


def test_invoke_claude_retries_once_then_succeeds(monkeypatch, tmp_path):
    """A transient first-attempt failure is retried; the second success wins."""
    from scripts.fuzzer import analyzer as analyzer_mod

    monkeypatch.setattr(analyzer_mod, "shallow_clone_at_sha", lambda *a, **kw: True)
    results = [
        MagicMock(returncode=1, stdout="", stderr="timeout after 1800s"),
        MagicMock(returncode=0, stdout='{"overall_status": "normal"}', stderr=""),
    ]
    calls = {"n": 0}

    def fake_run_agent(*a, **kw):
        out = results[calls["n"]]
        calls["n"] += 1
        return out

    monkeypatch.setattr(analyzer_mod, "run_agent", fake_run_agent)
    payload = analyzer_mod._invoke_claude(_ctx(head_sha="abc"), [], tmp_path)
    assert payload["overall_status"] == "normal"
    assert calls["n"] == 2


def test_invoke_claude_does_not_retry_on_parse_failure(monkeypatch, tmp_path):
    """A successful exit with unparseable output is deterministic, so it is
    raised immediately rather than burning a second Bedrock attempt."""
    from scripts.fuzzer import analyzer as analyzer_mod

    monkeypatch.setattr(analyzer_mod, "shallow_clone_at_sha", lambda *a, **kw: True)
    calls = {"n": 0}

    def fake_run_agent(*a, **kw):
        calls["n"] += 1
        return MagicMock(returncode=0, stdout="not json", stderr="")

    monkeypatch.setattr(analyzer_mod, "run_agent", fake_run_agent)
    with pytest.raises(ValueError):
        analyzer_mod._invoke_claude(_ctx(head_sha="abc"), [], tmp_path)
    assert calls["n"] == 1


def test_invoke_claude_raises_after_retry_exhausted(monkeypatch, tmp_path):
    """Both attempts failing surfaces the last error for the caller to handle."""
    from scripts.fuzzer import analyzer as analyzer_mod

    monkeypatch.setattr(analyzer_mod, "shallow_clone_at_sha", lambda *a, **kw: True)
    monkeypatch.setattr(
        analyzer_mod, "run_agent",
        lambda *a, **kw: MagicMock(returncode=1, stdout="", stderr="timeout after 1800s"),
    )
    with pytest.raises(RuntimeError, match="timeout after 1800s"):
        analyzer_mod._invoke_claude(_ctx(head_sha="abc"), [], tmp_path)


def test_invoke_claude_guards_against_zero_attempts(monkeypatch, tmp_path):
    """A misconfigured attempt count fails loud, never `raise None`."""
    from scripts.fuzzer import analyzer as analyzer_mod

    monkeypatch.setattr(analyzer_mod, "shallow_clone_at_sha", lambda *a, **kw: True)
    monkeypatch.setattr(analyzer_mod, "run_agent", lambda *a, **kw: None)
    monkeypatch.setattr(analyzer_mod, "_CLAUDE_MAX_ATTEMPTS", 0)
    with pytest.raises(RuntimeError, match=">= 1"):
        analyzer_mod._invoke_claude(_ctx(head_sha="abc"), [], tmp_path)


def test_agent_failure_analysis_carries_error_and_distinct_fingerprint():
    """A clean run + Claude failure produces an honest, triageable analysis
    that dedupes separately from artifact errors and real findings."""
    from scripts.fuzzer.analyzer import _build_agent_failure_analysis, _build_error_analysis
    af = _build_agent_failure_analysis(
        _ctx(seed="2153362329"), "Claude Code failed (rc=1): timeout after 1800s",
    )
    assert "timeout after 1800s" in af.summary
    assert af.triage_verdict == "needs-human-triage"
    assert af.analyzer_incomplete is True
    assert af.anomalies == []
    assert af.suggested_labels == []
    assert af.reproduction_hint == "valkey-fuzzer cluster --seed 2153362329"
    # Distinct dedup bucket from artifact-handling errors.
    err = _build_error_analysis(_ctx(), "no fuzzer artifact bundle found")
    assert err.analyzer_incomplete is True
    assert af.incident_fingerprint
    assert af.incident_fingerprint != err.incident_fingerprint


def test_agent_failure_analysis_fingerprint_is_stable():
    """Repeated Bedrock outages collapse into one issue."""
    from scripts.fuzzer.analyzer import _build_agent_failure_analysis
    a = _build_agent_failure_analysis(_ctx(), "timeout after 1800s")
    b = _build_agent_failure_analysis(_ctx(), "a totally different error string")
    assert a.incident_fingerprint == b.incident_fingerprint
