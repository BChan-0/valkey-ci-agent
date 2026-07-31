"""Tests for the AI analysis stage."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.ai.runtime import AgentRunResult, get_agent_profile
from scripts.failure_triage import analyze as analyze_mod
from scripts.failure_triage.analyze import build_prompt, triage
from scripts.failure_triage.models import FailureEvidence, TriageTarget


def _target(**kw) -> TriageTarget:
    defaults = dict(
        issue_number=1, issue_url="u", failure_type="assertion",
        test_name="my test", test_file="tests/x.tcl", error="boom", run_id=9,
    )
    defaults.update(kw)
    return TriageTarget(**defaults)


def _evidence(tmp_path: Path, **kw) -> FailureEvidence:
    defaults = dict(
        workdir=tmp_path, job_name="job-a",
        log_excerpt_path="evidence/job-log-excerpt.txt", excerpt_lines=42,
        source_path="valkey", source_sha="abc1234",
    )
    defaults.update(kw)
    return FailureEvidence(**defaults)


def _result(stdout: str, rc: int = 0) -> AgentRunResult:
    return AgentRunResult(
        profile="failure_triage_readonly", stdout=stdout, stderr="", returncode=rc,
        prompt_sha256="x", cwd="/tmp", allowed_tools="Read,Grep,Glob",
        model="fable", started_at="t0", finished_at="t1",
    )


def _stream(payload: dict) -> str:
    return json.dumps({"type": "result", "result": json.dumps(payload)})


_GOOD = {
    "summary": "A replica dropped during the diskless RDB transfer.",
    "root_cause": "The primary closed the pipe before the replica finished.",
    "failure_class": "product-bug",
    "confidence": "medium",
    "category": "replication-timing",
    "evidence": ["log: pipe closed at line 128", "src/replication.c handles the pipe"],
    "suspected_area": "src/replication.c:rdbPipeReadHandler",
    "fix_suggestion": "Wait for the replica ack before closing the pipe.",
    "fix_confidence": "medium",
    "reproduction_hint": "./runtest --single integration/replication",
}


def test_triage_parses_clean_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result(_stream(_GOOD)))
    result = triage(_target(), _evidence(tmp_path))
    assert not result.failed
    assert result.failure_class == "product-bug"
    assert result.confidence == "medium"
    assert result.has_fix_suggestion
    assert result.suspected_area == "src/replication.c:rdbPipeReadHandler"
    assert len(result.evidence) == 2


def test_triage_retries_once_then_succeeds(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _fake(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _result("stall", rc=1)
        return _result(_stream(_GOOD))

    monkeypatch.setattr(analyze_mod, "run_agent", _fake)
    result = triage(_target(), _evidence(tmp_path))
    assert calls["n"] == 2
    assert not result.failed


def test_triage_marks_failed_on_persistent_error(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result("nope", rc=1))
    result = triage(_target(), _evidence(tmp_path))
    assert result.failed
    assert "exited 1" in result.error


def test_triage_does_not_retry_unparseable_clean_exit(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _fake(*a, **k):
        calls["n"] += 1
        return _result("no json here", rc=0)

    monkeypatch.setattr(analyze_mod, "run_agent", _fake)
    result = triage(_target(), _evidence(tmp_path))
    assert calls["n"] == 1
    assert result.failed
    assert "no parseable verdict" in result.error


def test_triage_out_of_turns_is_not_a_hard_error(monkeypatch, tmp_path):
    stdout = json.dumps({"type": "result", "subtype": "error_max_turns"})
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result(stdout, rc=1))
    result = triage(_target(), _evidence(tmp_path))
    assert result.failed
    assert "turn budget" in result.error


def test_triage_out_of_turns_detected_on_clean_exit(monkeypatch, tmp_path):
    # The CLI can report the turn limit with a zero exit code; it is still a
    # non-conclusion, not a parseable verdict.
    stdout = json.dumps({"type": "result", "subtype": "error_max_turns"})
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result(stdout, rc=0))
    result = triage(_target(), _evidence(tmp_path))
    assert result.failed
    assert "turn budget" in result.error


def test_triage_marker_text_in_log_is_not_a_false_turn_limit(monkeypatch, tmp_path):
    # A successful verdict whose stream also contains the literal marker text
    # (e.g. echoed from a read log) must not be misread as a turn-limit outcome.
    good = json.dumps({"type": "result", "result": json.dumps(_GOOD)})
    noise = json.dumps({"type": "assistant", "message": {
        "content": [{"type": "text", "text": "the log mentions error_max_turns"}]}})
    monkeypatch.setattr(
        analyze_mod, "run_agent", lambda *a, **k: _result(noise + "\n" + good, rc=0),
    )
    result = triage(_target(), _evidence(tmp_path))
    assert not result.failed
    assert result.failure_class == "product-bug"


def test_triage_error_reason_falls_back_to_stream_text(monkeypatch, tmp_path):
    # The normal nonzero-exit path returns an empty stderr, so the reason shown
    # to a maintainer comes from the model's last text rather than being blank.
    stream = json.dumps({"type": "result", "result": "I could not read the log."})
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result(stream, rc=1))
    result = triage(_target(), _evidence(tmp_path))
    assert result.failed
    assert result.error.rstrip().endswith("I could not read the log.")
    assert not result.error.rstrip().endswith(":")


def test_triage_parses_verdict_with_only_summary(monkeypatch, tmp_path):
    # A verdict carrying a summary but no root_cause is valid; keying the parse
    # on failure_class rather than root_cause must still find it.
    payload = {"summary": "Something failed in setup.", "failure_class": "infrastructure",
               "confidence": "low"}
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result(_stream(payload)))
    result = triage(_target(), _evidence(tmp_path))
    assert not result.failed
    assert result.summary.startswith("Something failed")
    assert result.root_cause == ""


def test_triage_rejects_empty_verdict(monkeypatch, tmp_path):
    payload = {"summary": "", "root_cause": "", "failure_class": "undetermined"}
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result(_stream(payload)))
    result = triage(_target(), _evidence(tmp_path))
    assert result.failed
    assert "neither a summary nor a root cause" in result.error


def test_triage_coerces_out_of_range_enums(monkeypatch, tmp_path):
    payload = dict(_GOOD, failure_class="banana", confidence="certain", fix_confidence="x")
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result(_stream(payload)))
    result = triage(_target(), _evidence(tmp_path))
    assert result.failure_class == "undetermined"
    assert result.confidence == "low"
    assert result.fix_confidence == "low"
    # A low fix confidence withholds the suggestion even when text was returned.
    assert not result.has_fix_suggestion


def test_triage_treats_string_null_as_absent(monkeypatch, tmp_path):
    payload = dict(_GOOD, fix_suggestion="null", suspected_area="none")
    monkeypatch.setattr(analyze_mod, "run_agent", lambda *a, **k: _result(_stream(payload)))
    result = triage(_target(), _evidence(tmp_path))
    assert result.fix_suggestion == ""
    assert result.suspected_area == ""


def test_prompt_names_missing_source(tmp_path):
    evidence = _evidence(tmp_path, source_path="", source_sha="")
    prompt = build_prompt(_target(), evidence)
    assert "source is NOT available" in prompt
    assert "do not cite" in prompt.lower()


def test_prompt_names_missing_log(tmp_path):
    evidence = _evidence(
        tmp_path, log_excerpt_path="", job_name="",
        log_unavailable_reason="the job log has expired",
    )
    prompt = build_prompt(_target(), evidence)
    assert "No log excerpt is available" in prompt
    assert "expired" in prompt


def test_profile_is_read_only():
    profile = get_agent_profile("failure_triage_readonly")
    assert not profile.writes_allowed
    assert "Bash" not in profile.allowed_tools
    assert "Write" not in profile.allowed_tools
    assert "Edit" not in profile.allowed_tools
