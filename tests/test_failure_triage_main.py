"""Tests for the triage CLI orchestration."""
from __future__ import annotations

import json

import pytest

from scripts.failure_triage import main as main_mod
from scripts.failure_triage.models import FailureEvidence, TriageAnalysis, TriageTarget


def _target(number=1, **kw) -> TriageTarget:
    defaults = dict(
        issue_number=number,
        issue_url=f"https://github.com/valkey-io/valkey/issues/{number}",
        failure_type="assertion", test_name="my test", test_file="tests/x.tcl",
        error="boom", run_id=9,
    )
    defaults.update(kw)
    return TriageTarget(**defaults)


def _good_analysis() -> TriageAnalysis:
    return TriageAnalysis(summary="s", root_cause="r", failure_class="product-bug")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Stub every external stage so main's control flow can be tested alone."""
    state = {
        "targets": [_target()],
        "evidence": FailureEvidence(
            workdir=tmp_path, job_name="job-a",
            log_excerpt_path="evidence/x.txt", excerpt_lines=10,
            source_path="valkey", source_sha="abc1234",
        ),
        "analysis": _good_analysis(),
        "posted": [],
    }

    monkeypatch.setattr(main_mod, "Github", lambda **k: object())
    monkeypatch.setattr(main_mod, "ArtifactClient", lambda *a, **k: object())
    monkeypatch.setattr(
        main_mod.discover, "select_targets", lambda *a, **k: state["targets"],
    )
    monkeypatch.setattr(
        main_mod.evidence, "collect", lambda *a, **k: state["evidence"],
    )
    monkeypatch.setattr(
        main_mod.analyze, "triage", lambda *a, **k: state["analysis"],
    )

    def _post(gh, repo, target, analysis, **k):
        state["posted"].append(target.issue_number)
        return f"https://github.com/{repo}/issues/{target.issue_number}#c1"

    monkeypatch.setattr(main_mod.comment, "post", _post)
    return state


def test_run_comments_on_selected_issue(wired, capsys):
    rc = main_mod.run(github_token="t", repo_full_name="valkey-io/valkey")
    assert rc == 0
    assert wired["posted"] == [1]
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["action"] == "commented"


def test_run_dry_run_posts_nothing(wired, capsys):
    rc = main_mod.run(github_token="t", repo_full_name="valkey-io/valkey", dry_run=True)
    assert rc == 0
    assert wired["posted"] == []
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["action"] == "would-comment"


def test_run_skips_when_no_evidence(wired, monkeypatch, capsys):
    monkeypatch.setattr(
        main_mod.evidence, "collect",
        lambda *a, **k: FailureEvidence(
            workdir=wired["evidence"].workdir,
            log_unavailable_reason="the job log has expired",
        ),
    )
    rc = main_mod.run(github_token="t", repo_full_name="valkey-io/valkey")
    assert rc == 0
    assert wired["posted"] == []
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["action"] == "skipped-no-evidence"


def test_run_records_analysis_failure_but_still_comments(wired, monkeypatch, capsys):
    monkeypatch.setattr(
        main_mod.analyze, "triage",
        lambda *a, **k: TriageAnalysis(failed=True, error="did not conclude"),
    )
    rc = main_mod.run(github_token="t", repo_full_name="valkey-io/valkey")
    assert rc == 0
    # An honest "analysis unavailable" note is still posted.
    assert wired["posted"] == [1]
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["action"] == "analysis-failed"


def test_run_isolates_a_failing_target(wired, monkeypatch, capsys):
    wired["targets"] = [_target(1), _target(2)]

    def _post(gh, repo, target, analysis, **k):
        if target.issue_number == 1:
            raise RuntimeError("github down")
        wired["posted"].append(target.issue_number)
        return "u"

    monkeypatch.setattr(main_mod.comment, "post", _post)
    rc = main_mod.run(github_token="t", repo_full_name="valkey-io/valkey")
    # One target errored, so the run exits non-zero, but the other still posted.
    assert rc == 1
    assert wired["posted"] == [2]
    report = json.loads(capsys.readouterr().out)
    actions = {r["issue_number"]: r["action"] for r in report["results"]}
    assert actions == {1: "error", 2: "commented"}


def test_run_writes_output_file(wired, tmp_path):
    out = tmp_path / "result.json"
    rc = main_mod.run(
        github_token="t", repo_full_name="valkey-io/valkey", output=str(out),
    )
    assert rc == 0
    assert json.loads(out.read_text())["results"][0]["action"] == "commented"


def test_main_requires_token(monkeypatch, capsys):
    monkeypatch.delenv("TARGET_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        main_mod.main(["--repo", "valkey-io/valkey"])
    assert "TARGET_TOKEN" in capsys.readouterr().err


def test_main_rejects_zero_limit(monkeypatch):
    monkeypatch.setenv("TARGET_TOKEN", "t")
    with pytest.raises(SystemExit):
        main_mod.main(["--repo", "valkey-io/valkey", "--limit", "0"])


def test_parse_issue_numbers():
    assert main_mod._parse_issue_numbers("1, 2 ,3") == (1, 2, 3)
    assert main_mod._parse_issue_numbers("") == ()
    # Duplicates collapse so one issue is not commented on twice in a run.
    assert main_mod._parse_issue_numbers("5,5,5") == (5,)
    with pytest.raises(SystemExit):
        main_mod._parse_issue_numbers("1,notanumber")
    # int() would accept these; the stricter check must reject them.
    with pytest.raises(SystemExit):
        main_mod._parse_issue_numbers("1_0")
    with pytest.raises(SystemExit):
        main_mod._parse_issue_numbers("-5")
    with pytest.raises(SystemExit):
        main_mod._parse_issue_numbers("0")


def test_main_rejects_negative_max_age(monkeypatch):
    monkeypatch.setenv("TARGET_TOKEN", "t")
    with pytest.raises(SystemExit):
        main_mod.main(["--repo", "valkey-io/valkey", "--max-age-days", "-1"])


def test_run_comment_failure_keeps_analysis(wired, monkeypatch, capsys):
    def _post(gh, repo, target, analysis, **k):
        raise RuntimeError("github down")

    monkeypatch.setattr(main_mod.comment, "post", _post)
    rc = main_mod.run(github_token="t", repo_full_name="valkey-io/valkey")
    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    result = report["results"][0]
    assert result["action"] == "error"
    # The verdict survives in the report rather than being dropped on a post failure.
    assert result["analysis"] is not None
    assert result["analysis"]["failure_class"] == "product-bug"
