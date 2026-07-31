"""Tests for the triage data shapes."""
from __future__ import annotations

from scripts.failure_triage.models import (
    JobRef,
    RunReport,
    TriageAnalysis,
    TriageResult,
    TriageTarget,
)


def test_jobref_has_log_only_with_id():
    assert JobRef(name="j", url="u", job_id=5).has_log
    assert not JobRef(name="j", url="u", job_id=0).has_log


def test_target_first_job_with_log_skips_anchorless():
    target = TriageTarget(
        issue_number=1, issue_url="u", failure_type="assertion",
        test_name="t", test_file="f", error="e", run_id=9,
        jobs=(JobRef("a", "u1", 0), JobRef("b", "u2", 42)),
    )
    job = target.first_job_with_log()
    assert job is not None
    assert job.name == "b"


def test_target_first_job_with_log_none_when_all_anchorless():
    target = TriageTarget(
        issue_number=1, issue_url="u", failure_type="assertion",
        test_name="t", test_file="f", error="e", run_id=9,
        jobs=(JobRef("a", "u1", 0),),
    )
    assert target.first_job_with_log() is None


def test_target_display_name_falls_back_to_type():
    named = TriageTarget(
        issue_number=1, issue_url="u", failure_type="assertion",
        test_name="my test", test_file="t.tcl", error="", run_id=1,
    )
    assert named.display_name == "my test in t.tcl"

    nameless = TriageTarget(
        issue_number=2, issue_url="u", failure_type="sanitizer",
        test_name="", test_file="", error="", run_id=1,
    )
    assert nameless.display_name == "[sanitizer] in unknown"


def test_analysis_fix_shown_only_when_confident():
    base = dict(summary="s", root_cause="r", fix_suggestion="do the thing")
    assert TriageAnalysis(**base, fix_confidence="high").has_fix_suggestion
    assert TriageAnalysis(**base, fix_confidence="medium").has_fix_suggestion
    assert not TriageAnalysis(**base, fix_confidence="low").has_fix_suggestion
    assert not TriageAnalysis(
        summary="s", root_cause="r", fix_suggestion="", fix_confidence="high",
    ).has_fix_suggestion


def test_report_counts_by_action():
    report = RunReport(repo="r", run_id=1, dry_run=False, results=[
        TriageResult(1, "u", "d", "commented"),
        TriageResult(2, "u", "d", "commented"),
        TriageResult(3, "u", "d", "error"),
    ])
    assert report.count("commented") == 2
    assert report.count("error") == 1
    assert report.count("skipped-no-evidence") == 0
