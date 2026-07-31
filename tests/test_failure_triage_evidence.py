"""Tests for triage evidence gathering and log carving."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from scripts.failure_triage import evidence as evidence_mod
from scripts.failure_triage.evidence import carve_excerpt, collect
from scripts.failure_triage.models import JobRef, TriageTarget


def _target(**kw) -> TriageTarget:
    defaults = dict(
        issue_number=1,
        issue_url="https://github.com/valkey-io/valkey/issues/1",
        failure_type="assertion",
        test_name="diskless timeout replicas drop during rdb pipe",
        test_file="tests/integration/replication.tcl",
        error="log message of '*Diskless rdb transfer*' not found",
        run_id=30502672689,
        jobs=(JobRef("test-macos-latest", "u", 90745555745),),
    )
    defaults.update(kw)
    return TriageTarget(**defaults)


def _job_log(*, marker_line: str, before: int = 600, after: int = 200) -> bytes:
    """A synthetic job log with timestamped lines around one failure marker."""
    ts = "2026-07-30T01:49:29.8523740Z "
    lines = [f"{ts}server log line {i}" for i in range(before)]
    lines.append(f"{ts}{marker_line}")
    lines += [f"{ts}later line {i}" for i in range(after)]
    return ("\n".join(lines)).encode("utf-8")


def _cleaned(raw: bytes) -> list[str]:
    """Strip timestamps as the real carver does before searching."""
    return evidence_mod._clean_log_lines(raw)


def test_carve_excerpt_centers_on_named_test():
    target = _target()
    marker = f"[err]: {target.test_name} in {target.test_file}"
    lines = _cleaned(_job_log(marker_line=marker))
    excerpt = carve_excerpt(lines, target)
    assert excerpt is not None
    assert marker in excerpt.text
    assert not excerpt.truncated
    assert excerpt.text.splitlines()[0].startswith("Excerpt of the job log")


def test_carve_excerpt_none_without_marker():
    lines = ["nothing", "relevant", "here"]
    assert carve_excerpt(lines, _target()) is None


def test_carve_excerpt_anchors_on_error_when_name_absent():
    target = _target(test_name="", error="ERROR SUMMARY: 3 errors from 3 contexts")
    lines = [
        "setup",
        "ERROR SUMMARY: 3 errors from 3 contexts detected in the run",
        "teardown",
    ]
    excerpt = carve_excerpt(lines, target)
    assert excerpt is not None
    assert "ERROR SUMMARY" in excerpt.text


def test_carve_excerpt_prefers_failure_over_end_of_run_recap():
    # The runner prints "*** [err]: <test>" in an end-of-run recap; the excerpt
    # must anchor on the in-place failure, not its recap at the tail.
    target = _target(test_name="my flaky test", test_file="tests/x.tcl")
    verdict = "[err]: my flaky test in tests/x.tcl"
    lines = (
        ["setup line"] * 5
        + [verdict]
        + ["server log detail"] * 5
        + ["The following tests failed:"]
        + ["*** [err]: my flaky test in tests/x.tcl"]
    )
    excerpt = carve_excerpt(lines, target)
    assert excerpt is not None
    # The header names the kept line range; it must start at the in-place
    # failure near the top, not at the recap line near the end.
    first = excerpt.text.splitlines()[0]
    assert "server log detail" in excerpt.text
    assert "The following tests failed:" not in excerpt.text or "1-" in first


def test_carve_excerpt_keeps_preceding_server_logs_under_budget(monkeypatch):
    # The dumped server logs precede the verdict; a run of fat trailing lines
    # must not starve them out of the excerpt.
    monkeypatch.setattr(evidence_mod, "_MAX_EXCERPT_CHARS", 4000)
    monkeypatch.setattr(evidence_mod, "_HEADER_RESERVE", 100)
    target = _target(test_name="t", test_file="tests/x.tcl")
    marker = "[err]: t in tests/x.tcl"
    preceding = [f"server log {i}" for i in range(60)]
    following = ["x" * 300 for _ in range(60)]
    lines = preceding + [marker] + following
    excerpt = carve_excerpt(lines, target)
    assert excerpt is not None
    # Some preceding server-log context survived alongside the verdict.
    assert "server log" in excerpt.text
    assert marker in excerpt.text


def test_carve_excerpt_truncates_within_budget(monkeypatch):
    monkeypatch.setattr(evidence_mod, "_MAX_EXCERPT_CHARS", 2000)
    monkeypatch.setattr(evidence_mod, "_HEADER_RESERVE", 100)
    target = _target(test_name="t", test_file="tests/x.tcl")
    marker = "[err]: t in tests/x.tcl"
    long_lines = ["x" * 100 for _ in range(50)]
    lines = long_lines + [marker] + long_lines
    excerpt = carve_excerpt(lines, target)
    assert excerpt is not None
    assert excerpt.truncated
    assert "truncated" in excerpt.text
    assert marker in excerpt.text
    # The rendered excerpt stays within the budget.
    assert len(excerpt.text) <= 2000


def test_carve_excerpt_log_lines_excludes_header():
    target = _target(test_name="t", test_file="tests/x.tcl")
    marker = "[err]: t in tests/x.tcl"
    lines = ["a", "b", "c", marker, "d", "e"]
    excerpt = carve_excerpt(lines, target)
    assert excerpt is not None
    # log_lines counts kept log lines, not the header the carver prepends.
    assert excerpt.log_lines == len(lines)


def _artifact_client(log: bytes) -> MagicMock:
    client = MagicMock()
    client.download_job_log.return_value = log
    return client


def _job(name, job_id, conclusion="failure", url="https://x/job"):
    job = MagicMock()
    job.name = name
    job.id = job_id
    job.conclusion = conclusion
    job.html_url = url
    return job


def _gh(*, sha="", jobs=None):
    gh = MagicMock()
    run = MagicMock()
    run.head_sha = sha
    run.jobs.return_value = jobs if jobs is not None else []
    gh.get_repo.return_value.get_workflow_run.return_value = run
    return gh


def test_collect_writes_excerpt_and_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_mod, "shallow_clone_at_sha", lambda *a, **k: True)
    target = _target()
    marker = f"[err]: {target.test_name} in {target.test_file}"
    client = _artifact_client(_job_log(marker_line=marker))
    gh = _gh(
        sha="deadbeef1234567deadbeef1234567deadbeef12",
        jobs=[_job("test-macos-latest", 111)],
    )

    result = collect(
        target, tmp_path,
        repo_full_name="valkey-io/valkey", artifact_client=client, gh=gh,
    )

    assert result.has_log
    assert result.has_source
    assert result.source_sha.startswith("deadbeef")
    excerpt_path = tmp_path / result.log_excerpt_path
    assert marker in excerpt_path.read_text()
    facts = json.loads((tmp_path / "evidence" / "failure.json").read_text())
    assert facts["test_name"] == target.test_name
    # The job is resolved from the run's fresh job list, not the body's id.
    client.download_job_log.assert_called_once_with("valkey-io/valkey", 111)


def test_collect_resolves_fresh_job_by_name(tmp_path, monkeypatch):
    # The body records job id 999 (frozen at first occurrence); the run's live
    # jobs give the same-named job a fresh id, which is what must be fetched.
    monkeypatch.setattr(evidence_mod, "shallow_clone_at_sha", lambda *a, **k: True)
    target = _target(jobs=(JobRef("test-macos-latest", "u", 999),))
    marker = f"[err]: {target.test_name} in {target.test_file}"
    client = _artifact_client(_job_log(marker_line=marker))
    gh = _gh(sha="a" * 40, jobs=[_job("test-macos-latest", 222)])

    collect(
        target, tmp_path,
        repo_full_name="valkey-io/valkey", artifact_client=client, gh=gh,
    )
    client.download_job_log.assert_called_once_with("valkey-io/valkey", 222)


def test_collect_falls_back_to_body_job_when_run_jobs_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_mod, "shallow_clone_at_sha", lambda *a, **k: True)
    target = _target(jobs=(JobRef("test-macos-latest", "u", 999),))
    marker = f"[err]: {target.test_name} in {target.test_file}"
    client = _artifact_client(_job_log(marker_line=marker))
    gh = _gh(sha="a" * 40, jobs=[])  # no live jobs

    collect(
        target, tmp_path,
        repo_full_name="valkey-io/valkey", artifact_client=client, gh=gh,
    )
    # The body's job id serves as the last resort.
    client.download_job_log.assert_called_once_with("valkey-io/valkey", 999)


def test_collect_records_expired_log(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_mod, "shallow_clone_at_sha", lambda *a, **k: False)
    client = _artifact_client(b"")
    gh = _gh(sha="", jobs=[_job("test-macos-latest", 111)])

    result = collect(
        _target(), tmp_path,
        repo_full_name="valkey-io/valkey", artifact_client=client, gh=gh,
    )
    assert not result.has_log
    assert "expired" in result.log_unavailable_reason


def test_collect_no_resolvable_job(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_mod, "shallow_clone_at_sha", lambda *a, **k: True)
    target = _target(jobs=(JobRef("job-a", "u", 0),))
    client = _artifact_client(b"irrelevant")
    gh = _gh(sha="a" * 40, jobs=[])  # no live jobs, and body job has no id

    result = collect(
        target, tmp_path,
        repo_full_name="valkey-io/valkey", artifact_client=client, gh=gh,
    )
    assert not result.has_log
    assert "failing job could not be identified" in result.log_unavailable_reason
    client.download_job_log.assert_not_called()


def test_collect_skips_clone_without_sha(tmp_path, monkeypatch):
    clone_calls = {"n": 0}

    def _clone(*a, **k):
        clone_calls["n"] += 1
        return True

    monkeypatch.setattr(evidence_mod, "shallow_clone_at_sha", _clone)
    client = _artifact_client(b"")
    gh = _gh(sha="", jobs=[_job("test-macos-latest", 111)])

    result = collect(
        _target(), tmp_path,
        repo_full_name="valkey-io/valkey", artifact_client=client, gh=gh,
    )
    assert not result.has_source
    assert clone_calls["n"] == 0


def test_clean_log_lines_strips_prefix_and_bom():
    raw = "﻿2026-07-30T01:49:29.8523740Z [err]: boom\n".encode("utf-8")
    lines = evidence_mod._clean_log_lines(raw)
    assert lines == ["[err]: boom"]
