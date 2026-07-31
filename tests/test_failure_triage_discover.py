"""Tests for triage discovery and issue-body parsing."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from scripts.failure_triage import discover
from scripts.failure_triage.discover import (
    already_triaged,
    build_target,
    select_targets,
    triage_marker,
)


def _body(
    *,
    namespace="valkey-ci-agent:test-failure",
    test_name="Stacktraces generated on SIGALRM",
    test_file="tests/integration/logging.tcl",
    run_id=30411074547,
    job_id=90447044679,
    last_key=30411074547,
    error="Expected '0' to be between '1' and '999'",
):
    job_url = (
        f"https://github.com/valkey-io/valkey/actions/runs/{run_id}/job/{job_id}"
    )
    return "\n".join([
        f"<!-- {namespace}:1cabf8895d1ca826f36a -->",
        f"<!-- {namespace}:occurrences:1 -->",
        "",
        "**Summary**",
        "",
        f"`{test_name}` in `{test_file}` is failing in CI.",
        "",
        "**Failing test(s)**",
        "",
        f"- Test name: `{test_name}`",
        f"- Test file: `{test_file}`",
        "- CI link(s):",
        f"    - `test-fedoralatest-tls-module`: [CI link]({job_url})",
        "",
        "**Error stack trace**",
        "",
        "```",
        error,
        "```",
        "",
        "**Environments:** `test-fedoralatest-tls-module`",
        "",
        "---",
        "*Auto-created by Test Failure Detector*",
        f"<!-- {namespace}:last-key:{last_key} -->",
    ])


def _issue(number=4289, body=None, created_days_ago=1):
    issue = MagicMock()
    issue.number = number
    issue.body = body if body is not None else _body()
    issue.html_url = f"https://github.com/valkey-io/valkey/issues/{number}"
    issue.created_at = datetime.now(timezone.utc) - timedelta(days=created_days_ago)
    issue._rawData = {}
    return issue


def test_build_target_reads_fields():
    target = build_target(_issue(), "assertion")
    assert target is not None
    assert target.test_name == "Stacktraces generated on SIGALRM"
    assert target.test_file == "tests/integration/logging.tcl"
    assert target.run_id == 30411074547
    assert target.error.startswith("Expected '0'")
    assert len(target.jobs) == 1
    assert target.jobs[0].job_id == 90447044679
    assert target.jobs[0].has_log


def test_build_target_prefers_last_key_over_stale_body_link():
    # A recurrence: the body's CI link is frozen at the first occurrence (run
    # 111), but last-key names the freshest run (run 222). Evidence must target
    # the fresh run.
    body = _body(run_id=111, job_id=1, last_key=222)
    target = build_target(_issue(body=body), "assertion")
    assert target is not None
    assert target.run_id == 222


def test_build_target_falls_back_to_link_without_last_key():
    # A legacy issue predating the last-key marker still resolves via its link.
    body = "\n".join([
        "<!-- valkey-ci-agent:test-failure:abc -->",
        "- Test name: `t`",
        "- Test file: `f.tcl`",
        "- CI link(s):",
        "    - `job-a`: [CI link](https://github.com/valkey-io/valkey/actions/runs/333/job/9)",
    ])
    target = build_target(_issue(body=body), "assertion")
    assert target is not None
    assert target.run_id == 333


def test_build_target_reads_memory_error_namespace():
    body = _body(namespace="valkey-ci-agent:memory-error", test_name="[no test]",
                 test_file="[no test]")
    target = build_target(_issue(body=body), "memory-error")
    assert target is not None
    # The renderer's "[no test]" filler reads back as an empty identity.
    assert target.test_name == ""
    assert target.test_file == ""


def test_select_targets_recognizes_memory_error(monkeypatch):
    monkeypatch.setattr(discover, "already_triaged", lambda issue: False)
    issue = _issue(number=1, body=_body(namespace="valkey-ci-agent:memory-error"))
    gh = _gh_with_repo(_repo_with_issues([issue]))
    targets = select_targets(gh, "valkey-io/valkey")
    assert targets[0].failure_type == "memory-error"


def test_build_target_none_without_run_link():
    body = "\n".join([
        "<!-- valkey-ci-agent:test-failure:abc -->",
        "- Test name: `t`",
        "- Test file: `f.tcl`",
    ])
    assert build_target(_issue(body=body), "assertion") is None


def test_build_target_run_id_from_plain_run_url():
    body = "\n".join([
        "<!-- valkey-ci-agent:test-timeout:abc -->",
        "- Test name: ``",
        "- Test file: `tests/x.tcl`",
        "- CI link(s):",
        "    - `job-a`: [CI link](https://github.com/valkey-io/valkey/actions/runs/555)",
    ])
    target = build_target(_issue(body=body), "timeout")
    assert target is not None
    assert target.run_id == 555
    # A link with no job anchor yields no fetchable log.
    assert target.first_job_with_log() is None


def test_build_target_treats_placeholder_name_as_empty():
    body = _body(test_name="unknown")
    target = build_target(_issue(body=body), "startup")
    assert target is not None
    assert target.test_name == ""


def test_build_target_dedupes_repeated_jobs():
    job_url = "https://github.com/valkey-io/valkey/actions/runs/1/job/2"
    body = "\n".join([
        "<!-- valkey-ci-agent:test-failure:abc -->",
        "- Test name: `t`",
        "- Test file: `f.tcl`",
        "- CI link(s):",
        f"    - `job-a`: [CI link]({job_url})",
        f"    - `job-a`: [CI link]({job_url})",
        "    - `job-b`: [CI link](https://github.com/valkey-io/valkey/actions/runs/1/job/3)",
    ])
    target = build_target(_issue(body=body), "assertion")
    assert target is not None
    assert [j.name for j in target.jobs] == ["job-a", "job-b"]


def test_already_triaged_matches_current_version():
    issue = _issue()
    commented = MagicMock()
    commented.body = triage_marker(999) + "\n\nsome analysis"
    issue.get_comments.return_value = [commented]
    assert already_triaged(issue)


def test_already_triaged_false_without_marker():
    issue = _issue()
    other = MagicMock()
    other.body = "an unrelated human comment"
    issue.get_comments.return_value = [other]
    assert not already_triaged(issue)


def test_already_triaged_true_when_comments_unreadable():
    issue = _issue()
    issue.get_comments.side_effect = RuntimeError("api down")
    # Fails safe: an unreadable list is treated as triaged so no duplicate posts.
    assert already_triaged(issue)


def _repo_with_issues(issues):
    repo = MagicMock()
    repo.get_issues.return_value = issues
    repo.get_issue.side_effect = lambda n: next(i for i in issues if i.number == n)
    return repo


def _gh_with_repo(repo):
    gh = MagicMock()
    gh.get_repo.return_value = repo
    return gh


def test_select_targets_filters_by_run_and_marker(monkeypatch):
    monkeypatch.setattr(discover, "already_triaged", lambda issue: False)
    matching = _issue(number=1, body=_body(last_key=777))
    other_run = _issue(number=2, body=_body(last_key=888))
    no_marker = _issue(number=3, body="not a detector issue")
    gh = _gh_with_repo(_repo_with_issues([matching, other_run, no_marker]))

    targets = select_targets(gh, "valkey-io/valkey", run_id=777)
    assert [t.issue_number for t in targets] == [1]


def test_select_targets_skips_already_triaged(monkeypatch):
    monkeypatch.setattr(discover, "already_triaged", lambda issue: True)
    gh = _gh_with_repo(_repo_with_issues([_issue(number=1)]))
    assert select_targets(gh, "valkey-io/valkey") == []


def test_select_targets_force_bypasses_triaged_check(monkeypatch):
    called = {"n": 0}

    def _fake(issue):
        called["n"] += 1
        return True

    monkeypatch.setattr(discover, "already_triaged", _fake)
    gh = _gh_with_repo(_repo_with_issues([_issue(number=1)]))
    targets = select_targets(gh, "valkey-io/valkey", force=True)
    assert [t.issue_number for t in targets] == [1]
    assert called["n"] == 0


def test_select_targets_honors_limit(monkeypatch):
    monkeypatch.setattr(discover, "already_triaged", lambda issue: False)
    issues = [_issue(number=n, body=_body(last_key=n)) for n in (10, 11, 12, 13)]
    gh = _gh_with_repo(_repo_with_issues(issues))
    targets = select_targets(gh, "valkey-io/valkey", limit=2)
    assert len(targets) == 2


def test_select_targets_drops_old_issues(monkeypatch):
    monkeypatch.setattr(discover, "already_triaged", lambda issue: False)
    fresh = _issue(number=1, body=_body(last_key=1), created_days_ago=2)
    stale = _issue(number=2, body=_body(last_key=2), created_days_ago=90)
    gh = _gh_with_repo(_repo_with_issues([fresh, stale]))
    targets = select_targets(gh, "valkey-io/valkey", max_age_days=30)
    assert [t.issue_number for t in targets] == [1]


def test_select_targets_by_issue_number_ignores_age(monkeypatch):
    monkeypatch.setattr(discover, "already_triaged", lambda issue: False)
    stale = _issue(number=42, body=_body(last_key=42), created_days_ago=200)
    gh = _gh_with_repo(_repo_with_issues([stale]))
    targets = select_targets(gh, "valkey-io/valkey", issue_numbers=[42])
    assert [t.issue_number for t in targets] == [42]


def test_select_targets_longest_marker_wins(monkeypatch):
    # A body carrying only the timeout namespace must classify as timeout even
    # though the assertion namespace is a prefix of no other, guarding the
    # longest-match selection.
    monkeypatch.setattr(discover, "already_triaged", lambda issue: False)
    issue = _issue(number=1, body=_body(namespace="valkey-ci-agent:test-timeout"))
    gh = _gh_with_repo(_repo_with_issues([issue]))
    targets = select_targets(gh, "valkey-io/valkey")
    assert targets[0].failure_type == "timeout"
