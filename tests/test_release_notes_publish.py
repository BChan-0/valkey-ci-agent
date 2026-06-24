"""Tests for branch/PR publishing (mocked GitHub + git)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.release_notes import publish as publish_mod
from scripts.release_notes.models import MergedPR, PRDisposition
from scripts.release_notes.publish import (
    build_branch_name,
    build_pr_body,
    commit_and_push,
    find_existing_pr,
    open_or_update_pr,
)


class TestBuildBranchName:
    def test_namespaced(self) -> None:
        assert build_branch_name("9.1") == "agent/release-notes/9.1"

    def test_flattens_slashes(self) -> None:
        assert build_branch_name("release/9.1") == "agent/release-notes/release-9.1"


class TestCommitAndPush:
    def test_refuses_non_namespaced_branch(self) -> None:
        with pytest.raises(RuntimeError, match="non-namespaced"):
            commit_and_push("/repo", branch="evil", message="m", git_env={})

    def test_pushes_force_with_lease_to_origin(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(publish_mod, "run_git",
                            lambda *a, **k: calls.append((a, k)) or MagicMock())
        commit_and_push("/repo", branch="agent/release-notes/9.1", message="m", git_env={"X": "1"})
        push = [a for a, k in calls if a[1] == "push"][0]
        assert push[1:] == ("push", "--force-with-lease", "origin", "HEAD:agent/release-notes/9.1")
        # The push carries the GitAuth env; local ops do not.
        push_kwargs = [k for a, k in calls if a[1] == "push"][0]
        assert push_kwargs.get("env") == {"X": "1"}

    def test_configures_bot_identity(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(publish_mod, "run_git",
                            lambda *a, **k: calls.append(a) or MagicMock())
        commit_and_push("/repo", branch="agent/release-notes/9.1", message="m", git_env={})
        config = [a for a in calls if a[1] == "config"]
        assert any("user.name" in a and publish_mod.BOT_NAME in a for a in config)
        assert any("user.email" in a and publish_mod.BOT_EMAIL in a for a in config)


class TestFindExistingPr:
    def test_returns_first_open_pr(self, monkeypatch) -> None:
        repo = MagicMock()
        existing = MagicMock(number=5)
        repo.get_pulls.return_value = [existing]
        found = find_existing_pr(repo, base_repo="valkey-io/valkey", push_repo=None,
                                 branch="agent/release-notes/9.1")
        assert found is existing

    def test_returns_none_when_no_open_pr(self) -> None:
        repo = MagicMock()
        repo.get_pulls.return_value = []
        assert find_existing_pr(repo, base_repo="valkey-io/valkey", push_repo=None,
                                branch="agent/release-notes/9.1") is None


class TestOpenOrUpdatePr:
    def test_updates_existing(self) -> None:
        repo = MagicMock()
        existing = MagicMock(number=5, html_url="https://x/5")
        url = open_or_update_pr(repo, base_repo="o/r", push_repo=None,
                                branch="agent/release-notes/9.1", base_branch="9.1",
                                title="t", body="b", existing=existing)
        assert url == "https://x/5"
        existing.edit.assert_called_once()
        repo.create_pull.assert_not_called()

    def test_creates_when_absent(self) -> None:
        repo = MagicMock()
        repo.create_pull.return_value = MagicMock(number=9, html_url="https://x/9")
        url = open_or_update_pr(repo, base_repo="o/r", push_repo=None,
                                branch="agent/release-notes/9.1", base_branch="9.1",
                                title="t", body="b", existing=None)
        assert url == "https://x/9"
        repo.create_pull.assert_called_once()


class TestBuildPrBody:
    def _triage(self):
        return [MergedPR(number=7, title="Untagged | thing", author="bob", url="https://x/7",
                         disposition=PRDisposition.TRIAGE)]

    def test_includes_metadata(self) -> None:
        body = build_pr_body(base_tag="9.1.0-rc1", head_ref="9.1", included=3,
                             skipped=[], triage=[])
        assert "9.1.0-rc1" in body and "`9.1`" in body and "**3**" in body

    def test_triage_table_present_and_escaped(self) -> None:
        body = build_pr_body(base_tag="9.1.0-rc1", head_ref="9.1", included=0,
                             skipped=[], triage=self._triage())
        assert "Needs triage" in body
        assert "[#7](https://x/7)" in body
        assert "@bob" in body
        # The pipe in the title is escaped so it doesn't break the table.
        assert "Untagged \\| thing" in body

    def test_skipped_listed(self) -> None:
        body = build_pr_body(base_tag="t", head_ref="9.1", included=1, skipped=[5, 6], triage=[])
        assert "#5" in body and "#6" in body
