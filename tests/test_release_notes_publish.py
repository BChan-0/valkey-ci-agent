"""Tests for the PR-side publish primitives (mocked GitHub)."""

from __future__ import annotations

from unittest.mock import MagicMock

from scripts.release_notes.publish import escape_cell, find_existing_pr, open_or_update_pr


class TestFindExistingPr:
    def test_returns_first_open_pr(self) -> None:
        repo = MagicMock()
        existing = MagicMock(number=5)
        repo.get_pulls.return_value = [existing]
        found = find_existing_pr(repo, base_repo="valkey-io/valkey", push_repo=None,
                                 branch="agent/release-cut/9.1.0-rc1")
        assert found is existing

    def test_returns_none_when_no_open_pr(self) -> None:
        repo = MagicMock()
        repo.get_pulls.return_value = []
        assert find_existing_pr(repo, base_repo="valkey-io/valkey", push_repo=None,
                                branch="agent/release-cut/9.1.0-rc1") is None


class TestOpenOrUpdatePr:
    def test_updates_existing(self) -> None:
        repo = MagicMock()
        existing = MagicMock(number=5, html_url="https://x/5")
        url = open_or_update_pr(repo, base_repo="o/r", push_repo=None,
                                branch="agent/release-cut/9.1.0-rc1", base_branch="pre-release-9.1.0",
                                title="t", body="b", existing=existing)
        assert url == "https://x/5"
        existing.edit.assert_called_once()
        repo.create_pull.assert_not_called()

    def test_creates_when_absent(self) -> None:
        repo = MagicMock()
        repo.create_pull.return_value = MagicMock(number=9, html_url="https://x/9")
        url = open_or_update_pr(repo, base_repo="o/r", push_repo=None,
                                branch="agent/release-cut/9.1.0-rc1", base_branch="pre-release-9.1.0",
                                title="t", body="b", existing=None)
        assert url == "https://x/9"
        repo.create_pull.assert_called_once()


class TestEscapeCell:
    def test_escapes_pipe_and_newline(self) -> None:
        assert escape_cell("a | b\nc") == "a \\| b c"

    def test_plain_text_unchanged(self) -> None:
        assert escape_cell("normal title") == "normal title"
