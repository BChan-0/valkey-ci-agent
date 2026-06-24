"""Tests for the release-notes entry point (orchestration mocked)."""

from __future__ import annotations

import os
import shutil
from unittest.mock import MagicMock

import pytest

from scripts.release_notes import main as main_mod
from scripts.release_notes.main import main
from scripts.release_notes.models import (
    CategorizedBullet,
    DiscoveryResult,
    GenerationResult,
    MergedPR,
)

_FIXTURE_CLONE = os.path.join(os.path.dirname(__file__), "fixtures", "valkey_clone")


def _discovery(prs=()):
    return DiscoveryResult(base_tag="9.1.0-rc1", base_sha="s", head_ref="9.1", head_sha="h", prs=prs)


@pytest.fixture
def clone(tmp_path):
    """A throwaway copy of the fixture clone so a run never mutates the committed one."""
    dest = tmp_path / "clone"
    shutil.copytree(_FIXTURE_CLONE, dest)
    return str(dest)


@pytest.fixture
def patched(monkeypatch, clone):
    monkeypatch.setattr(main_mod, "Github", MagicMock())
    monkeypatch.setattr(main_mod, "retry_github_call", lambda op, **k: op())
    auth = MagicMock()
    auth.__enter__.return_value.env.return_value = {"GIT_PASSWORD": "x"}
    monkeypatch.setattr(main_mod, "GitAuth", lambda *a, **k: auth)
    monkeypatch.setattr(main_mod, "github_https_url", lambda name: f"https://github.com/{name}.git")
    monkeypatch.setattr(main_mod, "run_git", lambda *a, **k: MagicMock())
    monkeypatch.setattr(main_mod.tempfile, "mkdtemp", lambda *a, **k: clone)
    return monkeypatch


def test_missing_token_is_usage_error(monkeypatch):
    monkeypatch.delenv("RELEASE_NOTES_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("TARGET_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--head-ref", "9.1"])
    assert exc.value.code == 2


def test_missing_head_ref_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["--token", "t"])
    assert exc.value.code == 2


def test_empty_range_is_noop(patched):
    patched.setattr(main_mod.discover_mod, "discover", lambda *a, **k: _discovery(()))
    publish = MagicMock()
    patched.setattr(main_mod.publish_mod, "publish", publish)
    rc = main(["--token", "t", "--repo", "valkey-io/valkey", "--head-ref", "9.1"])
    assert rc == 0
    publish.assert_not_called()


def test_dry_run_prints_without_publishing(patched, capsys):
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    patched.setattr(main_mod.discover_mod, "discover", lambda *a, **k: _discovery(prs))
    patched.setattr(main_mod.generate_mod, "generate", lambda *a, **k: GenerationResult(
        bullets=(CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="fix"),)))
    publish = MagicMock()
    patched.setattr(main_mod.publish_mod, "publish", publish)
    rc = main(["--token", "t", "--head-ref", "9.1", "--dry-run"])
    assert rc == 0
    publish.assert_not_called()
    out = capsys.readouterr().out
    assert "## Unreleased" in out
    assert "* fix by @a (#40)" in out


def test_full_run_publishes_and_writes_file(patched, clone):
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    patched.setattr(main_mod.discover_mod, "discover", lambda *a, **k: _discovery(prs))
    patched.setattr(main_mod.generate_mod, "generate", lambda *a, **k: GenerationResult(
        bullets=(CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="fix"),)))

    # The clone is removed in main's finally block, so read the written file
    # from inside the publish mock (which runs before cleanup).
    seen = {}

    def _publish(repo, repo_dir, **kwargs):
        seen["written"] = open(os.path.join(repo_dir, "00-RELEASENOTES"), encoding="utf-8").read()
        return "https://x/pr/1"

    patched.setattr(main_mod.publish_mod, "publish", _publish)
    rc = main(["--token", "t", "--head-ref", "9.1"])
    assert rc == 0
    assert "* fix by @a (#40)" in seen["written"]
    # The temp clone is cleaned up after the run.
    assert not os.path.exists(clone)


class TestDefaultTagGlob:
    def test_bare_minor_line(self) -> None:
        assert main_mod._default_tag_glob("9.1") == "9.1.*"

    def test_prefixed_ref(self) -> None:
        assert main_mod._default_tag_glob("release/9.1") == "9.1.*"

    def test_non_version_ref_is_none(self) -> None:
        assert main_mod._default_tag_glob("unstable") is None

    def test_full_version_is_none(self) -> None:
        # A full M.m.p is not a release line; leave the glob unset.
        assert main_mod._default_tag_glob("9.1.0") is None


def test_default_tag_glob_passed_to_discover(patched):
    seen = {}

    def _discover(repo, repo_dir, head_ref, *, tag_glob=None):
        seen["tag_glob"] = tag_glob
        return _discovery(())

    patched.setattr(main_mod.discover_mod, "discover", _discover)
    main(["--token", "t", "--head-ref", "9.1"])
    assert seen["tag_glob"] == "9.1.*"


def test_all_skipped_refuses_to_blank_existing_block(patched, clone):
    # An included PR exists but the model returns zero bullets -> must NOT wipe
    # the (populated) block and must fail rather than publish an empty change.
    from scripts.release_notes import render as render_mod
    fmt = render_mod.load_format_module(clone)
    # Seed the fixture with an existing bullet so a wipe would be destructive.
    notes = os.path.join(clone, "00-RELEASENOTES")
    existing = open(notes, encoding="utf-8").read()
    seeded = render_mod.apply_to_file(
        existing,
        render_mod.group_bullets(
            [CategorizedBullet(pr_number=1, author="x", category="Bug Fixes", text="prior note")], fmt),
        fmt,
    )
    open(notes, "w", encoding="utf-8").write(seeded)

    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    patched.setattr(main_mod.discover_mod, "discover", lambda *a, **k: _discovery(prs))
    patched.setattr(main_mod.generate_mod, "generate",
                    lambda *a, **k: GenerationResult(bullets=(), skipped=(40,)))
    publish = MagicMock()
    patched.setattr(main_mod.publish_mod, "publish", publish)
    rc = main(["--token", "t", "--head-ref", "9.1"])
    assert rc == 1
    publish.assert_not_called()


def test_all_triage_still_publishes(patched):
    # No included PRs, but an untagged PR exists -> still open a PR with the triage table.
    prs = (MergedPR(number=50, title="untagged", author="z", url="u", labels=()),)
    patched.setattr(main_mod.discover_mod, "discover", lambda *a, **k: _discovery(prs))
    patched.setattr(main_mod.generate_mod, "generate", lambda *a, **k: GenerationResult())
    publish = MagicMock(return_value="https://x/pr/2")
    patched.setattr(main_mod.publish_mod, "publish", publish)
    rc = main(["--token", "t", "--head-ref", "9.1"])
    assert rc == 0
    publish.assert_called_once()
    _, kwargs = publish.call_args
    assert [p.number for p in kwargs["triage"]] == [50]
