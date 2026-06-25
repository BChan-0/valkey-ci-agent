"""Tests for the release-cut entry point (orchestration mocked).

main is now cut-only: it always dispatches to release_cut.cut(). The cut
internals are tested in test_release_notes_release_cut.py; here we cover
argument validation, the baseline-glob/base-ref resolution, and that the parsed
inputs reach cut().
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts.release_notes import main as main_mod
from scripts.release_notes.main import main


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "Github", MagicMock())
    monkeypatch.setattr(main_mod, "retry_github_call", lambda op, **k: op())
    auth = MagicMock()
    auth.__enter__.return_value.env.return_value = {"GIT_PASSWORD": "x"}
    monkeypatch.setattr(main_mod, "GitAuth", lambda *a, **k: auth)
    monkeypatch.setattr(main_mod, "github_https_url", lambda name: f"https://github.com/{name}.git")
    monkeypatch.setattr(main_mod, "run_git", lambda *a, **k: MagicMock())
    monkeypatch.setattr(main_mod.tempfile, "mkdtemp", lambda *a, **k: str(tmp_path / "clone"))
    monkeypatch.setattr(main_mod.shutil, "rmtree", lambda *a, **k: None)
    return monkeypatch


def _capture_cut(patched):
    captured = {}

    def _cut(repo, **kwargs):
        captured.update(kwargs)
        return 0

    patched.setattr(main_mod.cut_mod, "cut", _cut)
    return captured


# --- argument validation ---

def test_missing_token_is_usage_error(monkeypatch):
    monkeypatch.delenv("RELEASE_NOTES_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("TARGET_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--head-ref", "unstable", "--version", "9.1.0", "--stage", "rc1", "--urgency", "LOW"])
    assert exc.value.code == 2


def test_missing_head_ref_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["--token", "t", "--version", "9.1.0", "--stage", "rc1", "--urgency", "LOW"])
    assert exc.value.code == 2


def test_missing_version_stage_urgency_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["--token", "t", "--head-ref", "unstable"])
    assert exc.value.code == 2


# --- dispatch + arg threading ---

def test_dispatches_to_cut_with_parsed_args(patched):
    captured = _capture_cut(patched)
    rc = main(["--token", "t", "--head-ref", "unstable",
               "--version", "9.1.0", "--stage", "rc2", "--urgency", "HIGH"])
    assert rc == 0
    assert captured["version"] == "9.1.0"
    assert captured["stage"] == "rc2"
    assert captured["urgency"] == "HIGH"
    assert captured["source_ref"] == "unstable"


def test_dry_run_threads_through(patched):
    captured = _capture_cut(patched)
    main(["--token", "t", "--head-ref", "unstable", "--version", "9.1.0",
          "--stage", "rc1", "--urgency", "LOW", "--dry-run"])
    assert captured["dry_run"] is True


def test_cut_failure_returns_one(patched):
    def _cut(repo, **kwargs):
        raise RuntimeError("boom")

    patched.setattr(main_mod.cut_mod, "cut", _cut)
    rc = main(["--token", "t", "--head-ref", "unstable", "--version", "9.1.0",
               "--stage", "rc1", "--urgency", "LOW"])
    assert rc == 1


# --- baseline glob / base-ref resolution ---

class TestDefaultTagGlob:
    def test_rc2_makes_rc_glob(self) -> None:
        # rc2+ anchors to the prior RC of this version.
        assert main_mod._default_tag_glob("9.1.0", "rc2") == "9.1.0-rc*"
        assert main_mod._default_tag_glob("9.1.0", "rc10") == "9.1.0-rc*"

    def test_rc1_has_no_glob(self) -> None:
        # rc1 has no rc0 to anchor to -> no glob (uses base_ref instead).
        assert main_mod._default_tag_glob("9.1.0", "rc1") is None

    def test_ga_has_no_glob(self) -> None:
        # GA continues an existing line; the no-glob nearest tag is correct.
        assert main_mod._default_tag_glob("9.1.0", "ga") is None

    def test_non_version_is_none(self) -> None:
        assert main_mod._default_tag_glob("9.1", "rc2") is None


class TestDefaultBaseRefForRc1:
    def test_previous_minor_ga(self) -> None:
        assert main_mod._default_base_ref_for_rc1("9.1.0") == "9.0.0"

    def test_patch_release(self) -> None:
        assert main_mod._default_base_ref_for_rc1("9.2.3") == "9.1.0"

    def test_first_minor_of_major_has_none(self) -> None:
        # 9.0.0 has no obvious previous-minor GA on the same major.
        assert main_mod._default_base_ref_for_rc1("9.0.0") is None

    def test_non_version_is_none(self) -> None:
        assert main_mod._default_base_ref_for_rc1("9.1") is None


def test_rc2_default_glob_passed_to_cut(patched):
    captured = _capture_cut(patched)
    main(["--token", "t", "--head-ref", "unstable",
          "--version", "9.1.0", "--stage", "rc2", "--urgency", "LOW"])
    assert captured["tag_glob"] == "9.1.0-rc*"
    assert captured["base_ref"] is None


def test_rc1_without_base_ref_warns_and_defaults(patched, caplog):
    captured = _capture_cut(patched)
    import logging
    with caplog.at_level(logging.WARNING):
        main(["--token", "t", "--head-ref", "unstable",
              "--version", "9.1.0", "--stage", "rc1", "--urgency", "LOW"])
    # Defaults base_ref to the previous release, and suppresses the doomed glob.
    assert captured["base_ref"] == "9.0.0"
    assert captured["tag_glob"] is None
    assert any("rc1" in r.message and "9.0.0" in r.message for r in caplog.records)


def test_rc1_with_explicit_base_ref_no_override(patched, caplog):
    captured = _capture_cut(patched)
    import logging
    with caplog.at_level(logging.WARNING):
        main(["--token", "t", "--head-ref", "unstable", "--version", "9.1.0",
              "--stage", "rc1", "--urgency", "LOW", "--base-ref", "9.0.4"])
    assert captured["base_ref"] == "9.0.4"  # user value wins, not the derived default
    assert captured["tag_glob"] is None
    # No rc1 baseline warning when the user supplied one.
    assert not any("Defaulting --base-ref" in r.message for r in caplog.records)


def test_rc1_first_minor_warns_without_default(patched, caplog):
    captured = _capture_cut(patched)
    import logging
    with caplog.at_level(logging.WARNING):
        main(["--token", "t", "--head-ref", "unstable",
              "--version", "9.0.0", "--stage", "rc1", "--urgency", "LOW"])
    # No previous-minor could be derived -> base_ref stays None, loud warning.
    assert captured["base_ref"] is None
    assert any("no previous-minor release" in r.message for r in caplog.records)


def test_explicit_base_ref_overrides_glob(patched):
    captured = _capture_cut(patched)
    main(["--token", "t", "--head-ref", "feature/release-notes-automation",
          "--version", "9.1.0", "--stage", "rc2", "--urgency", "LOW", "--base-ref", "unstable"])
    assert captured["base_ref"] == "unstable"
    assert captured["tag_glob"] is None
