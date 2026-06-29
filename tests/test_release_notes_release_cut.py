"""Tests for the release-cut orchestration.

Branch resolution and the version/notes promotion are exercised against the
real valkey primitives in the test fixture; git and GitHub are mocked.
"""

from __future__ import annotations

import os
import shutil

import pytest

from scripts.release_notes import release_cut as rc
from scripts.release_notes.release_cut import (
    BranchPlan,
    commit_title,
    promote_and_bump,
    resolve_branch_plan,
    stage_release_name,
)

_FIXTURE_CLONE = os.path.join(os.path.dirname(__file__), "fixtures", "valkey_clone")


@pytest.fixture
def clone(tmp_path):
    dest = tmp_path / "clone"
    shutil.copytree(_FIXTURE_CLONE, dest)
    return str(dest)


class TestStageHelpers:
    def test_release_name_rc(self) -> None:
        assert stage_release_name("9.1.0", "rc2") == "9.1.0-rc2"

    def test_release_name_ga(self) -> None:
        assert stage_release_name("9.1.0", "ga") == "9.1.0"

    def test_commit_title_rc(self) -> None:
        assert commit_title("9.1.0", "rc2") == "Update version to 9.1.0-rc2 and add release notes"

    def test_commit_title_ga(self) -> None:
        assert commit_title("9.1.0", "ga") == "Add release notes entry for Valkey 9.1.0 GA"


class TestResolveBranchPlan:
    def _exists(self, monkeypatch, present):
        monkeypatch.setattr(rc, "_remote_branch_exists",
                            lambda repo_dir, branch: branch in present)

    def test_rc1_creates_pre_release(self, monkeypatch) -> None:
        self._exists(monkeypatch, set())
        plan = resolve_branch_plan("/d", version="9.1.0", stage="rc1", source_ref="unstable")
        assert plan == BranchPlan("rc1", "pre-release-9.1.0", "unstable", False, None)

    def test_rcN_continues_pre_release(self, monkeypatch) -> None:
        self._exists(monkeypatch, {"pre-release-9.1.0"})
        # Avoid the sequence-warning fetch by stubbing it.
        monkeypatch.setattr(rc, "_warn_rc_sequence", lambda *a, **k: None)
        plan = resolve_branch_plan("/d", version="9.1.0", stage="rc2", source_ref="unstable")
        assert plan.target == "pre-release-9.1.0"
        assert plan.base_ref == "pre-release-9.1.0"
        assert plan.continuing is True
        assert plan.rename_from is None

    def test_ga_renames_pre_release(self, monkeypatch) -> None:
        self._exists(monkeypatch, {"pre-release-9.1.0"})
        plan = resolve_branch_plan("/d", version="9.1.0", stage="ga", source_ref="unstable")
        assert plan.target == "9.1"
        assert plan.base_ref == "pre-release-9.1.0"
        assert plan.continuing is True
        assert plan.rename_from == "pre-release-9.1.0"

    def test_ga_continues_existing_minor(self, monkeypatch) -> None:
        self._exists(monkeypatch, {"9.1"})
        plan = resolve_branch_plan("/d", version="9.1.1", stage="ga", source_ref="unstable")
        assert plan.target == "9.1"
        assert plan.rename_from is None

    def test_ga_first_release_from_source(self, monkeypatch) -> None:
        self._exists(monkeypatch, set())
        plan = resolve_branch_plan("/d", version="9.1.0", stage="ga", source_ref="unstable")
        assert plan.target == "9.1"
        assert plan.base_ref == "unstable"
        assert plan.continuing is False

    def test_bad_stage_raises(self, monkeypatch) -> None:
        self._exists(monkeypatch, set())
        with pytest.raises(ValueError):
            resolve_branch_plan("/d", version="9.1.0", stage="beta", source_ref="unstable")

    def test_bad_version_raises(self, monkeypatch) -> None:
        self._exists(monkeypatch, set())
        with pytest.raises(ValueError):
            resolve_branch_plan("/d", version="9.1", stage="rc1", source_ref="unstable")


class TestPromoteAndBump:
    def _source_with_bullet(self, clone):
        from scripts.release_notes import render as render_mod
        fmt = render_mod.load_format_module(clone)
        from scripts.release_notes.models import CategorizedBullet
        existing = open(os.path.join(clone, "00-RELEASENOTES"), encoding="utf-8").read()
        return render_mod.apply_to_file(
            existing,
            render_mod.group_bullets(
                [CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="fix a crash")],
                fmt),
            fmt)

    def test_promotes_dated_section_and_bumps_version(self, clone, monkeypatch) -> None:
        # No contributor base -> skip the network lookup entirely.
        source = self._source_with_bullet(clone)
        version_text = open(os.path.join(clone, "src", "version.h"), encoding="utf-8").read()
        new_notes, new_version = promote_and_bump(
            clone,
            source_notes_text=source,
            dest_notes_text="",          # first cut: no prior changelog
            dest_version_text=version_text,
            version="9.1.0", stage_lc="rc1", urgency="LOW", date="2026-06-25",
            repo_full_name="valkey-io/valkey", contrib_base=None, token=None,
            security_fixes=None,
        )
        # Dated section rendered, bullet promoted, no ## Unreleased in drain mode.
        assert "Valkey 9.1.0-rc1" in new_notes
        assert "* fix a crash by @a (#40)" in new_notes
        assert "## Unreleased" not in new_notes
        # version.h macros bumped.
        assert '#define VALKEY_VERSION "9.1.0"' in new_version
        assert "#define VALKEY_VERSION_NUM 0x00090100" in new_version
        assert '#define VALKEY_RELEASE_STAGE "rc1"' in new_version

    def test_drains_prior_rc_notes(self, clone) -> None:
        # A prior rc1 dated section on the destination must survive into rc2.
        source = self._source_with_bullet(clone)
        prior = (
            "Valkey 9.1 release notes\n========================\n\n"
            "Valkey 9.1.0-rc1  -  Released 2026-06-01\n"
            "---------------------------------------\n\n"
            "Upgrade urgency LOW: ...\n\n### Bug Fixes\n* earlier fix by @x (#1)\n"
        )
        version_text = open(os.path.join(clone, "src", "version.h"), encoding="utf-8").read()
        new_notes, _ = promote_and_bump(
            clone, source_notes_text=source, dest_notes_text=prior,
            dest_version_text=version_text, version="9.1.0", stage_lc="rc2",
            urgency="LOW", date="2026-06-25", repo_full_name="valkey-io/valkey",
            contrib_base=None, token=None, security_fixes=None,
        )
        assert "Valkey 9.1.0-rc2" in new_notes
        assert "Valkey 9.1.0-rc1" in new_notes      # prior rc retained
        assert "* earlier fix by @x (#1)" in new_notes
        assert "* fix a crash by @a (#40)" in new_notes

    def test_contributor_list_included(self, clone, monkeypatch) -> None:
        source = self._source_with_bullet(clone)
        version_text = open(os.path.join(clone, "src", "version.h"), encoding="utf-8").read()
        # Stub the contributor lookup primitive so no network is touched.
        import scripts.release_notes.clone_tools as ct
        real_load = ct.load_releasetools_module

        def _fake_load(d, name):
            mod = real_load(d, name)
            if name == "gen_contributors":
                mod.list_contributors = lambda *a, **k: ["Jane Doe @jane", "Bob @bob"]
            return mod

        monkeypatch.setattr(rc, "load_releasetools_module", _fake_load)
        new_notes, _ = promote_and_bump(
            clone, source_notes_text=source, dest_notes_text="",
            dest_version_text=version_text, version="9.1.0", stage_lc="rc1",
            urgency="LOW", date="2026-06-25", repo_full_name="valkey-io/valkey",
            contrib_base="9.0.0", token=None, security_fixes=None,
        )
        assert "### Contributors" in new_notes
        assert "Jane Doe @jane" in new_notes

    def test_contributor_refs_resolved_for_compare_api(self, clone, monkeypatch) -> None:
        # Regression: the contributor base is a remote-tracking ref
        # (origin/unstable) and the head is the literal "HEAD". Both resolve for
        # git but 404 the GitHub compare API, which silently drops to the
        # names-only git-shortlog fallback. promote_and_bump must dereference both
        # to SHAs (via _compare_ref) before calling list_contributors, so the API
        # path, and thus the "Full Name @handle" format, is preserved.
        source = self._source_with_bullet(clone)
        version_text = open(os.path.join(clone, "src", "version.h"), encoding="utf-8").read()
        captured: dict = {}

        import scripts.release_notes.clone_tools as ct
        real_load = ct.load_releasetools_module

        def _fake_load(d, name):
            mod = real_load(d, name)
            if name == "gen_contributors":
                def _list(repo, base, head, token, *, repo_dir=None):
                    captured["base"] = base
                    captured["head"] = head
                    return ["Jane Doe @jane"]
                mod.list_contributors = _list
            return mod

        monkeypatch.setattr(rc, "load_releasetools_module", _fake_load)
        # Stub ref resolution so no real git repo is needed: prove the values
        # passed to list_contributors are what _compare_ref returned, not the
        # raw origin/unstable / HEAD refs.
        monkeypatch.setattr(rc, "_compare_ref",
                            lambda repo_dir, ref: {"origin/unstable": "base_sha", "HEAD": "head_sha"}[ref])
        promote_and_bump(
            clone, source_notes_text=source, dest_notes_text="",
            dest_version_text=version_text, version="9.1.0", stage_lc="rc1",
            urgency="LOW", date="2026-06-25", repo_full_name="valkey-io/valkey",
            contrib_base="origin/unstable", token="t", security_fixes=None,
        )
        assert captured["base"] == "base_sha"
        assert captured["head"] == "head_sha"  # never the literal "HEAD"

    def test_compare_ref_dereferences_to_sha(self, tmp_path) -> None:
        # _compare_ref turns a branch name into the commit SHA the compare API
        # wants; an unresolvable ref falls back to the ref as given.
        from scripts.common.proc import git_output, run_git
        repo = str(tmp_path / "r")
        os.makedirs(repo)
        run_git(repo, "init", "-q")
        run_git(repo, "config", "user.email", "t@e")
        run_git(repo, "config", "user.name", "t")
        (tmp_path / "r" / "f").write_text("x")
        run_git(repo, "add", "f")
        run_git(repo, "commit", "-q", "-m", "c")
        sha = git_output(repo, "rev-parse", "HEAD").strip()
        assert rc._compare_ref(repo, "HEAD") == sha
        assert rc._compare_ref(repo, "no-such-ref") == "no-such-ref"  # graceful fallback


class TestCutOrchestration:
    """End-to-end cut() with git + GitHub + pipeline mocked, real fixture worktree."""

    def _setup(self, monkeypatch, clone, *, line_exists, bullets=True, triage=(),
               stub_contrib_base=True):
        from scripts.release_notes import pipeline as pipeline_mod
        from scripts.release_notes import render as render_mod
        from scripts.release_notes.models import CategorizedBullet
        from scripts.release_notes.pipeline import RegenResult

        fmt = render_mod.load_format_module(clone)
        existing = open(os.path.join(clone, "00-RELEASENOTES"), encoding="utf-8").read()
        bl = ([CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="fix")]
              if bullets else [])
        updated = render_mod.apply_to_file(existing, render_mod.group_bullets(bl, fmt), fmt)
        monkeypatch.setattr(
            pipeline_mod, "regenerate_unreleased",
            lambda *a, **k: RegenResult(
                base_tag="9.0.0", existing_text=existing, updated_text=updated,
                included=1 if bullets else 0, bullet_count=len(bl), skipped=(),
                triage=tuple(triage), had_prs=True, wipes_existing=False),
        )
        # Record git commands; emulate worktree by copying the clone tree.
        calls = []

        def _fake_git(repo_dir, *args, **kwargs):
            calls.append(args)
            if args[:1] == ("worktree",) and args[1] == "add":
                dest = args[-2]
                shutil.copytree(clone, dest, dirs_exist_ok=True)
            from unittest.mock import MagicMock
            return MagicMock()

        monkeypatch.setattr(rc, "run_git", _fake_git)
        monkeypatch.setattr(rc, "_remote_branch_exists", lambda d, b: line_exists.get(b, False))
        if stub_contrib_base:
            monkeypatch.setattr(rc, "_contrib_base", lambda *a, **k: None)
        return calls

    def test_rc1_creates_line_and_prs_prep_branch_into_it(self, monkeypatch, clone):
        from unittest.mock import MagicMock
        calls = self._setup(monkeypatch, clone, line_exists={})
        repo = MagicMock()
        repo.get_pulls.return_value = []
        created = []

        def _create_pull(**kw):
            created.append(kw)
            return MagicMock(number=len(created), html_url=f"https://x/{len(created)}")

        repo.create_pull.side_effect = _create_pull
        monkeypatch.setattr(rc.publish_mod, "retry_github_call", lambda op, **k: op())

        rc.cut(
            repo, repo_full_name="valkey-io/valkey", source_clone_dir=clone,
            valkey_clone_dir=clone, source_ref="unstable", version="9.1.0", stage="rc1",
            urgency="LOW", date="2026-06-25", tag_glob=None, base_ref=None, contrib_base_ref=None,
            security_fixes=None, token="t", git_env={}, dry_run=False,
        )
        # The release line was created (it did not exist), and exactly one PR is
        # opened: the prep branch into the release line. No companion reset PR;
        # the source branch is never modified.
        pushed = [c for c in calls if c[:1] == ("push",)]
        assert any("refs/heads/pre-release-9.1.0" in " ".join(c) for c in pushed), pushed
        assert len(created) == 1
        assert created[0]["head"].startswith("agent/release-cut/")
        assert created[0]["base"] == "pre-release-9.1.0"
        # The source branch is never pushed to.
        assert not any("HEAD:unstable" in " ".join(c) or ":refs/heads/unstable" in " ".join(c)
                       for c in pushed)

    def test_triage_listed_in_release_pr_body(self, monkeypatch, clone):
        from unittest.mock import MagicMock

        from scripts.release_notes.models import MergedPR
        triage = (MergedPR(number=7, title="Untagged | thing", author="bob", url="https://x/7"),)
        calls = self._setup(monkeypatch, clone, line_exists={"pre-release-9.1.0": True}, triage=triage)
        monkeypatch.setattr(rc, "_warn_rc_sequence", lambda *a, **k: None)
        repo = MagicMock()
        repo.get_pulls.return_value = []
        created = []

        def _create_pull(**kw):
            created.append(kw)
            return MagicMock(number=1, html_url="https://x/1")

        repo.create_pull.side_effect = _create_pull
        monkeypatch.setattr(rc.publish_mod, "retry_github_call", lambda op, **k: op())

        rc.cut(
            repo, repo_full_name="valkey-io/valkey", source_clone_dir=clone,
            valkey_clone_dir=clone, source_ref="unstable", version="9.1.0", stage="rc2",
            urgency="LOW", date="2026-06-25", tag_glob=None, base_ref=None, contrib_base_ref=None,
            security_fixes=None, token="t", git_env={}, dry_run=False,
        )
        body = created[0]["body"]
        assert "Needs triage" in body
        assert "[#7](https://x/7)" in body
        assert "Untagged \\| thing" in body  # pipe escaped for the table

    def test_existing_line_not_recreated(self, monkeypatch, clone):
        from unittest.mock import MagicMock
        calls = self._setup(monkeypatch, clone, line_exists={"pre-release-9.1.0": True})
        monkeypatch.setattr(rc, "_warn_rc_sequence", lambda *a, **k: None)
        repo = MagicMock()
        repo.get_pulls.return_value = []
        repo.create_pull.return_value = MagicMock(number=2, html_url="https://x/2")
        monkeypatch.setattr(rc.publish_mod, "retry_github_call", lambda op, **k: op())

        rc.cut(
            repo, repo_full_name="valkey-io/valkey", source_clone_dir=clone,
            valkey_clone_dir=clone, source_ref="unstable", version="9.1.0", stage="rc2",
            urgency="LOW", date="2026-06-25", tag_glob=None, base_ref=None, contrib_base_ref=None,
            security_fixes=None, token="t", git_env={}, dry_run=False,
        )
        # No create-line push (branch already exists); only the prep-branch push.
        line_create = [c for c in calls
                       if c[:1] == ("push",) and "refs/heads/pre-release-9.1.0" in " ".join(c)]
        assert line_create == []

    def test_dry_run_pushes_nothing(self, monkeypatch, clone):
        from unittest.mock import MagicMock
        calls = self._setup(monkeypatch, clone, line_exists={})
        repo = MagicMock()
        rc.cut(
            repo, repo_full_name="valkey-io/valkey", source_clone_dir=clone,
            valkey_clone_dir=clone, source_ref="unstable", version="9.1.0", stage="rc1",
            urgency="LOW", date="2026-06-25", tag_glob=None, base_ref=None, contrib_base_ref=None,
            security_fixes=None, token="t", git_env={}, dry_run=True,
        )
        assert [c for c in calls if c[:1] == ("push",)] == []
        repo.create_pull.assert_not_called()

    def test_contrib_base_matches_notes_baseline(self, monkeypatch, clone):
        # The credits must span the same range as the bullets: the contributor
        # base passed to promote_and_bump equals regen.base_tag (9.0.0 here),
        # not whatever `git describe` would return from the source branch. _setup
        # leaves the real _contrib_base in place here so the wiring is exercised;
        # promote_and_bump is captured to read what it received.
        from unittest.mock import MagicMock
        self._setup(monkeypatch, clone, line_exists={"pre-release-9.1.0": True},
                    stub_contrib_base=False)
        monkeypatch.setattr(rc, "_warn_rc_sequence", lambda *a, **k: None)

        captured = {}

        def _promote(valkey_clone_dir, **kw):
            captured["contrib_base"] = kw["contrib_base"]
            return "NOTES", "VERSION"

        monkeypatch.setattr(rc, "promote_and_bump", _promote)
        repo = MagicMock()
        repo.get_pulls.return_value = []
        repo.create_pull.return_value = MagicMock(number=1, html_url="https://x/1")
        monkeypatch.setattr(rc.publish_mod, "retry_github_call", lambda op, **k: op())

        rc.cut(
            repo, repo_full_name="valkey-io/valkey", source_clone_dir=clone,
            valkey_clone_dir=clone, source_ref="unstable", version="9.1.0", stage="rc2",
            urgency="LOW", date="2026-06-25", tag_glob=None, base_ref=None, contrib_base_ref=None,
            security_fixes=None, token="t", git_env={}, dry_run=False,
        )
        # regen.base_tag is 9.0.0; the real _contrib_base must return it (via the
        # notes_base_ref branch), never reaching git describe.
        assert captured["contrib_base"] == "9.0.0"


class TestContribBase:
    _PLAN = BranchPlan("rc1", "pre-release-9.1.0", "unstable", False, None)

    def test_explicit_wins(self, monkeypatch) -> None:
        # Explicit --contrib-base-ref beats even the notes baseline.
        assert rc._contrib_base("/d", explicit="9.0.0", notes_base_ref="9.0.1",
                                plan=self._PLAN) == "9.0.0"

    def test_notes_base_ref_used_before_describe(self, monkeypatch) -> None:
        # The fix: the notes baseline anchors contributors, ahead of git describe.
        # describe would (wrongly) return an older nearest tag, but must not be hit.
        def _git(d, *a):
            raise AssertionError(f"git should not run when notes_base_ref is set: {a}")
        monkeypatch.setattr(rc, "git_output", _git)
        assert rc._contrib_base("/d", explicit=None, notes_base_ref="9.0.0",
                                plan=self._PLAN) == "9.0.0"

    def test_falls_back_to_last_tag_when_no_baseline(self, monkeypatch) -> None:
        # rc2+/ga path: notes baseline is a tag passed through, but if None we
        # still resolve via describe.
        monkeypatch.setattr(rc, "git_output",
                            lambda d, *a: "9.0.5\n" if a[0] == "describe" else "")
        assert rc._contrib_base("/d", explicit=None, notes_base_ref=None,
                                plan=self._PLAN) == "9.0.5"

    def test_falls_back_to_root_commit(self, monkeypatch) -> None:
        def _git(d, *a):
            if a[0] == "describe":
                raise RuntimeError("no tags")
            if a[0] == "rev-list":
                return "rootsha\n"
            return ""
        monkeypatch.setattr(rc, "git_output", _git)
        assert rc._contrib_base("/d", explicit=None, notes_base_ref=None,
                                plan=self._PLAN) == "rootsha"


class TestDedupAgainstDestination:
    """The tag-independent dedup: drop PRs the release line already credits.

    Without an RC tag to bound the range (the agent never pushes tags; a fork has
    none), discovery re-finds every PR on a continued cut, most visibly GA after
    the final RC. These cover the dedup that keeps promotion idempotent anyway.
    """

    _GA_PLAN = BranchPlan("ga", "9.1", "pre-release-9.1.0", True, "pre-release-9.1.0")

    def test_credited_reads_trailing_pr_refs(self) -> None:
        text = (
            "Valkey 9.1.0-rc1 - Released\n\n"
            "### Bug Fixes\n"
            "* fix a thing by @a (#44)\n"
            "* and another by @b (#51)\n"
        )
        assert rc._credited_pr_numbers(text) == {44, 51}

    def test_credited_ignores_non_bullet_and_inline_refs(self) -> None:
        # A "(#N)" in prose or a heading is not a credit; only a trailing ref on
        # a bullet line is. Mirrors the guidance comment that mentions "(#N)".
        text = (
            "See PR (#999) for context.\n"
            "## Heading mentioning (#998)\n"
            "* real credit by @a (#44)\n"
            "* a bullet with a mid-line (#7) ref but no trailing one\n"
        )
        assert rc._credited_pr_numbers(text) == {44}

    def test_drop_removes_only_overlapping_bullets(self) -> None:
        source = (
            "## Unreleased\n\n"
            "### Performance and Efficiency Improvements\n"
            "* already shipped by @a (#44)\n"
            "### Bug Fixes\n"
            "* genuinely new by @b (#60)\n"
        )
        filtered, dropped = rc._drop_already_credited(source, {44})
        assert dropped == [44]
        assert "(#44)" not in filtered
        assert "(#60)" in filtered                 # new PR survives
        assert "### Performance and Efficiency Improvements" in filtered  # empty header kept
        assert "### Bug Fixes" in filtered

    def test_drop_is_noop_without_overlap(self) -> None:
        source = "### Bug Fixes\n* new by @a (#60)\n"
        filtered, dropped = rc._drop_already_credited(source, set())
        assert dropped == []
        assert filtered == source

    def test_ga_after_final_rc_drops_all_and_warns(self, clone, monkeypatch) -> None:
        # End-to-end-ish: dest already credits #44; the source block re-found #44
        # (no tag to bound the range). The cut must drop it, render an empty dated
        # section, and warn in the PR body.
        from scripts.release_notes import pipeline as pipeline_mod
        from scripts.release_notes import render as render_mod
        from scripts.release_notes.models import CategorizedBullet
        from scripts.release_notes.pipeline import RegenResult

        fmt = render_mod.load_format_module(clone)
        existing = open(os.path.join(clone, "00-RELEASENOTES"), encoding="utf-8").read()
        bl = [CategorizedBullet(pr_number=44, author="a", category="Bug Fixes", text="fix")]
        updated = render_mod.apply_to_file(existing, render_mod.group_bullets(bl, fmt), fmt)
        monkeypatch.setattr(
            pipeline_mod, "regenerate_unreleased",
            lambda *a, **k: RegenResult(
                base_tag="unstable", existing_text=existing, updated_text=updated,
                included=1, bullet_count=1, skipped=(), triage=(), had_prs=True,
                wipes_existing=False,
            ),
        )
        # Destination line already credits #44 (carried from rc1).
        dest_notes = (
            "Valkey 9.1 release notes\n========================\n\n"
            "Valkey 9.1.0-rc1  -  Released 2026-06-01\n"
            "---------------------------------------\n\n"
            "Upgrade urgency LOW: ...\n\n### Bug Fixes\n* fix by @a (#44)\n"
        )
        captured = {}

        # Drive cut() with git/GitHub/promote stubbed; assert the dedup + warning.
        from scripts.release_notes import release_cut as rcmod
        monkeypatch.setattr(rcmod, "resolve_branch_plan", lambda *a, **k: self._GA_PLAN)
        monkeypatch.setattr(rcmod, "_remote_branch_exists", lambda d, b: True)
        monkeypatch.setattr(rcmod, "run_git", lambda *a, **k: None)
        monkeypatch.setattr(rcmod, "_read",
                            lambda p: dest_notes if p.endswith("00-RELEASENOTES")
                            else open(os.path.join(clone, "src", "version.h")).read())

        def _capture_promote(*a, **k):
            captured["source_notes"] = k["source_notes_text"]
            return ("NEWNOTES", "NEWVERSION")
        monkeypatch.setattr(rcmod, "promote_and_bump", _capture_promote)
        monkeypatch.setattr(rcmod, "_print_dry_run",
                            lambda *a, **k: captured.setdefault("already", a[5]))

        rcmod.cut(
            object(), repo_full_name="valkey-io/valkey", source_clone_dir=clone,
            valkey_clone_dir=clone, source_ref="unstable", version="9.1.0",
            stage="ga", urgency="LOW", date="2026-06-29", tag_glob=None,
            base_ref=None, contrib_base_ref=None, security_fixes=None,
            token="t", git_env={}, dry_run=True,
        )
        # #44 was dropped before promote saw the source block.
        assert "(#44)" not in captured["source_notes"]
        assert captured["already"] == [44]
