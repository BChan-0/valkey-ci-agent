"""Tests for the shared discover->classify->generate->render pipeline."""

from __future__ import annotations

import os
import shutil

import pytest

from scripts.release_notes import pipeline as pipeline_mod
from scripts.release_notes.models import (
    CategorizedBullet,
    DiscoveryResult,
    GenerationResult,
    MergedPR,
)

_FIXTURE_CLONE = os.path.join(os.path.dirname(__file__), "fixtures", "valkey_clone")


@pytest.fixture
def clone(tmp_path):
    dest = tmp_path / "clone"
    shutil.copytree(_FIXTURE_CLONE, dest)
    return str(dest)


def _patch(monkeypatch, *, prs, bullets=(), skipped=()):
    monkeypatch.setattr(pipeline_mod.discover_mod, "discover",
                        lambda *a, **k: DiscoveryResult(base_tag="9.1.0-rc1", base_sha="s",
                                                        head_ref="9.1", head_sha="h", prs=prs))
    monkeypatch.setattr(pipeline_mod.generate_mod, "generate",
                        lambda *a, **k: GenerationResult(bullets=bullets, skipped=skipped))


def _all_lines(grouped):
    return [line for lines in grouped.values() for line in lines]


def test_empty_range(monkeypatch, clone):
    _patch(monkeypatch, prs=())
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.had_prs is False
    assert r.grouped == {}


def test_generates_and_renders(monkeypatch, clone):
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs,
           bullets=(CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="fix"),))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.had_prs and r.included == 1 and r.bullet_count == 1
    assert r.grouped["Bug Fixes"] == ["* fix by @a (#40)"]


def test_triage_surfaced(monkeypatch, clone):
    prs = (MergedPR(number=50, title="untagged", author="z", url="u", labels=()),)
    _patch(monkeypatch, prs=prs)
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert [p.number for p in r.triage] == [50]
    assert r.included == 0


def test_no_usable_bullets_yields_empty_grouped(monkeypatch, clone):
    # Included PRs but generate produces nothing: bullet_count is 0 and grouped is
    # empty, which is what the cut's blank-cut guard (included and not bullet_count)
    # keys on to refuse the cut.
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs, bullets=(), skipped=(40,))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.bullet_count == 0
    assert r.grouped == {}


def test_reserved_only_bullets_count_as_zero(monkeypatch, clone):
    # Regression: bullet_count must reflect what group_bullets actually renders,
    # not what the model returned. If the model's only bullet is under a reserved
    # category ("Security Fixes", auto-generated at release), group_bullets drops
    # it -> grouped == {} -> bullet_count 0, so the cut's blank-cut guard
    # (included and not bullet_count) fires instead of silently cutting empty notes.
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs, bullets=(
        CategorizedBullet(pr_number=40, author="a", category="Security Fixes", text="hallucinated"),
    ))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.bullet_count == 0        # the reserved-category bullet was dropped, not rendered
    assert r.grouped == {}


def test_duplicate_pr_bullets_deduped_and_recorded(monkeypatch, clone):
    # The model emits two bullets for the same PR; only the first survives and the
    # PR number is recorded so the caller can flag it in the body.
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs, bullets=(
        CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="first"),
        CategorizedBullet(pr_number=40, author="a", category="New Features", text="second"),
    ))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.bullet_count == 1            # second dropped
    assert r.duplicate_prs == (40,)
    lines = _all_lines(r.grouped)
    assert any("first" in line for line in lines)
    assert not any("second" in line for line in lines)


def test_uncertain_bullet_surfaced(monkeypatch, clone):
    # A rendered bullet the model flagged uncertain is reported as an UncertainNote
    # so the cut can list it in the PR body; the bullet still renders normally.
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs, bullets=(
        CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="fix",
                          uncertain=True, uncertain_reason="could be Behavior Changes"),
    ))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.bullet_count == 1
    assert [(n.pr_number, n.category, n.reason) for n in r.uncertain] == [
        (40, "Bug Fixes", "could be Behavior Changes")
    ]


def test_confident_bullets_produce_no_uncertain_notes(monkeypatch, clone):
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs, bullets=(
        CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="fix"),
    ))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.uncertain == ()


def test_uncertain_dropped_bullet_not_surfaced(monkeypatch, clone):
    # A bullet flagged uncertain but dropped by group_bullets (reserved category)
    # must NOT appear in the uncertain notes: it isn't rendered, so there is
    # nothing for a reviewer to check. Only rendered notes are surfaced.
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs, bullets=(
        CategorizedBullet(pr_number=40, author="a", category="Security Fixes", text="dropped",
                          uncertain=True, uncertain_reason="should not surface"),
    ))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.grouped == {}
    assert r.uncertain == ()


def test_dedup_bullets_by_pr_keeps_first_preserves_order(monkeypatch):
    bl = [
        CategorizedBullet(pr_number=1, author="a", category="Bug Fixes", text="one"),
        CategorizedBullet(pr_number=2, author="b", category="Bug Fixes", text="two"),
        CategorizedBullet(pr_number=1, author="a", category="New Features", text="dup"),
    ]
    kept, dups = pipeline_mod._dedup_bullets_by_pr(bl)
    assert [b.pr_number for b in kept] == [1, 2]
    assert [b.text for b in kept] == ["one", "two"]
    assert dups == (1,)
