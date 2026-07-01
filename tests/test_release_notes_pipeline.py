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


def test_empty_range(monkeypatch, clone):
    _patch(monkeypatch, prs=())
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.had_prs is False
    assert r.updated_text == r.existing_text


def test_generates_and_renders(monkeypatch, clone):
    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs,
           bullets=(CategorizedBullet(pr_number=40, author="a", category="Bug Fixes", text="fix"),))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.had_prs and r.included == 1 and r.bullet_count == 1
    assert "* fix by @a (#40)" in r.updated_text
    assert not r.wipes_existing  # the fixture block was empty, so nothing is lost


def test_triage_surfaced(monkeypatch, clone):
    prs = (MergedPR(number=50, title="untagged", author="z", url="u", labels=()),)
    _patch(monkeypatch, prs=prs)
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert [p.number for p in r.triage] == [50]
    assert r.included == 0


def test_wipes_existing_detected(monkeypatch, clone):
    # Seed a populated block, then have generate produce nothing.
    from scripts.release_notes import render as render_mod
    fmt = render_mod.load_format_module(clone)
    notes = os.path.join(clone, "00-RELEASENOTES")
    existing = open(notes, encoding="utf-8").read()
    seeded = render_mod.apply_to_file(
        existing,
        render_mod.group_bullets(
            [CategorizedBullet(pr_number=1, author="x", category="Bug Fixes", text="prior")], fmt),
        fmt)
    open(notes, "w", encoding="utf-8").write(seeded)

    prs = (MergedPR(number=40, title="t", author="a", url="u", labels=("release-notes",)),)
    _patch(monkeypatch, prs=prs, bullets=(), skipped=(40,))
    r = pipeline_mod.regenerate_unreleased(object(), clone, head_ref="9.1", tag_glob=None)
    assert r.bullet_count == 0
    assert r.wipes_existing is True


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
    assert "first" in r.updated_text
    assert "second" not in r.updated_text


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
