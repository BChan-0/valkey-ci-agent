"""Shared AI-notes pipeline: discover -> classify -> generate -> render.

The release cut (:mod:`release_cut`) needs one step: take a release line's
clone, find the labelled PRs merged since its last tag, and produce the updated
``00-RELEASENOTES`` text with a freshly AI-written ``## Unreleased`` block. This
module owns that step.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from scripts.release_notes import discover as discover_mod
from scripts.release_notes import generate as generate_mod
from scripts.release_notes import render as render_mod
from scripts.release_notes.classify import classify
from scripts.release_notes.models import MergedPR

logger = logging.getLogger(__name__)

NOTES_FILE = "00-RELEASENOTES"


@dataclass(frozen=True)
class RegenResult:
    """Outcome of regenerating a release line's ``## Unreleased`` block."""

    base_tag: str
    existing_text: str          # the file before regeneration
    updated_text: str           # the file with the new ## Unreleased block
    included: int               # PRs included (labelled release-notes)
    bullet_count: int           # bullets the model actually produced
    skipped: tuple[int, ...]    # PR numbers the model declined
    triage: tuple[MergedPR, ...]  # untagged / double-labelled PRs
    had_prs: bool               # whether the range contained any PR at all
    wipes_existing: bool        # True if writing updated_text would blank a populated block


def regenerate_unreleased(
    repo: Any, clone_dir: str, *, head_ref: str, tag_glob: str | None,
    base_ref: str | None = None,
) -> RegenResult:
    """Discover the range, generate bullets, and render the updated notes text.

    Reads (does not write) ``clone_dir``'s ``00-RELEASENOTES``. Returns a
    :class:`RegenResult`. The cut caller promotes ``updated_text`` regardless of
    whether the range was empty (an RC->GA with no intervening PRs is a valid
    cut), but consults ``bullet_count``/``wipes_existing`` as a safety net and
    ``triage`` for the PR body.

    ``base_ref`` overrides tag-based baseline resolution (see :func:`discover`).
    """
    notes_path = os.path.join(clone_dir, NOTES_FILE)
    with open(notes_path, "r", encoding="utf-8") as fh:
        existing = fh.read()

    discovery = discover_mod.discover(
        repo, clone_dir, head_ref, tag_glob=tag_glob, base_ref=base_ref
    )
    if not discovery.prs:
        return RegenResult(
            base_tag=discovery.base_tag, existing_text=existing, updated_text=existing,
            included=0, bullet_count=0, skipped=(), triage=(), had_prs=False,
            wipes_existing=False,
        )

    include, _exclude, triage = classify(discovery.prs)
    logger.info(
        "%d included, %d excluded, %d triage", len(include),
        len(discovery.prs) - len(include) - len(triage), len(triage),
    )

    fmt = render_mod.load_format_module(clone_dir)
    gen = generate_mod.generate(include, repo_dir=clone_dir, categories=fmt.CATEGORIES)
    grouped = render_mod.group_bullets(gen.bullets, fmt)
    updated = render_mod.apply_to_file(existing, grouped, fmt)

    # Would writing this blank an already-populated block? True only when there
    # are no bullets and resetting the block changes the file (i.e. it had content).
    wipes = not grouped and render_mod.apply_to_file(existing, {}, fmt) != existing

    return RegenResult(
        base_tag=discovery.base_tag, existing_text=existing, updated_text=updated,
        included=len(include), bullet_count=len(gen.bullets), skipped=tuple(gen.skipped),
        triage=tuple(triage), had_prs=True, wipes_existing=wipes,
    )
