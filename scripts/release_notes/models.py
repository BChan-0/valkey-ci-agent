"""Typed data model for the release-notes generation pipeline.

The pipeline is a chain of small, explicit handoffs:

    discover -> DiscoveryResult     (git/GitHub: which PRs merged since the last tag)
    classify -> MergedPR.disposition (code: include / exclude / triage, from labels)
    generate -> GenerationResult    (AI: one categorized bullet per included PR)
    render   -> updated 00-RELEASENOTES text (code: canonical format, authoritative)
    publish  -> PR url              (code: branch + PR on valkey)

AI populates only the judgment fields (``CategorizedBullet`` category and
text); code populates every factual field (PR number, author, labels, the
trailing ``(#N)``, the ``by @handle`` attribution). The split is deliberate:
the model decides what to say and where it goes, never the dedup identity or
the format the downstream release tooling parses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PRDisposition(str, Enum):
    """What the labelling says to do with a discovered PR."""

    INCLUDE = "include"   # carries only 'release-notes'
    EXCLUDE = "exclude"   # carries only 'no-release-notes'
    TRIAGE = "triage"     # neither label, or both -> a human must decide


@dataclass(frozen=True)
class MergedPR:
    """One PR discovered in the release range. Entirely factual.

    ``author`` is a bare login (no leading ``@``); it may be ``""`` when GitHub
    returns no user (a deleted/ghost account), which render must tolerate.
    ``merge_commit_sha`` may be ``""`` for a PR resolved from a commit subject
    that was never confirmed against the API.
    """

    number: int
    title: str
    author: str
    url: str
    labels: tuple[str, ...] = ()
    merge_commit_sha: str = ""
    disposition: PRDisposition = PRDisposition.TRIAGE


@dataclass(frozen=True)
class CategorizedBullet:
    """One note line's content. ``category`` must be a canonical category.

    ``text`` is the human-readable description ONLY: it must not contain the
    ``(#N)`` reference or the ``by @handle`` attribution, which render appends
    so they land in the exact positions the release tooling's regexes expect.
    """

    pr_number: int
    author: str
    category: str
    text: str


@dataclass(frozen=True)
class GenerationResult:
    """The AI's output for the whole range. Pure judgment."""

    bullets: tuple[CategorizedBullet, ...] = ()
    skipped: tuple[int, ...] = ()   # PR numbers the model declined to summarize


@dataclass(frozen=True)
class DiscoveryResult:
    """Factual summary of the release range, from discover.py.

    ``prs`` is deduplicated to one entry per originating PR number, so a change
    cherry-picked across the range collapses to a single PR.
    """

    base_tag: str
    base_sha: str
    head_ref: str
    head_sha: str
    prs: tuple[MergedPR, ...] = field(default_factory=tuple)
