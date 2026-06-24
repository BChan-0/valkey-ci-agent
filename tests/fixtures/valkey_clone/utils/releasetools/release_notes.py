#!/usr/bin/env python3
"""Parse the 00-RELEASENOTES "## Unreleased" block.

The unstable branch keeps a "## Unreleased" block in 00-RELEASENOTES that
user-facing PRs append to as they merge. This module extracts and measures
that block; it is consumed by the release-notes CI check
(check_release_notes.py) to require a net-new entry on labelled PRs.

The module is pure (no I/O, no network) so it is cheap to unit test. Rendering
and promotion of the block into dated release sections live elsewhere and are
added alongside the release-cutting tooling.
"""

from __future__ import annotations

import datetime
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

# Canonical category order. Contributors append under these headers in the
# "## Unreleased" block; dated sections render them in this same order.
CATEGORIES: List[str] = [
    "Behavior Changes",
    "New Features and Enhanced Behavior",
    "Performance and Efficiency Improvements",
    "Bug Fixes",
    "Command and API Updates",
    "Module API Changes",
    "Observability and Logging",
    "Build and Tooling",
]

# Security fixes are never seeded in the unstable block: they are supplied at
# release-cut time from the embargo CVE list (prepare_release --security-fix) and
# render first, ahead of the canonical categories.
SECURITY_CATEGORY = "Security Fixes"

# The contributor list is generated from the merged-PR authors of the release
# range (gen_contributors.py), deduplicated and alpha-sorted, not hand-edited.
CONTRIBUTORS_SECTION = "Contributors"

# Sections that are populated automatically at release time and therefore must
# never be hand-added to the "## Unreleased" block. If one appears there it is
# *ignored* at render time (the generated section is the source of truth) rather
# than merged, which would otherwise emit a duplicate header. Callers (the CI
# check and the release cut) surface a non-blocking warning so a maintainer
# removes the stray section.
RESERVED_SECTIONS = (SECURITY_CATEGORY, CONTRIBUTORS_SECTION)

UNRELEASED_HEADER = "## Unreleased"

UNRELEASED_COMMENT = """<!--
Contributors: if your change is user-facing, add the `release-notes` label to your
PR and append a bullet under the matching category below, in the form:

    * <human-readable description> by @<your-github-handle> (#<PR number>)

If your change is not user-facing, add the `no-release-notes` label instead. A CI
check requires exactly one of these two labels, and a note here when `release-notes`
is set. The `.github/workflows/prepare-release.yml` workflow promotes this block into
a dated release section when a release is cut, so keep entries user-readable.

Do not add `### Security Fixes` or `### Contributors` sections here: they are
generated automatically when a release is cut (security fixes from the embargo
CVE list, contributors from the merged PRs), so anything you add under them is
dropped. The CI check warns if you do.
-->"""

# Upgrade urgency legend rendered at the top of a release-branch notes file.
URGENCY_LEGEND = """Upgrade urgency levels:

| Level    | Meaning                                                             |
|----------|---------------------------------------------------------------------|
| LOW      | No need to upgrade unless there are new features you want to use.   |
| MODERATE | Program an upgrade of the server, but it's not urgent.              |
| HIGH     | There is a critical bug that may affect a subset of users. Upgrade! |
| CRITICAL | There is a critical bug affecting MOST USERS. Upgrade ASAP.         |
| SECURITY | There are security fixes in the release.                            |"""

VALID_URGENCIES = ("LOW", "MODERATE", "HIGH", "CRITICAL", "SECURITY")

_BULLET_RE = re.compile(r"^\s*[*-]\s+\S")
_CATEGORY_RE = re.compile(r"^###\s+(.*\S)\s*$")
_H2_RE = re.compile(r"^##\s+\S")
_DATED_SECTION_RE = re.compile(r"^Valkey\s+\d+\.\d+\.\d+", re.MULTILINE)
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
# rcN, N starting at 1 with no leading zeros: "rc1", "rc12" but not "rc0"/"rc01".
_RC_STAGE_RE = re.compile(r"^rc([1-9]\d*)$")

_ORDINALS = [
    "zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
]


def parse_version(version: str) -> "tuple[int, int, int]":
    """Split ``"M.m.p"`` into integer ``(major, minor, patch)``.

    Each component must be an integer in the inclusive range 0-255 so it fits
    a single byte of ``VALKEY_VERSION_NUM`` (see bump_version.py).
    """
    match = _VERSION_RE.match(version.strip())
    if not match:
        raise ValueError(
            "version must be in the form MAJOR.MINOR.PATCH (e.g. 9.1.0), got {!r}".format(version)
        )
    parts = tuple(int(p) for p in match.groups())
    for component, value in zip(("major", "minor", "patch"), parts):
        if not 0 <= value <= 255:
            raise ValueError(
                "{} version {} is out of range 0-255".format(component, value)
            )
    return parts  # type: ignore[return-value]


def ordinal(n: int) -> str:
    """Return a small ordinal word ("first", "second", ...) or "Nth" fallback."""
    if 0 <= n < len(_ORDINALS):
        return _ORDINALS[n]
    return "{}th".format(n)


def parse_unreleased(text: str) -> "OrderedDict[str, List[str]]":
    """Extract the "## Unreleased" block as an ordered ``category -> bullets`` map.

    Only the region between the ``## Unreleased`` header and the next level-2
    header (or end of file) is considered. HTML comments and blank lines are
    skipped. Categories are returned in the order they appear; categories that
    appear in :data:`CATEGORIES` but are absent from the text are not added.
    """
    result: "OrderedDict[str, List[str]]" = OrderedDict()
    lines = text.splitlines()

    # Locate the "## Unreleased" header.
    start = None
    for i, line in enumerate(lines):
        if line.strip() == UNRELEASED_HEADER:
            start = i + 1
            break
    if start is None:
        return result

    current: Optional[str] = None
    in_comment = False
    for line in lines[start:]:
        stripped = line.strip()
        if _H2_RE.match(line) and stripped != UNRELEASED_HEADER:
            break  # next top-level section ends the Unreleased block
        if in_comment:
            if "-->" in stripped:
                in_comment = False
            continue
        if stripped.startswith("<!--"):
            if "-->" not in stripped:
                in_comment = True
            continue
        cat_match = _CATEGORY_RE.match(line)
        if cat_match:
            current = cat_match.group(1)
            result.setdefault(current, [])
            continue
        if current is not None and _BULLET_RE.match(line):
            result[current].append(stripped)
    return result


def is_unreleased_empty(notes: "Dict[str, List[str]]") -> bool:
    """True when no category in *notes* carries any bullet."""
    return not any(bullets for bullets in notes.values())


def count_bullets(notes: "Dict[str, List[str]]") -> int:
    """Total number of bullets across all categories."""
    return sum(len(bullets) for bullets in notes.values())


def unrecognized_categories(notes: "Dict[str, List[str]]") -> List[str]:
    """Return the names of bullet-bearing categories that are not canonical.

    A contributor may typo a header (``### Bug Fix`` for ``### Bug Fixes``) or
    invent one (``### Networking``). Such bullets are still rendered verbatim at
    promotion time (nothing is dropped), but they fall outside :data:`CATEGORIES`,
    so callers warn on them and ask a maintainer to recategorize. Reserved
    sections (:data:`RESERVED_SECTIONS`) are deliberately excluded -- they are not
    "miscategorized notes" to be promoted verbatim but auto-generated sections
    that should be removed from the block; :func:`reserved_sections_present`
    reports them separately. Categories with no bullets are ignored. Order
    follows *notes*.
    """
    known = set(CATEGORIES) | set(RESERVED_SECTIONS)
    return [
        category
        for category, bullets in notes.items()
        if bullets and category not in known
    ]


def reserved_sections_present(notes: "Dict[str, List[str]]") -> List[str]:
    """Return the names of :data:`RESERVED_SECTIONS` that carry bullets in *notes*.

    ``Security Fixes`` and ``Contributors`` are populated automatically at release
    time, so a contributor should never hand-add them to ``## Unreleased``. When
    one does, the bullets are ignored at render time (not promoted), so this lets
    callers warn that the stray section will be dropped and should be removed.
    Order follows :data:`RESERVED_SECTIONS`.
    """
    return [name for name in RESERVED_SECTIONS if notes.get(name)]


def _format_date(date: str) -> str:
    """Render *date* as ``"Tue 02 June 2026"``.

    Accepts an ISO ``YYYY-MM-DD`` string (reformatted) or any other string
    (returned unchanged, so callers may pass a pre-formatted display date).
    """
    try:
        parsed = datetime.date.fromisoformat(date.strip())
    except ValueError:
        return date.strip()
    return parsed.strftime("%a %d %B %Y")


def _normalize_stage(stage: str) -> str:
    s = stage.strip().lower()
    if s == "ga":
        return "ga"
    if _RC_STAGE_RE.match(s):
        return s
    raise ValueError("release stage must be 'ga' or 'rcN' (e.g. rc1), got {!r}".format(stage))


def render_header(major: int, minor: int) -> str:
    """Render the file title and urgency legend for a ``M.m`` release line."""
    title = "Valkey {}.{} release notes".format(major, minor)
    underline = "=" * len(title)
    return "{}\n{}\n\n{}".format(title, underline, URGENCY_LEGEND)


def _stage_heading(version: str, stage: str) -> str:
    if stage == "ga":
        return "Valkey {} GA".format(version)
    return "Valkey {}-{}".format(version, stage)


def _urgency_sentence(version: str, stage: str, urgency: str) -> str:
    major, minor, patch = parse_version(version)
    if stage == "ga":
        which = ordinal(patch + 1)  # M.m.0 is the first stable release of M.m
        return (
            "Upgrade urgency {}: This is the {} stable release of Valkey {}.{}.".format(
                urgency, which, major, minor
            )
        )
    rc_num = int(_RC_STAGE_RE.match(stage).group(1))  # type: ignore[union-attr]
    which = ordinal(rc_num)
    return (
        "Upgrade urgency {}: This is the {} release candidate of Valkey {}.".format(
            urgency, which, version
        )
    )


def render_version_section(
    version: str,
    stage: str,
    urgency: str,
    date: str,
    notes: "Dict[str, List[str]]",
    security_fixes: Optional[Sequence[str]] = None,
) -> str:
    """Render one dated release section in release-branch markdown form.

    *notes* maps category name to a list of bullet strings (already including
    the leading ``* ``). Only non-empty categories are emitted, in
    :data:`CATEGORIES` order, with ``Security Fixes`` (from *security_fixes*)
    rendered first. Any non-canonical category (a typo'd or invented header) is
    rendered verbatim *after* the canonical ones so its bullets are never dropped;
    callers warn on them via :func:`unrecognized_categories`. The reserved
    sections (:data:`RESERVED_SECTIONS`) are never read from *notes* -- a
    ``Security Fixes`` or ``Contributors`` section a contributor hand-added to the
    block is ignored here, since *security_fixes* is the source of truth for the
    former and the latter is rendered once for the whole file (not per section).
    *security_fixes* is an optional list of CVE bullet strings.

    Contributors are deliberately *not* rendered here: a single cumulative
    ``### Contributors`` footer for the whole file is rendered by
    :func:`render_contributors_footer` and assembled in :func:`promote`.
    """
    stage = _normalize_stage(stage)
    urgency = urgency.strip().upper()
    if urgency not in VALID_URGENCIES:
        raise ValueError(
            "urgency must be one of {}, got {!r}".format(", ".join(VALID_URGENCIES), urgency)
        )

    heading = "{}  -  Released {}".format(_stage_heading(version, stage), _format_date(date))
    underline = "-" * len(heading)
    out: List[str] = [heading, underline, "", _urgency_sentence(version, stage, urgency), ""]

    def emit_category(name: str, bullets: Sequence[str]) -> None:
        out.append("### {}".format(name))
        for bullet in bullets:
            bullet = bullet.strip()
            if not bullet.startswith(("* ", "- ")):
                bullet = "* " + bullet
            out.append(bullet)
        out.append("")

    # Security Fixes come only from *security_fixes* (the embargo CVE list), never
    # from *notes*: a hand-added "### Security Fixes" in the block is ignored so it
    # cannot duplicate this header (reserved_sections_present warns about it).
    if security_fixes:
        emit_category(SECURITY_CATEGORY, list(security_fixes))
    for category in CATEGORIES:
        bullets = notes.get(category)
        if bullets:
            emit_category(category, bullets)
    # Non-canonical categories (typo'd or invented headers) are rendered last,
    # verbatim and in their original order, so a miscategorized note is never
    # silently dropped; unrecognized_categories() lets callers warn about them.
    for category in unrecognized_categories(notes):
        emit_category(category, notes[category])

    return "\n".join(out).rstrip() + "\n"


def render_empty_unreleased() -> str:
    """Render the canonical empty ``## Unreleased`` block."""
    parts = [UNRELEASED_HEADER, "", UNRELEASED_COMMENT, ""]
    for category in CATEGORIES:
        parts.append("### {}".format(category))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def reset_unreleased(text: str) -> str:
    """Return *text* with the ``## Unreleased`` block reset to empty categories.

    Everything before ``## Unreleased`` is preserved verbatim; the block itself
    is replaced with :func:`render_empty_unreleased`.
    """
    idx = text.find("\n" + UNRELEASED_HEADER)
    if idx == -1:
        if text.startswith(UNRELEASED_HEADER):
            return render_empty_unreleased()
        # No Unreleased block at all — append a fresh one.
        return text.rstrip() + "\n\n" + render_empty_unreleased()
    return text[: idx + 1] + render_empty_unreleased()


_CONTRIBUTORS_HEADER_RE = re.compile(r"^###\s+Contributors\s*$", re.MULTILINE)


def _strip_bullet(line: str) -> str:
    """Return *line* trimmed of a leading ``* ``/``- `` bullet marker."""
    s = line.strip()
    if s.startswith(("* ", "- ")):
        return s[2:].strip()
    return s


def _split_contributors_footer(text: str) -> "tuple[str, List[str]]":
    """Split *text* at its trailing ``### Contributors`` section.

    Returns ``(body, contributors)`` where *body* is everything before the last
    ``### Contributors`` header (right-stripped) and *contributors* is the list of
    display names parsed from that section (bullet markers removed). When no such
    header exists, returns ``(text, [])``. Using the *last* header means a legacy
    per-section ``### Contributors`` is folded into the cumulative footer on the
    next promote(), migrating old files to the single-footer layout.
    """
    matches = list(_CONTRIBUTORS_HEADER_RE.finditer(text))
    if not matches:
        return text, []
    last = matches[-1]
    body = text[: last.start()].rstrip()
    names: List[str] = []
    for line in text[last.end():].splitlines():
        # The footer's bullets run until the next header. Stopping here matters in
        # legacy mode, where the footer precedes the ## Unreleased block: without
        # the break we would scoop up the example bullet inside that block's
        # guidance comment as if it were a contributor.
        if line.lstrip().startswith("#"):
            break
        if _BULLET_RE.match(line):
            names.append(_strip_bullet(line))
    return body, names


def render_contributors_footer(contributors: Sequence[str]) -> str:
    """Render the cumulative ``### Contributors`` footer, deduped and alpha-sorted.

    *contributors* is a list of display strings (``"Jane Doe @jdoe"``), possibly
    with duplicates carried across cuts. They are de-duplicated case-insensitively
    (first spelling wins) and sorted by the display-name portion before ``@``,
    matching gen_contributors. Returns ``""`` when the list is empty.
    """
    seen: set = set()
    unique: List[str] = []
    for entry in contributors:
        name = _strip_bullet(entry)
        if not name:
            continue
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    if not unique:
        return ""
    unique.sort(key=lambda e: e.split(" @", 1)[0].casefold())
    out = ["### Contributors"]
    out.extend("* {}".format(name) for name in unique)
    return "\n".join(out)


def _existing_dated_sections(before_unreleased: str) -> str:
    """Return the dated-section region of the text preceding ``## Unreleased``."""
    match = _DATED_SECTION_RE.search(before_unreleased)
    if not match:
        return ""
    return before_unreleased[match.start():].strip()


def promote(
    text: str,
    *,
    version: str,
    stage: str,
    urgency: str,
    date: str,
    contributors: Optional[Sequence[str]] = None,
    security_fixes: Optional[Sequence[str]] = None,
    prior_text: Optional[str] = None,
) -> str:
    """Promote a ``## Unreleased`` block into a new dated release section.

    The bullets to promote always come from the ``## Unreleased`` block of *text*.
    Two output shapes, selected by *prior_text*:

    **Two-source (drain) mode -- when *prior_text* is given.** *text* is the
    *source* branch's file (the base/feature branch, whose block accumulates the
    bullets) and *prior_text* is the *destination* branch's existing changelog
    (the pre-release branch, which carries earlier dated sections). The result is
    the destination's frozen changelog: title + legend, the new dated section,
    then *prior_text*'s previously dated sections -- and **no** ``## Unreleased``
    block, because the destination does not accumulate notes; the source branch
    does (and is emptied separately with :func:`reset_unreleased`). This is the
    rc1 -> rcN -> GA flow: every cut drains the base branch's block onto the
    running pre-release branch.

    **Single-source (legacy) mode -- when *prior_text* is None.** *text* supplies
    both the bullets and the prior dated sections, and the result re-emits an
    **emptied** ``## Unreleased`` block at the foot so a single file can keep
    accumulating between cuts. Retained for callers/tests that promote in place.

    The trailing block (legacy mode) sits *after* the dated sections on purpose:
    :func:`parse_unreleased` reads from ``## Unreleased`` to the next ``##`` header
    or EOF, so a foot-position block contains only its own categories and never
    bleeds into the dated sections above it.
    """
    major, minor, _ = parse_version(version)
    notes = parse_unreleased(text)
    dated = render_version_section(version, stage, urgency, date, notes, security_fixes)

    # Prior dated sections come from the destination changelog in drain mode, or
    # from the source file itself in legacy mode. Restrict to the region *before*
    # the ## Unreleased header first: a ``### Contributors`` header inside the
    # source's Unreleased block is a hand-added (reserved) section, not the running
    # footer, and must not be folded into the roll-up. Splitting on the header is a
    # no-op when it is absent (a frozen pre-release file), returning the whole text.
    prior_source = prior_text if prior_text is not None else text
    before_unreleased_raw = prior_source.split("\n" + UNRELEASED_HEADER, 1)[0]
    # Peel off any existing ``### Contributors`` footer so (a) it is not swept into
    # the dated region below, and (b) its names roll into the new cumulative footer
    # -- this is what dedups the roll-up across rc1..rcN..GA.
    before_unreleased, prior_contributors = _split_contributors_footer(before_unreleased_raw)
    existing = _existing_dated_sections(before_unreleased)

    parts: List[str] = [render_header(major, minor), "", dated.rstrip()]
    if existing:
        parts += ["", existing]

    # One cumulative ``### Contributors`` footer for the whole file: this cut's
    # contributors (commit authors over the release range, so everyone whose code
    # shipped is credited even without a release-note bullet) merged with those
    # carried on the prior changelog, deduped and alpha-sorted.
    merged = list(prior_contributors) + list(contributors or [])
    footer = render_contributors_footer(merged)
    if footer:
        parts += ["", footer]

    # Legacy mode re-emits an emptied ## Unreleased block so the single file keeps
    # accumulating; drain mode leaves the destination frozen (the source branch
    # holds the block). The block goes *last*: it is level-2, so a preceding
    # level-3 ``### Contributors`` footer reads as a dated-section trailer, while a
    # footer placed *after* the block would be parsed as a category inside it.
    if prior_text is None:
        parts += ["", render_empty_unreleased().rstrip()]

    return "\n".join(parts).rstrip() + "\n"
