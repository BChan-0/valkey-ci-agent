"""Render generated bullets into the canonical ``00-RELEASENOTES`` markdown.

The format is **authoritative in the valkey repo**, not here: this module
imports ``utils/releasetools/release_notes.py`` from the valkey clone at
runtime (via importlib) and reuses its ``CATEGORIES``, the exact ``##
Unreleased`` header, and the byte-for-byte contributor-guidance HTML comment.
The agent never re-encodes the category names or that comment, so a change to
the format in valkey flows through automatically.

What this module owns is purely mechanical: turning each
:class:`CategorizedBullet` into the canonical bullet line
``* <text> by @<handle> (#<N>)`` -- with the ``(#N)`` trailing and the
``by @handle`` present, exactly as ``check_release_notes.py`` requires -- and
splicing the filled block into the file in place of the existing one.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from scripts.release_notes.clone_tools import load_releasetools_module
from scripts.release_notes.models import CategorizedBullet

logger = logging.getLogger(__name__)

# A GitHub login is [A-Za-z0-9-]; the attribution regex check_release_notes uses
# is ``by @([\w-]+)``. Anything outside that set (a space, '.', '@', parens from
# a malformed author) would truncate or break the captured handle.
_HANDLE_SAFE_RE = re.compile(r"[^\w-]")


def _one_line(text: str) -> str:
    """Collapse *text* to a single physical line.

    A bullet and a category header are parsed line-by-line by the format
    module, so an embedded line break would split the bullet, terminate the
    whole block (a line starting ``"## "``), or inject a spurious category
    (``"### ..."``). We split on exactly the boundaries ``str.splitlines()``
    recognizes -- the same call ``parse_unreleased`` uses -- so our notion of
    "one line" cannot disagree with the parser's, then join with single spaces.
    """
    return " ".join(text.splitlines()).strip()


def load_format_module(valkey_clone_dir: str) -> Any:
    """Import the valkey ``release_notes`` format module from *valkey_clone_dir*.

    Thin wrapper over :func:`clone_tools.load_releasetools_module` kept for the
    existing call sites; the format stays authoritative in the valkey repo.
    """
    return load_releasetools_module(valkey_clone_dir, "release_notes")


def format_bullet(bullet: CategorizedBullet) -> str:
    """Render one canonical bullet line: ``* <text> by @<handle> (#<N>)``.

    The trailing ``(#N)`` and the ``by @handle`` are appended in this fixed
    order so they satisfy ``check_release_notes``'s ``_TRAILING_PR_REF_RE`` and
    ``_AUTHOR_RE``. When the author is unknown (a ghost account), the ``by @``
    segment is omitted -- the PR-number requirement still holds, and a missing
    attribution is a warning, not a hard failure, in the CI check.

    Both the text and the handle are sanitized: the text is collapsed to a
    single line (a newline would split the bullet or inject a ``##``/``###``
    line that terminates the block), a trailing ``(#...)`` the model left inside
    the text is removed so the appended reference is the only trailing one, and
    the handle is reduced to the ``[\\w-]`` login charset so a stray space or
    punctuation can't break the attribution.
    """
    text = _one_line(bullet.text)
    text = re.sub(r"\s*\(#[^)]*\)\s*$", "", text).strip()
    parts = [f"* {text}"]
    handle = _HANDLE_SAFE_RE.sub("", bullet.author)
    if handle:
        parts.append(f"by @{handle}")
    parts.append(f"(#{bullet.pr_number})")
    return " ".join(parts)


def group_bullets(
    bullets: Sequence[CategorizedBullet], fmt: Any
) -> "dict[str, list[str]]":
    """Group bullets into ``{category: [rendered line, ...]}``.

    Canonical categories (``fmt.CATEGORIES``) come first, in their canonical
    order; any non-canonical category the model emitted follows, in first-seen
    order, so a miscategorized note is never dropped (mirrors the format
    module's own behavior). Bullets the model placed under the reserved
    ``Security Fixes`` / ``Contributors`` sections are refused -- those are
    generated at release-cut time -- and logged.
    """
    reserved = set(getattr(fmt, "RESERVED_SECTIONS", ("Security Fixes", "Contributors")))
    canonical = set(fmt.CATEGORIES)
    grouped: "dict[str, list[str]]" = {}
    for bullet in bullets:
        category = _one_line(bullet.category)
        if category not in canonical:
            # A non-canonical category is rendered as a verbatim "### <name>"
            # header, so sanitize it the same way as a bullet: strip leading '#'
            # (which would otherwise let "## x" terminate the block) and skip it
            # entirely if nothing usable remains.
            category = category.lstrip("#").strip()
            if not category:
                logger.warning("Dropping PR #%s with empty category", bullet.pr_number)
                continue
        if category in reserved:
            logger.warning(
                "Refusing PR #%s under reserved section %r (auto-generated at release)",
                bullet.pr_number, category,
            )
            continue
        grouped.setdefault(category, []).append(format_bullet(bullet))

    # Re-key into canonical order first, then trailing non-canonical categories.
    ordered: "dict[str, list[str]]" = {}
    for name in fmt.CATEGORIES:
        if grouped.get(name):
            ordered[name] = grouped[name]
    for name, lines in grouped.items():
        if name not in ordered:
            ordered[name] = lines
    return ordered


def render_unreleased_block(grouped: "dict[str, list[str]]", fmt: Any) -> str:
    """Build the ``## Unreleased`` block: header + guidance comment + categories.

    Starts from the canonical empty block (``fmt.render_empty_unreleased()``) so
    the header and HTML comment are byte-exact, then re-emits each category
    header followed by its bullets. Canonical categories are always emitted (even
    when empty) to match the empty-block shape; non-canonical categories with
    bullets are appended after.
    """
    canonical = list(fmt.CATEGORIES)
    # Reuse the canonical empty block to capture everything up to the first
    # category header (the "## Unreleased" line, blank line, and HTML comment),
    # so that preamble stays byte-identical to what valkey emits.
    empty = fmt.render_empty_unreleased()
    first_header = f"### {canonical[0]}"
    preamble = empty.split(first_header, 1)[0].rstrip("\n")

    parts: list[str] = [preamble, ""]
    for name in canonical:
        parts.append(f"### {name}")
        parts.extend(grouped.get(name, []))
        parts.append("")
    for name, lines in grouped.items():
        if name not in canonical:
            parts.append(f"### {name}")
            parts.extend(lines)
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def apply_to_file(existing_text: str, grouped: "dict[str, list[str]]", fmt: Any) -> str:
    """Return *existing_text* with its ``## Unreleased`` block replaced.

    Everything before the block is preserved verbatim. We locate the block the
    way the format module's ``reset_unreleased`` does -- by the header preceded
    by a newline (``"\\n## Unreleased"``), NOT a bare ``find``. The intro prose
    mentions ``"## Unreleased"`` in quotes, so a bare search would match that
    mention and splice the block into the middle of the paragraph.
    """
    header = fmt.UNRELEASED_HEADER
    filled_block = render_unreleased_block(grouped, fmt)
    anchor = "\n" + header
    idx = existing_text.find(anchor)
    if idx == -1:
        if existing_text.startswith(header):
            return filled_block
        # No block at all -- append a fresh, filled one.
        return existing_text.rstrip() + "\n\n" + filled_block
    # Keep everything up to and including the newline before the header.
    return existing_text[: idx + 1] + filled_block
