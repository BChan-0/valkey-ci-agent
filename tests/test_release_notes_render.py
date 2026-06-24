"""Tests for canonical release-notes rendering.

Uses a fixture copy of valkey's ``utils/releasetools/release_notes.py`` (the
authoritative format module) under ``tests/fixtures/valkey_clone`` so the
rendered output is validated against the real parser, not a re-implementation.
"""

from __future__ import annotations

import os
import re

from scripts.release_notes.models import CategorizedBullet
from scripts.release_notes.render import (
    apply_to_file,
    format_bullet,
    group_bullets,
    load_format_module,
    render_unreleased_block,
)

_FIXTURE_CLONE = os.path.join(os.path.dirname(__file__), "fixtures", "valkey_clone")

# check_release_notes' rule-3 regexes, replicated to assert compliance.
_TRAILING_PR_RE = re.compile(r"\(#([^)]*)\)\s*$")
_AUTHOR_RE = re.compile(r"by @([\w-]+)")


def _fmt():
    return load_format_module(_FIXTURE_CLONE)


def _bullet(pr, author, category, text):
    return CategorizedBullet(pr_number=pr, author=author, category=category, text=text)


class TestFormatBullet:
    def test_canonical_form(self) -> None:
        line = format_bullet(_bullet(40, "BChan-0", "Bug Fixes", "fix a crash"))
        assert line == "* fix a crash by @BChan-0 (#40)"

    def test_trailing_pr_and_author_present(self) -> None:
        line = format_bullet(_bullet(7, "jdoe", "Bug Fixes", "x"))
        assert _TRAILING_PR_RE.search(line)
        assert _AUTHOR_RE.search(line)

    def test_ghost_author_omits_attribution_but_keeps_pr(self) -> None:
        line = format_bullet(_bullet(7, "", "Bug Fixes", "x"))
        assert line == "* x (#7)"
        assert _TRAILING_PR_RE.search(line)
        assert not _AUTHOR_RE.search(line)

    def test_newline_in_text_collapsed_to_single_line(self) -> None:
        line = format_bullet(_bullet(7, "a", "Bug Fixes", "line one\nline two"))
        assert "\n" not in line
        assert line == "* line one line two by @a (#7)"

    def test_text_with_h2_does_not_survive_as_its_own_line(self) -> None:
        # A '## ...' on its own line would terminate the Unreleased block.
        line = format_bullet(_bullet(7, "a", "Bug Fixes", "fixed\n## Injected"))
        assert "\n" not in line
        assert _TRAILING_PR_RE.search(line)

    def test_trailing_ref_in_text_not_duplicated(self) -> None:
        line = format_bullet(_bullet(40, "", "Bug Fixes", "see (#99)"))
        # The stray trailing ref is stripped; only the real (#40) remains at end.
        assert line == "* see (#40)"
        assert line.count("(#") == 1

    def test_author_handle_sanitized_to_login_charset(self) -> None:
        line = format_bullet(_bullet(7, "alice smith", "Bug Fixes", "x"))
        # The space would otherwise truncate _AUTHOR_RE's capture.
        assert "by @alicesmith" in line
        assert _AUTHOR_RE.search(line).group(1) == "alicesmith"


class TestGroupBullets:
    def test_canonical_order_then_noncanonical(self) -> None:
        fmt = _fmt()
        bullets = [
            _bullet(3, "a", "Bug Fixes", "b"),
            _bullet(1, "a", "Behavior Changes", "c"),
            _bullet(9, "a", "Networking", "n"),  # non-canonical
        ]
        grouped = group_bullets(bullets, fmt)
        keys = list(grouped.keys())
        # Behavior Changes precedes Bug Fixes (canonical order); Networking last.
        assert keys.index("Behavior Changes") < keys.index("Bug Fixes")
        assert keys[-1] == "Networking"

    def test_reserved_sections_refused(self) -> None:
        fmt = _fmt()
        grouped = group_bullets(
            [_bullet(1, "a", "Security Fixes", "x"), _bullet(2, "a", "Contributors", "y")], fmt
        )
        assert grouped == {}

    def test_multiple_bullets_same_category(self) -> None:
        fmt = _fmt()
        grouped = group_bullets(
            [_bullet(1, "a", "Bug Fixes", "one"), _bullet(2, "b", "Bug Fixes", "two")], fmt
        )
        assert len(grouped["Bug Fixes"]) == 2

    def test_noncanonical_category_with_h2_is_stripped(self) -> None:
        # A category like "## Injected" must not survive as a block-terminating header.
        fmt = _fmt()
        grouped = group_bullets([_bullet(1, "a", "## Injected", "x")], fmt)
        assert all(not k.startswith("#") for k in grouped)

    def test_empty_category_after_sanitize_dropped(self) -> None:
        fmt = _fmt()
        grouped = group_bullets([_bullet(1, "a", "###", "x")], fmt)
        assert grouped == {}


class TestMaliciousBulletCannotBreakBlock:
    def test_injected_heading_in_text_does_not_truncate_block(self) -> None:
        fmt = _fmt()
        bullets = [
            _bullet(40, "a", "Bug Fixes", "fixed\n## Unreleased\n### Bug Fixes\n* fake (#1)"),
            _bullet(41, "b", "Build and Tooling", "later category still parsed"),
        ]
        block = render_unreleased_block(group_bullets(bullets, fmt), fmt)
        parsed = {k: v for k, v in fmt.parse_unreleased(block).items() if v}
        # Both bullets survive; the injected heading did not split or end the block.
        assert "Bug Fixes" in parsed and "Build and Tooling" in parsed
        assert len(parsed["Bug Fixes"]) == 1


class TestRenderUnreleasedBlock:
    def test_roundtrips_through_parse_unreleased(self) -> None:
        fmt = _fmt()
        bullets = [
            _bullet(40, "BChan-0", "Bug Fixes", "fix crash"),
            _bullet(41, "jdoe", "New Features and Enhanced Behavior", "new opt"),
        ]
        block = render_unreleased_block(group_bullets(bullets, fmt), fmt)
        parsed = {k: v for k, v in fmt.parse_unreleased(block).items() if v}
        assert parsed == {
            "Bug Fixes": ["* fix crash by @BChan-0 (#40)"],
            "New Features and Enhanced Behavior": ["* new opt by @jdoe (#41)"],
        }

    def test_preamble_is_byte_exact(self) -> None:
        fmt = _fmt()
        block = render_unreleased_block({}, fmt)
        empty = fmt.render_empty_unreleased()
        # Everything up to the first category header must match the canonical block.
        first = "### Behavior Changes"
        assert block.split(first, 1)[0] == empty.split(first, 1)[0]

    def test_all_canonical_categories_present_when_empty(self) -> None:
        fmt = _fmt()
        block = render_unreleased_block({}, fmt)
        for name in fmt.CATEGORIES:
            assert f"### {name}" in block


class TestApplyToFile:
    def test_replaces_block_and_preserves_preamble(self) -> None:
        fmt = _fmt()
        existing = open(os.path.join(_FIXTURE_CLONE, "00-RELEASENOTES"), encoding="utf-8").read()
        grouped = group_bullets([_bullet(40, "BChan-0", "Bug Fixes", "fix crash")], fmt)
        result = apply_to_file(existing, grouped, fmt)
        # Intro paragraph (which itself mentions "## Unreleased") is preserved.
        assert result.startswith(existing[:80])
        parsed = {k: v for k, v in fmt.parse_unreleased(result).items() if v}
        assert parsed == {"Bug Fixes": ["* fix crash by @BChan-0 (#40)"]}

    def test_does_not_splice_into_quoted_mention(self) -> None:
        # Regression: the intro prose contains the quoted string "## Unreleased";
        # the block must be located by the newline-anchored header, not that mention.
        fmt = _fmt()
        existing = 'Intro mentions "## Unreleased" here.\n\n## Unreleased\n\n### Bug Fixes\n'
        grouped = group_bullets([_bullet(1, "a", "Bug Fixes", "z")], fmt)
        result = apply_to_file(existing, grouped, fmt)
        assert result.startswith('Intro mentions "## Unreleased" here.\n')
        assert "* z by @a (#1)" in result

    def test_bullets_satisfy_check_release_notes_rules(self) -> None:
        fmt = _fmt()
        existing = open(os.path.join(_FIXTURE_CLONE, "00-RELEASENOTES"), encoding="utf-8").read()
        grouped = group_bullets(
            [_bullet(40, "BChan-0", "Bug Fixes", "a"), _bullet(41, "", "Bug Fixes", "b")], fmt
        )
        result = apply_to_file(existing, grouped, fmt)
        for bullets in fmt.parse_unreleased(result).values():
            for bullet in bullets:
                assert _TRAILING_PR_RE.search(bullet), bullet
