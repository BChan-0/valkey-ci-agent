"""Tests for the Claude/Bedrock bullet generation, with a faked run_fn."""

from __future__ import annotations

import json

from scripts.release_notes.generate import build_prompt, generate
from scripts.release_notes.models import MergedPR

_CATEGORIES = [
    "Behavior Changes",
    "New Features and Enhanced Behavior",
    "Performance and Efficiency Improvements",
    "Bug Fixes",
    "Command and API Updates",
    "Module API Changes",
    "Observability and Logging",
    "Build and Tooling",
]


def _pr(number: int, author: str = "alice") -> MergedPR:
    return MergedPR(number=number, title=f"PR {number}", author=author, url=f"https://x/{number}",
                    labels=("release-notes",))


def _stream(obj: dict) -> str:
    """Wrap a JSON object in a stream-json 'result' event, like the claude CLI."""
    return json.dumps({"type": "result", "result": json.dumps(obj)})


def _fake_run(obj, *, exit_code: int = 0):
    """Build a run_fn that returns the given object as stream-json output."""
    def _run(prompt, **kwargs):
        return _stream(obj), "", exit_code
    return _run


class TestBuildPrompt:
    def test_includes_categories_and_pr_numbers(self) -> None:
        prompt = build_prompt([_pr(40), _pr(41)], categories=_CATEGORIES, repo_path="/clone")
        for name in _CATEGORIES:
            assert name in prompt
        assert "40" in prompt and "41" in prompt
        assert "/clone" in prompt

    def test_does_not_leak_author_into_text_instruction(self) -> None:
        prompt = build_prompt([_pr(1)], categories=_CATEGORIES, repo_path="/c")
        # The instruction explicitly forbids the model from adding (#N)/by @.
        assert "(#N)" in prompt or "(#" in prompt


class TestGenerate:
    def test_parses_bullets_and_stamps_author(self) -> None:
        prs = [_pr(40, "alice"), _pr(41, "bob")]
        obj = {"bullets": [
            {"pr": 40, "category": "Bug Fixes", "text": "fix a"},
            {"pr": 41, "category": "Behavior Changes", "text": "change b"},
        ], "skipped": []}
        result = generate(prs, repo_dir="/c", categories=_CATEGORIES, run_fn=_fake_run(obj))
        assert {b.pr_number for b in result.bullets} == {40, 41}
        # Author is the factual PR author, never from the model output.
        by_num = {b.pr_number: b for b in result.bullets}
        assert by_num[40].author == "alice"
        assert by_num[41].author == "bob"

    def test_drops_bullet_for_unknown_pr(self) -> None:
        obj = {"bullets": [
            {"pr": 40, "category": "Bug Fixes", "text": "ok"},
            {"pr": 999, "category": "Bug Fixes", "text": "invented"},
        ]}
        result = generate([_pr(40)], repo_dir="/c", categories=_CATEGORIES, run_fn=_fake_run(obj))
        assert {b.pr_number for b in result.bullets} == {40}

    def test_keeps_noncanonical_category_verbatim(self) -> None:
        obj = {"bullets": [{"pr": 40, "category": "Networking", "text": "n"}]}
        result = generate([_pr(40)], repo_dir="/c", categories=_CATEGORIES, run_fn=_fake_run(obj))
        assert result.bullets[0].category == "Networking"

    def test_records_skipped(self) -> None:
        obj = {"bullets": [], "skipped": [40, 41]}
        result = generate([_pr(40), _pr(41)], repo_dir="/c", categories=_CATEGORIES,
                          run_fn=_fake_run(obj))
        assert set(result.skipped) == {40, 41}

    def test_unparseable_output_marks_batch_skipped(self) -> None:
        def _bad_run(prompt, **kwargs):
            return "not json at all", "boom", 1
        result = generate([_pr(40), _pr(41)], repo_dir="/c", categories=_CATEGORIES, run_fn=_bad_run)
        assert result.bullets == ()
        assert set(result.skipped) == {40, 41}

    def test_nonzero_exit_with_valid_output_still_parsed(self) -> None:
        # Turn-budget exhaustion yields a nonzero exit but valid output.
        obj = {"bullets": [{"pr": 40, "category": "Bug Fixes", "text": "ok"}]}
        result = generate([_pr(40)], repo_dir="/c", categories=_CATEGORIES,
                          run_fn=_fake_run(obj, exit_code=1))
        assert {b.pr_number for b in result.bullets} == {40}

    def test_empty_input_no_call(self) -> None:
        called = {"n": 0}
        def _run(prompt, **kwargs):
            called["n"] += 1
            return _stream({"bullets": []}), "", 0
        result = generate([], repo_dir="/c", categories=_CATEGORIES, run_fn=_run)
        assert result.bullets == () and result.skipped == ()
        assert called["n"] == 0

    def test_empty_text_bullet_dropped(self) -> None:
        obj = {"bullets": [{"pr": 40, "category": "Bug Fixes", "text": ""}]}
        result = generate([_pr(40)], repo_dir="/c", categories=_CATEGORIES, run_fn=_fake_run(obj))
        assert result.bullets == ()
