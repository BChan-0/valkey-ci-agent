"""Entry point for the AI release-notes generator.

Driven by ``workflow_dispatch``: a maintainer supplies the release line
(``--head-ref``, e.g. ``9.1``) and the workflow regenerates that line's
``## Unreleased`` notes from the labelled PRs merged since its last tag, then
opens or updates a PR on valkey carrying the change.

Pipeline: clone valkey (full depth + tags) -> :mod:`discover` the range ->
:mod:`classify` by label -> :mod:`generate` bullets via Claude/Bedrock ->
:mod:`render` into the canonical ``00-RELEASENOTES`` -> :mod:`publish` the PR.

Returns 0 on success or a benign no-op (empty range), 1 on failure, and 2 on a
usage error (argparse). Orchestration is wrapped so a GitHub/AI error is logged
and surfaced as a non-zero exit rather than an uncaught crash.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from github import Auth, Github

from scripts.common.git_auth import GitAuth, github_https_url
from scripts.common.github_client import retry_github_call
from scripts.common.proc import run_git
from scripts.release_notes import discover as discover_mod
from scripts.release_notes import generate as generate_mod
from scripts.release_notes import publish as publish_mod
from scripts.release_notes import render as render_mod
from scripts.release_notes.classify import classify

logger = logging.getLogger(__name__)

# Config via env so the workflow can pass GitHub Actions context directly; the
# RELEASE_NOTES_ prefix mirrors the CI_FIX_/FUZZER_ convention.
_REPO = os.environ.get("RELEASE_NOTES_REPO", "valkey-io/valkey")
_HEAD_REF = os.environ.get("RELEASE_NOTES_HEAD_REF", "")
_BASE_BRANCH = os.environ.get("RELEASE_NOTES_BASE_BRANCH", "")
_TAG_GLOB = os.environ.get("RELEASE_NOTES_TAG_GLOB", "")


def _token() -> str:
    """Resolve the GitHub token: env chain (CLI override applied in main)."""
    return (
        os.environ.get("RELEASE_NOTES_GITHUB_TOKEN", "")
        or os.environ.get("TARGET_TOKEN", "")
        or os.environ.get("GITHUB_TOKEN", "")
    )


def _default_tag_glob(head_ref: str) -> str | None:
    """Derive a baseline-tag match glob from a ``M.m`` release line.

    ``git describe`` returns the tag with the shortest graph distance to HEAD
    *regardless of release line*, so without a glob a 9.1 branch could pick up a
    9.0 tag that happens to be a nearer ancestor. A release line named like
    ``9.1`` (or ``release/9.1``) constrains the baseline to ``9.1.*`` tags. A
    ref that is not a bare ``M.m`` returns None -- the caller may still pass an
    explicit ``--tag-glob``.
    """
    tail = head_ref.rsplit("/", 1)[-1]
    if re.fullmatch(r"\d+\.\d+", tail):
        return f"{tail}.*"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=_token(), help="GitHub token (App installation or PAT)")
    parser.add_argument("--repo", default=_REPO, help="Target repo, owner/name")
    parser.add_argument("--head-ref", default=_HEAD_REF,
                        help="Release line branch name, e.g. 9.1 (a short branch/tag name, "
                             "not a full ref or SHA -- it is passed to `git clone --branch`)")
    parser.add_argument("--base-branch", default=_BASE_BRANCH,
                        help="PR base branch (defaults to --head-ref)")
    parser.add_argument("--tag-glob", default=_TAG_GLOB,
                        help="Optional --match glob restricting the baseline tag, e.g. '9.1.*'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the notes without pushing or opening a PR")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.token:
        parser.error("a GitHub token is required (--token or RELEASE_NOTES_GITHUB_TOKEN/GITHUB_TOKEN)")
    if not args.head_ref:
        parser.error("--head-ref (release line, e.g. 9.1) is required")

    base_branch = args.base_branch or args.head_ref
    tag_glob = args.tag_glob or _default_tag_glob(args.head_ref)

    try:
        return _run(
            token=args.token,
            repo_full_name=args.repo,
            head_ref=args.head_ref,
            base_branch=base_branch,
            tag_glob=tag_glob,
            dry_run=args.dry_run,
        )
    except Exception:  # noqa: BLE001 - never crash the workflow uncaught
        logger.exception("Release-notes generation failed")
        return 1


def _run(
    *,
    token: str,
    repo_full_name: str,
    head_ref: str,
    base_branch: str,
    tag_glob: str | None,
    dry_run: bool,
) -> int:
    gh = Github(auth=Auth.Token(token))
    repo = retry_github_call(
        lambda: gh.get_repo(repo_full_name), retries=3, description=f"get repo {repo_full_name}",
    )

    with GitAuth(token, prefix="release-notes-git-askpass-") as auth:
        git_env = auth.env()
        clone_dir = tempfile.mkdtemp(prefix="release-notes-")
        try:
            # Full depth + tags: shallow clones break `git describe` and the
            # range walk that discovery depends on.
            run_git(None, "clone", "--branch", head_ref, github_https_url(repo_full_name), clone_dir,
                    env=git_env)
            run_git(clone_dir, "fetch", "--tags", "origin", env=git_env)
            return _generate_in_clone(
                repo, clone_dir,
                repo_full_name=repo_full_name, head_ref=head_ref, base_branch=base_branch,
                tag_glob=tag_glob, dry_run=dry_run, git_env=git_env,
            )
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)


def _generate_in_clone(
    repo: object,
    clone_dir: str,
    *,
    repo_full_name: str,
    head_ref: str,
    base_branch: str,
    tag_glob: str | None,
    dry_run: bool,
    git_env: dict[str, str],
) -> int:
    result = discover_mod.discover(repo, clone_dir, head_ref, tag_glob=tag_glob)
    if not result.prs:
        logger.info("No PRs in %s..%s; nothing to do.", result.base_tag, head_ref)
        return 0

    include, _exclude, triage = classify(result.prs)
    logger.info(
        "%d included, %d excluded, %d triage", len(include),
        len(result.prs) - len(include) - len(triage), len(triage),
    )

    fmt = render_mod.load_format_module(clone_dir)
    gen = generate_mod.generate(include, repo_dir=clone_dir, categories=fmt.CATEGORIES)
    grouped = render_mod.group_bullets(gen.bullets, fmt)

    notes_path = os.path.join(clone_dir, publish_mod.NOTES_FILE)
    with open(notes_path, "r", encoding="utf-8") as fh:
        existing = fh.read()
    updated = render_mod.apply_to_file(existing, grouped, fmt)

    # Refuse to blank an existing block: if PRs were included but the model
    # produced no bullets (every PR skipped, or a batch failed to parse), the
    # rendered block is empty. Writing it would *delete* notes already in the
    # file. Treat that as a failure unless there is genuinely nothing to lose.
    wipes_existing = not grouped and render_mod.apply_to_file(existing, {}, fmt) != existing
    if include and not gen.bullets and wipes_existing:
        logger.error(
            "%d PR(s) were included but no bullets were generated; refusing to blank %s. "
            "Skipped: %s", len(include), publish_mod.NOTES_FILE, list(gen.skipped),
        )
        return 1

    if dry_run:
        logger.info("[dry-run] would write %s and open a PR on %s", publish_mod.NOTES_FILE, base_branch)
        print(updated)
        if triage:
            print("\n=== Needs triage ===")
            for pr in triage:
                print(f"  #{pr.number} {pr.title} (@{pr.author or 'unknown'})")
        return 0

    if updated == existing and not triage:
        logger.info("No change to %s and nothing to triage; skipping PR.", publish_mod.NOTES_FILE)
        return 0

    with open(notes_path, "w", encoding="utf-8") as fh:
        fh.write(updated)

    url = publish_mod.publish(
        repo, clone_dir,
        base_repo=repo_full_name, push_repo=None,
        head_ref=head_ref, base_branch=base_branch, base_tag=result.base_tag,
        included=len(gen.bullets), skipped=list(gen.skipped), triage=triage,
        git_env=git_env,
    )
    logger.info("Release-notes PR ready: %s", url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
