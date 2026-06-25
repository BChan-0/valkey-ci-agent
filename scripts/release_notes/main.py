"""Entry point for the AI release-notes cut.

Driven by ``workflow_dispatch``: a maintainer supplies the source branch
(``--head-ref``), the target ``--version``/``--stage``/``--urgency``, and the
agent cuts the release in one shot. There is no accumulated ``## Unreleased``
block -- the notes for a release are generated all at once from the labelled
PRs in range, promoted into a dated section, and the version is bumped.

Pipeline: clone valkey (full depth + tags) -> :mod:`discover` the range (the
``release-notes``-labelled PRs from HEAD back to the most recent reachable RC
tag) -> :mod:`generate` bullets via Claude/Bedrock -> :mod:`release_cut`
promotes them onto the release line (dated section + ``src/version.h`` bump +
running contributor list, draining prior RCs) and opens the PR.

Returns 0 on success or a benign no-op (empty range), 1 on failure, and 2 on a
usage error (argparse). Orchestration is wrapped so a GitHub/AI error is logged
and surfaced as a non-zero exit rather than an uncaught crash.
"""

from __future__ import annotations

import argparse
import datetime
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
from scripts.release_notes import release_cut as cut_mod

logger = logging.getLogger(__name__)

# Config via env so the workflow can pass GitHub Actions context directly; the
# RELEASE_NOTES_ prefix mirrors the CI_FIX_/FUZZER_ convention.
# EDIT BEFORE PR: change the default repo back to valkey-io/valkey. Pointed at
# the BChan-0 fork for fork testing (the workflow also passes RELEASE_NOTES_REPO
# explicitly, so this default only matters for a bare CLI invocation).
_REPO = os.environ.get("RELEASE_NOTES_REPO", "BChan-0/valkey")
_HEAD_REF = os.environ.get("RELEASE_NOTES_HEAD_REF", "")
_TAG_GLOB = os.environ.get("RELEASE_NOTES_TAG_GLOB", "")
_BASE_REF = os.environ.get("RELEASE_NOTES_BASE_REF", "")


def _token() -> str:
    """Resolve the GitHub token: env chain (CLI override applied in main)."""
    return (
        os.environ.get("RELEASE_NOTES_GITHUB_TOKEN", "")
        or os.environ.get("TARGET_TOKEN", "")
        or os.environ.get("GITHUB_TOKEN", "")
    )


def _default_tag_glob(version: str, stage: str) -> str | None:
    """Derive the baseline-tag match glob for this cut, or None.

    ``git describe`` returns the tag with the shortest graph distance to HEAD
    *regardless of release line*, so a glob is needed to pin the baseline to the
    intended boundary. The boundary depends on the stage:

    * **rc2+** -- the prior RC of *this* version: ``<version>-rc*`` (so a cut of
      9.1.0-rc3 walks back only to 9.1.0-rc2).
    * **rc1 / ga / anything else** -- ``None``. rc1 has no prior same-version RC
      to anchor to (there is no rc0), and its true baseline is the *previous*
      release, which is not reachable from the source branch in valkey's
      fork-at-freeze model. So rc1 cannot resolve a tag automatically and must
      use ``--base-ref`` (see :func:`_default_base_ref_for_rc1`); ga continues an
      existing release line where the no-glob nearest tag is already correct.

    A version that is not ``M.m.p`` also returns None.
    """
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.strip())
    if not m:
        return None
    rc = re.fullmatch(r"rc([1-9]\d*)", stage.strip().lower())
    if rc and int(rc.group(1)) >= 2:
        return f"{version.strip()}-rc*"
    return None


def _default_base_ref_for_rc1(version: str) -> str | None:
    """Best-effort previous-release baseline for an rc1 cut, e.g. 9.1.0 -> 9.0.0.

    rc1 of ``M.m.p`` covers everything since the previous minor's GA. We can only
    *guess* that tag's name (``M.(m-1).0``); whether it is actually reachable as
    a range base is checked at clone time. Returns None when the version is not
    ``M.m.p`` or there is no previous minor (``M.0.*``), in which case the user
    must supply ``--base-ref`` explicitly.
    """
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.strip())
    if not m:
        return None
    major, minor, _ = (int(g) for g in m.groups())
    if minor == 0:
        return None  # first minor of a major: no obvious previous-minor GA
    return f"{major}.{minor - 1}.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=_token(), help="GitHub token (App installation or PAT)")
    parser.add_argument("--repo", default=_REPO, help="Target repo, owner/name")
    parser.add_argument("--head-ref", default=_HEAD_REF,
                        help="Source branch whose merged PRs are cut, e.g. unstable "
                             "(a short branch/tag name -- it is passed to `git clone --branch`)")
    parser.add_argument("--version", default=os.environ.get("RELEASE_NOTES_VERSION", ""),
                        help="Target version MAJOR.MINOR.PATCH, e.g. 9.1.0")
    parser.add_argument("--stage", default=os.environ.get("RELEASE_NOTES_STAGE", ""),
                        help="Release stage: rc1..rcN or ga")
    parser.add_argument("--urgency", default=os.environ.get("RELEASE_NOTES_URGENCY", ""),
                        help="Upgrade urgency: LOW, MODERATE, HIGH, CRITICAL, SECURITY")
    parser.add_argument("--date", default=os.environ.get("RELEASE_NOTES_DATE", ""),
                        help="Release date YYYY-MM-DD (default: today)")
    parser.add_argument("--tag-glob", default=_TAG_GLOB,
                        help="Optional --match glob restricting the baseline tag, e.g. '9.1.0-rc*'")
    parser.add_argument("--base-ref", default=_BASE_REF,
                        help="Explicit baseline ref (branch/tag/SHA) overriding tag resolution. "
                             "Use when the line has no reachable tag, e.g. a fork.")
    parser.add_argument("--contrib-base-ref", default=os.environ.get("RELEASE_NOTES_CONTRIB_BASE", ""),
                        help="Contributor range start (default: last tag, else root commit)")
    parser.add_argument("--security-fix", action="append", default=None, dest="security_fixes",
                        help="A Security Fixes bullet (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print the cut without pushing or opening a PR")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.token:
        parser.error("a GitHub token is required (--token or RELEASE_NOTES_GITHUB_TOKEN/GITHUB_TOKEN)")
    if not args.head_ref:
        parser.error("--head-ref (source branch, e.g. unstable) is required")
    if not (args.version and args.stage and args.urgency):
        parser.error("--version, --stage, and --urgency are required")

    base_ref = args.base_ref or None

    # rc1 has no prior same-version RC tag and, in valkey's fork-at-freeze model,
    # no reachable release tag from the source branch -- so tag resolution can't
    # find its baseline. rc1's true baseline is the previous release. If the user
    # did not pass --base-ref, warn loudly and default to the derived previous
    # release (M.(m-1).0); the cut still fails clearly at clone time if that ref
    # is absent, but most cuts get a sensible default instead of a hard error.
    if args.stage.strip().lower() == "rc1" and base_ref is None:
        derived = _default_base_ref_for_rc1(args.version)
        if derived:
            logger.warning(
                "rc1 of %s has no reachable baseline tag (there is no rc0, and release "
                "tags are not reachable from %r). Defaulting --base-ref to the previous "
                "release %r. Pass --base-ref explicitly to override (e.g. the previous "
                "release tag or its branch).",
                args.version, args.head_ref, derived,
            )
            base_ref = derived
        else:
            logger.warning(
                "rc1 of %s has no reachable baseline tag and no previous-minor release "
                "could be derived. Pass --base-ref explicitly (the previous release tag "
                "or branch); the cut will otherwise fail to resolve a baseline.",
                args.version,
            )

    # An explicit (or rc1-defaulted) base_ref overrides tag resolution, so don't
    # also derive a glob.
    tag_glob = None if base_ref else (args.tag_glob or _default_tag_glob(args.version, args.stage))

    try:
        return _run_cut(
            token=args.token,
            repo_full_name=args.repo,
            source_ref=args.head_ref,
            version=args.version,
            stage=args.stage,
            urgency=args.urgency,
            date=args.date or None,
            tag_glob=tag_glob,
            base_ref=base_ref,
            contrib_base_ref=args.contrib_base_ref or None,
            security_fixes=args.security_fixes,
            dry_run=args.dry_run,
        )
    except Exception:  # noqa: BLE001 - never crash the workflow uncaught
        logger.exception("Release cut failed")
        return 1


def _run_cut(
    *,
    token: str,
    repo_full_name: str,
    source_ref: str,
    version: str,
    stage: str,
    urgency: str,
    date: str | None,
    tag_glob: str | None,
    base_ref: str | None,
    contrib_base_ref: str | None,
    security_fixes: list[str] | None,
    dry_run: bool,
) -> int:
    gh = Github(auth=Auth.Token(token))
    repo = retry_github_call(
        lambda: gh.get_repo(repo_full_name), retries=3, description=f"get repo {repo_full_name}",
    )
    resolved_date = date or datetime.date.today().isoformat()
    with GitAuth(token, prefix="release-cut-git-askpass-") as auth:
        git_env = auth.env()
        clone_dir = tempfile.mkdtemp(prefix="release-cut-")
        try:
            run_git(None, "clone", "--branch", source_ref, github_https_url(repo_full_name),
                    clone_dir, env=git_env)
            run_git(clone_dir, "fetch", "--tags", "origin", env=git_env)
            return cut_mod.cut(
                repo,
                repo_full_name=repo_full_name,
                source_clone_dir=clone_dir,
                valkey_clone_dir=clone_dir,
                source_ref=source_ref,
                version=version, stage=stage, urgency=urgency, date=resolved_date,
                tag_glob=tag_glob, base_ref=base_ref, contrib_base_ref=contrib_base_ref,
                security_fixes=security_fixes, token=token, git_env=git_env, dry_run=dry_run,
            )
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
