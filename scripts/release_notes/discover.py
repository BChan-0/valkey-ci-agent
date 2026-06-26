"""Discover the PRs a release line has accrued since its last tag.

Selection is by **graph reachability**, never by date: we resolve the most
recent tag reachable from the release-line tip and walk ``tag..head``. A
backport cherry-picked onto the line is a distinct commit with its own date,
so a date window would either miss it or double-count it; a graph range counts
it exactly when it is part of this line's history.

Each commit is resolved to its **originating PR number** so the set is
deduplicated by change identity rather than by commit SHA. The squash-merge
subject's trailing ``(#N)`` is the cheap offline path
(:func:`scripts.backport.utils.pr_numbers_from_commit_subjects`); commits
without one fall back to the GitHub "PRs associated with a commit" API.

Cross-line dedup (the same change shipping as a different PR on ``unstable``)
is intentionally out of scope: within a single release line, the PR that
merged the change *onto this line* is the right identity. A change that should
have been release-noted on another line surfaces as a triage signal, not an
auto-merge.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from scripts.backport.utils import pr_numbers_from_commit_subjects
from scripts.common.github_client import retry_github_call
from scripts.common.proc import git_output
from scripts.release_notes.models import DiscoveryResult, MergedPR

logger = logging.getLogger(__name__)

# NUL is illegal in a git ref/subject, so it is a safe field separator for the
# ``%H%x00%s`` log format (a subject may itself contain tabs or pipes).
_LOG_FORMAT = "%H%x00%s"


def resolve_last_tag(repo_dir: str, head_ref: str, *, tag_glob: str | None = None) -> tuple[str, str]:
    """Return ``(tag_name, tag_sha)`` for the most recent tag reachable from *head_ref*.

    "Most recent" is by graph distance, not date: ``git describe --tags
    --abbrev=0`` reports the nearest tag that is an ancestor of *head_ref*.
    ``tag_glob`` (e.g. ``"9.1.*"``) restricts matching to one release line via
    ``--match``. Raises :class:`ValueError` if no tag is reachable.
    """
    args = ["describe", "--tags", "--abbrev=0"]
    if tag_glob:
        args += ["--match", tag_glob]
    args.append(head_ref)
    try:
        tag = git_output(repo_dir, *args).strip()
    except Exception as exc:  # noqa: BLE001 - normalize any git failure to a clear error
        raise ValueError(
            f"no tag reachable from {head_ref!r}"
            + (f" matching {tag_glob!r}" if tag_glob else "")
        ) from exc
    if not tag:
        raise ValueError(f"no tag reachable from {head_ref!r}")
    # Dereference the (possibly annotated) tag to the commit it points at.
    tag_sha = git_output(repo_dir, "rev-list", "-n", "1", tag).strip()
    logger.info("Last tag reachable from %s: %s (%s)", head_ref, tag, tag_sha[:12])
    return tag, tag_sha


def list_range_commits(repo_dir: str, base: str, head_ref: str) -> list[tuple[str, str]]:
    """Return ``[(sha, subject), ...]`` for commits in ``base..head_ref``, oldest first.

    ``base`` is the prior tag (or its SHA); the range excludes it and includes
    everything reachable from *head_ref* that it does not reach -- exactly the
    line's new history.
    """
    out = git_output(
        repo_dir, "log", "--reverse", f"--format={_LOG_FORMAT}", f"{base}..{head_ref}"
    )
    commits: list[tuple[str, str]] = []
    # Split on "\n" only -- NOT str.splitlines(), which also breaks on \v, \f,
    # \x85, U+2028/2029 etc. A subject legitimately containing one of those
    # would otherwise be torn into a bogus extra record.
    for line in out.split("\n"):
        if not line:
            continue
        sha, _, subject = line.partition("\x00")
        commits.append((sha, subject))
    logger.info("%d commit(s) in %s..%s", len(commits), base, head_ref)
    return commits


def resolve_commit_prs(repo: Any, commits: list[tuple[str, str]]) -> dict[int, str]:
    """Map originating PR number -> representative commit SHA, deduplicated.

    Two-tier resolution:

    1. **Subject parse** (offline, free): the trailing ``(#N)`` of a squash-merge
       subject is the PR that merged the commit onto this line. This catches the
       overwhelming majority and is what makes a cherry-picked change collapse
       onto one key when its subject preserves the source ``(#N)``.
    2. **API fallback**: for a commit whose subject has no trailing ``(#N)``
       (a hand-applied cherry-pick, a merge commit, very old history), ask
       GitHub for the PRs associated with that SHA and take the first.

    The first commit seen per PR number wins; later occurrences collapse onto
    it. Commits that resolve to no PR are dropped with a warning -- they are
    invisible to dedup and carry no PR reference for a note.
    """
    pr_to_sha: dict[int, str] = {}
    for sha, subject in commits:
        numbers = pr_numbers_from_commit_subjects([subject])
        if not numbers:
            number = _pr_from_commit_api(repo, sha)
            numbers = {number} if number is not None else set()
        if not numbers:
            logger.warning("Commit %s has no resolvable PR (subject: %s)", sha[:12], subject[:80])
            continue
        for number in numbers:
            pr_to_sha.setdefault(number, sha)
    logger.info("Resolved %d unique PR(s) from %d commit(s)", len(pr_to_sha), len(commits))
    return pr_to_sha


def _pr_from_commit_api(repo: Any, sha: str) -> int | None:
    """Return the first PR number associated with *sha* via the GitHub API, or None.

    Uses ``GET /repos/{owner}/{repo}/commits/{sha}/pulls`` (PyGithub
    ``Commit.get_pulls()``). Only the first page's first item is consulted; a
    commit belongs to at most one merge in practice.
    """
    def _lookup() -> int | None:
        commit = repo.get_commit(sha)
        for pull in commit.get_pulls():
            return int(pull.number)
        return None

    try:
        return retry_github_call(_lookup, retries=3, description=f"PRs for commit {sha[:12]}")
    except Exception as exc:  # noqa: BLE001 - a lookup miss must not abort discovery
        logger.warning("Could not resolve PR for commit %s: %s", sha[:12], exc)
        return None


def hydrate_prs(repo: Any, pr_to_sha: dict[int, str]) -> list[MergedPR]:
    """Fetch title/author/labels for each PR number, returning :class:`MergedPR`.

    Disposition is left at its default (TRIAGE) here; :mod:`classify` assigns
    the real value. A number that 404s (an issue reference, or a ``(#N)`` from a
    different repo) is skipped with a warning rather than aborting the run.
    """
    prs: list[MergedPR] = []
    for number in sorted(pr_to_sha):
        sha = pr_to_sha[number]
        try:
            pull = retry_github_call(
                lambda: repo.get_pull(number), retries=3, description=f"get PR #{number}"
            )
        except Exception as exc:  # noqa: BLE001 - skip an unresolvable reference
            logger.warning("Skipping PR #%s (could not fetch): %s", number, exc)
            continue
        author = ""
        if pull.user is not None and pull.user.login:
            author = pull.user.login
        labels = tuple(label.name for label in pull.labels)
        prs.append(
            MergedPR(
                number=number,
                title=pull.title or "",
                author=author,
                url=pull.html_url or "",
                labels=labels,
                merge_commit_sha=pull.merge_commit_sha or sha,
            )
        )
    return prs


def _resolve_base_ref(repo_dir: str, base_ref: str) -> str:
    """Return a ref name for *base_ref* that resolves in *repo_dir*.

    The clone is made with ``git clone --branch <source>``, so only the source
    branch becomes a local ref; every other branch exists solely as its
    remote-tracking ref ``origin/<name>``. A ``--base-ref`` naming such a branch
    (e.g. a fork passing ``unstable``) therefore fails a bare ``rev-parse``. Try
    the name as given first -- it covers tags, SHAs, and the source branch -- and
    fall back to ``origin/<name>`` for any other branch. The returned name is
    used both to resolve the base SHA and as the range/contributor baseline, so
    every downstream ``base..head`` walk sees a name git can resolve.
    """
    try:
        git_output(repo_dir, "rev-parse", "--verify", "--quiet", f"{base_ref}^{{commit}}")
        return base_ref
    except subprocess.CalledProcessError:
        remote = f"origin/{base_ref}"
        git_output(repo_dir, "rev-parse", "--verify", "--quiet", f"{remote}^{{commit}}")
        logger.info("Base ref %r resolved via remote-tracking ref %r", base_ref, remote)
        return remote


def discover(
    repo: Any, repo_dir: str, head_ref: str, *,
    tag_glob: str | None = None, base_ref: str | None = None,
) -> DiscoveryResult:
    """Resolve the release range and return a deduplicated :class:`DiscoveryResult`.

    ``repo`` is a PyGithub repository; ``repo_dir`` is a full-depth local clone
    of the same repo with tags fetched (a shallow clone breaks ``describe`` and
    the range walk). Dispositions are unset -- :func:`classify.classify` fills
    them.

    ``base_ref`` is an explicit baseline (a branch, tag, or SHA) that overrides
    tag resolution -- the escape hatch for a line with no reachable tag (e.g. a
    fork that carries no release tags). When set, the range is ``base_ref..head``
    directly; otherwise the most recent tag (optionally filtered by ``tag_glob``)
    is used.
    """
    if base_ref:
        # Resolve to a name git can use in a fresh --branch clone (the bare
        # branch may only exist as origin/<name>); reuse it as the range and
        # contributor baseline so every base..head walk resolves identically.
        base_tag = _resolve_base_ref(repo_dir, base_ref)
        base_sha = git_output(repo_dir, "rev-parse", base_tag).strip()
    else:
        base_tag, base_sha = resolve_last_tag(repo_dir, head_ref, tag_glob=tag_glob)
    head_sha = git_output(repo_dir, "rev-parse", head_ref).strip()
    commits = list_range_commits(repo_dir, base_tag, head_ref)
    pr_to_sha = resolve_commit_prs(repo, commits)
    prs = hydrate_prs(repo, pr_to_sha)
    return DiscoveryResult(
        base_tag=base_tag,
        base_sha=base_sha,
        head_ref=head_ref,
        head_sha=head_sha,
        prs=tuple(prs),
    )
