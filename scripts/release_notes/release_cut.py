"""Cut a release: generate notes, promote them, bump the version, drain prior RCs.

Each cut regenerates the notes from the labelled PRs in range and promotes them 
in one shot. The orchestration mirrors valkey's
``utils/releasetools/prepare_release.py`` but reuses that repo's primitives
(``promote``, ``set_version``, ``version_num``, ``list_contributors``), loaded
from the clone at runtime (:mod:`clone_tools`). This keeps the version macros,
the dated-section format, and the contributor list authoritative in valkey.

The release-line branch model (one long-running branch per minor line):

    rc1 of M.m.p   -> create  pre-release-M.m.p  from the source branch
    rcN (N>1)      -> continue pre-release-M.m.p (keeps its prior dated notes)
    GA  of M.m.p   -> create  M.m carrying pre-release-M.m.p's history, then
                      delete pre-release-M.m.p (a rename)
    later patches  -> continue the existing M.m branch

The AI generates the bullets in a transient ``## Unreleased`` block, built in
memory by the discover/generate/render pipeline and never written to a branch.
``promote`` then drains that block into a new dated section on the release line,
prepends prior RCs' dated sections, appends the running contributor list, and
bumps ``src/version.h``.

Successive RCs do not double-note. Each cut discovers PRs by graph range from
HEAD back to the most recent reachable RC tag, so a PR captured by rc1's tag is
outside rc2's range. The source branch is never modified. The promoted commit
lands on an agent prep branch that opens a PR into the release line, so the cut
is reviewed before the line advances.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from scripts.common.proc import BOT_EMAIL, BOT_NAME, git_output, run_git
from scripts.release_notes import pipeline as pipeline_mod
from scripts.release_notes import publish as publish_mod
from scripts.release_notes.clone_tools import load_releasetools_module

logger = logging.getLogger(__name__)

NOTES_FILE = "00-RELEASENOTES"
VERSION_FILE = os.path.join("src", "version.h")

# Branch namespace. The release line (pre-release-M.m.p / M.m) is long-running
# and only advanced by merging a PR; the agent never force-pushes it directly.
# The cut's promoted commit lands on a throwaway agent prep branch that PRs into
# the line. The source branch is never modified.
PREP_BRANCH_PREFIX = "agent/release-cut"

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_RC_STAGE_RE = re.compile(r"^rc([1-9]\d*)$")
# Matches "Valkey M.m.p-rcN" headings in a running pre-release changelog, to
# tell which rc numbers already shipped on it.
_DATED_RC_RE_TMPL = r"^Valkey {major}\.{minor}\.{patch}-rc(\d+)"

# A rendered note bullet ends with "(#N)" naming the PR it credits. The
# bullet-line guard keeps a "(#N)" in prose or a heading from being read as a
# credit. Used to dedup a cut's notes against the PRs the destination release
# line already lists (see _drop_already_credited).
_BULLET_LINE_RE = re.compile(r"^\s*[*-]\s+\S")
_TRAILING_PR_RE = re.compile(r"\(#(\d+)\)\s*$")


@dataclass(frozen=True)
class BranchPlan:
    """How a cut maps onto the release-line branch model."""

    stage: str                 # normalized: 'ga' or 'rcN'
    target: str                # branch to write/push, e.g. pre-release-9.1.0 or 9.1
    base_ref: str              # ref the target is (re)based on
    continuing: bool           # True if the target line already exists (drain prior notes)
    rename_from: Optional[str]  # pre-release branch to delete after a GA rename, else None


def _split_version(version: str) -> tuple[int, int, int]:
    m = _VERSION_RE.match(version.strip())
    if not m:
        raise ValueError(f"version must be MAJOR.MINOR.PATCH (e.g. 9.1.0), got {version!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _normalize_stage(stage: str) -> str:
    s = stage.strip().lower()
    if s == "ga" or _RC_STAGE_RE.match(s):
        return s
    raise ValueError(f"stage must be 'ga' or 'rcN' (e.g. rc1), got {stage!r}")


def _remote_branch_exists(repo_dir: str, branch: str) -> bool:
    """True if ``refs/heads/<branch>`` exists on ``origin``."""
    out = git_output(repo_dir, "ls-remote", "--heads", "origin", f"refs/heads/{branch}")
    return bool(out.strip())


def resolve_branch_plan(repo_dir: str, *, version: str, stage: str, source_ref: str) -> BranchPlan:
    """Resolve the destination branch and base for this cut.

    Mirrors valkey's prepare-release branch resolution: rc stages target the
    long-running ``pre-release-M.m.p``; ``ga`` targets ``M.m`` and, when only the
    rc branch exists, renames it (carry its history, delete the rc branch). An
    existing line is continued (its prior dated sections are drained); otherwise
    it starts from ``source_ref``.
    """
    stage_lc = _normalize_stage(stage)
    major, minor, patch = _split_version(version)
    pre_branch = f"pre-release-{major}.{minor}.{patch}"
    ga_branch = f"{major}.{minor}"

    if stage_lc == "ga":
        if _remote_branch_exists(repo_dir, ga_branch):
            return BranchPlan(stage_lc, ga_branch, ga_branch, True, None)
        if _remote_branch_exists(repo_dir, pre_branch):
            # Carry the rc line's history onto M.m, then delete the rc branch.
            return BranchPlan(stage_lc, ga_branch, pre_branch, True, pre_branch)
        return BranchPlan(stage_lc, ga_branch, source_ref, False, None)

    # rc stages
    if _remote_branch_exists(repo_dir, pre_branch):
        _warn_rc_sequence(repo_dir, pre_branch, stage_lc, major, minor, patch)
        return BranchPlan(stage_lc, pre_branch, pre_branch, True, None)
    return BranchPlan(stage_lc, pre_branch, source_ref, False, None)


def _warn_rc_sequence(repo_dir: str, pre_branch: str, stage_lc: str, major: int, minor: int, patch: int) -> None:
    """Log (non-blocking) if a continued rc number is out of sequence.

    A continued rc should be exactly one past the highest rc already recorded on
    the running branch; a repeat (re-cut) or a gap is probably a mis-dispatched
    stage. This only warns; the caller still cuts what was asked.
    """
    m = _RC_STAGE_RE.match(stage_lc)
    if not m:
        return
    requested = int(m.group(1))
    try:
        run_git(repo_dir, "fetch", "--quiet", "origin", pre_branch)
        notes = git_output(repo_dir, "show", "FETCH_HEAD:00-RELEASENOTES")
    except Exception:  # noqa: BLE001 - best-effort; absence just means "no prior rc"
        return
    pattern = re.compile(_DATED_RC_RE_TMPL.format(major=major, minor=minor, patch=patch), re.MULTILINE)
    seen = [int(x) for x in pattern.findall(notes)]
    highest = max(seen) if seen else 0
    expected = highest + 1
    if requested != expected:
        logger.warning(
            "Dispatched %s but %s records up to rc%d (expected rc%d). Cutting anyway: "
            "a repeat re-cuts an existing rc; a gap skips one.",
            stage_lc, pre_branch, highest, expected,
        )


def stage_release_name(version: str, stage_lc: str) -> str:
    """``9.1.0`` at ga, else ``9.1.0-rcN``."""
    return version if stage_lc == "ga" else f"{version}-{stage_lc}"


def commit_title(version: str, stage_lc: str) -> str:
    """Match valkey's release commit titles."""
    if stage_lc == "ga":
        return f"Add release notes entry for Valkey {version} GA"
    return f"Update version to {version}-{stage_lc} and add release notes"


def promote_and_bump(
    valkey_clone_dir: str,
    *,
    source_notes_text: str,
    dest_notes_text: str,
    dest_version_text: str,
    version: str,
    stage_lc: str,
    urgency: str,
    date: str,
    repo_full_name: str,
    contrib_base: Optional[str],
    token: Optional[str],
    security_fixes: Optional[Sequence[str]],
) -> tuple[str, str]:
    """Drain *source_notes_text*'s block onto the destination changelog and bump the version.

    Returns ``(new_dest_notes, new_version_h)``. The valkey primitives make all
    formatting decisions: ``promote`` in drain mode (``prior_text`` is the
    destination's running changelog) produces a frozen dated changelog with no
    ``## Unreleased`` block, and ``set_version`` rewrites the three version
    macros. The contributor list is generated over ``contrib_base..HEAD`` and
    merged into the cumulative footer by ``promote``.
    """
    rn = load_releasetools_module(valkey_clone_dir, "release_notes")
    bv = load_releasetools_module(valkey_clone_dir, "bump_version")

    contributors: list[str] = []
    if contrib_base:
        gc = load_releasetools_module(valkey_clone_dir, "gen_contributors")
        # Resolve both ends to SHAs the GitHub compare API accepts. contrib_base
        # is typically a remote-tracking ref (origin/unstable) and the head is the
        # literal "HEAD"; both 404 the API and silently fall back to git shortlog
        # (names only, no @handle, bots not filtered). See _compare_ref.
        base_sha = _compare_ref(valkey_clone_dir, contrib_base)
        head_sha = _compare_ref(valkey_clone_dir, "HEAD")
        contributors = gc.list_contributors(
            repo_full_name, base_sha, head_sha, token, repo_dir=valkey_clone_dir
        )
        logger.info("Collected %d contributor(s) over %s..HEAD", len(contributors), contrib_base)
    else:
        logger.warning("No contributor base ref/tag found; skipping contributor list")

    new_notes = rn.promote(
        source_notes_text,
        version=version,
        stage=stage_lc,
        urgency=urgency,
        date=date,
        contributors=contributors,
        security_fixes=list(security_fixes) if security_fixes else None,
        prior_text=dest_notes_text,
    )
    new_version = bv.set_version(dest_version_text, version, stage_lc)
    logger.info(
        "version.h -> VALKEY_VERSION=%s VALKEY_VERSION_NUM=%s VALKEY_RELEASE_STAGE=%s",
        version, bv.version_num(version), stage_lc,
    )
    return new_notes, new_version


def _contrib_base(
    repo_dir: str, *, explicit: Optional[str], notes_base_ref: Optional[str], plan: BranchPlan
) -> Optional[str]:
    """Pick the contributor-range start.

    Order: explicit ``--contrib-base-ref``, then the notes baseline, then last
    tag, then root commit. The contributor list must span the same range as the
    bullets, or the credits diverge from the notes. So whenever the notes
    baseline was pinned (``notes_base_ref``, an explicit ``--base-ref`` or rc1's
    derived previous release), it is used here before ``git describe``: on a
    branch following valkey's fork-at-freeze model, ``describe`` returns a far
    older nearest tag (e.g. 8.0.8 from unstable) than the real baseline (9.0.0),
    crediting a whole extra minor of history. The describe/root fallbacks remain
    for the tag-resolved path (rc2+/ga), where the notes baseline is a tag and
    ``notes_base_ref`` is None. Matches prepare_release's chain so the
    ``### Contributors`` list is never silently empty.
    """
    if explicit:
        return explicit
    if notes_base_ref:
        return notes_base_ref
    try:
        tag = git_output(repo_dir, "describe", "--tags", "--abbrev=0").strip()
        if tag:
            return tag
    except Exception:  # noqa: BLE001 - no tag reachable; fall through to root
        pass
    try:
        roots = git_output(repo_dir, "rev-list", "--max-parents=0", "HEAD").split("\n")
        roots = [r for r in roots if r.strip()]
        if roots:
            return roots[-1].strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _compare_ref(repo_dir: str, ref: str) -> str:
    """Resolve *ref* to a commit SHA the GitHub compare API can use.

    ``gen_contributors.list_contributors`` hits ``GET /compare/{base}...{head}``,
    which only accepts refs the server knows: a branch/tag name or a full commit
    SHA. The contributor base and head we have locally are neither. The base is a
    remote-tracking ref (``origin/unstable``, because the clone is
    ``--branch <source>`` so other branches exist only as ``origin/<name>``) and
    the head is the literal ``HEAD``. Both resolve fine for git but 404 the
    compare API, which silently drops to the ``git shortlog`` fallback:
    names-only, no ``@handle``, no ``[bot]`` filtering. Dereferencing each to its
    SHA here keeps the API path, and thus the ``Full Name @handle`` format,
    working. Falls back to the ref as given if it cannot be resolved (e.g. no
    local clone), so the contributor step degrades rather than crashing.
    """
    try:
        return git_output(repo_dir, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip() or ref
    except subprocess.CalledProcessError:
        return ref


def _credited_pr_numbers(notes_text: str) -> set[int]:
    """Return the PR numbers a release-line changelog already credits.

    Reads every bullet line's trailing ``(#N)`` from *notes_text* (a destination
    changelog: the dated sections of pre-release-M.m.p or M.m). This is the dedup
    key for promotion. Upstream, discovery excludes prior-RC PRs via the RC tag
    it walks back to, but the agent never pushes those tags and a fork carries
    none, so on GA (or any continued cut) discovery re-walks the whole source
    branch and re-finds PRs the line already shipped. Deduping the cut's bullets
    against this set makes promotion idempotent regardless of tags: a PR the line
    already lists is dropped instead of double-noted.
    """
    credited: set[int] = set()
    for line in notes_text.splitlines():
        if not _BULLET_LINE_RE.match(line):
            continue
        m = _TRAILING_PR_RE.search(line)
        if m:
            credited.add(int(m.group(1)))
    return credited


def _drop_already_credited(source_notes_text: str, credited: set[int]) -> tuple[str, list[int]]:
    """Drop bullets whose trailing ``(#N)`` is in *credited* from the source block.

    Returns ``(filtered_text, dropped_numbers)``. Only bullet lines are touched;
    headers, prose, and blank lines pass through unchanged, so the block still
    renders through the canonical format. A category left with no bullets stays
    as an empty ``### Header``; promote() and the format module already omit
    empty categories from the dated section, so no extra cleanup is needed here.
    """
    if not credited:
        return source_notes_text, []
    kept: list[str] = []
    dropped: list[int] = []
    for line in source_notes_text.split("\n"):
        if _BULLET_LINE_RE.match(line):
            m = _TRAILING_PR_RE.search(line)
            if m and int(m.group(1)) in credited:
                dropped.append(int(m.group(1)))
                continue
        kept.append(line)
    return "\n".join(kept), dropped


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def cut(
    repo: Any,
    *,
    repo_full_name: str,
    source_clone_dir: str,
    valkey_clone_dir: str,
    source_ref: str,
    version: str,
    stage: str,
    urgency: str,
    date: str,
    tag_glob: Optional[str],
    base_ref: Optional[str],
    contrib_base_ref: Optional[str],
    security_fixes: Optional[Sequence[str]],
    token: str,
    git_env: dict[str, str],
    dry_run: bool,
) -> int:
    """Cut a release: regenerate source notes with AI, drain onto the release line, open PRs.

    ``source_clone_dir`` is a clone of the source branch; it doubles as
    ``valkey_clone_dir`` for loading the release primitives. The destination
    release branch is materialized in a worktree under it. Returns 0 on success,
    1 on failure.
    """
    plan = resolve_branch_plan(
        source_clone_dir, version=version, stage=stage, source_ref=source_ref
    )
    logger.info(
        "Plan: stage=%s target=%s base=%s continuing=%s rename_from=%s",
        plan.stage, plan.target, plan.base_ref, plan.continuing, plan.rename_from or "<none>",
    )

    # 1. Regenerate the source branch's ## Unreleased block from labelled PRs.
    regen = pipeline_mod.regenerate_unreleased(
        repo, source_clone_dir, head_ref=source_ref, tag_glob=tag_glob, base_ref=base_ref
    )
    if regen.included and not regen.bullet_count and regen.wipes_existing:
        logger.error(
            "%d PR(s) included but no bullets generated; refusing to cut with a blanked block.",
            regen.included,
        )
        return 1
    source_notes = regen.updated_text  # source block now carries the fresh bullets

    # 2. Materialize a throwaway worktree at the release line's base. We never
    #    check out (or force-push) the real release branch; instead we build the
    #    promoted commit on an agent-namespaced prep branch and PR it into the
    #    release line, so the line only advances when a human merges. The prep
    #    branch starts from origin/<base_ref> so the PR diff is exactly the cut.
    run_git(source_clone_dir, "fetch", "origin", plan.base_ref, env=git_env)
    prep_branch = f"{PREP_BRANCH_PREFIX}/{version}-{plan.stage}"
    dest_dir = os.path.join(source_clone_dir, ".release-dest")
    run_git(source_clone_dir, "worktree", "add", "--force", "-B", prep_branch, dest_dir,
            f"origin/{plan.base_ref}")
    try:
        # A first cut of a line has no prior dated changelog to prepend.
        dest_notes_path = os.path.join(dest_dir, NOTES_FILE)
        dest_notes_text = _read(dest_notes_path) if plan.continuing else ""
        dest_version_text = _read(os.path.join(dest_dir, VERSION_FILE))

        # Drop bullets the destination changelog already credits. The tag-based
        # dedup in discovery cannot engage without RC tags (the agent never
        # pushes them; a fork has none), so a continued cut (most visibly GA
        # after the final RC) otherwise re-notes every PR the line already
        # shipped. With nothing new, the dated section renders empty (heading +
        # version bump only) and the PR body says so. This is a no-op upstream,
        # where discovery already returns only new PRs.
        already_credited = sorted(
            _credited_pr_numbers(dest_notes_text)
            & _credited_pr_numbers(source_notes)
        )
        if already_credited:
            source_notes, _dropped = _drop_already_credited(
                source_notes, set(already_credited)
            )
            logger.info(
                "Dropped %d PR(s) already credited on %s: %s",
                len(already_credited), plan.target, already_credited,
            )

        # Anchor contributors to the same baseline the bullets used (regen.base_tag
        # is the resolved tag for rc2+/ga, or the pinned base_ref / rc1 default),
        # so the credits never span a different range than the notes.
        contrib_base = _contrib_base(
            source_clone_dir, explicit=contrib_base_ref,
            notes_base_ref=regen.base_tag, plan=plan,
        )

        # 3. Drain source bullets -> dated section on dest; bump version.h.
        new_dest_notes, new_version = promote_and_bump(
            valkey_clone_dir,
            source_notes_text=source_notes,
            dest_notes_text=dest_notes_text,
            dest_version_text=dest_version_text,
            version=version, stage_lc=plan.stage, urgency=urgency, date=date,
            repo_full_name=repo_full_name, contrib_base=contrib_base, token=token,
            security_fixes=security_fixes,
        )

        if dry_run:
            _print_dry_run(plan, version, new_dest_notes, new_version, regen, already_credited)
            return 0

        # 4. Ensure the release line exists to PR into. When starting a new line
        #    (rc1, first GA, or a GA rename carrying the rc history), create it at
        #    origin/<base_ref> with a non-force push so a race can't clobber it.
        if not _remote_branch_exists(source_clone_dir, plan.target):
            run_git(source_clone_dir, "push", "origin",
                    f"origin/{plan.base_ref}:refs/heads/{plan.target}", env=git_env)
            logger.info("Created release line %s at origin/%s", plan.target, plan.base_ref)

        # 5. Commit the promoted notes + bumped version on the prep branch, push
        #    it (agent-namespaced, force-with-lease), and PR it into the line. The
        #    source branch is never modified: there is no ## Unreleased block to
        #    empty, so no companion PR. Each cut rediscovers PRs from the last RC
        #    tag, so prior RCs' PRs are excluded by the graph range, not by reset.
        _write(dest_notes_path, new_dest_notes)
        _write(os.path.join(dest_dir, VERSION_FILE), new_version)
        release_url = _commit_push_release_pr(
            repo, dest_dir, repo_full_name=repo_full_name, plan=plan,
            version=version, prep_branch=prep_branch, triage=regen.triage,
            already_credited=already_credited, git_env=git_env,
        )
        # 6. GA rename: delete the old pre-release branch (best-effort). The M.m
        #    line was created from it above, so its history is already carried.
        if plan.rename_from:
            _delete_remote_branch(source_clone_dir, plan.rename_from, git_env)

        logger.info("Release PR: %s", release_url)
        return 0
    finally:
        run_git(source_clone_dir, "worktree", "remove", "--force", dest_dir)


def _print_dry_run(plan, version, dest_notes, version_h, regen, already_credited) -> None:
    print(f"\n===== release plan ({version} {plan.stage}) =====")
    print(f"target branch: {plan.target}  base: {plan.base_ref}  continuing: {plan.continuing}")
    if plan.rename_from:
        print(f"GA rename: would delete {plan.rename_from}")
    if already_credited:
        print(f"already credited on {plan.target} (dropped): {already_credited}")
    if regen.triage:
        print(f"triage PRs (untagged): {[p.number for p in regen.triage]}")
    print(f"\n===== {NOTES_FILE} (release branch, dry run) =====\n{dest_notes}")
    print(f"\n===== {VERSION_FILE} (dry run) =====\n{version_h}")


def _commit_push_release_pr(
    repo: Any, dest_dir: str, *, repo_full_name: str, plan: BranchPlan, version: str,
    prep_branch: str, triage: Sequence[Any], already_credited: Sequence[int],
    git_env: dict[str, str],
) -> str:
    """Commit the cut on the prep branch, push it, and open/update a PR into the line.

    The PR is ``head=prep_branch`` into ``base=plan.target`` (the release line),
    so it shows exactly the promoted diff and merges into the line, never the
    self-referential merge-back-into-source shape the release line must avoid.
    The prep branch is agent-namespaced, so force-with-lease on it is safe.
    *triage* (untagged / double-labelled PRs in range) is listed in the PR body
    for a maintainer to label. *already_credited* (PRs the line already lists,
    dropped from this cut) is surfaced so an empty dated section is explained.
    """
    run_git(dest_dir, "config", "user.name", BOT_NAME)
    run_git(dest_dir, "config", "user.email", BOT_EMAIL)
    run_git(dest_dir, "add", NOTES_FILE, VERSION_FILE)
    run_git(dest_dir, "commit", "-s", "-m", commit_title(version, plan.stage))
    if not prep_branch.startswith(f"{PREP_BRANCH_PREFIX}/"):
        raise RuntimeError(f"Refusing to push to non-namespaced prep branch: {prep_branch!r}")
    run_git(dest_dir, "push", "--force-with-lease", "origin", f"HEAD:{prep_branch}", env=git_env)

    title = commit_title(version, plan.stage)
    body = (
        f"Cuts **{stage_release_name(version, plan.stage)}** onto release line "
        f"`{plan.target}`.\n\n"
        f"- Promotes the release notes into a dated section, bumps "
        f"`src/version.h`, and refreshes the running contributor list.\n"
        + (f"- GA: carries `{plan.rename_from}`'s history; that branch is deleted by this run.\n"
           if plan.rename_from else "")
        + _no_new_prs_section(already_credited, plan)
        + _triage_section(triage)
        + "\n*Generated by valkey-ci-agent. Review before merging into the release line.*"
    )
    existing = publish_mod.find_existing_pr(
        repo, base_repo=repo_full_name, push_repo=None, branch=prep_branch
    )
    return publish_mod.open_or_update_pr(
        repo, base_repo=repo_full_name, push_repo=None, branch=prep_branch,
        base_branch=plan.target, title=title, body=body, existing=existing,
    )


def _no_new_prs_section(already_credited: Sequence[int], plan: BranchPlan) -> str:
    """Warn in the PR body when every PR in range was already credited on the line.

    Returns an empty string unless some PR was dropped as a duplicate. When the
    drop leaves the dated section with no bullets (the common GA-after-final-RC
    case), the cut is version-bump-only, and the reader needs to know the empty
    notes are intentional rather than a generation miss.
    """
    if not already_credited:
        return ""
    refs = ", ".join(f"#{n}" for n in already_credited)
    return (
        "\n### No new release notes\n\n"
        f"Every release-noted PR in range is already credited on `{plan.target}` "
        f"(carried from an earlier cut): {refs}. They were dropped to avoid "
        "duplicate entries, so this cut only adds the dated heading and the "
        "`src/version.h` bump. If you expected new notes here, confirm the new "
        "PRs merged into the source branch and carry the `release-notes` label.\n"
    )


def _triage_section(triage: Sequence[Any]) -> str:
    """Render a Markdown table of untagged/double-labelled PRs for the PR body."""
    if not triage:
        return ""
    lines = [
        "",
        "### Needs triage",
        "",
        "These merged PRs in range carry neither `release-notes` nor "
        "`no-release-notes` (or carry both) and were not included. A maintainer "
        "should label them:",
        "",
        "| PR | Title | Author |",
        "|----|-------|--------|",
    ]
    for pr in triage:
        author = f"@{pr.author}" if pr.author else "(unknown)"
        lines.append(f"| [#{pr.number}]({pr.url}) | {publish_mod.escape_cell(pr.title)} | {author} |")
    lines.append("")
    return "\n".join(lines)


def _delete_remote_branch(repo_dir: str, branch: str, git_env: dict[str, str]) -> None:
    """Delete a remote branch (best-effort: a missing branch is not an error)."""
    try:
        run_git(repo_dir, "push", "origin", "--delete", branch, env=git_env)
        logger.info("Deleted remote branch %s (GA rename)", branch)
    except Exception as exc:  # noqa: BLE001
        logger.info("Could not delete %s (already gone?): %s", branch, exc)
