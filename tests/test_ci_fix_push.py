"""Push-path tests, focused on the PORT path preserving upstream authorship.

A port carries an already-merged upstream commit, so the pushed commit must keep
the original author and sign-off rather than being re-authored as the bot.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.ci_fix import push as push_mod
from scripts.ci_fix.push import PushRefused, commit_and_push_port


def _git(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, **kw).stdout


def _make_origin(tmp_path):
    """A repo whose default branch (unstable) carries a fix authored by a
    distinct human, and a release branch that diverged before the fix and is
    missing it. Returns (work, bare, fix_sha, head_sha).

    ``unstable`` is set as the default branch (origin/HEAD) so the port path's
    ancestry check resolves it the way a real clone would.
    """
    work = tmp_path / "origin"
    work.mkdir()
    _git(work, "init", "-q", "-b", "unstable")
    _git(work, "config", "user.email", "base@t")
    _git(work, "config", "user.name", "Base")
    (work / "f.txt").write_text("base\n")
    _git(work, "add", "f.txt")
    _git(work, "commit", "-qm", "base")
    base_sha = _git(work, "rev-parse", "HEAD").strip()

    # The release branch diverges at base, before the fix lands.
    _git(work, "branch", "agent/backport/sweep/9.0", base_sha)

    # The upstream fix on unstable (the default branch), authored by a human.
    (work / "g.txt").write_text("the fix\n")
    _git(work, "add", "g.txt")
    _git(work, "-c", "user.email=author@example.com", "-c", "user.name=Real Author",
         "commit", "-qm", "the upstream fix\n\nSigned-off-by: Real Author <author@example.com>")
    fix_sha = _git(work, "rev-parse", "HEAD").strip()

    head_sha = _git(work, "rev-parse", "agent/backport/sweep/9.0").strip()
    _git(work, "checkout", "-q", "agent/backport/sweep/9.0")

    bare = tmp_path / "remote.git"
    _git(work, "clone", "-q", "--bare", str(work), str(bare))
    # Point the bare remote's HEAD at unstable so a clone resolves it as default.
    _git(bare, "symbolic-ref", "HEAD", "refs/heads/unstable")
    return work, bare, fix_sha, head_sha


def test_port_push_preserves_original_authorship(tmp_path, monkeypatch):
    work, bare, fix_sha, head_sha = _make_origin(tmp_path)

    # Clone from the bare remote (it has both branches, the fix commit, and
    # HEAD->unstable), and push back to it, so no network or real GitHub is used.
    def fake_clone(_full_name, dest: Path):
        _git(dest.parent, "clone", "-q", str(bare), str(dest))

    monkeypatch.setattr(push_mod, "_clone_clean", fake_clone)
    monkeypatch.setattr(push_mod, "github_https_url", lambda _n: str(bare))

    pushed_sha = commit_and_push_port(
        str(work),
        head_repo_full_name="valkey-io/valkey",
        head_branch="agent/backport/sweep/9.0",
        head_sha=head_sha,
        unstable_fix_commit=fix_sha,
        git_env={},
    )

    # The pushed commit on the bare remote keeps the original author, not the bot.
    author = _git(bare, "log", "-1", "--format=%an <%ae>", pushed_sha).strip()
    assert author == "Real Author <author@example.com>"
    body = _git(bare, "log", "-1", "--format=%B", pushed_sha)
    assert "Signed-off-by: Real Author <author@example.com>" in body
    assert "cherry picked from commit" in body  # the -x trailer


def test_port_push_refuses_commit_not_on_default_branch(tmp_path, monkeypatch):
    """A SHA the model points at that is not reachable from the default branch
    is refused, not ported: code owns 'this is a merged upstream fix'."""
    work, bare, _fix_sha, head_sha = _make_origin(tmp_path)

    # A commit that exists but lives only on a side branch, never merged to
    # the default branch.
    _git(work, "checkout", "-qb", "rogue", head_sha)
    (work / "rogue.txt").write_text("not upstream\n")
    _git(work, "add", "rogue.txt")
    _git(work, "commit", "-qm", "rogue commit")
    rogue_sha = _git(work, "rev-parse", "HEAD").strip()
    _git(work, "checkout", "-q", "agent/backport/sweep/9.0")
    # Refresh the bare so the clone can fetch the rogue object by SHA.
    rebare = tmp_path / "remote2.git"
    _git(work, "clone", "-q", "--bare", str(work), str(rebare))
    _git(rebare, "symbolic-ref", "HEAD", "refs/heads/unstable")

    def fake_clone(_full_name, dest: Path):
        _git(dest.parent, "clone", "-q", str(rebare), str(dest))

    monkeypatch.setattr(push_mod, "_clone_clean", fake_clone)
    monkeypatch.setattr(push_mod, "github_https_url", lambda _n: str(rebare))

    with pytest.raises(PushRefused, match="not reachable from"):
        commit_and_push_port(
            str(work),
            head_repo_full_name="valkey-io/valkey",
            head_branch="agent/backport/sweep/9.0",
            head_sha=head_sha,
            unstable_fix_commit=rogue_sha,
            git_env={},
        )


def test_port_push_refuses_commit_already_on_head(tmp_path, monkeypatch):
    """Porting a commit already present on the PR head is a no-op and refused."""
    work, bare, fix_sha, _head_sha = _make_origin(tmp_path)

    # Make the head branch already contain the fix (fast-forward it to unstable).
    head_with_fix = fix_sha

    def fake_clone(_full_name, dest: Path):
        _git(dest.parent, "clone", "-q", str(bare), str(dest))

    monkeypatch.setattr(push_mod, "_clone_clean", fake_clone)
    monkeypatch.setattr(push_mod, "github_https_url", lambda _n: str(bare))

    with pytest.raises(PushRefused, match="already present on the PR head"):
        commit_and_push_port(
            str(work),
            head_repo_full_name="valkey-io/valkey",
            head_branch="agent/backport/sweep/9.0",
            head_sha=head_with_fix,
            unstable_fix_commit=fix_sha,
            git_env={},
        )


def test_port_push_refuses_non_namespaced_branch(tmp_path):
    with pytest.raises(PushRefused):
        commit_and_push_port(
            str(tmp_path), head_repo_full_name="valkey-io/valkey",
            head_branch="main", head_sha="a" * 40, unstable_fix_commit="b" * 40, git_env={},
        )


def test_port_push_refuses_malformed_commit(tmp_path):
    with pytest.raises(PushRefused):
        commit_and_push_port(
            str(tmp_path), head_repo_full_name="valkey-io/valkey",
            head_branch="agent/backport/sweep/9.0", head_sha="a" * 40,
            unstable_fix_commit="not-a-sha", git_env={},
        )
