"""Typed data passed through the scheduled backport sweep."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProjectBackportCandidate:
    source_pr_number: int
    source_pr_title: str
    source_pr_url: str
    target_branch: str
    merge_commit_sha: str | None = None
    commit_shas: list[str] = field(default_factory=list)
    merged_at: str = ""


@dataclass
class CandidateResult:
    source_pr_number: int
    source_pr_title: str
    # One of: applied, skipped-existing, skipped-conflict,
    # skipped-validation-failed, error.
    outcome: str
    detail: str = ""


# Detail string used when a candidate PR is already cherry-picked onto the
# backport sweep branch. Reporting treats this as "on the branch", unlike
# empty cherry-picks that mean "already on the release branch".
DETAIL_ALREADY_ON_SWEEP_BRANCH = "already on backport branch"


@dataclass
class BranchSweepResult:
    target_branch: str
    candidates_found: int = 0
    results: list[CandidateResult] = field(default_factory=list)
    pr_url: str = ""
    error: str = ""

    @property
    def applied_count(self) -> int:
        """Number of candidates that were cherry-picked onto the branch."""
        return sum(1 for item in self.results if item.outcome == "applied")
