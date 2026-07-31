"""Data shapes passed between the triage stages.

One target flows through the pipeline in order: :class:`TriageTarget` names the
issue and the failure recorded on it, :class:`FailureEvidence` describes the
files laid out for the agent to read, and :class:`TriageAnalysis` holds the
verdict that gets rendered into the issue comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class JobRef:
    """A CI job a failure was reported in.

    ``job_id`` is the numeric id parsed out of the job's URL, which is what the
    per-job log endpoint takes. It is None when the issue recorded a plain run
    URL with no job anchor, in which case that job's log cannot be fetched.
    """

    name: str
    url: str
    job_id: int = 0

    @property
    def has_log(self) -> bool:
        return self.job_id > 0


@dataclass(frozen=True)
class TriageTarget:
    """An issue to analyze, with the failure facts read off its body."""

    issue_number: int
    issue_url: str
    failure_type: str
    test_name: str
    test_file: str
    error: str
    run_id: int
    jobs: tuple[JobRef, ...] = ()

    @property
    def display_name(self) -> str:
        if self.test_name:
            return f"{self.test_name} in {self.test_file}"
        return f"[{self.failure_type}] in {self.test_file or 'unknown'}"

    def first_job_with_log(self) -> JobRef | None:
        """The first recorded job whose log can be fetched, or None."""
        return next((job for job in self.jobs if job.has_log), None)


@dataclass
class FailureEvidence:
    """The files laid out for the agent, and what could not be gathered.

    ``log_excerpt_path`` and ``source_path`` are relative to ``workdir``, which
    is the agent's working directory, so the prompt can name them as the agent
    will see them.
    """

    workdir: Path
    job_name: str = ""
    log_excerpt_path: str = ""
    source_path: str = ""
    source_sha: str = ""
    excerpt_lines: int = 0
    log_truncated: bool = False
    log_unavailable_reason: str = ""

    @property
    def has_log(self) -> bool:
        return bool(self.log_excerpt_path)

    @property
    def has_source(self) -> bool:
        return bool(self.source_path)


@dataclass
class TriageAnalysis:
    """The agent's verdict on one failure.

    ``failed`` marks an analysis that could not be produced at all, which is
    reported on the issue as an honest note rather than being dropped silently.
    """

    summary: str = ""
    root_cause: str = ""
    failure_class: str = "undetermined"
    confidence: str = "low"
    category: str = ""
    evidence: tuple[str, ...] = ()
    suspected_area: str = ""
    fix_suggestion: str = ""
    fix_confidence: str = "low"
    reproduction_hint: str = ""
    failed: bool = False
    error: str = ""

    @property
    def has_fix_suggestion(self) -> bool:
        """Whether a fix suggestion should be shown.

        A suggestion is withheld unless the agent also stood behind it. The
        model is told to return null when it would be guessing, and low
        confidence is the same statement made in the confidence field, so both
        are treated as no suggestion.
        """
        return bool(self.fix_suggestion) and self.fix_confidence in {"high", "medium"}


@dataclass
class TriageResult:
    """What happened for one target, for the run's JSON output and summary."""

    issue_number: int
    issue_url: str
    display_name: str
    action: str
    analysis: TriageAnalysis | None = None
    detail: str = ""


@dataclass
class RunReport:
    """Every target's outcome, plus the counts the job summary reports."""

    repo: str
    run_id: int
    dry_run: bool
    results: list[TriageResult] = field(default_factory=list)

    def count(self, action: str) -> int:
        return sum(1 for result in self.results if result.action == action)
