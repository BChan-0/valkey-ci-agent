"""AI triage for the issues the Test Failure Detector filed.

Runs after a detector run and analyzes the issues that run recorded, or with an
explicit issue list to analyze issues filed earlier.

Kept separate from the detector rather than folded into it for two reasons. The
detector's job is to make sure no failure goes untracked, and that signal should
not go red because Bedrock was unavailable. And because triage finds its work by
reading published issues, analyzing an older issue is the same code path as
analyzing a fresh one, with a different filter.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from github import Auth, Github

from scripts.common.job_summary import emit_job_summary
from scripts.common.workflow_artifacts import ArtifactClient
from scripts.failure_triage import analyze, comment, discover, evidence
from scripts.failure_triage.models import RunReport, TriageResult, TriageTarget

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 10
_DEFAULT_MAX_AGE_DAYS = 30

# Wall-clock budget for the whole run. One analysis can take up to an hour
# (two attempts at the profile's 30-minute timeout), so a full batch could
# otherwise outlast the job timeout and lose the report and evidence uploads.
# Once this passes, no new target is started and the report is written. Set
# below the workflow's job timeout to leave room for the uploads.
_DEFAULT_MAX_RUNTIME_SECONDS = 150 * 60


def run(
    *,
    github_token: str,
    repo_full_name: str,
    run_id: int | None = None,
    issue_numbers: tuple[int, ...] = (),
    limit: int = _DEFAULT_LIMIT,
    max_age_days: int = _DEFAULT_MAX_AGE_DAYS,
    force: bool = False,
    dry_run: bool = False,
    triage_run_url: str = "",
    output: str | None = None,
    max_runtime_seconds: float = _DEFAULT_MAX_RUNTIME_SECONDS,
    verbose: bool = False,
) -> int:
    """Triage the selected issues and return the process exit code.

    :param github_token: token with issues:write and actions:read on the target.
    :param repo_full_name: repository whose issues are triaged.
    :param run_id: analyze only the issues this detector run recorded.
    :param issue_numbers: analyze these issues, ignoring the age filter.
    :param limit: most issues to analyze in one run.
    :param max_age_days: skip issues older than this, whose logs have expired.
    :param force: analyze issues that already carry a triage comment.
    :param dry_run: analyze and report without commenting.
    :param triage_run_url: URL of this workflow run, linked in the comment.
    :param output: write the JSON report here instead of stdout.
    :param max_runtime_seconds: stop starting new targets past this budget.
    :param verbose: enable debug logging.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    gh = Github(auth=Auth.Token(github_token))
    artifact_client = ArtifactClient(gh, token=github_token)

    targets = discover.select_targets(
        gh,
        repo_full_name,
        run_id=run_id,
        issue_numbers=issue_numbers,
        limit=limit,
        max_age_days=max_age_days,
        force=force,
    )

    report = RunReport(repo=repo_full_name, run_id=run_id or 0, dry_run=dry_run)

    deadline = time.monotonic() + max_runtime_seconds
    for index, target in enumerate(targets):
        # Stop before an analysis that would likely outlast the job timeout and
        # lose the report and evidence uploads. The already-started targets are
        # reported; the rest are picked up on the next run.
        if index > 0 and time.monotonic() >= deadline:
            logger.warning(
                "Runtime budget reached; %d of %d issue(s) not started",
                len(targets) - index, len(targets),
            )
            break

        # Isolate each target. An unreadable log, a clone failure, or a GitHub
        # error that outlasts its retries must not drop the issues after it.
        try:
            result = _triage_one(
                gh,
                repo_full_name,
                target,
                artifact_client=artifact_client,
                dry_run=dry_run,
                triage_run_url=triage_run_url,
            )
        except Exception as exc:
            logger.warning(
                "Could not triage issue #%s; skipping it",
                target.issue_number, exc_info=True,
            )
            result = TriageResult(
                issue_number=target.issue_number,
                issue_url=target.issue_url,
                display_name=target.display_name,
                action="error",
                detail=str(exc),
            )
        report.results.append(result)

    _write_report(report, output)
    emit_job_summary(_build_job_summary(report))

    # A target that errored got no comment, so the run is not clean. Exit non-zero
    # to surface that in the Actions tab rather than leaving it in the JSON.
    return 1 if report.count("error") else 0


def _triage_one(
    gh: Github,
    repo_full_name: str,
    target: TriageTarget,
    *,
    artifact_client: ArtifactClient,
    dry_run: bool,
    triage_run_url: str,
) -> TriageResult:
    """Gather evidence, analyze, and comment for one issue."""
    logger.info(
        "Triaging issue #%s: %s", target.issue_number, target.display_name,
    )

    # The working directory holds the excerpt and the source clone, and is
    # discarded afterwards: it is the agent's sandbox, not an artifact.
    with tempfile.TemporaryDirectory(prefix="failure-triage-") as tmp:
        workdir = Path(tmp)
        collected = evidence.collect(
            target,
            workdir,
            repo_full_name=repo_full_name,
            artifact_client=artifact_client,
            gh=gh,
        )

        if not collected.has_log and not collected.has_source:
            reason = (
                collected.log_unavailable_reason
                or "neither the job log nor the source could be retrieved"
            )
            logger.info(
                "No evidence for issue #%s: %s", target.issue_number, reason,
            )
            return TriageResult(
                issue_number=target.issue_number,
                issue_url=target.issue_url,
                display_name=target.display_name,
                action="skipped-no-evidence",
                detail=reason,
            )

        analysis = analyze.triage(target, collected)

    if dry_run:
        logger.info(
            "Dry run; not commenting on issue #%s", target.issue_number,
        )
        return TriageResult(
            issue_number=target.issue_number,
            issue_url=target.issue_url,
            display_name=target.display_name,
            action="would-comment",
            analysis=analysis,
        )

    try:
        comment_url = comment.post(
            gh,
            repo_full_name,
            target,
            analysis,
            triage_run_url=triage_run_url,
        )
    except Exception as exc:
        # Carry the analysis on the result even though the post failed, so the
        # verdict, which cost a full agent run, survives in the report instead of
        # being redone on the next run.
        logger.warning(
            "Could not comment on issue #%s", target.issue_number, exc_info=True,
        )
        return TriageResult(
            issue_number=target.issue_number,
            issue_url=target.issue_url,
            display_name=target.display_name,
            action="error",
            analysis=analysis,
            detail=f"comment failed: {exc}",
        )

    return TriageResult(
        issue_number=target.issue_number,
        issue_url=target.issue_url,
        display_name=target.display_name,
        action="analysis-failed" if analysis.failed else "commented",
        analysis=analysis,
        detail=comment_url,
    )


def _write_report(report: RunReport, output: str | None) -> None:
    """Write the run's JSON report to *output* or stdout."""
    rendered = json.dumps(asdict(report), indent=2, sort_keys=True)
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


def _build_job_summary(report: RunReport) -> str:
    """The markdown summary shown on the workflow run."""
    lines = [
        "## AI Failure Triage",
        "",
        f"**Repository:** [{report.repo}](https://github.com/{report.repo})",
    ]
    if report.run_id:
        lines.append(
            f"**Detector run:** "
            f"[{report.run_id}](https://github.com/{report.repo}/actions/runs/{report.run_id})"
        )
    if report.dry_run:
        lines.append("**Mode:** dry run, no comments posted")
    lines.extend([
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Issues selected | {len(report.results)} |",
        f"| Analyses commented | {report.count('commented')} |",
        f"| Analyses that would be posted | {report.count('would-comment')} |",
        f"| Analyses unavailable | {report.count('analysis-failed')} |",
        f"| Skipped, no evidence | {report.count('skipped-no-evidence')} |",
        f"| Errors | {report.count('error')} |",
        "",
    ])

    if report.results:
        lines.extend(["| Issue | Failure | Outcome |", "|---|---|---|"])
        for result in report.results:
            lines.append(
                f"| [#{result.issue_number}]({result.issue_url}) "
                f"| {_escape_cell(result.display_name)} | {result.action} |"
            )
        lines.append("")

    return "\n".join(lines)


def _escape_cell(text: str) -> str:
    """Keep a failure name from breaking the summary table it sits in."""
    return text.replace("|", "\\|").replace("\n", " ")


def _parse_issue_numbers(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated issue list, ignoring blanks.

    Each token must be plain digits. ``int()`` alone would accept ``1_0`` (which
    silently means issue 10) and signed values, so a stricter check is used.
    Duplicates are dropped, keeping first position, so one issue is not triaged,
    and commented on, twice in a run.
    """
    numbers: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise SystemExit(f"invalid issue number: {token!r}")
        number = int(token)
        if number == 0:
            raise SystemExit("issue number must be greater than 0")
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return tuple(numbers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post AI failure analysis on Test Failure Detector issues.",
    )
    parser.add_argument(
        "--token", default=None,
        help="GitHub token (falls back to the TARGET_TOKEN environment variable).",
    )
    parser.add_argument(
        "--repo", required=True,
        help="Repository whose issues are triaged (e.g. valkey-io/valkey).",
    )
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Only triage issues recorded by this detector run.",
    )
    parser.add_argument(
        "--issues", default="",
        help="Comma-separated issue numbers to triage, ignoring the age filter.",
    )
    parser.add_argument(
        "--limit", type=int, default=_DEFAULT_LIMIT,
        help=f"Most issues to triage in one run (default: {_DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--max-age-days", type=int, default=_DEFAULT_MAX_AGE_DAYS,
        help=(
            "Skip issues older than this many days, whose run logs are likely "
            f"expired (default: {_DEFAULT_MAX_AGE_DAYS}, 0 disables the filter)."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Triage issues that already carry a triage comment.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze and report without commenting on any issue.",
    )
    parser.add_argument(
        "--triage-run-url", default="",
        help="URL of this workflow run, linked from the posted comment.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write the JSON report to this path instead of stdout.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("TARGET_TOKEN", "")
    if not token:
        parser.error("--token or the TARGET_TOKEN environment variable is required")

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.max_age_days < 0:
        parser.error("--max-age-days must be 0 or greater (0 disables the filter)")

    if args.run_id is not None and args.run_id <= 0:
        parser.error("--run-id must be a positive run id")

    return run(
        github_token=token,
        repo_full_name=args.repo,
        run_id=args.run_id,
        issue_numbers=_parse_issue_numbers(args.issues),
        limit=args.limit,
        max_age_days=args.max_age_days,
        force=args.force,
        dry_run=args.dry_run,
        triage_run_url=args.triage_run_url,
        output=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    raise SystemExit(main())
