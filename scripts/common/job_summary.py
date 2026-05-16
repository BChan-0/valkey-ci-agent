"""Write to GitHub Actions job summaries."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def emit_job_summary(text: str) -> None:
    """Write *text* to the GitHub Actions job summary file.

    The path is read from the ``GITHUB_STEP_SUMMARY`` environment variable.
    If the variable is unset or the write fails, the error is logged but
    does not raise.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        logger.info("GITHUB_STEP_SUMMARY not set; skipping job summary.")
        return
    try:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
        logger.info("Wrote job summary to %s.", summary_path)
    except OSError as exc:
        logger.warning("Failed to write job summary: %s", exc)
