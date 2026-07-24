"""Parse and deduplicate test failures from the consolidated artifact.

Handles multiple failure types: assertion errors (the original case),
sanitizer/valgrind memory errors, timeouts, exceptions, server startup
failures, and gtest unit-test failures.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    """Classification of a CI test failure."""

    ASSERTION = "assertion"
    SANITIZER = "sanitizer"
    VALGRIND = "valgrind"
    TIMEOUT = "timeout"
    STARTUP = "startup"
    EXCEPTION = "exception"
    MEMORY_LEAK = "memory-leak"
    UNITTEST = "unittest"


# Patterns stripped from error text to produce a stable identity across runs.
_VOLATILE_PATTERNS = (
    # ANSI escape codes
    re.compile(r"\033\[[0-9;]*m"),
    # Valgrind PID annotations: ==12345==
    re.compile(r"==\d+==\s*"),
    # Hex addresses: 0xDEADBEEF
    re.compile(r"0x[0-9a-fA-F]+"),
    # Temp paths: /tmp/foo/bar.123
    re.compile(r"/tmp/[^\s:]+"),
    # PID/port annotations: pid 12345, port=6379, port 6379
    re.compile(r"\b(?:pid|port)[=\s]+\d+", re.IGNORECASE),
    # Bare large numbers (>=4 digits) that are likely PIDs/timestamps/addresses
    re.compile(r"\b\d{4,}\b"),
)

# Lines containing these keywords are considered "significant" for identity.
_SIGNIFICANT_KEYWORDS = (
    "error:", "Error:", "ERROR:",
    "Invalid", "Mismatched", "uninitialized",
    "runtime error", "Sanitizer", "SUMMARY:",
    "fishy", "overlap", "Can't start",
    "heap-buffer-overflow", "heap-use-after-free",
    "stack-buffer-overflow", "use-after-poison",
    "definitely lost", "LEAK SUMMARY",
)

# Report boilerplate that matches a significance keyword but is identical in
# every valgrind run, so it carries no bug identity. The "error:" keyword
# would otherwise match the tool banner via the runner's wrapper prefix
# ("Valgrind error: ==123== Memcheck, ...").
_BOILERPLATE_SUBSTRINGS = (
    "Memcheck, a memory error detector",
    "HEAP SUMMARY:",
    "LEAK SUMMARY:",
)

# Heap-layout coordinates and allocation sizes: the same leak moves between
# loss records and can vary in size run to run, so these must not feed the
# fingerprint identity.
_VOLATILE_COUNT_PATTERNS = (
    re.compile(r"\bin loss record \d[\d,]* of \d[\d,]*"),
    re.compile(r"\b\d[\d,]*\s+(?:bytes?|blocks?)\b"),
    # Any remaining thousands-separated number is a count/size.
    re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),
)

# A valgrind stack frame after volatile stripping: "at : malloc (...)" or
# "by : sdsdup (sds.c:190)". The function names are the stable identity
# anchor for a leak; addresses, sizes, and loss records around them are not.
_STACK_FRAME_RE = re.compile(r"^(?:at|by)\s*:\s*(?P<func>[^\s(]+)")


def _extract_stack_anchor(lines: list[str]) -> str:
    """Function-name chain of the first stack block in a valgrind report.

    Distinguishes two different leaks whose report lines are otherwise
    identical after count scrubbing (e.g. same "definitely lost" shape but
    allocated from debugCommand vs clusterCommand). Returns "" when the
    error has no at/by frames (assertions, startup failures).
    """
    frames: list[str] = []
    in_stack = False
    for line in lines:
        match = _STACK_FRAME_RE.match(line)
        if match:
            in_stack = True
            frames.append(match.group("func"))
            if len(frames) >= 8:
                break
        elif in_stack:
            # First stack block ended; a second block would belong to a
            # different loss record and make the identity order-sensitive.
            break
    if not frames:
        return ""
    return "stack: " + " > ".join(frames)

# Test names that carry no real test identity: they're volatile artifacts of
# whatever the runner happened to be doing when it timed out (spawning a
# server, between tests). Using them as identity would mint a fresh
# fingerprint (and a fresh issue) every run.
_VOLATILE_TEST_NAME_RE = re.compile(
    r"^(?:"
    r"pid:\d+"           # server PID annotation: "pid:92663"
    r"|hang"             # generic "hang in <file> (last state: ...)"
    r")$"
)


def normalize_error_identity(error: str) -> str:
    """Extract a stable identity from an error message for fingerprinting.

    Strips run-specific volatile tokens (PIDs, hex addresses, temp paths,
    timestamps) and extracts the first few meaningful lines that characterize
    the error type and location.

    Two runs that hit the same bug with different PIDs/addresses will produce
    the same normalized identity.
    """
    text = error
    for pattern in _VOLATILE_PATTERNS:
        text = pattern.sub("", text)
    for pattern in _VOLATILE_COUNT_PATTERNS:
        text = pattern.sub("", text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Extract up to 3 significant lines for the identity, skipping tool
    # boilerplate that is identical in every run of every bug.
    significant: list[str] = []
    for line in lines[:30]:
        if any(bp in line for bp in _BOILERPLATE_SUBSTRINGS):
            continue
        if any(kw in line for kw in _SIGNIFICANT_KEYWORDS):
            significant.append(line)
            if len(significant) >= 3:
                break

    # The stack frames pin the identity to the code path (two leaks with
    # identical report lines but different allocation sites stay distinct).
    stack_anchor = _extract_stack_anchor(lines)
    if significant:
        return "\n".join([*significant, stack_anchor] if stack_anchor else significant)
    if stack_anchor:
        return stack_anchor
    # Fall back to first 3 non-empty lines
    return "\n".join(lines[:3])


@dataclass
class JobReference:
    """A reference to a specific CI job where a test failed."""

    job: str
    suite: str
    url: str = ""


@dataclass
class UniqueFailure:
    """A deduplicated test failure that may appear across multiple jobs."""

    test_name: str
    test_file: str
    failure_type: FailureType = FailureType.ASSERTION
    error: str = ""
    jobs: list[JobReference] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        if self.test_name:
            return f"{self.test_name} in {self.test_file}"
        return f"[{self.failure_type.value}] in {self.test_file or 'unknown'}"

    @property
    def has_test_identity(self) -> bool:
        """Whether this failure has a meaningful test_name for fingerprinting."""
        return bool(self.test_name)


def parse_and_deduplicate(
    all_failures: dict[str, Any],
    job_urls: dict[str, str],
) -> list[UniqueFailure]:
    """Parse the all-test-failures JSON and deduplicate.

    Args:
        all_failures: The parsed all-test-failures.json content.
            Structure: {job_name: {suite_name: [{test_name, test_file, type?, error}]}}
        job_urls: Mapping of job name -> HTML URL for CI links.

    Returns:
        List of UniqueFailure objects, deduplicated across jobs.

    Grouping logic:
        - Failures WITH a test_name: grouped by (failure_type, test_name, test_file)
        - Failures WITHOUT a test_name (sanitizer/valgrind/startup): grouped by
          (failure_type, normalized_error_identity). The test_file is intentionally
          excluded so the same leak detected after different test files produces
          one issue, not N duplicates.
    """
    grouped: dict[tuple, UniqueFailure] = {}

    if not isinstance(all_failures, dict):
        logger.warning(
            "Unexpected top-level format: expected dict, got %s",
            type(all_failures).__name__,
        )
        return []

    for job_name, suites in all_failures.items():
        if not isinstance(suites, dict):
            logger.warning(
                "Unexpected format for job %r: expected dict, got %s",
                job_name, type(suites).__name__,
            )
            continue

        for suite_name, entries in suites.items():
            if not isinstance(entries, list):
                logger.warning(
                    "Unexpected format for %s/%s: expected list, got %s",
                    job_name, suite_name, type(entries).__name__,
                )
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                test_name = entry.get("test_name", "")
                test_file = entry.get("test_file", "")
                error = entry.get("error", "")
                raw_type = entry.get("type", "assertion")

                try:
                    failure_type = FailureType(raw_type)
                except ValueError:
                    # The producer emits a type this enum doesn't know yet.
                    # Exception is the catch-all for non-assertion errors;
                    # warn so producer/consumer drift is visible in run logs.
                    logger.warning(
                        "Unknown failure type %r, classifying as exception", raw_type
                    )
                    failure_type = FailureType.EXCEPTION

                # Volatile test names (bare PIDs, "hang") are transient
                # runner state, not real test identity. Demote them so the
                # grouping key is stable across runs.
                if test_name and _VOLATILE_TEST_NAME_RE.fullmatch(test_name):
                    logger.info(
                        "Demoted volatile test name %r to nameless "
                        "(type=%s, file=%s, job=%s)",
                        test_name, failure_type.value, test_file, job_name,
                    )
                    test_name = ""

                # Determine grouping key
                if test_name:
                    key: tuple = (failure_type, test_name, test_file)
                elif failure_type == FailureType.TIMEOUT and test_file:
                    # Nameless timeouts (volatile PID/hang demoted above, or
                    # captured without a test body running) group by file:
                    # the error text is generic ("Test timed out") across all
                    # timeouts, so without the file every timeout in the run
                    # would collapse into one issue.
                    key = (failure_type, test_file)
                elif error:
                    identity = normalize_error_identity(error)
                    key = (failure_type, identity)
                else:
                    logger.debug(
                        "Skipping entry with no test_name and no error: %s", entry
                    )
                    continue

                if key not in grouped:
                    grouped[key] = UniqueFailure(
                        test_name=test_name,
                        test_file=test_file,
                        failure_type=failure_type,
                        error=error,
                    )

                failure = grouped[key]
                if not any(j.job == job_name for j in failure.jobs):
                    failure.jobs.append(
                        JobReference(
                            job=job_name,
                            suite=suite_name,
                            url=job_urls.get(job_name, ""),
                        )
                    )
                    logger.debug("%s in %s/%s", failure.display_name, job_name, suite_name)

    unique_failures = list(grouped.values())
    if unique_failures:
        type_counts = {}
        for f in unique_failures:
            type_counts[f.failure_type.value] = type_counts.get(f.failure_type.value, 0) + 1
        logger.info("Total unique failures: %d (by type: %s)", len(unique_failures), type_counts)
    else:
        logger.info("Total unique failures: 0")
    return unique_failures
