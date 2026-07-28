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

# Report boilerplate that matches a significance keyword but carries no bug
# identity. The "error:" keyword would otherwise match the tool banner via the
# runner's wrapper prefix ("Valgrind error: ==123== Memcheck, ..."). Valgrind's
# "ERROR SUMMARY: N errors from N contexts" matches "SUMMARY:" but its count
# varies run to run and its position drifts, so it must not reach the identity
# (two runs of one leak whose summary lands inside vs outside the line window
# would otherwise fingerprint differently). The sanitizer's meaningful summary
# is "SUMMARY: AddressSanitizer: ...", which has no "ERROR" prefix and survives.
_BOILERPLATE_SUBSTRINGS = (
    "Memcheck, a memory error detector",
    "HEAP SUMMARY:",
    "LEAK SUMMARY:",
    "ERROR SUMMARY:",
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

# A macOS /usr/bin/leaks root-leak line: "1 (48 bytes) ROOT LEAK: <malloc in
# sdsnewlen 0x600001d1c100> [48]". These are the only lines in a leaks report
# that name the allocation site, so they are the identity anchor that keeps
# two different leaks in the same test file as two issues. Addresses and
# sizes around the symbol are scrubbed before this is applied.
_ROOT_LEAK_RE = re.compile(r"ROOT LEAK:\s*(?P<site>[^\[]+)")

# The startup blob's config dump: "Can't start <exe>\nCONFIGURATION:\n<full
# config file>\nERROR:\n<reason>". The config is dozens of lines shared by
# every startup failure; left in place it fills the significant-line window
# before the ERROR: reason is reached, collapsing all startup causes into one
# identity. The reason after ERROR: is the identity; the dump is not.
_STARTUP_CONFIG_SECTION_RE = re.compile(
    r"\nCONFIGURATION:\n.*?\nERROR:\n", re.DOTALL
)

# A stack frame after volatile stripping. Valgrind: "at : malloc (...)" or
# "by : sdsdup (sds.c:190)". Sanitizer: "#1  in ztrymalloc_usable_internal
# /.../zmalloc.c:172" (the "#N 0xADDR in func" shape with the address
# scrubbed). The function names are the stable identity anchor; addresses,
# sizes, and loss records around them are not.
_STACK_FRAME_RE = re.compile(
    r"^(?:at|by)\s*:\s*(?P<func>[^\s(]+)"
    r"|^#\d+\s+in\s+(?P<san_func>\S+)"
)


def _extract_root_leak_anchor(lines: list[str]) -> str:
    """Allocation-site chain of a macOS leaks report, or "".

    A leaks blob has no stack frames, so without this anchor every leak in
    one test file normalizes to the same boilerplate and two distinct leaks
    collapse into one issue. Unsymbolicated roots (bare scrubbed addresses)
    yield an empty site and are skipped; sorted so report order does not
    change the identity.
    """
    sites: set[str] = set()
    for line in lines:
        match = _ROOT_LEAK_RE.search(line)
        if not match:
            continue
        site = match.group("site").strip()
        if site:
            sites.add(site)
    if not sites:
        return ""
    return "roots: " + " > ".join(sorted(sites))


def _extract_stack_anchor(lines: list[str]) -> str:
    """Function-name chain of the first stack block in a valgrind or
    sanitizer report.

    Distinguishes two different leaks whose report lines are otherwise
    identical after count scrubbing (e.g. same "definitely lost" shape but
    allocated from debugCommand vs clusterCommand). Returns "" when the
    error has no stack frames (assertions, startup failures).
    """
    frames: list[str] = []
    in_stack = False
    for line in lines:
        match = _STACK_FRAME_RE.match(line)
        if match:
            in_stack = True
            frames.append(match.group("func") or match.group("san_func"))
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
    text = _STARTUP_CONFIG_SECTION_RE.sub("\nERROR:\n", error)
    for pattern in _VOLATILE_PATTERNS:
        text = pattern.sub("", text)
    for pattern in _VOLATILE_COUNT_PATTERNS:
        text = pattern.sub("", text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Extract up to 3 significant lines for the identity, skipping tool
    # boilerplate that is identical in every run of every bug.
    significant: list[str] = []
    for index, line in enumerate(lines[:30]):
        if any(bp in line for bp in _BOILERPLATE_SUBSTRINGS):
            continue
        if any(kw in line for kw in _SIGNIFICANT_KEYWORDS):
            if line == "ERROR:" and index + 1 < len(lines):
                # A bare "ERROR:" header (startup blob's stderr separator)
                # matches the keyword but names no bug; the fatal reason
                # follows it, possibly behind a "*** FATAL ... ***" banner
                # that is identical across causes. Take the first non-banner
                # line after the header, not the blob's last line: a trailing
                # server-log tail would bind the identity to volatile text
                # and mint a fresh issue per run.
                reason = next(
                    (
                        following
                        for following in lines[index + 1 :]
                        if not following.startswith("***")
                    ),
                    "",
                )
                if reason:
                    line = f"ERROR: {reason}"
            significant.append(line)
            if len(significant) >= 3:
                break

    # The stack frames (or a macOS leaks report's root-leak sites) pin the
    # identity to the code path, so two leaks with identical report lines but
    # different allocation sites stay distinct.
    stack_anchor = _extract_stack_anchor(lines) or _extract_root_leak_anchor(lines)
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


def _coerce_str(value: Any) -> str:
    """Return *value* if it is a string, else "".

    Non-string artifact values (an int PID, null) carry no usable test
    identity or error text, so they are treated as absent rather than
    stringified into a bogus identity.
    """
    return value if isinstance(value, str) else ""


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

                # Field values are producer-controlled; a non-string (int PID,
                # null) must degrade to one bad entry, not a TypeError that
                # aborts the whole batch in the regex calls below.
                test_name = _coerce_str(entry.get("test_name", ""))
                test_file = _coerce_str(entry.get("test_file", ""))
                error = _coerce_str(entry.get("error", ""))
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
        type_counts: dict[str, int] = {}
        for f in unique_failures:
            type_counts[f.failure_type.value] = type_counts.get(f.failure_type.value, 0) + 1
        logger.info("Total unique failures: %d (by type: %s)", len(unique_failures), type_counts)
    else:
        logger.info("Total unique failures: 0")
    return unique_failures
