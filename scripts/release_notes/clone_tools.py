"""Load valkey's ``utils/releasetools`` modules from a clone at runtime.

The release-notes format and the release-cut primitives (promotion, version
bump, contributor list) are **authoritative in the valkey repo**. Rather than
duplicate them, the agent imports them from the clone by file path so a change
upstream flows through automatically and the agent can never disagree with the
format the release tooling parses.

Some of these modules import their siblings (e.g. ``bump_version`` does
``from release_notes import parse_version`` via a ``try`` shim), so the
releasetools directory is placed on ``sys.path`` for the duration of the load
and the module is registered under its bare name, letting those sibling imports
resolve. Both are undone afterwards so the agent's own namespace is untouched.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

_RELEASETOOLS_REL = os.path.join("utils", "releasetools")


def load_releasetools_module(valkey_clone_dir: str, module_name: str) -> Any:
    """Import ``utils/releasetools/<module_name>.py`` from *valkey_clone_dir*.

    ``module_name`` is the bare stem, e.g. ``"release_notes"`` or
    ``"bump_version"``. The releasetools dir is temporarily on ``sys.path`` and
    the module is temporarily registered under its bare name so a sibling import
    inside it (``from release_notes import ...``) resolves; both are reverted
    before returning. Raises :class:`FileNotFoundError` if the clone predates the
    release tooling, :class:`ImportError` if the spec cannot be built.
    """
    tools_dir = os.path.join(valkey_clone_dir, _RELEASETOOLS_REL)
    path = os.path.join(tools_dir, f"{module_name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"valkey releasetools module not found at {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)

    added_path = tools_dir not in sys.path
    if added_path:
        sys.path.insert(0, tools_dir)
    prior = sys.modules.get(module_name)
    # Register before exec so a sibling import of this very module resolves.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if prior is not None:
            sys.modules[module_name] = prior
        else:
            sys.modules.pop(module_name, None)
        if added_path:
            try:
                sys.path.remove(tools_dir)
            except ValueError:
                pass
    return module
