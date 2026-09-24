# SPDX-License-Identifier: MIT
"""The API page must account for every module, one way or the other.

A module that is neither listed nor deliberately excluded is simply invisible
to a reader of the documentation, and nothing else notices: Sphinx does not
warn about a module you never mentioned. That is how eleven of them went
undocumented, two of them added the same week this test was written.

The rule here is not "document everything". It is "decide about everything":
a module is either on the page or in EXCLUDED below with a reason.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
API_PAGE = REPO / "docs" / "api" / "index.rst"
PACKAGE = REPO / "src" / "so101_mvbench"

# Modules that are deliberately NOT on the API page. Keep the reason with the
# entry, and keep it in sync with the note at the top of docs/api/index.rst.
EXCLUDED = {
    # Command modules: they parse argv and (the first five) boot Isaac Sim at
    # import, so they are documented in getting-started/cli_overview.md.
    "so101_mvbench.recording.teleop_recorder": "command module",
    "so101_mvbench.recording.multicam_replay": "command module",
    "so101_mvbench.evaluation.async_eval": "command module",
    "so101_mvbench.audit.determinism": "command module",
    "so101_mvbench.audit.cube_replay": "command module",
    "so101_mvbench.tools.drift_configs": "command module",
    # Isaac-bound scene configuration, not an importable API.
    "so101_mvbench.tasks.lift_cube_env_cfg": "Isaac-bound env config",
    # One articulation-config constant and no module docstring, so automodule
    # would render an empty section.
    "so101_mvbench.assets.so101_white": "declarative config, no module docstring",
}


def _package_modules() -> set[str]:
    """Every importable module of the package, vendored code excluded."""
    found = set()
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        if path.name == "__init__.py" or "vendor" in parts:
            continue
        found.add(".".join(parts))
    return found


def _documented_modules() -> set[str]:
    return set(re.findall(r"^\.\.\s+automodule::\s+(\S+)", API_PAGE.read_text(), re.M))


def test_every_module_is_either_documented_or_excluded() -> None:
    unaccounted = _package_modules() - _documented_modules() - set(EXCLUDED)
    assert not unaccounted, (
        "these modules are on neither list. Add an automodule entry to "
        f"docs/api/index.rst, or an EXCLUDED entry here with a reason: "
        f"{sorted(unaccounted)}"
    )


def test_the_exclusion_list_has_no_stale_entries() -> None:
    """A renamed or deleted module must not keep a silent exemption."""
    gone = set(EXCLUDED) - _package_modules()
    assert not gone, f"EXCLUDED names modules that no longer exist: {sorted(gone)}"


def test_nothing_is_both_documented_and_excluded() -> None:
    both = _documented_modules() & set(EXCLUDED)
    assert not both, f"listed on the API page despite being excluded: {sorted(both)}"


def test_the_api_page_note_names_the_same_exclusions() -> None:
    """The prose a reader sees must match the list this test enforces."""
    note = API_PAGE.read_text()
    missing = [m for m in EXCLUDED if m.split("so101_mvbench.")[-1] not in note]
    assert not missing, (
        f"excluded here but not mentioned in the note on the page: {missing}"
    )


def test_the_version_is_not_duplicated_by_hand() -> None:
    """__init__ may fall back to a literal, but it must equal pyproject's.

    The metadata path only fires in an installed tree, so in a bare source
    checkout the fallback is what everyone sees. Two literals that can disagree
    are exactly the drift this replaced.
    """
    import so101_mvbench

    declared = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    assert so101_mvbench.__version__ == declared, (
        f"package reports {so101_mvbench.__version__}, pyproject declares {declared}"
    )


@pytest.mark.parametrize("path", ["LICENSE"])
def test_the_license_file_exists_and_names_a_holder(path: str) -> None:
    """SPDX headers in the sources are a claim; this file is the artifact."""
    text = (REPO / path).read_text()
    assert "MIT License" in text
    assert re.search(r"Copyright \(c\) \d{4} \S", text), "no copyright holder named"
