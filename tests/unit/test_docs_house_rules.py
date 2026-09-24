# SPDX-License-Identifier: MIT
"""What Sphinx does not check about this documentation.

`sphinx-build -W` validates {doc} cross-references and {image} paths. It says
nothing about video assets embedded through raw HTML, about whether a command
block would survive being pasted into a terminal, or about the house rules this
documentation sets itself. Every rule here was written after the corresponding
defect had already shipped at least once.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
STATIC = DOCS / "_static"


def _pages() -> list[Path]:
    """Every documentation page, plus the README, which follows the same rules."""
    pages = [p for p in DOCS.rglob("*.md") if "_build" not in p.parts
             and "_static" not in p.parts]
    pages += [p for p in DOCS.rglob("*.rst") if "_build" not in p.parts]
    return sorted(pages) + [REPO / "README.md"]


def _bash_blocks(page: Path) -> list[tuple[int, str]]:
    """(line number of the opening fence, block body) for every ```bash block."""
    text = page.read_text()
    return [(text[:m.start()].count("\n") + 1, m.group(1))
            for m in re.finditer(r"```bash\n(.*?)```", text, re.S)]


def _rel(page: Path) -> str:
    return str(page.relative_to(REPO))


# --- raw HTML video assets: Sphinx never resolves these -----------------------

def _video_refs() -> list[tuple[Path, int, str]]:
    refs = []
    for page in _pages():
        text = page.read_text()
        for m in re.finditer(r'(?:poster|src)="([^"]+\.(?:jpg|png|mp4|gif))"', text):
            refs.append((page, text[:m.start()].count("\n") + 1, m.group(1)))
    return refs


@pytest.mark.parametrize("page,line,ref", _video_refs(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_raw_html_media_resolves(page: Path, line: int, ref: str) -> None:
    """A poster or source that 404s is invisible until someone opens the page."""
    target = (page.parent / ref).resolve()
    assert target.is_file(), f"{_rel(page)}:{line} points at a missing file: {ref}"


def test_no_orphan_media() -> None:
    """Media nothing embeds is dead weight in a repository people clone."""
    referenced = {(p.parent / r).resolve() for p, _, r in _video_refs()}
    for page in _pages():
        text = page.read_text()
        # MyST fence directive, reStructuredText directive, and markdown image
        for pat in (r"```\{image\}\s*(\S+)", r"^\.\.\s+(?:image|figure)::\s*(\S+)",
                    r"!\[[^\]]*\]\((\S+?\.(?:png|jpg|gif))\)"):
            for m in re.finditer(pat, text, re.M):
                referenced.add((page.parent / m.group(1)).resolve())
    # .drawio files are editable sources, documented as such in the diagrams README.
    on_disk = {p.resolve() for p in STATIC.rglob("*")
               if p.suffix in {".png", ".jpg", ".mp4", ".gif"} and p.is_file()}
    orphans = sorted(p.relative_to(REPO) for p in on_disk - referenced)
    assert not orphans, f"under _static but embedded nowhere: {orphans}"


# --- command blocks: would they survive a paste? ------------------------------

# Directory trees are tagged bash on purpose: the highlighter colours the "#"
# annotations, which is why they read well. They are not commands.
_TREE = re.compile(r"^[│├└]", re.M)

# Shell variables a block may use without setting: the reader's environment.
_AMBIENT = {"PATH", "HOME", "PWD", "TMPDIR", "PYTHONPATH", "PYTHONNOUSERSITE",
            "OMNI_KIT_ACCEPT_EULA", "SHELL", "USER"}


def _command_blocks() -> list[tuple[Path, int, str]]:
    return [(p, line, body) for p in _pages() for line, body in _bash_blocks(p)
            if not _TREE.search(body)]


@pytest.mark.parametrize("page,line,body", _command_blocks(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_command_block_is_valid_shell(page: Path, line: int, body: str) -> None:
    result = subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True)
    assert result.returncode == 0, (
        f"{_rel(page)}:{line} is not valid shell: {result.stderr.strip()}")


@pytest.mark.parametrize("page,line,body", _command_blocks(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_command_block_has_no_shell_breaking_placeholder(
        page: Path, line: int, body: str) -> None:
    """`<REPO_ID>` is input redirection, so the paste fails before the command runs.

    docs/getting-started/cli_overview.md promises this cannot happen.
    """
    bad = re.findall(r"<[A-Z][A-Z_]{2,}>", body)
    assert not bad, f"{_rel(page)}:{line} would break the shell on: {bad}"


def test_command_blocks_do_not_inherit_variables_silently() -> None:
    """A block using a variable it never sets targets the wrong path when pasted alone.

    The README is exempt: it states in prose that its numbered steps are one
    sequence in one terminal, which is a different contract from the docs pages.
    """
    offenders = []
    for page, line, body in _command_blocks():
        if page.name == "README.md":
            continue
        used = set(re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", body))
        assigned = set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=", body, re.M))
        loops = set(re.findall(r"for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in", body))
        missing = used - assigned - loops - _AMBIENT
        if missing:
            offenders.append(f"{_rel(page)}:{line} uses {sorted(missing)}")
    assert not offenders, "\n".join(offenders)


# --- house rules --------------------------------------------------------------

def test_every_opening_fence_declares_a_language() -> None:
    offenders = []
    for page in _pages():
        inside = False
        for n, raw in enumerate(page.read_text().splitlines(), 1):
            if raw.lstrip().startswith("```"):
                if not inside and not raw[3:].strip():
                    offenders.append(f"{_rel(page)}:{n}")
                inside = not inside
    assert not offenders, f"bare opening fence: {offenders}"


def test_no_prose_semicolons() -> None:
    """A semicolon splicing two clauses reads as machine-written, not as English."""
    offenders = []
    for page in _pages():
        inside = False
        for n, raw in enumerate(page.read_text().splitlines(), 1):
            if raw.lstrip().startswith("```"):
                inside = not inside
                continue
            if inside or raw.lstrip().startswith((">", "|")):
                continue
            if re.search(r"[a-z]; +[a-zA-Z]", raw):
                offenders.append(f"{_rel(page)}:{n}: {raw.strip()[:70]}")
    assert not offenders, "\n".join(offenders)


# Spellings this documentation does not use. British where the two differ, and
# one canonical name per concept, so the same thing is not introduced twice
# under two names.
_BANNED = {
    "randomization": "randomisation",
    "randomized": "randomised",
    "generalization": "generalisation",
    "modeled": "modelled",
    "third-person camera": "external camera",
    "campaign orchestrator": "campaign runner",
}


def test_terminology_is_consistent() -> None:
    offenders = []
    for page in _pages():
        inside = False
        for n, raw in enumerate(page.read_text().splitlines(), 1):
            if raw.lstrip().startswith("```"):
                inside = not inside
                continue
            if inside:
                continue          # code and identifiers keep their own spelling
            low = raw.lower()
            for bad, good in _BANNED.items():
                if bad in low:
                    offenders.append(f"{_rel(page)}:{n}: {bad!r} -> {good!r}")
    assert not offenders, "\n".join(offenders)
