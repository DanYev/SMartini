"""Helpers for comparing GROMACS ITP files section by section."""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
REFS_DIR = TESTS_DIR / "refs"
OUTPUT_DIR = TESTS_DIR / "output"
PIPELINE_SCRIPT = REPO_ROOT / "1_gen_cg_topo.py"

_SECTION_HEADER_RE = re.compile(r"^\s*\[\s*(?P<name>[^\]]+?)\s*\]\s*$")


def discover_molecules(refs_dir: Path = REFS_DIR) -> list[str]:
    """Return molecule names that have a complete reference input set."""
    molecules = []
    for mol_dir in sorted(refs_dir.iterdir()):
        if not mol_dir.is_dir():
            continue
        molname = mol_dir.name
        expected = (
            mol_dir / f"{molname}_initial.itp",
            mol_dir / f"{molname}.sdf",
            mol_dir / "config.yml",
        )
        if all(path.is_file() for path in expected):
            molecules.append(molname)
    return molecules


def split_itp_sections(text: str) -> dict[str, str]:
    """Split ITP text into its header plus each ``[ section ]`` block.

    Every value keeps the raw lines of that block (including the section
    directive and blank lines), so an exact string comparison still catches
    formatting changes.  The header contains everything before the first
    section directive.

    Duplicate section names are preserved by suffixing ``#2``, ``#3``, ...
    so repeated sections cannot silently overwrite one another.
    """
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    seen: dict[str, int] = {}

    for line in text.splitlines(keepends=True):
        match = _SECTION_HEADER_RE.match(line)
        if match is not None:
            name = match.group("name").strip().lower()
            seen[name] = seen.get(name, 0) + 1
            current = name if seen[name] == 1 else f"{name}#{seen[name]}"
            sections[current] = []
        sections[current].append(line)

    return {name: "".join(lines) for name, lines in sections.items()}
