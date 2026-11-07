"""Regression tests for the initial CG topology generation step.

Each molecule in ``tests/refs`` is run through ``1_gen_cg_topo.py`` and the
generated ITP is compared with the reference **one section at a time** (header,
moleculetype, atoms, bonds, constraints, angles, dihedrals, exclusions,
position_restraints, ...).  This makes a failure point directly at the part of
the topology that changed instead of reporting one opaque whole-file diff.
"""

from __future__ import annotations

import difflib

import pytest

from itp_utils import OUTPUT_DIR, REFS_DIR, discover_molecules, split_itp_sections

MOLECULES = discover_molecules()


def _section_params():
    """Build ``(molname, section)`` parametrizations from the references."""
    params = []
    for molname in MOLECULES:
        reference = REFS_DIR / molname / f"{molname}_initial.itp"
        for section in split_itp_sections(reference.read_text(encoding="utf-8")):
            params.append(
                pytest.param(molname, section, id=f"{molname}-{section}")
            )
    return params


@pytest.mark.parametrize("molname, section", _section_params())
def test_initial_itp_section_matches_reference(run_pipeline, molname, section):
    """Compare one ITP section (or the header) with its reference."""
    reference_path = REFS_DIR / molname / f"{molname}_initial.itp"
    generated_path = run_pipeline(molname) / f"{molname}_initial.itp"

    expected_sections = split_itp_sections(
        reference_path.read_text(encoding="utf-8")
    )
    actual_sections = split_itp_sections(
        generated_path.read_text(encoding="utf-8")
    )

    assert section in actual_sections, (
        f"{molname}: generated ITP is missing a block for '{section}'. "
        f"Found: {list(actual_sections)}"
    )

    expected = expected_sections[section]
    actual = actual_sections[section]

    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"refs/{molname}/{section}",
            tofile=f"output/{molname}/{section}",
        )
    )
    assert actual == expected, (
        f"{molname}: ITP block '{section}' differs from the reference.\n{diff}"
    )


@pytest.mark.parametrize("molname", MOLECULES)
def test_initial_itp_has_same_sections_in_order(run_pipeline, molname):
    """Ensure no section was silently added, removed, or reordered."""
    reference_path = REFS_DIR / molname / f"{molname}_initial.itp"
    generated_path = run_pipeline(molname) / f"{molname}_initial.itp"

    expected = list(
        split_itp_sections(reference_path.read_text(encoding="utf-8"))
    )
    actual = list(
        split_itp_sections(generated_path.read_text(encoding="utf-8"))
    )

    assert actual == expected, (
        f"{molname}: ITP section list differs.\n"
        f"  reference: {expected}\n"
        f"  generated: {actual}"
    )


@pytest.mark.parametrize("molname", MOLECULES)
def test_pipeline_writes_expected_files(run_pipeline, molname):
    """All artifacts produced by 1_gen_cg_topo.py must exist under tests/output."""
    out_dir = run_pipeline(molname)

    expected_files = [
        out_dir / f"{molname}_initial.itp",
        out_dir / f"{molname}.pdb",
        out_dir / f"{molname}.map",
        out_dir / f"{molname}_aa.pdb",
    ]
    missing = [str(path) for path in expected_files if not path.is_file()]
    assert not missing, f"{molname}: missing generated files: {missing}"
    assert out_dir == OUTPUT_DIR / molname
