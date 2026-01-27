import importlib
import os
import re

import pytest


SMILES_LIST = ["N=Cc1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O", ] # "Clc1ccc(cc1)CN(c2nnnn2)Cc3ccc(Cl)cc3"]


def _normalize_itp(text: str) -> str:
    """Normalize .itp text so new-vs-legacy comparisons are stable.

    We drop comment-only lines and collapse whitespace.
    """

    out_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Drop comments/header chatter (typically starts with ';')
        if s.startswith(";"):
            continue
        # Normalize whitespace inside non-comment lines
        s = re.sub(r"\s+", " ", s)
        out_lines.append(s)
    return "\n".join(out_lines) + "\n"


def _generate_itp(smiles: str, molname: str, mode: str) -> str:
    """Generate ITP string using requested optimization backend.

    mode: "legacy" or "new"
    """

    if mode == "legacy":
        os.environ["AUTO_MARTINI_OPTIMIZATION"] = "legacy"
    else:
        os.environ.pop("AUTO_MARTINI_OPTIMIZATION", None)

    import auto_martiniM3

    importlib.reload(auto_martiniM3)

    # Force solver.py to pick up the aliased optimization module.
    import auto_martiniM3.solver

    importlib.reload(auto_martiniM3.solver)

    mol, _ = auto_martiniM3.topology.gen_molecule_smi(smiles)
    cg = auto_martiniM3.solver.Cg_molecule(
        mol,
        smiles,
        molname,
        topfname=None,  # don't write to disk
        forcepred=True,
        min_beads=None,
        max_beads=None,
    )
    assert cg.topout is not None
    return cg.topout
    

def test_itp_matches_legacy():
    """Compatibility test: new optimization should match legacy ITP for a known SMILES."""
    for smiles in SMILES_LIST:
        itp_new = _normalize_itp(_generate_itp(smiles, "TEST", mode="new"))
        itp_legacy = _normalize_itp(
            _generate_itp(smiles, "TEST", mode="legacy")
        )   
        assert itp_new == itp_legacy
