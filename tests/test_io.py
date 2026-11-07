"""Tests for ligand input handling (SDF files and SMILES fallback)."""

from __future__ import annotations

import shutil

import pytest
from rdkit import Chem

from itp_utils import OUTPUT_DIR
from smartini.io import ensure_ligand_sdf

PIT_SMILES = "Oc1cc(O)cc(/C=C/c2ccc(O)c(O)c2)c1"


def test_ensure_ligand_sdf_generates_from_smiles():
    work_dir = OUTPUT_DIR / "smiles_io"
    shutil.rmtree(work_dir, ignore_errors=True)

    sdf_path = ensure_ligand_sdf(work_dir, "PIT", smiles=PIT_SMILES)

    assert sdf_path == work_dir / "PIT.sdf"
    assert sdf_path.is_file()

    molecule = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    assert molecule is not None
    assert molecule.GetNumAtoms() > molecule.GetNumHeavyAtoms()


def test_ensure_ligand_sdf_requires_sdf_or_smiles():
    work_dir = OUTPUT_DIR / "smiles_io_missing"
    shutil.rmtree(work_dir, ignore_errors=True)

    with pytest.raises(FileNotFoundError, match="Ligand SDF not found"):
        ensure_ligand_sdf(work_dir, "MISSING")
