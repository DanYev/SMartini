"""Ligand input/output helpers.

The pipeline accepts either a pre-built ``<molname>.sdf`` file or a SMILES
string in the molecule config (``smiles: ...``).  When no SDF exists, the
SMILES is converted into a 3D SDF with explicit hydrogens so every downstream
step can keep reading the same file format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def molecule_from_smiles(smiles: str, random_seed: int = 1) -> Chem.Mol:
    """Build an RDKit molecule with explicit hydrogens and 3D coordinates."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    molecule = Chem.AddHs(molecule)
    result = AllChem.EmbedMolecule(
        molecule, randomSeed=random_seed, useRandomCoords=True
    )
    if result != 0:
        raise RuntimeError(
            f"Could not generate 3D coordinates for SMILES: {smiles}"
        )
    AllChem.UFFOptimizeMolecule(molecule)
    return molecule


def generate_sdf_from_smiles(
    smiles: str,
    sdf_path: str | Path,
    molname: str = "MOL",
    random_seed: int = 1,
) -> Path:
    """Write a 3D SDF with explicit hydrogens generated from ``smiles``."""
    sdf_path = Path(sdf_path)
    sdf_path.parent.mkdir(parents=True, exist_ok=True)

    molecule = molecule_from_smiles(smiles, random_seed=random_seed)
    molecule.SetProp("_Name", molname)

    writer = Chem.SDWriter(str(sdf_path))
    try:
        writer.write(molecule)
    finally:
        writer.close()

    logger.info("Generated %s from SMILES", sdf_path)
    return sdf_path


def ensure_ligand_sdf(
    wdir: str | Path,
    molname: str,
    smiles: Optional[str] = None,
    random_seed: int = 1,
) -> Path:
    """Return ``<wdir>/<molname>.sdf``, creating it from SMILES if needed.

    Parameters
    ----------
    wdir:
        Molecule working directory.
    molname:
        Molecule name, also used as the SDF stem.
    smiles:
        SMILES string used when the SDF does not exist yet.
    random_seed:
        Seed for 3D embedding, matching the seed used by the main pipeline.

    Raises
    ------
    FileNotFoundError
        If the SDF is missing and no SMILES was provided.
    """
    sdf_path = Path(wdir) / f"{molname}.sdf"
    if sdf_path.is_file():
        return sdf_path

    if not smiles:
        raise FileNotFoundError(
            f"Ligand SDF not found: {sdf_path}. Provide {sdf_path.name} or "
            "set 'smiles' in the molecule config."
        )

    return generate_sdf_from_smiles(
        smiles, sdf_path, molname=molname, random_seed=random_seed
    )
