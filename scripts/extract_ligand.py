#!/usr/bin/env python3
"""
Extract a ligand by residue name from a PDB file and convert it to SDF format.
"""

import logging
import sys

from pathlib import Path

# Ensure the project root is on sys.path so that 'config' can be found
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import MDAnalysis as mda
import smartini

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

logger = logging.getLogger("smartini")
smartini.setup_logging(level=logging.INFO)


def extract_ligand(pdb_path: Path, ligand_name: str, output_path: Path) -> None:
    """
    Extract a ligand by residue name from a PDB file and save as SDF.

    Parameters
    ----------
    pdb_path : Path
        Path to the input PDB file.
    ligand_name : str
        Residue name of the ligand to extract (e.g. "HEM").
    output_path : Path
        Path where the SDF file will be saved.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PDB: %s", pdb_path)
    universe = mda.Universe(str(pdb_path))

    # Select ligand atoms by residue name
    selection = universe.select_atoms(f"resname {ligand_name}")
    if len(selection) == 0:
        raise ValueError(
            f"No atoms found with residue name '{ligand_name}' in {pdb_path}"
        )
    logger.info(
        "Found %d atoms for ligand '%s' (residue %s)",
        len(selection),
        ligand_name,
        selection.residues[0].resid,
    )

    # Write ligand atoms to a temporary PDB file
    tmp_pdb = output_path.with_suffix(".pdb")
    selection.write(str(tmp_pdb))
    logger.info("Wrote ligand PDB: %s", tmp_pdb)

    # Read the PDB with RDKit and write to SDF
    mol = Chem.MolFromPDBFile(str(tmp_pdb), removeHs=False, sanitize=False)
    if mol is None:
        raise RuntimeError(f"RDKit could not parse the ligand PDB: {tmp_pdb}")

    # Attempt to determine bonds from the 3D geometry (for metal complexes like heme)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        logger.info("Bonds determined by rdDetermineBonds")
    except Exception as e:
        logger.warning("rdDetermineBonds failed (%s); bonds may be incomplete", e)

    # Sanitize the molecule
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        logger.warning("SanitizeMol warning (%s); proceeding anyway", e)

    # Write SDF
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    logger.info("Saved ligand SDF: %s", output_path)

    # Clean up temporary PDB
    tmp_pdb.unlink(missing_ok=True)
    logger.info("Removed temporary PDB: %s", tmp_pdb)


if __name__ == "__main__":
    pdb_path = Path("examples/1TQN/1TQN.pdb")
    ligand_name = "HEM"
    output_path = Path("examples/1TQN") / ligand_name / f"{ligand_name}.sdf"
    extract_ligand(pdb_path, ligand_name, output_path)
