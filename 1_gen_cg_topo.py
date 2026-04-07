import logging
import MDAnalysis as mda
import rdkit
import smartini

from pathlib import Path
from openff.toolkit import Molecule
from rdkit import Chem
from config import CFG

logger = logging.getLogger("smartini")
smartini.setup_logging(level=logging.INFO)


def gen_aa_molecule(molname, from_file=None, from_smiles=None):
    if from_file is not None:
        logger.info("Reading molecule from file: %s", from_file)
        mol = Molecule.from_file(str(from_file))
    elif from_smiles is not None:
        logger.info("Generating molecule from SMILES: %s", from_smiles)
        mol = Molecule.from_smiles(from_smiles)
        mol.generate_conformers(n_conformers=1)
    else:
        raise ValueError("Must provide either from_file or from_smiles")
    molecule = mol.to_rdkit()
    Chem.SanitizeMol(molecule)
    Chem.AddHs(molecule)
    Chem.AllChem.EmbedMolecule(molecule, randomSeed=1, useRandomCoords=True)  # Set Seed for random coordinate generation = 1.
    Chem.AllChem.UFFOptimizeMolecule(molecule)
    logger.info("Generated molecule with %d atoms", mol.n_atoms)
    return molecule, mol.to_rdkit()


if __name__ == "__main__":
    molname = CFG.molname
    n_beads = CFG.n_beads
    wdir = CFG.wdir
    mol_dir = CFG.mol_dir
    mol_dir.mkdir(parents=True, exist_ok=True)

    ligand_sdf = wdir / f"{molname}.sdf"
    mol, raw_mol = gen_aa_molecule(molname, from_file=ligand_sdf)
    # mol, raw_mol = gen_aa_molecule(molname, from_smiles=smiles)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    Chem.MolToPDBFile(raw_mol, mol_dir / f"{molname}_aa.pdb")
    
    # Generate the CG molecule
    cg_mol = smartini.solver.Cg_molecule(mol, smiles, molname, 
        specify_beads=CFG.specify_beads,
        use_vsites=CFG.use_vsites,
        symmetrize_rings=CFG.symmetrize_rings,
        min_beads=n_beads, 
        max_beads=n_beads, 
        raw_molecule=raw_mol)
    cg_mol.process()
    
    # Write .itp file
    itp_path = mol_dir / f"{molname}_initial.itp"
    cg_mol.to_itp(itp_path)  
    logging.info(f"Wrote: {itp_path}")

    # Save CG structure (.pdb)
    pdb_path = mol_dir / f"{molname}.pdb"
    cg_mol.to_pdb(str(pdb_path))
    logging.info(f"Wrote: {pdb_path}")

    # Make .map file
    map_path = mol_dir / f"{molname}.map"
    cg_mol.output_map(str(map_path))
    logging.info(f"Wrote: {map_path}")