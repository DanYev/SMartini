import logging
import MDAnalysis as mda
import rdkit
import AutoMartini as am

from pathlib import Path
from openff.toolkit import ForceField, Molecule, Topology 
from rdkit import Chem
from config import CFG

logger = logging.getLogger("AutoMartini")
logger.setLevel(logging.INFO)  # or DEBUG


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


def extract_ligand_from_pdb(pdb_file, selection, out_file="ligand.pdb"):
    logger.info("Extracting ligand from PDB: %s with selection: %s", pdb_file, selection)
    u = mda.Universe(str(pdb_file))
    ligand = u.select_atoms(selection)
    if len(ligand) == 0:
        raise ValueError(f"No atoms found for selection: {selection}")
    ligand.write(out_file)
    logger.info("Extracted ligand with %d atoms", len(ligand))
    return ligand


if __name__ == "__main__":
    molname = CFG.molname
    n_beads = CFG.n_beads
    wdir = CFG.wdir
    mol_dir = CFG.mol_dir
    mol_dir.mkdir(parents=True, exist_ok=True)

    # smiles = "CCC"
    # smiles = "CC1CCC(C(C1)O)C(C)C" # Menthol
    # smiles = "Cc1cc[nH]c1" # Toluene
    # smiles = "C1=NC2=NC=NC(=C2N1)N" # Adenine
    # smiles = "N=Cc1ccccc1" # Benzylimine
    # smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
    # smiles = "Clc1ccc(cc1)CN(c2nnnn2)Cc3ccc(Cl)cc3"  
    # smiles = "N#C/C(=C/Nc1ccc(Nc2ccccc2)cc1)c3n[nH]nn3" # FTA
    # smiles = "Nc1ncnc2n(cnc12)[C@@H]3O[C@H](CO[P](O)(=O)O[P](O)(=O)N[P](O)(O)=O)[C@@H](O)[C@H]3O" # ANP
    # mol, _ = am.topology.gen_molecule_smi(smiles)
    # raw_molecule = None

    ligand_sdf = wdir / f"{molname}.sdf"
    mol, raw_mol = gen_aa_molecule(molname, from_file=ligand_sdf)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    Chem.MolToPDBFile(raw_mol, mol_dir / f"{molname}_aa.pdb")
    
    # Use auto_martiniM3's built-in .itp writer via topfname
    itp_path = mol_dir / f"{molname}.itp"
    cg = am.solver.Cg_molecule(mol, smiles, molname, topfname=str(itp_path), forcepred=True, 
        min_beads=n_beads, max_beads=n_beads, raw_molecule=raw_mol)
    logging.info(f"Wrote: {itp_path}")

    # Save CG structure (.pdb)
    pdb_path = mol_dir / f"{molname}.pdb"
    cg.output_cg_pdb(str(pdb_path))
    logging.info(f"Wrote: {pdb_path}")

    # Save CG structure (.pdb)
    pdb_path = mol_dir / f"{molname}.pdb"
    cg.output_cg_pdb(str(pdb_path))
    logging.info(f"Wrote: {pdb_path}")

    # Make .map file
    map_path = mol_dir / f"{molname}.map"
    am.output.make_map_from_itp(str(itp_path), str(map_path), resname=molname)
    logging.info(f"Wrote: {map_path}")