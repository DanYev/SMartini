import logging
from pathlib import Path
import AutoMartini as am
import rdkit
from rdkit import Chem

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    force=True,  # override any prior logging config set by imported libs
)
logging.getLogger("AutoMartini").setLevel(logging.INFO)  # or DEBUG


if __name__ == "__main__":
    molname = "ANP"
    in_dir = Path("ligands") / "anp"
    smiles = "CCC"
    # smiles = "N=Cc1ccccc1"
    # smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    # smiles = "Clc1ccc(cc1)CN(c2nnnn2)Cc3ccc(Cl)cc3"
    # smiles = "N#C/C(=C/Nc1ccc(Nc2ccccc2)cc1)c3n[nH]nn3"
    # mol_am, _ = am.topology.gen_molecule_smi(smiles)
    sdf_file = "anp.sdf"
    n_beads = 9
    # mol_aa = Chem.MolFromPDBFile(str(pdb_file), removeHs=False, sanitize=True)
    mol_am = am.topology.gen_molecule_sdf(str(sdf_file))
    smiles = str(Chem.MolToSmiles(mol_am, isomericSmiles=False))
    outdir = Path("output") / molname
    outdir.mkdir(parents=True, exist_ok=True)

    # Save the atomistic RDKit molecule to SDF
    sdf_path = outdir / f"{molname.lower()}_aa.sdf"
    w = Chem.SDWriter(str(sdf_path))
    w.write(mol_am)
    w.close()
    print(f"Wrote: {sdf_path}")
    
    # Use auto_martiniM3's built-in .itp writer via topfname
    itp_path = f"{molname.lower()}.itp"
    cg = am.solver.Cg_molecule(mol_am, smiles, molname, topfname=str(itp_path), forcepred=True, 
        min_beads=n_beads, max_beads=n_beads)
    print(f"Wrote: {itp_path}")

    # Save CG structure (.gro)
    gro_path = outdir / f"{molname.lower()}.gro"
    cg.output_cg(str(gro_path))
    print(f"Wrote: {gro_path}")

    # Save A structure (.gro)
    gro_path = outdir / f"{molname.lower()}_aa.gro"
    cg.output_aa(str(gro_path))
    print(f"Wrote: {gro_path}")