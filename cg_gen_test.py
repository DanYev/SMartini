import logging
from pathlib import Path
import AutoMartini as am
# import auto_martiniM3 as am
import rdkit
from rdkit import Chem

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    force=True,  # override any prior logging config set by imported libs
)
logging.getLogger("AutoMartini").setLevel(logging.INFO)  # or DEBUG


if __name__ == "__main__":
    molname = "FTA"
    outdir = Path("output") / molname
    outdir.mkdir(parents=True, exist_ok=True)

    smiles = "CCC"
    smiles = "CC1CCC(C(C1)O)C(C)C" # Menthol
    # smiles = "N=Cc1ccccc1"
    # smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    # smiles = "Clc1ccc(cc1)CN(c2nnnn2)Cc3ccc(Cl)cc3"
    # smiles = "N#C/C(=C/Nc1ccc(Nc2ccccc2)cc1)c3n[nH]nn3"
    # mol_am, _ = am.topology.gen_molecule_smi(smiles)

    sdf_file = Path("ligands") / "FTA.sdf"
    n_beads = None
    mol_am = am.topology.gen_molecule_sdf(str(sdf_file))
    smiles = str(Chem.MolToSmiles(mol_am, isomericSmiles=False))

    # Save the atomistic RDKit molecule to SDF
    sdf_path = outdir / f"{molname.lower()}_aa.sdf"
    w = Chem.SDWriter(str(sdf_path))
    w.write(mol_am)
    w.close()
    print(f"Wrote: {sdf_path}")
    
    # Use auto_martiniM3's built-in .itp writer via topfname
    itp_path = outdir / f"ligand_{molname}.itp"
    cg = am.solver.Cg_molecule(mol_am, smiles, molname, topfname=str(itp_path), forcepred=True, 
        min_beads=n_beads, max_beads=n_beads)
    # cg = am.solver.Cg_molecule(mol_am, smiles, molname, topfname=str(itp_path), forcepred=True)
    print(f"Wrote: {itp_path}")

    # Save CG structure (.gro)
    gro_path = outdir / f"{molname.lower()}.gro"
    cg.output_cg(str(gro_path))
    print(f"Wrote: {gro_path}")

    # Save AA structure (.gro)
    gro_path = outdir / f"{molname.lower()}_aa.gro"
    cg.output_aa(str(gro_path))
    print(f"Wrote: {gro_path}")

    # Make .map file
    map_path = outdir / f"{molname.lower()}.map"
    am.output.make_map_from_itp(str(itp_path), str(map_path), resname=molname)
    print(f"Wrote: {map_path}")