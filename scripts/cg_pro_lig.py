import logging
import shutil
import numpy as np
import MDAnalysis as mda
from pathlib import Path

from smartini import setup_logging
logger = logging.getLogger("smartini")
setup_logging(level=logging.INFO)

from reforge.mdsystem.gmxmd import GmxSystem, GmxRun, get_ntomp
from reforge.utils import clean_dir
from reforge.forge.topology import Topology


# Global settings
DT = 0.020  # Time step in picoseconds
TOTAL_TIME = 1000  # Total simulation time in nanoseconds
NSTEPS = int(TOTAL_TIME * 1e3 / DT)  # Number of MD steps for production run
SYSDIR = Path("protein_systems").resolve()
WDIR = SYSDIR / "1TQN"
LIGAND_NAME = "HEM"
LIGAND_SRC = Path("examples") / LIGAND_NAME  # Source directory for ligand .itp/.map files


def postprocess_ligand():
    pass


def setup(sysdir, sysname, ligand_src=None):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    molname = sysname
    ligand_src = ligand_src or LIGAND_SRC
    mdsys = GmxSystem(sysdir, sysname)
    input_pdb = mdsys.root / f"{sysname}.pdb"
    mdsys.prepare_files(pour_martini=True) # be careful it can overwrite later files
    mdsys.clean_pdb_mm(input_pdb, add_missing_atoms=True, add_hydrogens=False, pH=7.0) # Generates Amber ff names in PDB
    # shutil.copy(input_pdb, mdsys.inpdb)  # Copy source PDB to inpdb.pdb (bypasses clean_pdb_mm)

    # Martinizing
    mdsys.martinize_proteins_en(append=True) # SWITCH APPEND TO TRUE IF ALREADY DONE
    # shutil.copy(mdsys.inpdb, mdsys.prodir / f"{molname}.pdb")
    # mdsys.martinize_proteins_go(go_eps=12.0, go_low=0.3, go_up=1.1, ff="martini3001",
    #     p="backbone", pf="500",  text="", append=True) 
    # shutil.copy(mdsys.topdir / f"{molname}.itp", mdsys.topdir / "tmp.itp") 
    shutil.copy(mdsys.topdir / "tmp.itp", mdsys.topdir / f"{molname}.itp") 

    # LIGANDS 
    # Copy ligand .itp and .map files from source to the system's ligands directory
    lig_dir = mdsys.root / "ligands" / LIGAND_NAME
    lig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ligand_src / f"{LIGAND_NAME}.itp", lig_dir / f"{LIGAND_NAME}.itp")
    shutil.copy(ligand_src / f"{LIGAND_NAME}.map", lig_dir / f"{LIGAND_NAME}.map")
    # !!!!!!
    # LIGANDS MUST BE IN ALPHABETICAL ORDER FOR NOW. I'LL FIX THIS LATER
    mdsys.martinize_ligands(input_pdb=input_pdb, ligands=[LIGAND_NAME], merge_with=molname)
    # !!!!!!!
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    _add_protein_ligand_bonds(mdsys, molname, ligand_bead_names=["N08", "N18"])
    
    # PROTEIN+WATER SYSTEMS:
    mdsys.make_box(d="5.0", bt="dodecahedron", center="0 0 0")
    solvent = mdsys.root / "water.gro"
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17") # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.10, pname="NA", nname="CL")
    # CENTERING AND PBC CORRECTIONS
    s = mdsys.sysgro
    mdsys.gmx("trjconv", s=s, f=s, o=s, pbc="whole", center="", clinput="1\n 0\n")
    mdsys.gmx("trjconv", s=s, f=s, o=s, pbc="atom", ur="compact", clinput="0\n")

    # 1.4. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])


def _add_protein_ligand_bonds(mdsys, molname, ligand_bead_names) -> None:
    """Find closest protein beads to specified ligand beads using solute.pdb.
    
    Parameters
    ----------
    mdsys : GmxSystem
        The molecular dynamics system object
    ligand_bead_names : list, optional
        List of ligand bead names (e.g., ["D01", "MG"]).
        If None, uses a default list.
    """
    u = mda.Universe(str(mdsys.solupdb))
    protein_atoms = u.select_atoms("name BB* or name SC*") # Martini backbone and sidechain beads
    restraints = [] 
    # For each ligand bead name, find matching atoms and their closest protein partner
    for bead_name in ligand_bead_names:
        ligand_beads = u.select_atoms(f"name {bead_name}")
        for ligand_bead in ligand_beads:
            lig_pos = ligand_bead.position
            # Find closest protein atom
            distances = 0.1 * np.array([np.linalg.norm(lig_pos - p.position) for p in protein_atoms])
            closest_idx = np.argmin(distances)
            closest_protein = protein_atoms[closest_idx]
            distance = distances[closest_idx]
            # Use serial numbers from PDB (1-indexed)
            ligand_id = ligand_bead.index + 1
            protein_id = closest_protein.index + 1
            restraints.append(((ligand_id, protein_id), (1, distance, 1000), "BONDED DISTANCE RESTRAINT"))
            logger.info(f"Bond: protein atom {protein_id} ({closest_protein.name}) <-> "
                        f"ligand atom {ligand_id} ({ligand_bead.name}), distance: {distance:.2f} nm")
    # Update topology with generated restraints
    itp_file = mdsys.topdir / f"{molname}.itp"
    target_topo = Topology.from_itp(itp_file)
    target_topo.bonds.extend(restraints)
    target_topo.write_to_itp(itp_file)
    logger.info("Saved topology with %d bonded restraints to %s", len(restraints), itp_file)

     
def md_npt(sysdir, sysname, runname, nsteps=None): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.prepare_files()
    ntomp = get_ntomp()
    mdrun.empp(f=mdrun.mdpdir / "em_cg.mdp", c=mdrun.sysgro, r=mdrun.sysgro)
    mdrun.mdrun(deffnm="em", ntomp=ntomp)
    mdrun.hupp(f=mdrun.mdpdir / "hu_cg.mdp", c="em.gro", r="em.gro")
    mdrun.mdrun(deffnm="hu", ntomp=ntomp)
    mdrun.eqpp(f=mdrun.mdpdir / "eq_cg.mdp", c="hu.gro", r="hu.gro")
    mdrun.mdrun(deffnm="eq", ntomp=ntomp)
    mdrun.mdpp(f=mdrun.mdpdir / "md_cg.mdp", maxwarn="1")    
    mdrun.mdrun(deffnm="md", ntomp=ntomp, nsteps=NSTEPS, ) # bonded="gpu")
    
    
def extend(sysdir, sysname, runname, nsteps=None):    
    mdrun = GmxRun(sysdir, sysname, runname)
    ntomp = get_ntomp()
    if nsteps is None:
        t_ext = 10000 # nanoseconds
        nsteps = int(t_ext * 1e3 / DT)
    mdrun.mdrun(deffnm="md", cpi="md.cpt", ntomp=ntomp, nsteps=nsteps, ) 
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault("b", 0) # in ps
    kwargs.setdefault("dt", 200) # in ps
    kwargs.setdefault("e", 1e7) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    k = 1 # k=1 to remove solvent, k=2 for backbone analysis, k=4 to include ions
    mdrun.convert_tpr(clinput=f"{k}\n", s="md.tpr", n=mdrun.sysndx, o="topology.tpr")
    topology = "topology.tpr" # mdrun.solupdb
    mdrun.trjconv(clinput=f"{k}\n {k}\n", s="md.tpr", f="md.xtc", o="conv.xtc", n=mdrun.sysndx, pbc="atom", ur="compact", **kwargs)
    mdrun.trjconv(clinput="0\n 0\n", s=topology, f="conv.xtc", o="conv.xtc", pbc="nojump")
    mdrun.trjconv(clinput="0\n 0\n", s=topology, f="conv.xtc", o="topology.pdb", fit="rot+trans", e=0)
    mdrun.trjconv(clinput="0\n 0\n", s=topology, f="conv.xtc", o="samples.xtc", fit="rot+trans")
    clean_dir(mdrun.rundir)


if __name__ == "__main__":
    sysdir = SYSDIR
    sysname = "1TQN"
    setup(sysdir, sysname, ligand_src=LIGAND_SRC) 