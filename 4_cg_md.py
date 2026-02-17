
from pathlib import Path
import shutil
import MDAnalysis as mda
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun, get_ntomp
from reforge.utils import clean_dir, get_logger
from reforge.forge.topology import Topology

from ligpar_config import CFG

logger = get_logger()

# Global settings
INPDB = 'KDA.pdb'
DT = 0.020  # Time step in picoseconds
total_time = 1000  # Total simulation time in nanoseconds
NSTEPS = int(total_time * 1e3 / DT)  # Number of MD steps for production run

ligand = CFG.molname
sysdir = str(CFG.wdir())
sysname = CFG.cg_sysname
runname = CFG.cg_runname


def setup_martini(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)
    mdsys.prepare_files(pour_martini=True)
    shutil.copy("md_cg.mdp", mdsys.mdpdir / "md_cg.mdp")
   
    # LIGANDS 
    pdb_file = Path(sysdir) / "mapping" / f"{ligand}.pdb"
    itp_file = Path(sysdir) / "mapping" / f"{ligand}_updated.itp"
    shutil.copy(itp_file, mdsys.topdir / f"ligand_{ligand}.itp") # copy .itp to mdsys.itpdir so it can be included in the system topology
    shutil.copy(pdb_file, mdsys.solupdb) # copy .pdb to mdsys.root so it can be included in the system structure
    shutil.copy(pdb_file, mdsys.inpdb) # copy .pdb to mdsys.root so it can be included in the system structure

    mdsys.molecules[f"ligand_{ligand}"] = 1
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    
    # 1.3. Coarse graining is *hopefully* done. Need to add solvent and ions
        # 1.3. Coarse graining is *hopefully* done. Need to add solvent and ions
    mdsys.make_box(d="1.2", bt="dodecahedron")
    solvent = mdsys.root / "water.gro"
    mdsys.solvate(cp=mdsys.solupdb, cs=solvent, radius="0.17") # all kwargs go to gmx solvate command
    mdsys.add_bulk_ions(conc=0.0, pname="NA", nname="CL")

    # 1.4. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])

    
def md_npt(sysdir, sysname, runname, nsteps=None): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.rundir = mdrun.root / "mdrun"
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    ntomp = get_ntomp()
    mdrun.empp(f=mdrun.mdpdir / "em_cg.mdp")
    mdrun.mdrun(deffnm="em", ntomp=ntomp)
    mdrun.eqpp(f=mdrun.mdpdir / "eq_cg.mdp", c="em.gro", r="em.gro", maxwarn="1") 
    mdrun.mdrun(deffnm="eq", ntomp=ntomp)
    mdrun.mdpp(f=mdrun.mdpdir / "md_cg.mdp", maxwarn="1")    
    if nsteps is None:
        nsteps = NSTEPS
    mdrun.mdrun(deffnm="md", ntomp=ntomp, nsteps=nsteps, ) # bonded="gpu")
    
    
def trjconv(sysdir, sysname, runname, **kwargs):
    kwargs.setdefault("b", 0) # in ps
    kwargs.setdefault("dt", 2) # in ps
    kwargs.setdefault("e", 1e6) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.rundir = mdrun.root / "mdrun"
    k = 1 # k=1 to remove solvent, k=2 for backbone analysis, k=4 to include ions
    # mdrun.trjconv(clinput=f"0\n 0\n", s="eq.tpr", f="eq.gro", o="viz.pdb", n=mdrun.sysndx, pbc="atom", ur="compact", e=0)
    mdrun.convert_tpr(clinput=f"{k}\n", s="md.tpr", n=mdrun.sysndx, o="topology.tpr")
    mdrun.trjconv(clinput=f"{k}\n {k}\n", s="md.tpr", f="md.xtc", o="conv.xtc", n=mdrun.sysndx, pbc="cluster", ur="compact", **kwargs)
    mdrun.trjconv(clinput="0\n 0\n", s="topology.tpr", f="conv.xtc", o="topology.pdb", fit="rot+trans", e=0)
    mdrun.trjconv(clinput="0\n 0\n", s="topology.tpr", f="conv.xtc", o="samples.xtc", fit="rot+trans")
    clean_dir(mdrun.rundir)


if __name__ == "__main__":
    setup_martini(sysdir, sysname)
    md_npt(sysdir, sysname, runname, nsteps=-2)
    trjconv(sysdir, sysname, runname, b=0, dt=2, e=1e6)


    