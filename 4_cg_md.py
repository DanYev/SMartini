
from pathlib import Path
import shutil
import MDAnalysis as mda
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun, get_ntomp
from reforge.utils import clean_dir, get_logger
from reforge.forge.topology import Topology

logger = get_logger()

# Global settings
INPDB = 'KDA.pdb'
DT = 0.020  # Time step in picoseconds
total_time = 1000  # Total simulation time in nanoseconds
NSTEPS = int(total_time * 1e3 / DT)  # Number of MD steps for production run

ligand_name = "FTA"
sysdir = f"systems/{ligand_name}"
sysname = "cg_md"
runname = "mdrun"


def setup_martini(sysdir, sysname):
    ### FOR CG PROTEIN+/RNA SYSTEMS ###
    mdsys = GmxSystem(sysdir, sysname)
    input_pdb = Path(sysdir) / "system.pdb"
    mdsys.prepare_files(pour_martini=True)
   
    # LIGANDS 
    itp_file = Path(sysdir) / "mapping" / f"{ligand_name}_updated.itp"
    map_file = Path(sysdir) / "mapping" / f"{ligand_name}.map"
    mdsys.martinize_ligands(input_pdb=input_pdb, ligand="UNK", itp_file=itp_file, map_file=map_file) # ligand identified by resname and it is "UNK"
    mdsys.make_cg_structure() # CG structure. Returns mdsys.solupdb ("solute.pdb") file
    mdsys.make_cg_topology() # CG topology. Returns mdsys.systop ("mdsys.top") file
    
    # 1.3. Coarse graining is *hopefully* done. Need to add solvent and ions
    solute = "solute.pdb"
    solvent = "water.gro"
    topo = "system.top"
    system_gro = "system_cg.gro"
    mdsys.gmx("editconf", f=solute, o=solute, d=1.0, bt="dodecahedron")
    mdsys.gmx("solvate",cp=solute, cs=solvent, radius="0.17")
    mdsys.gmx("grompp", f="mdp/ions.mdp", c=system_gro, p=topo, o="ions.tpr")
    mdsys.gmx("genion", clinput=f"{solvent}\n", s="ions.tpr",  p=topo, o=system_gro, conc=0.0, pname="NA", nname="CL")
    clean_dir(mdsys.root, "ions.tpr")

    # 1.4. Need index files to make selections with GROMACS. Very annoying but wcyd. Order:
    # 1.System 2.Solute 3.Backbone 4.Solvent 5...chains. Can add custom groups using AtomList.write_to_ndx()
    mdsys.make_system_ndx(backbone_atoms=["BB", "BB2"])

    
def md_npt(sysdir, sysname, runname, nsteps=None): 
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.rundir = mdrun.root 
    ntomp = get_ntomp()
    mdrun.empp(f=mdrun.mdpdir / "em_cg.mdp")
    mdrun.mdrun(deffnm="em", ntomp=ntomp)
    mdrun.hupp(f=mdrun.mdpdir / "hu_cg.mdp", c="em.gro", r="em.gro", maxwarn="1") 
    mdrun.mdrun(deffnm="hu", ntomp=ntomp)
    mdrun.eqpp(f=mdrun.mdpdir / "eq_cg.mdp", c="hu.gro", r="hu.gro", maxwarn="1") 
    mdrun.mdrun(deffnm="eq", ntomp=ntomp)
    mdrun.mdpp(f=mdrun.mdpdir / "md_cg.mdp", maxwarn="1")    
    if nsteps is None:
        nsteps = NSTEPS
    mdrun.mdrun(deffnm="md", ntomp=ntomp, nsteps=nsteps, ) # bonded="gpu")
    
    
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
    kwargs.setdefault("e", 10000000) # in ps
    mdrun = GmxRun(sysdir, sysname, runname)
    mdrun.rundir = mdrun.root 
    k = 1 # k=1 to remove solvent, k=2 for backbone analysis, k=4 to include ions
    # mdrun.trjconv(clinput=f"0\n 0\n", s="eq.tpr", f="eq.gro", o="viz.pdb", n=mdrun.sysndx, pbc="atom", ur="compact", e=0)
    mdrun.convert_tpr(clinput=f"{k}\n", s="md.tpr", n=mdrun.sysndx, o="topology.tpr")
    mdrun.trjconv(clinput=f"{k}\n {k}\n", s="md.tpr", f="md.xtc", o="conv.xtc", n=mdrun.sysndx, pbc="cluster", ur="compact", **kwargs)
    mdrun.trjconv(clinput="0\n 0\n", s="topology.tpr", f="conv.xtc", o="topology.pdb", fit="rot+trans", e=0)
    mdrun.trjconv(clinput="0\n 0\n", s="topology.tpr", f="conv.xtc", o="samples.xtc", fit="rot+trans")
    clean_dir(mdrun.rundir)


if __name__ == "__main__":
    setup_martini(sysdir, sysname)

    