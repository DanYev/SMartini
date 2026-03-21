import logging
import shutil
import sys
from pathlib import Path
from auto_martini.AutoMartini.utils import change_directory, clean_dir, get_ntomp, gmx

from config import CFG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Use configuration from config.py
ligand = CFG.molname
sysdir = CFG.wdir
outdir = CFG.mol_dir
sysname = CFG.cg_sysname
runname = CFG.cg_runname

# Compute NSTEPS from config
NSTEPS = int(CFG.cg_total_time_ns * 1e3 / CFG.cg_dt)


def setup(sysdir, sysname):
    root = Path(sysdir).resolve() / sysname
    topdir = root / "topol"
    mdpdir = root / "mdp"
    solupdb = root / "solute.pdb"
    inpdb = root / "inpdb.pdb"
    syspdb = root / "system.pdb"
    sysgro = root / "system.gro"
    systop = root / "system.top"
    sysndx = root / "system.ndx"

    root.mkdir(parents=True, exist_ok=True)
    topdir.mkdir(parents=True, exist_ok=True)
    mdpdir.mkdir(parents=True, exist_ok=True)

    src_md = Path(__file__).resolve().parent / "md_cg.mdp"
    shutil.copy(src_md, mdpdir / "md_cg.mdp")

    if not (mdpdir / "em_cg.mdp").exists() or not (mdpdir / "ions.mdp").exists():
        raise FileNotFoundError(
            f"Missing required mdp templates in {mdpdir}: need em_cg.mdp and ions.mdp"
        )

    pdb_file = outdir / f"{ligand}.pdb"
    itp_file = outdir / f"{ligand}.itp"
    ligand_itp = topdir / f"ligand_{ligand}.itp"
    shutil.copy(itp_file, ligand_itp)
    shutil.copy(pdb_file, solupdb)
    shutil.copy(pdb_file, inpdb)

    if "md" not in sys.argv:
        martini_itps = sorted(topdir.glob("martini*.itp"))
        if not martini_itps:
            raise FileNotFoundError(
                f"No Martini force-field .itp files found in {topdir}."
            )

        with open(systop, "w", encoding="utf-8") as f:
            for ff in martini_itps:
                f.write(f'#include "topol/{ff.name}"\n')
            f.write(f'\n#include "topol/{ligand_itp.name}"\n\n')
            f.write("[ system ]\n")
            f.write(f"Martini system for {sysname} in water\n\n")
            f.write("[ molecules ]\n")
            f.write("; name\t\tnumber\n")
            f.write(f"{ligand}\t\t1\n")

        solvent = root / "water.gro"
        if not solvent.exists():
            raise FileNotFoundError(f"Missing solvent structure file: {solvent}")

        with change_directory(root):
            gmx("editconf", f=solupdb, o=solupdb, d="1.0", bt="cubic")
            gmx("solvate", cp=solupdb, cs=solvent, p=systop, o=syspdb, radius="0.17")
            gmx("grompp", f=mdpdir / "ions.mdp", c=syspdb, p=systop, o="ions.tpr")
            gmx("genion", clinput="W\n", s="ions.tpr", p=systop, o=syspdb, conc=0.0, pname="NA", nname="CL")
            gmx("editconf", f=syspdb, o=sysgro)

        u_system = mda.Universe(str(syspdb))
        u_solute = mda.Universe(str(solupdb))
        n_system = len(u_system.atoms)
        n_solute = len(u_solute.atoms)
        system_idx = list(range(1, n_system + 1))
        solute_idx = list(range(1, n_solute + 1))
        backbone_idx = solute_idx.copy()
        solvent_idx = list(range(n_solute + 1, n_system + 1))

        def _write_group(handle, name, indices, wrap=15):
            handle.write(f"[ {name} ]\n")
            for i in range(0, len(indices), wrap):
                handle.write(" ".join(str(x) for x in indices[i:i + wrap]) + "\n")
            handle.write("\n")

        with open(sysndx, "w", encoding="utf-8") as ndx:
            _write_group(ndx, "System", system_idx)
            _write_group(ndx, "Solute", solute_idx)
            _write_group(ndx, "Backbone", backbone_idx)
            if solvent_idx:
                _write_group(ndx, "Solvent", solvent_idx)

    
def md_npt(sysdir, sysname, runname, nsteps=NSTEPS): 
    root = Path(sysdir).resolve() / sysname
    rundir = root / "mdrun"
    mdpdir = root / "mdp"
    sysgro = root / "system.gro"
    systop = root / "system.top"
    sysndx = root / "system.ndx"
    ntomp = get_ntomp()
    rundir.mkdir(parents=True, exist_ok=True)
    with change_directory(rundir):
        gmx("grompp", f=mdpdir / "em_cg.mdp", c=sysgro, r=sysgro, p=systop, n=sysndx, o="em.tpr")
        gmx("mdrun", deffnm="em", ntomp=ntomp)
        gmx("grompp", f=mdpdir / "md_cg.mdp", c="em.gro", r="em.gro", p=systop, n=sysndx, o="md.tpr", maxwarn="1")
        gmx("mdrun", deffnm="md", ntomp=ntomp, nsteps=nsteps)  # bonded="gpu"
    
    
def trjconv(sysdir, sysname, runname):
    root = Path(sysdir).resolve() / sysname
    rundir = root / "mdrun"
    sysndx = root / "system.ndx"
    with change_directory(rundir):
        gmx("convert-tpr", clinput=f"1\n", s="md.tpr", n=sysndx, o="topology.tpr")
        gmx("trjconv", clinput="1\n 1\n", s="md.tpr", f="md.xtc", o="samples.xtc", n=sysndx, fit="rot+trans")
        gmx("trjconv", clinput="1\n 1\n", s="md.tpr", f="md.xtc", o="topology.pdb", n=sysndx, fit="rot+trans", e=0)
    clean_dir(rundir)


if __name__ == "__main__":
    nsteps = -2
    if "nsteps" in sys.argv:
        nsteps = int(sys.argv[sys.argv.index("nsteps") + 1])
    setup(sysdir, sysname)
    md_npt(sysdir, sysname, runname, nsteps=nsteps)
    trjconv(sysdir, sysname, runname)


    