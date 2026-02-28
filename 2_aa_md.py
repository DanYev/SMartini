import logging
from pathlib import Path
import shutil
import sys
import numpy as np
import openmm as mm
from openmm import app, unit
from openff.toolkit import ForceField, Molecule, Topology 
from openff.interchange import Interchange
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem
from pdbfixer import PDBFixer
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter, convert_trajectories, get_platform_info

from ligpar_config import CFG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global settings
# Production parameters
TEMPERATURE = 300 * unit.kelvin  # for equilibration
GAMMA = 1 / unit.picosecond
PRESSURE = 1 * unit.bar
# Either steps or time
TOTAL_TIME = 1000 * unit.nanoseconds # USED BY DEFAULT. NSTEPS = TOTAL_TIME / TSTEP
TSTEP = 2 * unit.femtoseconds
TOTAL_STEPS = 100000 
# Reporting: save every NOUT steps
TRJ_NOUT = 1000 # normally you want ~10000 here
LOG_NOUT = 10000 # 100000 or more
CHK_NOUT = 100000 
OUT_SELECTION = "resname UNK" # "all" "not resname HOH" "protein"
TRJEXT = 'xtc' # 'xtc' if don't need velocities or 'trr' if do
# Analysis and trjconv
SELECTION = OUT_SELECTION 
#########

ligand = CFG.molname
sysdir = str(CFG.wdir())
sysname = CFG.aa_sysname
runname = "."


def process_ligand(sysdir, sysname, ligand_name):
    # INPUTS
    mdsys = MmSystem(sysdir, sysname)
    logger.info("Processing ligand: %s", ligand_name)
    wdir = Path("systems") / ligand_name
    wdir.mkdir(parents=True, exist_ok=True)
    logger.info("Ligand working directory: %s", wdir)
    input_file = CFG.ligands_dir / f"{ligand_name}.sdf"
    logger.info("Reading ligand file: %s", input_file)
    # Generate ligand topology and structure using OpenFF Toolkit and Interchange
    ligand = Molecule.from_file(str(input_file))
    smirnoff = SMIRNOFFTemplateGenerator(molecules=[ligand])
    forcefield = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
    # Ligand FF
    forcefield.registerTemplateGenerator(smirnoff.generator)
    ff = ForceField("openff-2.1.0.offxml")
    interchange = Interchange.from_smirnoff(ff, ligand.to_topology())
    ligand_topology = interchange.to_openmm_topology()
    ligand_positions = interchange.positions.to_openmm()
    model = app.Modeller(ligand_topology, ligand_positions)
    logger.info("Adding solvent and ions")
    model.addSolvent(forcefield, 
        model='tip3p', 
        boxShape='cube', #  ‘cube’, ‘dodecahedron’, and ‘octahedron’
        padding=1.5 * unit.nanometer,
        ionicStrength=0.0 * unit.molar,
        positiveIon='Na+',
        negativeIon='Cl-')    
    with open(mdsys.syspdb, "w", encoding="utf-8") as file:
        app.PDBFile.writeFile(model.topology, model.positions, file, keepIds=True)    
    logger.info("Generating topology...")
    system = forcefield.createSystem(
        model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        removeCMMotion=False,     # important for strict NVE
        ewaldErrorTolerance=1e-5
    )
    _save_system_to_xml(system, mdsys.sysxml)


def md_npt(sysdir, sysname, runname, CudaDeviceIndex="0"): 
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    mdrun.rundir = mdrun.root / runname
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # Log platform info
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {
        "CudaDeviceIndex": CudaDeviceIndex, # IF multiple GPUs
        "CudaPrecision": "mixed"
    }
    get_platform_info()
    # Prep
    logger.info("Preparing the system...")
    logger.info("Loading the PDB file...")
    pdb = app.PDBFile(str(mdsys.syspdb))
    # Create system object
    logger.info("Loading the XML file...")
    system = _load_system_from_xml(mdsys.sysxml)
    _add_bb_restraints(system, pdb, bb_aname='C*')
    # Create simulation object
    integrator = mm.LangevinMiddleIntegrator(0, GAMMA, 1*unit.femtosecond)  
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    # Reporters
    reporters = _get_reporters(mdrun, prefix="eq")
    simulation.reporters.extend(reporters)
    # Minimization
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    simulation.saveState(str(mdrun.rundir / "em.xml"))
    # Heatup
    logger.info("Heating up the system...")
    n_cycles = 10
    steps_per_cycle = 500
    for i in range(n_cycles):
        simulation.integrator.setTemperature(TEMPERATURE*i/n_cycles)
        simulation.step(steps_per_cycle)
    simulation.saveState(str(mdrun.rundir / "hu.xml"))
    # NPT Equilibration
    logger.info("NPT Equilibration")
    add_extra_forces(simulation.system)
    simulation.integrator.setTemperature(TEMPERATURE)
    simulation.context.reinitialize(preserveState=True)
    mdrun.eq(simulation, n_cycles=100, steps_per_cycle=1000)
    # MD
    logger.info("Production...")
    # add_extra_forces(simulation.system) # IF STARING FROM EQ
    simulation.integrator.setTemperature(TEMPERATURE)
    simulation.integrator.setStepSize(TSTEP)
    simulation.loadState(str(mdrun.rundir / "eq.xml"))
    # Reporters
    logger.info(f'Saving reference PDB with selection: {OUT_SELECTION}')
    mda.Universe(mdsys.syspdb).select_atoms(OUT_SELECTION).write(mdrun.rundir / "md.pdb")
    simulation.reporters = []  # clear existing reporters
    reporters = _get_reporters(mdrun, append=False, prefix='md')
    simulation.reporters.extend(reporters)
    # Run
    # simulation.context.reinitialize(preserveState=True) # ONLY FOR TESTING
    nsteps = int(TOTAL_TIME / TSTEP)
    simulation.step(nsteps)
    simulation.saveState(str(mdrun.rundir / "md.xml"))
    logger.info("Done!")


def trjconv(sysdir, sysname, runname):
    system = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
    mdrun.rundir = mdrun.root / runname
    logger.info(f"WDIR: %s", mdrun.rundir)
    # INPUT
    top = mdrun.rundir / "md.pdb"
    # top = mdrun.root / "system.pdb"
    traj = mdrun.rundir / f"md.{TRJEXT}"
    ext_trajs = sorted([f for f in mdrun.rundir.glob(f"md_*.{TRJEXT}")])
    trajs = [traj] + ext_trajs
    logger.info(f'Input trajectory files: {trajs}')
    out_top = mdrun.rundir / "topology.pdb"
    out_traj = mdrun.rundir / f"samples.{TRJEXT}"
    # CONVERT
    convert_trajectories(top, trajs, out_top, out_traj, selection=SELECTION, start=0, stop=None, step=1, fit=True)
    logger.info("Done!")


def add_extra_forces(system): # for NPT
    # COM remover
    com_remover = mm.CMMotionRemover()
    com_remover.setFrequency(100)
    system.addForce(com_remover)
    logger.info("Added center of mass drift remover")
    # Barostat
    barostat = mm.MonteCarloBarostat(
        PRESSURE,          # pressure
        TEMPERATURE        # temperature
    )
    system.addForce(barostat)
    logger.info("Added barostat")


def _save_system_to_xml(system, filename):
    with open(str(filename), "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    logger.info(f"Saved system to {filename}")


def _load_system_from_xml(filename):
    with open(str(filename), 'r') as file:
        system = mm.XmlSerializer.deserialize(file.read())
    logger.info(f"Loaded system from {filename}")
    return system


def _get_reporters(mdrun, append=False, prefix="md"):
    """Get reporters for MD simulation using custom MmReporter for velocities"""
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    # Log reporter (file)
    log_reporter = app.StateDataReporter(
        str(mdrun.rundir / f"{prefix}.log"), 
        LOG_NOUT, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Error reporter (stderr)
    err_reporter = app.StateDataReporter(
        sys.stderr, LOG_NOUT, time=True, step=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Custom trajectory reporter with velocities using MmReporter
    logger.info(f'Setting up trajectory reporter with selection: {OUT_SELECTION}')
    traj_reporter = MmReporter(str(mdrun.rundir / f"{prefix}.{TRJEXT}"), 
        reportInterval=TRJ_NOUT, selection=OUT_SELECTION)
    # State/checkpoint reporter
    state_reporter = app.CheckpointReporter(str(mdrun.rundir / f"{prefix}.xml"), CHK_NOUT, writeState=True)
    return log_reporter, err_reporter, traj_reporter, state_reporter


if __name__ == "__main__":
    process_ligand(sysdir, sysname, ligand_name)
    md_npt(sysdir, sysname, runname, CudaDeviceIndex="0")
    trjconv(sysdir, sysname, runname)
