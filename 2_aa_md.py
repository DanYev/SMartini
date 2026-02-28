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
TSTEP = 2 * unit.femtoseconds
TOTAL_STEPS = int(1e6)
# Reporting: save every NOUT steps
TRJ_NOUT = 1000 # normally you want ~10000 here
LOG_NOUT = 10000 # 100000 or more
CHK_NOUT = 100000 
OUT_SELECTION = "resname UNK" # "all" "not resname HOH" "protein"
TRJEXT = 'xtc' # 'xtc' if don't need velocities or 'trr' if do
# Analysis and trjconv
SELECTION = OUT_SELECTION 
#########

ligand_name = CFG.molname
sysdir = CFG.systems_dir
wdir = CFG.wdir
aa_dir = CFG.aa_dir
system_pdb = aa_dir / "system.pdb"
system_xml = aa_dir / "system.xml"
runname = "."


def process_ligand(ligand_name):
    # INPUTS
    logger.info("Working directory: %s", wdir)
    logger.info("Processing ligand: %s", ligand_name)
    # Generate ligand topology and structure using OpenFF Toolkit and Interchange
    input_file = wdir / f"{ligand_name}_ideal.sdf"
    logger.info("Reading ligand file: %s", input_file)
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
        padding=1.0 * unit.nanometer,
        ionicStrength=0.0 * unit.molar,
        positiveIon='Na+',
        negativeIon='Cl-')    
    with open(system_pdb, "w", encoding="utf-8") as file:
        app.PDBFile.writeFile(model.topology, model.positions, file, keepIds=True)    
    logger.info("Generating topology...")
    system = forcefield.createSystem(
        model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        removeCMMotion=True,     
        ewaldErrorTolerance=1e-5
    )
    _save_system_to_xml(system, system_xml)


def md_npt(): 
    # Log platform info
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {
        "CudaDeviceIndex": "0", # IF multiple GPUs
        "CudaPrecision": "mixed"
    }
    get_platform_info()
    # Prep
    logger.info("Preparing the system...")
    logger.info("Loading the PDB file...")
    pdb = app.PDBFile(str(system_pdb))
    # Create system object
    logger.info("Loading the XML file...")
    system = _load_system_from_xml(system_xml)
    # Create simulation object
    integrator = mm.LangevinMiddleIntegrator(0, GAMMA, 1*unit.femtosecond)  
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    # Minimization
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    # Eqilibration
    logger.info("Equilibrating...")
    simulation.integrator.setTemperature(TEMPERATURE)
    barostat = mm.MonteCarloBarostat(PRESSURE, TEMPERATURE)
    system.addForce(barostat)
    simulation.step(5000)
    # MD
    logger.info("Production...")
    # add_extra_forces(simulation.system) # IF STARING FROM EQ
    simulation.integrator.setStepSize(TSTEP)
    # Reporters
    logger.info(f'Saving reference PDB with selection: {OUT_SELECTION}')
    mda.Universe(system_pdb).select_atoms(OUT_SELECTION).write(str(aa_dir / "md.pdb"))
    reporters = _get_reporters(append=False, prefix='md')
    simulation.reporters = reporters
    # Run
    nsteps = int(TOTAL_STEPS)
    simulation.step(nsteps)
    logger.info("Done!")


def trjconv():
    # INPUT
    top = aa_dir / "md.pdb"
    # top = mdrun.root / "system.pdb"
    traj = aa_dir / f"md.{TRJEXT}"
    ext_trajs = sorted([f for f in aa_dir.glob(f"md_*.{TRJEXT}")])
    trajs = [traj] + ext_trajs
    logger.info(f'Input trajectory files: {trajs}')
    out_top = aa_dir / "topology.pdb"
    out_traj = aa_dir / f"samples.{TRJEXT}"
    # CONVERT
    convert_trajectories(top, trajs, out_top, out_traj, selection=SELECTION, start=0, stop=None, step=1, fit=True)
    logger.info("Done!")


def _save_system_to_xml(system, filename):
    with open(str(filename), "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    logger.info(f"Saved system to {filename}")


def _load_system_from_xml(filename):
    with open(str(filename), 'r') as file:
        system = mm.XmlSerializer.deserialize(file.read())
    logger.info(f"Loaded system from {filename}")
    return system


def _get_reporters(append=False, prefix="md"):
    """Get reporters for MD simulation using custom MmReporter for velocities"""
    # Log reporter (file)
    log_reporter = app.StateDataReporter(
        str(aa_dir / f"{prefix}.log"), 
        LOG_NOUT, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Error reporter (stderr)
    err_reporter = app.StateDataReporter(
        sys.stderr, LOG_NOUT, time=True, step=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Custom trajectory reporter with velocities using MmReporter
    logger.info(f'Setting up trajectory reporter with selection: {OUT_SELECTION}')
    traj_reporter = MmReporter(str(aa_dir / f"{prefix}.{TRJEXT}"), 
        reportInterval=TRJ_NOUT, selection=OUT_SELECTION)
    return log_reporter, err_reporter, traj_reporter


if __name__ == "__main__":
    # process_ligand(ligand_name)
    md_npt()
    trjconv()
