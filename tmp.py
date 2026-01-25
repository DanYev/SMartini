from pathlib import Path
import shutil
import sys
import numpy as np
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit
from reforge.martini import martini_openmm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter, convert_trajectories, get_platform_info
from reforge.utils import clean_dir, get_logger

logger = get_logger()

# Global settings
# Production parameters
TEMPERATURE = 300 * unit.kelvin  # for equilibraion
GAMMA = 1 / unit.picosecond
PRESSURE = 1 * unit.bar
# Either steps or time
TOTAL_TIME = 1000 * unit.nanoseconds
TSTEP = 2 * unit.femtoseconds
TOTAL_STEPS = 100000 
# Reporting: save every NOUT steps
TRJ_NOUT = 1000 # normally you want ~10000 here
LOG_NOUT = 1000 # 100000 or more
CHK_NOUT = 100000 
OUT_SELECTION = "not resname HOH" # "all" "not resname HOH" "protein"
TRJEXT = 'xtc' # 'xtc' if don't need velocities or 'trr' if do
# Analysis and trjconv
SELECTION = "protein or resname XI" 


def setup(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    input_pdb = Path("pdb") / f"{sysname}.pdb"
    mdsys.prepare_files()
    shutil.copy(input_pdb, mdsys.inpdb)
    # mdsys.clean_pdb(input_pdb, add_missing_atoms=True, add_hydrogens=True)
    pdb = app.PDBFile(str(mdsys.inpdb))
    model = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml", "amber14/lipid17.xml")
    get_platform_info()
    logger.info("Adding membrane...")
    model.addMembrane(forcefield, 
        lipidType='POPC',
        membraneCenterZ=14.5 * unit.nanometer,
        minimumPadding=0.8 * unit.nanometer,
        ionicStrength=0.1 * unit.molar,
        positiveIon='Na+',
        negativeIon='Cl-')
    tmp_pdb = mdsys.root / "system_tmp.pdb"
    logger.info("Writing temporary system PDB: %s", tmp_pdb)
    with open(tmp_pdb, "w", encoding="utf-8") as file:
        app.PDBFile.writeFile(model.topology, model.positions, file, keepIds=True)
    logger.info("Centering/wrapping system ...")
    _center_in_unitcell(tmp_pdb, mdsys.syspdb, center_selection='protein', wrap_compound='segments')
    logger.info("Saved bilayer system to %s", mdsys.syspdb)
    # Build a system WITHOUT any motion remover/barostat/thermostat. Add them later as needed.
    logger.info("Generating topology...")
    system = forcefield.createSystem(
        model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        removeCMMotion=False, # important for strict NVE, added later for NPT
        ewaldErrorTolerance=1e-5,
        rigidWater=True,
    )
    _save_system_to_xml(system, mdsys.sysxml)


def setup_gmx(sysdir, sysname):
    mdsys = GmxSystem(sysdir, sysname)
    mdsys.prepare_files()
    mdsys.gmx("pdb2gmx", f=mdsys.syspdb, p=mdsys.root / "system.top", ignh='yes')
    return mdsys


def add_extra_forces(system): # for NPT
    # COM remover
    com_remover = mm.CMMotionRemover()
    com_remover.setFrequency(100)
    system.addForce(com_remover)
    logger.info("Added center of mass drift remover")
    # Barostat
    barostat = mm.MonteCarloMembraneBarostat(
        PRESSURE,          # pressure
        0.0*unit.bar*unit.nanometer,  # surface tension (0 = tensionless)
        TEMPERATURE,       # temperature
        mm.MonteCarloMembraneBarostat.XYIsotropic,
        mm.MonteCarloMembraneBarostat.ZFree
    )
    system.addForce(barostat)
    logger.info("Added barostat")


def md_npt(sysdir, sysname, runname, CudaDeviceIndex="0,1"): 
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDIR: %s", mdrun.rundir)
    # Log platform info
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {
        "CudaDeviceIndex": CudaDeviceIndex, # IF multiple GPUs
        "CudaPrecision": "mixed"
    }
    # get_platform_info(platform=platform)
    # Prep
    logger.info("Preparing the system...")
    logger.info("Loading the PDB file...")
    pdb = app.PDBFile(str(mdsys.syspdb))
    # Create system object
    logger.info("Loading the XML file...")
    system = _load_system_from_xml(mdsys.sysxml)
    _add_bb_restraints(system, pdb, bb_aname='CA')
    # Create simulation object
    integrator = mm.LangevinMiddleIntegrator(0, GAMMA, 1*unit.femtosecond)  
    simulation = app.Simulation(pdb.topology, system, integrator, 
        platform=platform, platformProperties=properties)
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
    n_cycles = 100
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


def extend(sysdir, sysname, runname, CudaDeviceIndex="0"):    
    """ For NPT runs """
    mdsys = MmSystem(sysdir, sysname)
    mdrun = MmRun(sysdir, sysname, runname)
    logger.info(f"WDIR: %s", mdrun.rundir)
    pdb = app.PDBFile(str(mdrun.syspdb))
    system = _load_system_from_xml(mdsys.sysxml)
    # Platform
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {
        "CudaDeviceIndex": CudaDeviceIndex, # IF multiple GPUs
        "CudaPrecision": "mixed"
    }
    get_platform_info()
    # Add COM remover barostat
    add_extra_forces(system)
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, GAMMA, TSTEP)
    simulation = app.Simulation(pdb.topology, system, integrator,
        platform=platform, platformProperties=properties)
    # Reporters
    curr_prefix, next_prefix = _get_run_prefix(mdrun)
    logger.info(f"Current prefix for trajectory: {curr_prefix}, Next prefix: {next_prefix}")
    reporters = _get_reporters(mdrun, append=False, prefix=next_prefix)
    simulation.reporters.extend(reporters)
    # Run
    mdrun.extend(simulation, curr_prefix=curr_prefix, next_prefix=next_prefix, until_time=TOTAL_TIME)


def trjconv(sysdir, sysname, runname):
    system = MDSystem(sysdir, sysname)
    mdrun = MDRun(sysdir, sysname, runname)
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
    convert_trajectories(top, trajs, out_top, out_traj, selection=SELECTION, start=5000, stop=None, step=10, fit=True)
    logger.info("Done!")

###############################################################################################
### PRIVATE FUNCTIONS ###
###############################################################################################

def _save_system_to_xml(system, filename):
    with open(str(filename), "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    logger.info(f"Saved system to {filename}")


def _load_system_from_xml(filename):
    with open(str(filename), 'r') as file:
        system = mm.XmlSerializer.deserialize(file.read())
    logger.info(f"Loaded system from {filename}")
    return system


def _center_in_unitcell(
        input_pdb: Path,
        output_pdb: Path,
        center_selection: str = "protein",
        wrap_compound: str = "segments",
    ):
    """Center and wrap coordinates using MDAnalysis, then write a PDB.

    This is used as an alternative to doing coordinate wrapping directly in OpenMM.
    We load the PDB (with CRYST1), apply MDAnalysis transformations, then write the
    resulting coordinates back *via OpenMM* to preserve ids and CRYST1 consistently.

    Parameters
    ----------
    input_pdb, output_pdb
        Paths.
    center_selection
        Selection to center (usually 'protein').
    wrap_compound
        'segments' keeps chains together if segids are present; fallback to 'residues'
        if your PDB has no segids.
    """
    from MDAnalysis.transformations import center_in_box, wrap
    u = mda.Universe(str(input_pdb))
    ag_center = u.select_atoms(center_selection)
    # GROMACS-like: center on protein, then wrap everything with chains kept together.
    u.trajectory.add_transformations(
        center_in_box(ag_center, center='mass', wrap=False),
        wrap(u.atoms, compound=wrap_compound),
    )
    u.trajectory[0]
    coords_ang = u.atoms.positions.copy()  # Angstrom
    # Write using OpenMM PDBFile to keep ids consistent.
    pdb = app.PDBFile(str(input_pdb))
    positions_nm = unit.Quantity(coords_ang / 10.0, unit.nanometer)
    with open(output_pdb, "w", encoding="utf-8") as f:
        app.PDBFile.writeFile(pdb.topology, positions_nm, f, keepIds=True)


def _add_bb_restraints(system, pdb, bb_aname='CA'):
    restraint = mm.CustomExternalForce('bb_fc*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.setName('BackboneRestraint')
    restraint.addGlobalParameter('bb_fc', 1000.0*unit.kilojoules_per_mole/unit.nanometer)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')
    system.addForce(restraint)
    for atom in pdb.topology.atoms():
        if atom.name == bb_aname:
            restraint.addParticle(atom.index, pdb.positions[atom.index])


def _get_reporters(mdrun, append=False, prefix="md"):
    mdrun.rundir.mkdir(parents=True, exist_ok=True)
    log_reporter = app.StateDataReporter(
            str(mdrun.rundir / f"{prefix}.log"), 
            LOG_NOUT, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, speed=True, append=append)
    err_reporter =  app.StateDataReporter(
            sys.stderr, LOG_NOUT, time=True, step=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, speed=True, append=append)
    logger.info(f'Setting up trajectory reporter with selection: {OUT_SELECTION}')
    traj_reporter = MmReporter(str(mdrun.rundir / f"{prefix}.{TRJEXT}"), 
            reportInterval=TRJ_NOUT, selection=OUT_SELECTION)
    state_reporter = app.CheckpointReporter(str(mdrun.rundir / f"{prefix}.xml"), CHK_NOUT, writeState=True)
    return log_reporter, err_reporter, traj_reporter, state_reporter


def _get_run_prefix(mdrun):
    existing_md = list(mdrun.rundir.glob(f"md*.{TRJEXT}")) 
    if not existing_md:
        return "eq", "md"
    nums = [int(f.stem.split('md_')[-1]) for f in existing_md if 'md_' in f.stem]
    if not nums:
        return "md", "md_1"
    max_num = max(nums)
    return f"md_{max_num}", f"md_{max_num+1}"


def check_mps(*args):
    """Check if CUDA MPS is properly configured and running."""
    import os
    import subprocess
    try:
        # Verify CUDA availability
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        
        # Check MPS environment variable
        mps_pipe = os.getenv('CUDA_MPS_PIPE_DIRECTORY')
        if not mps_pipe:
            print("Warning: CUDA_MPS_PIPE_DIRECTORY not set")
            return False
            
        # Verify MPS control file
        control_file = os.path.join(mps_pipe, "control")
        if not os.path.exists(control_file):
            print("Warning: MPS control file not found")
            return False
            
        return True
        
    except subprocess.CalledProcessError:
        print("Warning: Unable to verify CUDA setup")
        return False


if __name__ == "__main__":
    from reforge.cli import run_command
    run_command()