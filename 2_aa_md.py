import logging
import sys
import smartini
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit
from openff.toolkit import ForceField, Molecule, Topology 
from openff.interchange import Interchange
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from config import CFG

logger = logging.getLogger(__name__)
smartini.setup_logging(level=logging.INFO)

"""Atomistic MD setup and production workflow for a single ligand system.

Pipeline:
1. Build solvated OpenMM system from ligand SDF.
2. Run minimization, heating, equilibration, and production NPT.
3. Export selected trajectory/topology for downstream CG fitting.
"""

# Use configuration from config.py
ligand_name = CFG.molname
sysdir = CFG.systems_dir
wdir = CFG.wdir
aa_dir = CFG.aa_dir
system_pdb = aa_dir / "system.pdb"
system_xml = aa_dir / "system.xml"


def process_ligand():
    """Build and solvate the ligand AA system, then write topology artifacts.

    Writes:
    - ``system.pdb`` and ``system.xml`` for simulation,
    - ``md.pdb`` with ``CFG.aa_selection`` for trajectory output reference.
    """
    # INPUTS
    ligand_name = CFG.molname
    logger.info("Working directory: %s", wdir)
    logger.info("Processing ligand: %s", ligand_name)
    # Generate ligand topology and structure using OpenFF Toolkit and Interchange
    aa_dir.mkdir(parents=True, exist_ok=True)
    input_file = wdir / f"{ligand_name}.sdf"
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
        padding=1.2 * unit.nanometer,
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
    logger.info(f'Saving reference PDB with selection: {CFG.aa_selection}')
    mda.Universe(system_pdb).select_atoms(CFG.aa_selection).write(str(aa_dir / "md.pdb"))


def md_npt(): 
    """Run AA MD in OpenMM: minimize, heat, equilibrate, then produce trajectory."""
    # Prep
    logger.info("Loading the PDB file...")
    pdb = app.PDBFile(str(system_pdb))
    logger.info("Loading the XML file...")
    system = _load_system_from_xml(system_xml)
    # Create simulation object
    integrator = mm.LangevinMiddleIntegrator(0, CFG.aa_gamma / unit.picosecond, 1*unit.femtosecond)  
    simulation = app.Simulation(pdb.topology, system, integrator) 
    simulation.context.setPositions(pdb.positions)
    reporters = _get_reporters(append=False, prefix='md')
    # Minimization
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    # Heatup
    logger.info("Heating up...")
    n_cycles = 10
    steps_per_cycle = 1000
    for i in range(n_cycles):
        current_temp = (i + 1) * CFG.temperature * unit.kelvin / n_cycles
        simulation.integrator.setTemperature(current_temp)
        simulation.step(steps_per_cycle)
    # Eqilibration
    logger.info("Equilibrating...")
    barostat = mm.MonteCarloBarostat(CFG.aa_pressure_bar * unit.bar, CFG.temperature * unit.kelvin)
    system.addForce(barostat)
    simulation.integrator.setTemperature(CFG.temperature * unit.kelvin)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(10000)
    # MD
    logger.info("Production...")
    # state = simulation.context.getState(getPositions=True, getVelocities=True)
    simulation.integrator.setStepSize(CFG.aa_timestep_fs * unit.femtoseconds)
    simulation.context.reinitialize(preserveState=True)
    simulation.reporters = reporters
    simulation.step(int(CFG.aa_total_steps))
    logger.info("Done!")


def trjconv(start=0, stop=None, step=1, fit=True):
    """Post-process AA trajectory and write aligned sampled outputs.

    Parameters
    ----------
    start, stop, step : int or None
        Frame slicing applied to ``md.xtc``.
    fit : bool
        If ``True``, perform rotational/translational fitting to the selected
        atom group before writing.
    """
    top = aa_dir / "md.pdb"
    traj = aa_dir / "md.xtc"
    out_top = aa_dir / "topology.pdb"
    out_traj = aa_dir / "samples.xtc"
    universe = mda.Universe(str(top), str(traj))
    atom_group = universe.select_atoms(CFG.aa_selection)

    if fit:
        ref_universe = mda.Universe(str(top))
        ref_atoms = ref_universe.select_atoms(CFG.aa_selection)
        from MDAnalysis.transformations import fit_rot_trans
        universe.trajectory.add_transformations(fit_rot_trans(atom_group, ref_atoms))

    atom_group.write(str(out_top))
    with mda.Writer(str(out_traj), n_atoms=atom_group.n_atoms) as writer:
        for _ in universe.trajectory[start:stop:step]:
            writer.write(atom_group)
    logger.info("Done!")


def _save_system_to_xml(system, filename):
    """Serialize an OpenMM ``System`` to XML."""
    with open(str(filename), "w", encoding="utf-8") as file:
        file.write(mm.XmlSerializer.serialize(system))
    logger.info(f"Saved system to {filename}")


def _load_system_from_xml(filename):
    """Load an OpenMM ``System`` from XML."""
    with open(str(filename), 'r') as file:
        system = mm.XmlSerializer.deserialize(file.read())
    logger.info(f"Loaded system from {filename}")
    return system


def _get_reporters(append=False, prefix="md"):
    """Get reporters for MD simulation using OpenMM reporters."""
    # Log reporter (file)
    log_reporter = app.StateDataReporter(
        str(aa_dir / f"{prefix}.log"), 
        CFG.aa_log_nout, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)
    # Error reporter (stderr)
    err_reporter = app.StateDataReporter(
        sys.stderr, CFG.aa_log_nout, time=True, step=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True, append=append)

    # Position-only trajectory reporter (XTC) with atom subset
    if CFG.aa_selection == "all":
        atom_subset = None
        logger.info("Setting up XTC reporter for all atoms")
    else:
        universe = mda.Universe(str(system_pdb))
        atom_subset = universe.select_atoms(CFG.aa_selection).indices.tolist()
        logger.info(
            "Setting up XTC reporter with selection '%s' (%d atoms)",
            CFG.aa_selection,
            len(atom_subset),
        )
    traj_reporter = app.XTCReporter(
        str(aa_dir / f"{prefix}.xtc"),
        CFG.aa_trj_nout,
        append=append,
        atomSubset=atom_subset,
    )
    return log_reporter, err_reporter, traj_reporter


if __name__ == "__main__":
    process_ligand()
    md_npt()
    trjconv()
