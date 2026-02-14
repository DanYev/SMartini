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
from reforge.martini import martini_openmm
from reforge.mdsystem.mdsystem import MDSystem, MDRun
from reforge.mdsystem.gmxmd import GmxSystem, GmxRun
from reforge.mdsystem.mmmd import MmSystem, MmRun, MmReporter, convert_trajectories, get_platform_info
from reforge.utils import clean_dir, get_logger

logger = get_logger()

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
TRJ_NOUT = 10000 # normally you want ~10000 here
LOG_NOUT = 10000 # 100000 or more
CHK_NOUT = 100000 
OUT_SELECTION = "protein or resname ANP or resname MG" # "all" "not resname HOH" "protein"
TRJEXT = 'xtc' # 'xtc' if don't need velocities or 'trr' if do
# Analysis and trjconv
SELECTION = OUT_SELECTION 
#########

sysdir = "systems"
sysname = "KDA"
runname = "mdrun"

def setup(sysdir, sysname):
    mdsys = MmSystem(sysdir, sysname)
    logger.info(f"System directory set up at: {mdsys.root}")
    ligand_resname = "ANP"
    prepare_protein_ligand_system(
        pdb_file=Path(sysdir) / f"{sysname}.pdb",
        ligand_resname=ligand_resname,
        ligand_smiles=None,
        ligand_sdf=Path(sysdir) / f"{ligand_resname}.sdf",
        output_dir=mdsys.root 
    )


def prepare_protein_ligand_system(
    pdb_file,
    ligand_resname="UNK",
    ligand_smiles=None,
    ligand_sdf=None,
    protein_ff="amber14-all.xml",
    water_ff="amber14/tip3pfb.xml",
    openff_version="openff-2.2.1.offxml",
    add_solvent=True,
    box_padding=1.0 * unit.nanometer,
    ionic_strength=0.10 * unit.molar,
    nonbonded_method=app.PME,
    nonbonded_cutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    output_dir=None
):
    """
    Prepare an OpenMM system from a protein-ligand complex PDB file.
    
    This function separates the protein and ligand, parameterizes them with 
    appropriate force fields (AMBER for protein, OpenFF for ligand), and 
    combines them into a single OpenMM system while maintaining the ligand's 
    original position.
    
    Parameters
    ----------
    pdb_file : str or Path
        Path to the PDB file containing the protein-ligand complex
    ligand_resname : str, default="UNK"
        Residue name of the ligand in the PDB file
    ligand_smiles : str, optional
        SMILES string of the ligand. If None, will attempt to extract from PDB
    ligand_sdf : str or Path, optional
        Path to SDF file with ligand structure for better geometry/bonding info
    protein_ff : str, default="amber14-all.xml"
        OpenMM force field XML file for the protein
    water_ff : str, default="amber14/tip3pfb.xml"
        OpenMM force field XML file for water
    openff_version : str, default="openff-2.2.1.offxml"
        OpenFF force field version for the ligand
    add_solvent : bool, default=True
        Whether to add solvent and ions to the system
    box_padding : Quantity, default=1.0*nm
        Padding around the solute when adding solvent box
    ionic_strength : Quantity, default=0.15*M
        Ionic strength for neutralizing ions
    nonbonded_method : app method, default=app.PME
        Method for nonbonded interactions
    nonbonded_cutoff : Quantity, default=1.0*nm
        Cutoff distance for nonbonded interactions
    constraints : app constraints, default=app.HBonds
        Constraints to apply (None, HBonds, AllBonds, HAngles)
    output_dir : str or Path, optional
        Directory to save output files (PDB, XML). If None, returns objects only
        
    Returns
    -------
    system : openmm.System
        The parameterized OpenMM system
    topology : openmm.app.Topology
        The system topology
    positions : list of Vec3
        Atomic positions
    """
    logger.info(f"Preparing protein-ligand system from {pdb_file}")
    pdb_file = Path(pdb_file)
    
    # Load the PDB file
    pdb = app.PDBFile(str(pdb_file))
    logger.info(f"Loaded PDB with {pdb.topology.getNumAtoms()} atoms")
    
    # Separate protein and ligand using MDAnalysis for easier manipulation
    u = mda.Universe(str(pdb_file))
    protein_atoms = u.select_atoms(f"protein or (resname HOH NA CL K MG)")
    ligand_atoms = u.select_atoms(f"resname {ligand_resname}")
    
    logger.info(f"Found {len(protein_atoms)} protein/solvent atoms")
    logger.info(f"Found {len(ligand_atoms)} ligand atoms")
    
    if len(ligand_atoms) == 0:
        raise ValueError(f"No ligand found with resname '{ligand_resname}'")
    
    # Save temporary files for protein and ligand
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    protein_pdb = temp_dir / "protein.pdb"
    ligand_pdb = temp_dir / "ligand.pdb"
    
    protein_atoms.write(str(protein_pdb))
    ligand_atoms.write(str(ligand_pdb))
    
    # Fix protein structure (add missing atoms, terminal groups, etc.)
    logger.info("Fixing protein structure with PDBFixer...")
    fixer = PDBFixer(str(protein_pdb))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # pH 7.0
    
    # Save fixed protein
    fixed_protein_pdb = temp_dir / "protein_fixed.pdb"
    with open(fixed_protein_pdb, 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    logger.info(f"Fixed protein saved to {fixed_protein_pdb}")
    
    # Get ligand molecule for OpenFF parameterization
    if ligand_sdf:
        logger.info(f"Loading ligand from SDF file: {ligand_sdf}")
        ligand_mol = Molecule.from_file(str(ligand_sdf))
    elif ligand_smiles:
        logger.info(f"Creating ligand from SMILES: {ligand_smiles}")
        ligand_mol = Molecule.from_smiles(ligand_smiles, allow_undefined_stereo=True)
        # Generate 3D coordinates and align to PDB positions
        rdmol = ligand_mol.to_rdkit()
        AllChem.EmbedMolecule(rdmol, randomSeed=42)
        ligand_mol = Molecule.from_rdkit(rdmol)
    else:
        logger.info("Attempting to infer ligand molecule from PDB coordinates")
        # Try to create molecule from PDB (may have issues with bond orders)
        try:
            ligand_mol = Molecule.from_file(str(ligand_pdb), file_format='pdb')
        except Exception as e:
            logger.error(f"Failed to load ligand from PDB: {e}")
            raise ValueError(
                "Could not determine ligand molecule. Please provide either "
                "ligand_smiles or ligand_sdf parameter."
            )
    
    # Store original ligand positions from PDB
    original_ligand_positions = [pdb.positions[atom.index] 
                                  for atom in pdb.topology.atoms() 
                                  if atom.residue.name == ligand_resname]
    
    logger.info("Setting up force fields...")
    
    # Create SMIRNOFF template generator for the ligand
    smirnoff_generator = SMIRNOFFTemplateGenerator(molecules=[ligand_mol])
    
    # Create the main force field with protein parameters
    forcefield = app.ForceField(protein_ff, water_ff)
    
    # Register the ligand template generator
    forcefield.registerTemplateGenerator(smirnoff_generator.generator)
    
    # Load fixed protein and combine with ligand
    logger.info("Combining fixed protein with ligand...")
    fixed_protein = app.PDBFile(str(fixed_protein_pdb))
    ligand_pdb_obj = app.PDBFile(str(ligand_pdb))
    
    # Create modeler with fixed protein
    modeller = app.Modeller(fixed_protein.topology, fixed_protein.positions)
    
    # Add ligand to the system
    modeller.add(ligand_pdb_obj.topology, ligand_pdb_obj.positions)
    
    # Add solvent if requested
    if add_solvent:
        logger.info(f"Adding solvent with {box_padding} padding and {ionic_strength} ionic strength")
        modeller.addSolvent(
            forcefield,
            model='tip3p',
            padding=box_padding,
            ionicStrength=ionic_strength,
            positiveIon='Na+',
            negativeIon='Cl-'
        )
        logger.info(f"System now has {modeller.topology.getNumAtoms()} atoms after solvation")
    
    # Create the OpenMM system
    logger.info("Creating OpenMM system...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=constraints,
        rigidWater=True,
        removeCMMotion=True
    )
    
    logger.info(f"System created with {system.getNumParticles()} particles")
    
    # Save outputs if directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDB
        pdb_out = output_dir / "system.pdb"
        with open(pdb_out, 'w') as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
        logger.info(f"Saved system PDB to {pdb_out}")
        
        # Save system XML
        xml_out = output_dir / "system.xml"
        with open(xml_out, 'w') as f:
            f.write(mm.XmlSerializer.serialize(system))
        logger.info(f"Saved system XML to {xml_out}")
    
    # Clean up temporary files
    shutil.rmtree(temp_dir)
    
    logger.info("System preparation complete!")
    
    return system, modeller.topology, modeller.positions


def process_ligand(sysdir, sysname, ligand_name):
    mdsys = MmSystem(sysdir, sysname)
    logger.info("Processing ligand: %s", ligand_name)
    wdir = Path("systems") / ligand_name
    wdir.mkdir(parents=True, exist_ok=True)
    logger.info("Ligand working directory: %s", wdir)
    input_file = Path(sysdir) / f"{ligand_name}.sdf"
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
        padding=1.5 * unit.nanometer,
        ionicStrength=0.1 * unit.molar,
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
    exit()
    # forcefield = ForceField("openff-2.2.1.offxml")
    # topology = ligand.to_topology()
    # logger.info("Creating interchange (OpenFF -> OpenMM)")
    # ic = forcefield.create_interchange(topology)
    # pdb_out = wdir / "ligand.pdb"
    # logger.info("Writing ligand PDB: %s", pdb_out)
    # ic.to_pdb(pdb_out)
    # logger.info("Creating OpenMM System")
    # mm_sys = ic.to_openmm_system()
    # xml_out = wdir / "ligand_sys.xml"
    # logger.info("Writing OpenMM System XML: %s", xml_out)
    # _save_system_to_xml(mm_sys, xml_out)
    # ic.to_top(wdir / "ligand.itp")
    # ic.to_mdp(wdir / "ligand.mdp")
    # ic.to_gro(wdir / "ligand.gro")
    # logger.info("Done processing ligand: %s", ligand_name)


def md_npt(sysdir, sysname, runname, CudaDeviceIndex="0"): 
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
    get_platform_info()
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
    convert_trajectories(top, trajs, out_top, out_traj, selection=SELECTION, start=0000, stop=None, step=10, fit=True)
    logger.info("Done!")


if __name__ == "__main__":
    # setup(sysdir, sysname)
    # md_npt(sysdir, sysname, runname, CudaDeviceIndex="0")
    trjconv(sysdir, sysname, runname)