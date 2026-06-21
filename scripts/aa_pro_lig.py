import logging
import sys
import smartini
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit
from openff.toolkit import ForceField, Molecule, Topology 
from openff.interchange import Interchange
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from smartini.config import CFG

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
    forcefield = app.ForceField("amber19-all.xml", "amber19/opc.xml")
    # Ligand FF
    forcefield.registerTemplateGenerator(smirnoff.generator)
    ff = ForceField("openff-2.1.0.offxml")
    interchange = Interchange.from_smirnoff(ff, ligand.to_topology())
    ligand_topology = interchange.to_openmm_topology()
    ligand_positions = interchange.positions.to_openmm()
    model = app.Modeller(ligand_topology, ligand_positions)
    logger.info("Adding solvent and ions")
    model.addSolvent(forcefield, 
        model='opc', 
        boxShape='dodecahedron', #  ‘cube’, ‘dodecahedron’, and ‘octahedron’
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
        ewaldErrorTolerance=1e-5,
        rigidWater=True,
    )
    _save_system_to_xml(system, system_xml)
    logger.info(f'Saving reference PDB with selection: {CFG.aa_selection}')
    mda.Universe(system_pdb).select_atoms(CFG.aa_selection).write(str(aa_dir / "md.pdb"))


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
    # process_ligand()
    md_npt()
    trjconv()
