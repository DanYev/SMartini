import logging
import sys
import tempfile
from pathlib import Path
import shutil

import numpy as np
import MDAnalysis as mda
import openmm as mm
from openmm import app, unit
from openff.toolkit import ForceField, Molecule
from openff.interchange import Interchange
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem
from rdkit.Chem import AllChem
from pdbfixer import PDBFixer

import smartini
from smartini.config import CFG

logger = logging.getLogger(__name__)
smartini.setup_logging(level=logging.INFO)

"""Atomistic MD setup and production workflow for protein+ligand systems.

Pipeline:
1. Clean input PDB and separate protein from ligand.
2. Parameterize protein (AMBER ff) and ligand (OpenFF) separately.
3. Combine, solvate, and build the full OpenMM system.
4. Run minimization, heating, equilibration, and production NPT.
5. Export selected trajectory/topology for downstream analysis.

Also includes standalone ligand workflow (``process_ligand``) inherited
from the original SMartini pipeline.
"""

# ---------------------------------------------------------------------------
# Protein + Ligand system configuration
# ---------------------------------------------------------------------------
SYSDIR = Path("protein_systems").resolve()
SYSNAME = "1TQN"                        # system name / PDB basename
LIGAND_RESNAME = "HEM"                   # residue name of the ligand in the PDB
LIGAND_SDF = Path("examples/HEM/HEM.sdf")  # SDF for OpenFF parameterization
RUNNAME = "aa_md"                        # subdirectory for AA MD run

# AA MD settings
AA_TEMPERATURE = 300.0                   # Kelvin
AA_PRESSURE = 1.0                        # bar
AA_GAMMA = 1.0                           # friction coefficient (1/ps)
AA_TIMESTEP_FS = 2.0                     # femtoseconds
AA_TOTAL_STEPS = int(5e6)                # 10 ns at 2 fs
AA_TRJ_NOUT = 10000                      # trajectory output interval
AA_LOG_NOUT = 10000                      # log output interval
AA_CHK_NOUT = 50000                      # checkpoint interval
AA_SELECTION = "all"                     # MDAnalysis selection for trajectory output

# Standalone-ligand globals (from CFG, used by process_ligand)
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
    
    # --- Pre-flight check: SMIRNOFF reference molecule vs PDB residue ---
    # SMIRNOFF does exact graph matching — the number of atoms in the
    # reference molecule (from SDF/SMILES) must equal the number of atoms
    # in the PDB residue.  If they differ (e.g. because the PDB was
    # pre-processed and has extra/missing hydrogens), matching will fail.
    ref_n_atoms = ligand_mol.n_atoms
    pdb_lig_n_atoms = len(ligand_atoms)
    if ref_n_atoms != pdb_lig_n_atoms:
        msg = (
            f"Atom count mismatch between SMIRNOFF reference molecule "
            f"({ref_n_atoms} atoms from SDF/SMILES) and PDB residue "
            f"'{ligand_resname}' ({pdb_lig_n_atoms} atoms).\n"
            f"SMIRNOFF requires an exact match.  Possible fixes:\n"
            f"  1. Use a PDB that has NOT been pre-processed "
            f"(no added hydrogens, no missing atoms).\n"
            f"  2. Provide an SDF/SMILES whose protonation state matches "
            f"the PDB residue exactly.\n"
            f"  3. Use GAFFTemplateGenerator instead of "
            f"SMIRNOFFTemplateGenerator (more forgiving)."
        )
        logger.error(msg)
        raise ValueError(msg)
    logger.info("Pre-flight OK: %d atoms in reference molecule matches PDB residue", ref_n_atoms)

    # Load fixed protein and combine with ligand
    logger.info("Combining fixed protein with ligand...")
    fixed_protein = app.PDBFile(str(fixed_protein_pdb))
    ligand_pdb_obj = app.PDBFile(str(ligand_pdb))
    modeller = app.Modeller(fixed_protein.topology, fixed_protein.positions)
    modeller.add(ligand_pdb_obj.topology, ligand_pdb_obj.positions)
    with open("temp.pdb", 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    exit()
    
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


def run_aa_md(system, topology, positions, output_dir,
              temperature=AA_TEMPERATURE,
              pressure=AA_PRESSURE,
              gamma=AA_GAMMA,
              timestep_fs=AA_TIMESTEP_FS,
              total_steps=AA_TOTAL_STEPS,
              log_nout=AA_LOG_NOUT,
              trj_nout=AA_TRJ_NOUT,
              chk_nout=AA_CHK_NOUT):
    """Run AA MD in OpenMM: minimize, heat, equilibrate, produce trajectory.

    Parameters
    ----------
    system : openmm.System
    topology : openmm.app.Topology
    positions : list of Vec3
    output_dir : Path
        Directory for output files (log, xtc, chk, xml checkpoint).
    temperature, pressure, gamma, timestep_fs, total_steps : float/int
    log_nout, trj_nout, chk_nout : int
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Integrator ---
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        gamma / unit.picosecond,
        1.0 * unit.femtoseconds,  # small step for minimization/heat-up
    )
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # --- Minimization ---
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=10000, tolerance=10 * unit.kilojoule_per_mole)
    state = simulation.context.getState(getEnergy=True)
    logger.info(f"  Energy after minimization: {state.getPotentialEnergy()}")

    # --- Heat-up (NVT) ---
    logger.info("Heating up...")
    n_cycles = 50
    steps_per_cycle = 200
    for i in range(n_cycles):
        current_temp = (i + 1) * temperature * unit.kelvin / n_cycles
        simulation.integrator.setTemperature(current_temp)
        simulation.step(steps_per_cycle)

    # --- Equilibration (NPT) ---
    logger.info("Equilibrating (NPT)...")
    barostat = mm.MonteCarloBarostat(pressure * unit.bar, temperature * unit.kelvin)
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    simulation.integrator.setTemperature(temperature * unit.kelvin)
    simulation.step(5000)

    # --- Production ---
    logger.info("Production MD (%d steps)...", total_steps)
    simulation.integrator.setStepSize(timestep_fs * unit.femtoseconds)
    simulation.context.reinitialize(preserveState=True)

    # Reporters
    log_file = output_dir / "md.log"
    chk_file = output_dir / "md.chk"
    xtc_file = output_dir / "md.xtc"

    reporters = [
        app.StateDataReporter(
            str(log_file), log_nout,
            step=True, time=True,
            potentialEnergy=True, kineticEnergy=True,
            temperature=True, speed=True,
            append=False,
        ),
        app.StateDataReporter(
            sys.stderr, log_nout,
            step=True, time=True,
            potentialEnergy=True, kineticEnergy=True,
            temperature=True, speed=True,
            append=False,
        ),
        app.CheckpointReporter(str(chk_file), chk_nout),
    ]

    # XTC reporter with atom selection
    if AA_SELECTION == "all":
        atom_subset = None
    else:
        # Use the system.pdb written by prepare_protein_ligand_system
        system_pdb = output_dir.parent / "system.pdb"
        u_sel = mda.Universe(str(system_pdb))
        atom_subset = u_sel.select_atoms(AA_SELECTION).indices.tolist()
        logger.info("XTC subset: %d atoms", len(atom_subset))

    reporters.append(
        app.XTCReporter(str(xtc_file), trj_nout,
                        append=False, atomSubset=atom_subset)
    )

    simulation.reporters = reporters
    simulation.step(int(total_steps))

    # Save final state as XML
    state_xml = output_dir / "md.xml"
    with open(str(state_xml), "w") as f:
        f.write(mm.XmlSerializer.serialize(system))
    logger.info("Production MD complete. Final system saved to %s", state_xml)

    return simulation


def trjconv_aa(top_pdb, traj_xtc, output_dir,
               start=0, stop=None, step=1, fit=True,
               selection=AA_SELECTION):
    """Post-process AA trajectory: fit, subsample, and write outputs.

    Parameters
    ----------
    top_pdb : Path
        Reference PDB for topology and fitting.
    traj_xtc : Path
        Input trajectory.
    output_dir : Path
        Directory for ``topology.pdb`` and ``samples.xtc``.
    start, stop, step : int or None
    fit : bool
    selection : str
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_top = output_dir / "topology.pdb"
    out_traj = output_dir / "samples.xtc"

    universe = mda.Universe(str(top_pdb), str(traj_xtc))
    atom_group = universe.select_atoms(selection)
    logger.info("Converting trajectory: %d atoms selected", atom_group.n_atoms)

    if fit:
        ref_universe = mda.Universe(str(top_pdb))
        ref_atoms = ref_universe.select_atoms(selection)
        from MDAnalysis.transformations import fit_rot_trans
        universe.trajectory.add_transformations(
            fit_rot_trans(atom_group, ref_atoms)
        )

    atom_group.write(str(out_top))
    with mda.Writer(str(out_traj), n_atoms=atom_group.n_atoms) as writer:
        for _ in universe.trajectory[start:stop:step]:
            writer.write(atom_group)
    logger.info("Trajectory conversion complete: %s, %s", out_top, out_traj)


# Standalone-ligand workflow helpers (kept for backwards compatibility)

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



if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Protein + Ligand AA MD workflow for 1TQN with HEM
    # -----------------------------------------------------------------------
    pdb_input = SYSDIR / SYSNAME / "inpdb.pdb"
    aa_out_dir = SYSDIR / SYSNAME / RUNNAME

    logger.info("=" * 60)
    logger.info("Building protein+ligand system for %s", SYSNAME)
    logger.info("  PDB: %s", pdb_input)
    logger.info("  Ligand: %s", LIGAND_RESNAME)
    logger.info("  Output: %s", aa_out_dir)
    logger.info("=" * 60)

    # Step 1: Prepare the system
    system, topol, pos = prepare_protein_ligand_system(
        pdb_file=pdb_input,
        ligand_resname=LIGAND_RESNAME,
        ligand_sdf=LIGAND_SDF,
        add_solvent=True,
        box_padding=1.0 * unit.nanometer,
        ionic_strength=0.10 * unit.molar,
        output_dir=aa_out_dir,
    )

    # Step 2: Run AA MD (minimize, heat, equilibrate, production)
    run_aa_md(
        system, topol, pos,
        output_dir=aa_out_dir,
    )

    # Step 3: Post-process trajectory
    md_pdb = aa_out_dir / "system.pdb"
    md_xtc = aa_out_dir / "md.xtc"
    trjconv_aa(
        md_pdb, md_xtc, aa_out_dir,
        start=0, stop=None, step=1, fit=True,
    )
