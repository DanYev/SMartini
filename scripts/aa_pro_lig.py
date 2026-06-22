import logging
import sys
import tempfile
from pathlib import Path
import shutil
import pickle
import hashlib

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
PROTEIN = "1TQN"
LIGAND_RESNAME = "HEM"                   # residue name of the ligand in the PDB
SYSNAME = f"{PROTEIN}_aa"                   # system name / PDB basename
LIGAND_SDF = Path(f"examples/{LIGAND_RESNAME}/{LIGAND_RESNAME}.sdf")  # SDF for OpenFF parameterization
RUNNAME = "mdrun_1"                        # subdirectory for AA MD run
SYS_OUT_DIR = SYSDIR / SYSNAME                      # system topology & PDB
MD_OUT_DIR = SYSDIR / SYSNAME / "mdruns" / RUNNAME  # trajectory & logs
PDB_INPUT = SYSDIR / f"{PROTEIN}.pdb"               # input PDB file

# MD settings
TEMPERATURE = 300.0                   # Kelvin
PRESSURE = 1.0                        # bar
GAMMA = 1.0                           # friction coefficient (1/ps)
TIMESTEP_FS = 2.0                     # femtoseconds
TOTAL_STEPS = int(5e7)                # 10 ns at 2 fs for 5e6
TRJ_NOUT = 10000                      # trajectory output interval
LOG_NOUT = 10000                      # log output interval
CHK_NOUT = 50000                      # checkpoint interval
SELECTION = "all"                     # MDAnalysis selection for trajectory output



def prepare_protein_ligand_system(
    pdb_file=PDB_INPUT,
    ligand_resname=LIGAND_RESNAME,
    ligand_smiles=None,
    ligand_sdf=LIGAND_SDF,
    protein_ff="amber14-all.xml",
    water_ff="amber14/tip3pfb.xml",
    openff_version="openff-2.2.1.offxml",
    add_solvent=True,
    box_padding=1.0 * unit.nanometer,
    ionic_strength=0.10 * unit.molar,
    nonbonded_method=app.PME,
    nonbonded_cutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    fit_ligand=True,
    output_dir=SYS_OUT_DIR
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
    fit_ligand : bool, default=True
        If True, translate OpenFF ligand centroid to match the original PDB ligand.
        If False, use OpenFF positions as-is.
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
    
    # Save fixed protein (temp copy for internal use)
    fixed_protein_pdb = temp_dir / "protein_fixed.pdb"
    with open(fixed_protein_pdb, 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    # Save to output dir if requested
    if output_dir:
        protein_out = Path(output_dir) / "protein.pdb"
        protein_out.parent.mkdir(parents=True, exist_ok=True)
        with open(protein_out, 'w') as f:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        logger.info(f"Saved fixed protein PDB to {protein_out}")

    # Create the AMBER force field for protein + water
    forcefield = app.ForceField(protein_ff, water_ff)

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

    # Register the ligand FF template on the AMBER force field so that
    # forcefield.createSystem() can parameterize the ligand atoms.
    smirnoff = SMIRNOFFTemplateGenerator(molecules=[ligand_mol])
    forcefield.registerTemplateGenerator(smirnoff.generator)

    # Generate ligand topology *from the SDF molecule* via OpenFF Interchange.
    # This guarantees the atom count matches the SMIRNOFF reference molecule
    # (the raw PDB residue may have a different protonation / H count).
    logger.info("Generating ligand topology from SDF via OpenFF Interchange...")
    cached = _load_interchange_cache(ligand_sdf or ligand_smiles or "ligand",
                                     openff_version, work_dir=output_dir)
    if cached is not None:
        interchange, ligand_topology, ligand_positions = cached
    else:
        ff = ForceField(openff_version)
        interchange = Interchange.from_smirnoff(ff, ligand_mol.to_topology())
        ligand_topology = interchange.to_openmm_topology()
        ligand_positions = interchange.positions.to_openmm()
        _save_interchange_cache(ligand_sdf or ligand_smiles or str(temp_dir / "ligand"),
                                openff_version, interchange,
                                ligand_topology, ligand_positions, work_dir=output_dir)


    # Save generated ligand PDB to output dir if requested
    if output_dir:
        ligand_out = Path(output_dir) / "ligand.pdb"
        ligand_out.parent.mkdir(parents=True, exist_ok=True)
        with open(ligand_out, 'w') as f:
            app.PDBFile.writeFile(ligand_topology, ligand_positions, f, keepIds=True)
        logger.info(f"Saved generated ligand PDB to {ligand_out}")

    # Load fixed protein and combine with ligand
    logger.info("Combining fixed protein with ligand...")
    fixed_protein = app.PDBFile(str(fixed_protein_pdb))

    modeller = app.Modeller(fixed_protein.topology, fixer.positions)
    modeller.add(ligand_topology, ligand_atoms.positions * unit.angstrom)

    # Save combined protein+ligand structure (before solvation)
    if output_dir:
        complex_out = Path(output_dir) / "complex.pdb"
        complex_out.parent.mkdir(parents=True, exist_ok=True)
        with open(complex_out, 'w') as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
        logger.info(f"Saved combined complex PDB to {complex_out}")
    
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

        # Save system PDB
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


def run_md(output_dir=MD_OUT_DIR,
           temperature=TEMPERATURE,
           pressure=PRESSURE,
           gamma=GAMMA,
           timestep_fs=TIMESTEP_FS,
           total_steps=TOTAL_STEPS,
           log_nout=LOG_NOUT,
           trj_nout=TRJ_NOUT,
           chk_nout=CHK_NOUT):
    """Run MD in OpenMM: minimize, heat, equilibrate, produce trajectory.

    Loads ``system.xml`` and ``system.pdb`` from ``SYS_OUT_DIR``.

    Parameters
    ----------
    output_dir : Path
        Directory for output files (log, xtc, chk, xml checkpoint).
    temperature, pressure, gamma, timestep_fs, total_steps : float/int
    log_nout, trj_nout, chk_nout : int
    """
    xml_path = SYS_OUT_DIR / "system.xml"
    pdb_path = SYS_OUT_DIR / "system.pdb"
    logger.info("Loading system from %s", xml_path)
    with open(str(xml_path), 'r') as f:
        system = mm.XmlSerializer.deserialize(f.read())
    pdb = app.PDBFile(str(pdb_path))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Integrator ---
    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        gamma / unit.picosecond,
        1.0 * unit.femtoseconds,  # small step for minimization/heat-up
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # --- Minimization ---
    logger.info("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=10000, tolerance=10 * unit.kilojoule_per_mole / unit.nanometer)
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
    if SELECTION == "all":
        atom_subset = None
    else:
        u_sel = mda.Universe(str(pdb_path))
        atom_subset = u_sel.select_atoms(SELECTION).indices.tolist()
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


def trjconv(top_pdb=SYS_OUT_DIR / "system.pdb",
            traj_xtc=MD_OUT_DIR / "md.xtc",
            output_dir=MD_OUT_DIR,
            start=0, stop=None, step=1, fit=True,
            selection=SELECTION):
    """Post-process trajectory: fit, subsample, and write outputs.

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _cache_key(ligand_path, openff_version):
    """Build a deterministic cache key from the SDF/SMILES source and FF version."""
    raw = f"{ligand_path}:{openff_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_dir(work_dir=None):
    """Return the cache directory, creating it if needed."""
    if work_dir is not None:
        d = Path(work_dir) / ".smartini_cache"
    else:
        d = Path(tempfile.gettempdir()) / "smartini_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_interchange_cache(ligand_path, openff_version, interchange,
                            ligand_topology, ligand_positions, work_dir=None):
    """Pickle Interchange results to disk so they can be reloaded later."""
    cdir = _cache_dir(work_dir)
    key = _cache_key(ligand_path, openff_version)
    cache_file = cdir / f"interchange_{key}.pkl"
    data = {
        "interchange": interchange,
        "ligand_topology": ligand_topology,
        "ligand_positions": ligand_positions,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Cached Interchange results to %s", cache_file)


def _load_interchange_cache(ligand_path, openff_version, work_dir=None):
    """Load previously cached Interchange results, or return None on miss."""
    cdir = _cache_dir(work_dir)
    key = _cache_key(ligand_path, openff_version)
    cache_file = cdir / f"interchange_{key}.pkl"
    if not cache_file.exists():
        return None
    logger.info("Loading cached Interchange results from %s", cache_file)
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data["interchange"], data["ligand_topology"], data["ligand_positions"]


if __name__ == "__main__":
    prepare_protein_ligand_system()
    run_md()
    trjconv()
