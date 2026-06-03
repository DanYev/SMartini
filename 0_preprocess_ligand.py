import logging
import sys
import smartini
import MDAnalysis as mda

from pathlib import Path
from openmm import app, unit
from openff.toolkit import ForceField, Molecule, Topology 
from openff.interchange import Interchange
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from rdkit import Chem
from smartini.config import CFG

logger = logging.getLogger(__name__)
smartini.setup_logging(level=logging.INFO)

# Use configuration from config.py
ligand_name = CFG.molname
sysdir = CFG.systems_dir
wdir = CFG.wdir
aa_dir = CFG.aa_dir
system_pdb = aa_dir / "system.pdb"
system_xml = aa_dir / "system.xml"


def print_ligand_info(sdf_path: Path, ligand_name: str) -> None:
    """Print chemical information about the ligand to screen and ``info.txt``.

    Reads the ligand SDF with RDKit and reports:
    - molecular formula, weight, atom/rotatable-bond/heavy-atom counts
    - H-bond donors / acceptors, logP, TPSA
    - ring count and, most importantly, the atom-index list of each ring
    """
    # --- Load molecule from SDF ------------------------------------------------
    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    if mol is None:
        logger.error("RDKit could not read ligand from %s", sdf_path)
        return

    # Add hydrogens if missing (RDKit sometimes strips them from SDF)
    mol = Chem.AddHs(mol) if mol.GetNumAtoms() == mol.GetNumHeavyAtoms() else mol

    # --- Gather chemical descriptors -------------------------------------------
    from rdkit.Chem import Descriptors, rdMolDescriptors

    formula       = Chem.rdMolDescriptors.CalcMolFormula(mol)
    mw            = Descriptors.MolWt(mol)
    num_atoms     = mol.GetNumAtoms()
    num_heavy     = mol.GetNumHeavyAtoms()
    num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_hbd       = rdMolDescriptors.CalcNumHBD(mol)
    num_hba       = rdMolDescriptors.CalcNumHBA(mol)
    logp          = Descriptors.MolLogP(mol)
    tpsa          = rdMolDescriptors.CalcTPSA(mol)
    charge        = Chem.GetFormalCharge(mol)

    # --- Ring information (THE key output) -------------------------------------
    ring_info = mol.GetRingInfo()
    rings     = ring_info.AtomRings()   # tuple of tuples of atom indices

    # Build the ring-listing lines
    ring_lines: list[str] = []
    for i, ring in enumerate(rings, start=1):
        # RDKit indices are 0-based → report 1-based for human readability
        atoms_str = ", ".join(str(idx + 1) for idx in ring)
        ring_lines.append(f"  Ring {i}: atoms {atoms_str}  (size={len(ring)})")

    num_rings = len(rings)

    # --- Assemble output text --------------------------------------------------
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"  Ligand Information: {ligand_name}")
    lines.append("=" * 60)
    lines.append(f"  Source file      : {sdf_path}")
    lines.append(f"  Molecular formula: {formula}")
    lines.append(f"  Molecular weight : {mw:.3f} Da")
    lines.append(f"  Formal charge    : {charge}")
    lines.append(f"  Total atoms      : {num_atoms}")
    lines.append(f"  Heavy atoms      : {num_heavy}")
    lines.append(f"  Rotatable bonds  : {num_rot_bonds}")
    lines.append(f"  H-bond donors    : {num_hbd}")
    lines.append(f"  H-bond acceptors : {num_hba}")
    lines.append(f"  logP             : {logp:.3f}")
    lines.append(f"  TPSA             : {tpsa:.2f} Å²")
    lines.append(f"  Number of rings  : {num_rings}")
    lines.append("-" * 60)
    if ring_lines:
        lines.append("  Ring atom lists (1-based indices):")
        lines.extend(ring_lines)
    else:
        lines.append("  (no rings detected)")
    lines.append("=" * 60)

    output = "\n".join(lines)

    # --- Print to screen -------------------------------------------------------
    logger.info("Ligand info for %s:\n%s", ligand_name, output)

    # --- Write info.txt --------------------------------------------------------
    info_path = sdf_path.parent / "info.txt"
    with open(info_path, "w", encoding="utf-8") as fh:
        fh.write(output)
        fh.write("\n")
    logger.info("Ligand info written to %s", info_path)


def process_to_ff():
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

    # Print chemical information (rings, formula, etc.) to screen + info.txt
    print_ligand_info(input_file, ligand_name)

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


if __name__ == "__main__":
    print_ligand_info(wdir / f"{ligand_name}.sdf", ligand_name)
