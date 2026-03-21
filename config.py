from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass()
class LigParConfig:
    # Identity / layout
    molname: str = "ANP"
    specify_beads: list[list[int]] = None
    # specify_beads: tuple[list[int]] = ([4, 5, 8],) # FOR CLA
    # specify_beads: tuple[list[int]] = ([3, 6], ) # FOR DMBI
    # specify_beads: tuple[list[int]] = ([9, 15], ) # FOR THC
    max_combs_merged: int = 1000
    n_beads: Optional[int] = None  # if None, will be determined by AutoMartini
    use_vsites: bool = False
    symmetrize_rings: bool = True

    systems_dir: Path = Path("systems")
    ligands_dir: Path = Path("ligands")

    wdir: Path = systems_dir / molname
    mol_dir: Path = wdir / "molecule"

    # Common subfolders
    aa_sysname: str = "aa_md"
    cg_sysname: str = "cg_md"
    cg_runname: str = "mdrun"
    aa_dir: Path = wdir / aa_sysname
    cg_dir: Path = wdir / cg_sysname

    # Selections
    aa_selection: str = "resname UNK"
    cg_selection: str = "all"

    # Sampling defaults
    cg_traj_stop: int = 2000

    # Fitting defaults
    temperature: float = 300.0
    fc_scale: float = 0.5  # Scaling factor for force constants to roughly account for coupling of the potentials

    # Type-9 dihedral (Gromacs) fitting parameters
    type9_max_n: int = 6
    type9_bins: int = 120
    type9_min_prob: float = 1e-12

    # Post-fit filtering / topology cleanup
    constraint_k_cutoff: float = 50000.0
    bond_lower_cutoff: float = 4000.0
    bond_upper_cutoff: float = 50000.0
    angle_k_lower_cutoff: float = 3.0
    angle_k_upper_cutoff: float = 2000.0
    dihedral_k_lower_cutoff: float = 0.0
    dihedral_k_upper_cutoff: float = 1000.0
    angle_cutoff: float = 155.0

    # Refinement guardrails
    alpha_max: float = 0.25
    alpha_min: float = 0.01
    refine_max_k_scale: float = 25.0
    refine_dihedral_shift_scale: float = 1.0

CFG = LigParConfig()

# Configure logging once for all scripts
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    force=True,
)


