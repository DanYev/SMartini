from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LigParConfig:
    # Identity / layout
    molname: str = "CLA"
    max_combs_merged: int = 100
    n_beads: Optional[int] = None  # if None, will be determined by AutoMartini

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
    fc_scale: float = 0.8 # Scaling factor for force constants to roughly account for coupling of the potentials

    # Type-9 dihedral (Gromacs) fitting parameters
    type9_max_n: int = 6
    type9_bins: int = 360
    type9_min_prob: float = 1e-6

    # Post-fit filtering / topology cleanup
    constraint_k_cutoff: float = 50000.0
    angle_k_cutoff: float = 20.0
    dihedral_k_cutoff: float = 0.0
    angle_cutoff: float = 150.0

    # Refinement guardrails
    refine_max_k_scale: float = 25.0
    refine_dihedral_shift_scale: float = 1.0

CFG = LigParConfig()

# Configure logging once for all scripts
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    force=True,
)


