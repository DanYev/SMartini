from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LigParConfig:
    # Identity / layout
    molname: str = "ANP"
    systems_dir: Path = Path("systems")
    ligands_dir: Path = Path("ligands")

    # AutoMartini / mapping
    n_beads: int = 10

    # Common subfolders
    aa_sysname: str = "aa_md_pro"
    cg_sysname: str = "cg_md"
    cg_runname: str = "mdrun"

    # Sampling defaults
    cg_traj_stop: int = 2000

    # Fitting defaults
    temperature: float = 300.0

    # Type-9 dihedral (Gromacs) fitting parameters
    type9_max_n: int = 18
    type9_bins: int = 360
    type9_min_prob: float = 1e-6

    # Post-fit filtering / topology cleanup
    constraint_k_cutoff: float = 20000.0
    angle_k_cutoff: float = 0.0
    dihedral_k_cutoff: float = 0.0

    # Refinement guardrails
    refine_max_k_scale: float = 25.0
    refine_dihedral_shift_scale: float = 1.0

    def wdir(self) -> Path:
        return self.systems_dir / self.molname

    def mapping_dir(self) -> Path:
        return self.wdir() / "mapping"

    def aa_dir(self) -> Path:
        return self.wdir() / self.aa_sysname

    def cg_dir(self) -> Path:
        return self.wdir() / self.cg_sysname


CFG = LigParConfig()

# Configure logging once for all scripts
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    force=True,
)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a module."""
    return logging.getLogger(name)
