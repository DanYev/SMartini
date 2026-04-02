from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass()
class LigParConfig:
    # ============================================================================
    # Identity, coarse graining and partitioning settings
    # ============================================================================
    molname: str = "CLA"
    specify_beads: Optional[list[list[int]]] = None
    # specify_beads: tuple[list[int]] = ([4, 5, 8],) # FOR CLA
    # specify_beads: tuple[list[int]] = ([3, 6], ) # FOR DMBI
    # specify_beads: tuple[list[int]] = ([9, 15], ) # FOR THC
    n_beads: Optional[int] = None 
    use_vsites: bool = True
    symmetrize_rings: bool = False
    keep_rings_together: bool = False
    max_combs_merged: int = 1000
    max_ring_len: int = 12  # Large rings are usually not aromatic and can be broken up
    max_mappings_to_keep: int = 500  # Keep top mappings to avoid combinatorial explosion
    max_bead_size: int = 4
    max_ring_bead_size: int = 3

    # ============================================================================
    # Working folders
    # ============================================================================
    systems_dir: Path = Path("systems")
    ligands_dir: Path = Path("ligands")
    wdir: Path = systems_dir / molname
    mol_dir: Path = wdir / "molecule"
    aa_sysname: str = "aa_md"
    cg_sysname: str = "cg_md"
    cg_runname: str = "mdrun"
    aa_dir: Path = wdir / aa_sysname
    cg_dir: Path = wdir / cg_sysname

    # ============================================================================
    # AA MD settings (All-Atom Molecular Dynamics)
    # ============================================================================
    aa_selection: str = "resname UNK"
    aa_gamma: float = 1.0  # Friction coefficient (1/picosecond)
    aa_pressure_bar: float = 1.0  # Pressure in bar
    aa_timestep_fs: float = 2.0  # Timestep in femtoseconds
    aa_total_steps: int = int(1e6)  # Total MD steps (1e6 = 2 ns with 2 fs timestep)
    aa_trj_nout: int = 1000  # Trajectory output frequency (frames every N steps)
    aa_log_nout: int = 10000  # Log output frequency (every N steps)
    aa_chk_nout: int = 100000  # Checkpoint output frequency (every N steps)

    # ============================================================================
    # CG MD settings (Coarse-Grained Molecular Dynamics)
    # ============================================================================
    cg_dt: float = 0.020  # Timestep in picoseconds
    cg_total_time_ns: float = 1000.0  # Total simulation time in nanoseconds
    cg_traj_stop: int = 2000  # Trajectory sampling cutoff
    cg_selection: str = "all"

    # ============================================================================
    # Refinement settings (Fitting, filtering, and optimization)
    # ============================================================================
    # Fitting /filtering defaults 
    temperature: float = 300.0
    bond_k_lower: float = 2000.0
    bond_k_upper: float = 20000.0
    angle_k_lower: float = 3.0
    angle_k_upper: float = 2000.0
    dihedral_k_lower: float = 0.0
    dihedral_k_upper: float = 1000.0
    ill_defined_angle_cutoff: float = 155.0
    use_type11_for_linear: bool = True  # Whether to use type 11 dihedrals for linear angles
    type9_max_n: int = 6
    nbins: int = 120
    min_prob: float = 1e-12
    fc_scale: float = 0.5  # Scaling factor for initial force constants to roughly account for coupling of the potentials
    # Refinement guardrails
    alpha_max: float = 0.3
    alpha_min: float = 0.01


CFG = LigParConfig()


