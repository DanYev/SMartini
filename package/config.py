"""Runtime configuration for the ligand fitting pipeline.

Expected inputs by default:
- Required SDF: <systems_dir>/<MOLNAME>/<MOLNAME>.sdf
- Optional config overrides: <systems_dir>/<MOLNAME>/config.yml (or config.yaml)

Set SM_MOLNAME to choose the molecule name and optionally set SM_CONFIG_YML
to point to a specific YAML file.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass()
class SMConfig:
    # ============================================================================
    # Identity, coarse graining and partitioning settings
    # ============================================================================
    molname: str = "HEM"
    smiles: Optional[str] = None
    specify_beads: Optional[list[list[int]]] = None
    n_beads: Optional[int] = None 
    use_vsites: bool = False
    symmetrize_rings: list[int] = None # list of which rings to symmetrize (from info.txt)
    keep_rings_together: bool = True
    max_combs_merged: int = 1000
    max_ring_len: int = 12  # Large rings are usually not aromatic and can be broken up
    max_mappings_to_keep: int = 500  # Keep top mappings to avoid combinatorial explosion
    max_bead_size: int = 4
    max_ring_bead_size: int = 3

    # ============================================================================
    # Working folders
    # ============================================================================
    systems_dir: Path = Path("examples")
    wdir: Path = systems_dir / molname
    mol_dir: Path = wdir
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
    aa_total_steps: int = int(1e7)  # Total MD steps (1e7 = 20 ns with 2 fs timestep)
    aa_trj_nout: int = 10000  # Trajectory output frequency (frames every N steps)
    aa_log_nout: int = 10000  # Log output frequency (every N steps)
    aa_chk_nout: int = 100000  # Checkpoint output frequency (every N steps)

    # ============================================================================
    # CG MD settings (Coarse-Grained Molecular Dynamics)
    # ============================================================================
    cg_dt: float = 0.010  # Timestep in picoseconds
    cg_total_time_ns: float = 1000.0  # Total simulation time in nanoseconds
    cg_traj_stop: int = 2000  # Trajectory sampling cutoff
    cg_selection: str = "all"

    # ============================================================================
    # Refinement settings (Fitting, filtering, and optimization)
    # ============================================================================
    # Fitting /filtering defaults 
    temperature: float = 300.0
    bond_k_lower: float = 2000.0
    bond_k_upper: float = 50000.0
    angle_k_lower: float = 3.0
    angle_k_upper: float = 2000.0
    dihedral_k_lower: float = 0.0
    dihedral_k_upper: float = 1000.0
    ill_defined_angle_cutoff: float = 160.0
    type9_max_n: int = 6
    use_type11_for_linear: bool = True  # Whether to use type 11 dihedrals for linear angles
    scale_by_sin3_for_type11: bool = False  # Whether to scale type 11 dihedrals by sin^3(theta)
    nbins: int = 120
    min_prob: float = 1e-12
    fc_scale: float = 0.3  # Scaling factor for initial force constants to roughly account for coupling of the potentials
    # Refinement guardrails
    alpha_max: float = 0.20
    alpha_min: float = 0.02


def _default_config_path(base_cfg: SMConfig) -> Path:
    molname = os.environ.get("SM_MOLNAME", base_cfg.molname)
    return base_cfg.systems_dir / molname / "config.yml"


def _resolve_config_path(base_cfg: SMConfig) -> Path:
    env_path = os.environ.get("SM_CONFIG_YML")
    if env_path:
        return Path(env_path)

    default_yml = _default_config_path(base_cfg)
    if default_yml.exists():
        return default_yml

    default_yaml = default_yml.with_suffix(".yaml")
    return default_yaml


def _load_overrides(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _apply_overrides(cfg: SMConfig, overrides: dict) -> SMConfig:
    path_fields = {"systems_dir", "", "wdir", "mol_dir", "aa_dir", "cg_dir"}
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key in YAML: {key}")
        if key in path_fields and value is not None:
            value = Path(value)
        setattr(cfg, key, value)
    return cfg


def _refresh_paths(cfg: SMConfig) -> SMConfig:
    cfg.wdir = cfg.systems_dir / cfg.molname
    cfg.mol_dir = cfg.wdir
    cfg.aa_dir = cfg.wdir / cfg.aa_sysname
    cfg.cg_dir = cfg.wdir / cfg.cg_sysname
    return cfg


def load_config() -> SMConfig:
    cfg = SMConfig()
    cfg = _apply_overrides(cfg, _load_overrides(_resolve_config_path(cfg)))
    cfg = _refresh_paths(cfg)
    return cfg


CFG = load_config()
