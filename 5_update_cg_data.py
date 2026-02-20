import copy
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import AutoMartini as am

from lpmath import (
    read_cg_trajectory,
    read_cog_trajectory,
    calculate_internal_coordinates,
    circular_mean_deg,
    wrap_to_180,
)
from plots import plot_internal_coordinates_overlay
from ligpar_config import CFG, get_logger

logger = get_logger(__name__)

InternalCoords = Dict[Tuple[int, ...], np.ndarray]


def _stats(values: np.ndarray, value_type: str) -> Tuple[float, float]:
    if value_type == "dihedral":
        mu = float(circular_mean_deg(values))
        centered = wrap_to_180(values - mu)
        sigma = float(np.std(centered))
        return mu, sigma

    mu = float(np.mean(values))
    sigma = float(np.std(values))
    return mu, sigma


def _dihedral_mode_deg(values: np.ndarray, center_deg: float, bins: int = 72) -> float:
    """Estimate dihedral 'location' robustly as a mode near a reference center.

    We histogram wrapped residuals (value - center) in [-180, 180] and take the
    most populated bin center; then map back by adding center.
    """
    if values is None or len(values) == 0:
        return float(wrap_to_180(center_deg))

    residual = wrap_to_180(values - center_deg)
    hist, edges = np.histogram(residual, bins=bins, range=(-180, 180))
    if np.all(hist == 0):
        return float(wrap_to_180(circular_mean_deg(values)))

    idx = int(np.argmax(hist))
    bin_center = float(0.5 * (edges[idx] + edges[idx + 1]))
    return float(wrap_to_180(center_deg + bin_center))


def _k_rescale(k_old: float, sigma_target: float, sigma_current: float, max_scale: float) -> float:
    if not np.isfinite(k_old) or k_old <= 0:
        return k_old
    if not np.isfinite(sigma_target) or not np.isfinite(sigma_current):
        return k_old
    if sigma_target <= 0 or sigma_current <= 0:
        return k_old

    scale = (sigma_current / sigma_target) ** 2
    scale = float(np.clip(scale, 1.0 / max_scale, max_scale))
    return float(k_old * scale)


def update_bonds(topo, aa_internal: InternalCoords, cg_internal: InternalCoords):
    """Update bonds and constraints by adjusting equilibrium values and force constants.
    
    Strategy
    --------
    - Equilibrium values: shift by (mu_AA - mu_CG)
    - Force constants: rescale by (sigma_CG / sigma_AA)^2 (harmonic approximation)
    """
    n_bonds_updated = 0
    n_constraints_updated = 0
    
    # Bonds
    new_bonds = []
    for bond in topo.bonds:
        i, j = int(bond[0]), int(bond[1])
        key = (i, j, "bond")
        aa_vals = aa_internal.get(key)
        cg_vals = cg_internal.get(key)
        if aa_vals is None or cg_vals is None:
            new_bonds.append(bond)
            continue

        mu_aa, sigma_aa = _stats(aa_vals, "bond")
        mu_cg, sigma_cg = _stats(cg_vals, "bond")
        delta = mu_aa - mu_cg

        r0_old = float(bond[3])
        k_old = float(bond[4]) if len(bond) >= 5 else None
        r0_new = float(r0_old + delta)

        if k_old is not None:
            k_new = _k_rescale(k_old, sigma_target=sigma_aa, sigma_current=sigma_cg, max_scale=CFG.refine_max_k_scale)
            new_bonds.append([i, j, bond[2], r0_new, k_new])
        else:
            new_bonds.append([i, j, bond[2], r0_new])

        n_bonds_updated += 1

    topo.bonds = new_bonds

    # Constraints (length only)
    new_constraints = []
    for constraint in topo.constraints:
        i, j = int(constraint[0]), int(constraint[1])
        key = (i, j, "constraint")
        aa_vals = aa_internal.get(key)
        cg_vals = cg_internal.get(key)
        if aa_vals is None or cg_vals is None:
            new_constraints.append(constraint)
            continue

        mu_aa, _ = _stats(aa_vals, "constraint")
        mu_cg, _ = _stats(cg_vals, "constraint")
        delta = mu_aa - mu_cg

        r0_old = float(constraint[3])
        r0_new = float(r0_old + delta)
        new_constraints.append([i, j, constraint[2], r0_new])
        n_constraints_updated += 1

    topo.constraints = new_constraints
    
    return n_bonds_updated, n_constraints_updated


def update_angles(topo, aa_internal: InternalCoords, cg_internal: InternalCoords):
    """Update angles by adjusting equilibrium values and force constants.
    
    Strategy
    --------
    - Equilibrium values: shift by (mu_AA - mu_CG)
    - Force constants: rescale by (sigma_CG / sigma_AA)^2 (harmonic approximation)
    - Optionally remove angles with force constants below threshold
    """
    n_angles_updated = 0
    n_angles_removed = 0
    
    new_angles = []
    for angle in topo.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        key = (i, j, k, "angle")
        aa_vals = aa_internal.get(key)
        cg_vals = cg_internal.get(key)
        if aa_vals is None or cg_vals is None:
            # Optional pruning based on existing k
            if CFG.angle_k_cutoff is not None and len(angle) >= 6 and float(angle[5]) < CFG.angle_k_cutoff:
                n_angles_removed += 1
                continue
            new_angles.append(angle)
            continue

        mu_aa, sigma_aa = _stats(aa_vals, "angle")
        mu_cg, sigma_cg = _stats(cg_vals, "angle")
        delta = mu_aa - mu_cg

        theta0_old = float(angle[4])
        theta0_new = float(theta0_old + delta)
        theta0_new = float(np.clip(theta0_new, 0.0, 180.0))

        k_old = float(angle[5]) if len(angle) >= 6 else None
        if k_old is not None:
            k_new = _k_rescale(k_old, sigma_target=sigma_aa, sigma_current=sigma_cg, max_scale=CFG.refine_max_k_scale)
            if CFG.angle_k_cutoff is not None and k_new < CFG.angle_k_cutoff:
                n_angles_removed += 1
                continue
            new_angles.append([i, j, k, angle[3], theta0_new, k_new])
        else:
            new_angles.append([i, j, k, angle[3], theta0_new])

        n_angles_updated += 1

    topo.angles = new_angles
    
    return n_angles_updated, n_angles_removed


def update_dihedrals(topo, aa_internal: InternalCoords, cg_internal: InternalCoords):
    """Update dihedrals by adjusting phi_0 and force constants.
    
    Strategy
    --------
    - Equilibrium angles: shift by (mu_AA - mu_CG) using circular statistics
    - Force constants: rescale by (sigma_CG / sigma_AA)^2 (harmonic approximation)
    - Optionally remove dihedrals with force constants below threshold
    
    Notes
    -----
    - Uses circular mean + std of wrapped residuals for dihedrals
    - Each dihedral term is adjusted independently
    """
    n_dihedrals_updated = 0
    n_dihedrals_removed = 0
    
    new_dihedrals = []
    dihedrals_by_key = {}
    for dihedral in topo.dihedrals:
        key = (int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3]))
        dihedrals_by_key.setdefault(key, []).append(dihedral)

    for (i, j, k, l), terms in dihedrals_by_key.items():
        key = (i, j, k, l, "dihedral")
        aa_vals = aa_internal.get(key)
        cg_vals = cg_internal.get(key)
        
        if aa_vals is None or cg_vals is None:
            # No AA or CG data, keep existing terms (with optional pruning)
            for term in terms:
                if CFG.dihedral_k_cutoff is not None and len(term) >= 7 and abs(float(term[6])) < CFG.dihedral_k_cutoff:
                    n_dihedrals_removed += 1
                    continue
                new_dihedrals.append(term)
            continue
        
        # Calculate circular statistics for AA and CG trajectories
        mu_aa, sigma_aa = _stats(aa_vals, "dihedral")
        mu_cg, sigma_cg = _stats(cg_vals, "dihedral")
        delta = wrap_to_180(mu_aa - mu_cg)
        
        # Update each term for this dihedral
        for term in terms:
            if len(term) < 7:
                # Unexpected format, keep as-is
                new_dihedrals.append(term)
                continue
            
            phi0_old = float(term[5])
            k_old = float(term[6])
            mult = int(term[7]) if len(term) >= 8 else 1
            
            # Adjust phi_0 by the circular mean difference
            phi0_new = wrap_to_180(phi0_old + delta)
            
            # Rescale force constant based on sigma ratio
            k_new = _k_rescale(k_old, sigma_target=sigma_aa, sigma_current=sigma_cg, max_scale=CFG.refine_max_k_scale)
            
            # Optional pruning based on minimum k threshold
            if CFG.dihedral_k_cutoff is not None and abs(k_new) < CFG.dihedral_k_cutoff:
                n_dihedrals_removed += 1
                continue
            
            new_dihedrals.append([i, j, k, l, 9, float(phi0_new), k_new, mult])
            n_dihedrals_updated += 1

    topo.dihedrals = new_dihedrals
    
    return n_dihedrals_updated, n_dihedrals_removed


def refine_topology_from_cg_vs_aa(
    topo,
    aa_internal: InternalCoords,
    cg_internal: InternalCoords,
    output_itp,
):
    """Refine an existing CG topology by matching CG distributions to AA ones.

    Strategy
    --------
    - Equilibrium values: shift by (mu_AA - mu_CG)
    - Force constants: rescale by (sigma_CG / sigma_AA)^2 (harmonic approximation)

    Notes
    -----
    - Bonds/constraints use linear mean/std.
    - Dihedrals use circular mean + std of wrapped residuals.
    - Writes a NEW ITP and returns the updated topology.
    """
    updated = copy.deepcopy(topo)

    # Update bonds and constraints
    n_bonds_updated, n_constraints_updated = update_bonds(updated, aa_internal, cg_internal)
    
    # Update angles
    n_angles_updated, n_angles_removed = update_angles(updated, aa_internal, cg_internal)
    
    # Update dihedrals
    n_dihedrals_updated, n_dihedrals_removed = update_dihedrals(updated, aa_internal, cg_internal)

    logger.info(
        "Refined topology: bonds %s, constraints %s, angles %s (removed %s), dihedrals %s (removed %s)",
        n_bonds_updated,
        n_constraints_updated,
        n_angles_updated,
        n_angles_removed,
        n_dihedrals_updated,
        n_dihedrals_removed,
    )

    updated.to_itp(out_file=output_itp)
    logger.info("Wrote refined ITP to %s", output_itp)

    return updated


if __name__ == "__main__":
	molname = CFG.molname
	wdir = CFG.wdir()

	itp_updated = wdir / "mapping" / f"{molname}_updated.itp"
	itp_default = wdir / "mapping" / f"{molname}.itp"
	in_itp = itp_updated if itp_updated.exists() else itp_default
	logger.info("Reading topology from %s", in_itp)
	topo = am.topology.read_itp(str(in_itp))
	unique_dihedrals = {(int(d[0]), int(d[1]), int(d[2]), int(d[3])) for d in topo.dihedrals}
	logger.info(
		"Loaded %s dihedral terms across %s unique torsions",
		len(topo.dihedrals),
		len(unique_dihedrals),
	)

	aa_dir = CFG.aa_dir()
	aa_pdb = aa_dir / "md.pdb"
	aa_xtc = aa_dir / "md.xtc"

	cg_dir = CFG.cg_dir()
	cg_pdb = cg_dir / CFG.cg_runname / "topology.pdb"
	cg_xtc = cg_dir / CFG.cg_runname / "samples.xtc"

	logger.info("Reading AA trajectory from %s", aa_dir)
	aa_traj = read_cog_trajectory(aa_pdb, aa_xtc, topo.partitioning)
	aa_internal = calculate_internal_coordinates(aa_traj, topo)

	logger.info("Reading CG trajectory from %s", cg_dir)
	cg_traj = read_cg_trajectory(cg_pdb, cg_xtc, start=0, stop=CFG.cg_traj_stop) # start and step can be adjusted to speed up processing for long trajectories
	cg_internal = calculate_internal_coordinates(cg_traj, topo)

	plot_internal_coordinates_overlay(
		aa_internal,
		cg_internal,
		topo,
		output_file=wdir / "png" / "cg_vs_aa.png",
	)

	# Refine the CG topology based on CG-vs-AA distribution mismatch.
	# Writes a new file and leaves the original ITP unchanged.
	# out_refined_itp = wdir / "mapping" / f"{molname}_cgrefined.itp"
	out_refined_itp = itp_updated
	refine_topology_from_cg_vs_aa(topo, aa_internal, cg_internal, out_refined_itp)
