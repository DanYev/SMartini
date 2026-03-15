import copy
import pickle
import logging
import shutil
import sys

import numpy as np
import AutoMartini as am

from pathlib import Path
from typing import Dict, Tuple, Optional
from lpmath import (
    read_cg_trajectory,
    read_cog_trajectory,
    calculate_internal_coordinates,
    circular_mean,
    wrap_to_180,
)
from plots import plot_internal_coordinates_overlay
from config import CFG

logger = logging.getLogger(__name__)

InternalCoords = Dict[Tuple[int, ...], np.ndarray]


def _stats(values: np.ndarray, value_type: str) -> Tuple[float, float]:
    if value_type == "dihedral":
        mu = float(circular_mean(values))
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
        return float(wrap_to_180(circular_mean(values)))

    idx = int(np.argmax(hist))
    bin_center = float(0.5 * (edges[idx] + edges[idx + 1]))
    return float(wrap_to_180(center_deg + bin_center))


def _circular_distance_deg(a: float, b: float) -> float:
    return float(abs(wrap_to_180(a - b)))


def _bimodal_dihedral_stats(
    values: np.ndarray,
    bins: int = 72,
    min_separation_deg: float = 30.0,
    min_cluster_size: int = 5,
):
    """Estimate up to two dihedral modes and their sigmas using a histogram.

    Returns a list of (center_deg, sigma_deg) sorted by population.
    """
    if values is None or len(values) == 0:
        return []

    wrapped = wrap_to_180(np.asarray(values, dtype=float))
    hist, edges = np.histogram(wrapped, bins=bins, range=(-180, 180))
    if np.all(hist == 0):
        return []

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    order = np.argsort(hist)[::-1]

    picked_centers = []
    for idx in order:
        if hist[idx] == 0:
            break
        candidate = float(bin_centers[idx])
        if not picked_centers:
            picked_centers.append(candidate)
            if len(picked_centers) == 2:
                break
            continue
        if _circular_distance_deg(candidate, picked_centers[0]) >= min_separation_deg:
            picked_centers.append(candidate)
            break

    if not picked_centers:
        return []

    # Assign samples to nearest picked center
    centers = picked_centers
    clusters = {c: [] for c in centers}
    for val in wrapped:
        nearest = min(centers, key=lambda c: _circular_distance_deg(val, c))
        clusters[nearest].append(val)

    stats = []
    for center in centers:
        cluster_vals = np.asarray(clusters[center], dtype=float)
        if len(cluster_vals) < min_cluster_size:
            continue
        residual = wrap_to_180(cluster_vals - center)
        sigma = float(np.std(residual))
        stats.append((float(center), sigma, len(cluster_vals)))

    if not stats:
        return []

    stats.sort(key=lambda x: x[2], reverse=True)
    return [(c, s) for c, s, _ in stats]


def _pair_mode_centers(aa_centers, cg_centers):
    if len(aa_centers) != 2 or len(cg_centers) != 2:
        return None
    a0, a1 = aa_centers
    c0, c1 = cg_centers
    d_same = _circular_distance_deg(a0, c0) + _circular_distance_deg(a1, c1)
    d_cross = _circular_distance_deg(a0, c1) + _circular_distance_deg(a1, c0)
    if d_same <= d_cross:
        return {c0: a0, c1: a1}
    return {c0: a1, c1: a0}


def _k_rescale(k_old: float, sigma_target: float, sigma_current: float) -> float:
    if not np.isfinite(k_old) or k_old <= 0:
        return k_old
    if not np.isfinite(sigma_target) or not np.isfinite(sigma_current):
        return k_old
    if sigma_target <= 0 or sigma_current <= 0:
        return k_old

    scale = (sigma_current / sigma_target) ** 2
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
        aa_key = (i, j, "constraint")
        key = (i, j, "bond")
        aa_vals = aa_internal.get(aa_key)
        cg_vals = cg_internal.get(key)
        if aa_vals is None or cg_vals is None:
            new_bonds.append(bond)
            continue

        mu_aa, sigma_aa = _stats(aa_vals, "bond")
        mu_cg, sigma_cg = _stats(cg_vals, "bond")
        updated = list(bond)

        delta = mu_aa - mu_cg
        updated[3] = float(updated[3]) + delta

        k_new = _k_rescale(
            float(updated[4]),
            sigma_target=sigma_aa,
            sigma_current=sigma_cg,
        )
        updated[4] = min(float(k_new), CFG.constraint_k_cutoff)
        
        new_bonds.append(updated)
        n_bonds_updated += 1

    topo.bonds = new_bonds

    # Constraints (length only)
    new_constraints = []
    for constraint in topo.constraints:
        i, j = int(constraint[0]), int(constraint[1])
        key = (i, j, "constraint")
        aa_vals = aa_internal.get(key)
        cg_vals = cg_internal.get(key)

        mu_aa, _ = _stats(aa_vals, "constraint")
        mu_cg, _ = _stats(cg_vals, "constraint")
        delta = mu_aa - mu_cg

        updated = list(constraint)
        updated[3] = float(updated[3]) + delta
        new_constraints.append(updated)
        n_constraints_updated += 1

    topo.constraints = new_constraints
    
    return n_bonds_updated, n_constraints_updated


def update_angles(topo, aa_internal: InternalCoords, cg_internal: InternalCoords):
    """Update angles by adjusting equilibrium values and force constants.
    
    Strategy
    --------
    - Equilibrium values: shift by (mu_AA - mu_CG)
    - Force constants: rescale by (sigma_CG / sigma_AA)^2 (harmonic approximation)
    - Never removes angles; only updates parameters when data exist.
    """
    n_angles_updated = 0
    
    new_angles = []
    for angle in topo.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        key = (i, j, k, "angle")
        aa_vals = aa_internal.get(key)
        cg_vals = cg_internal.get(key)
        if aa_vals is None or cg_vals is None:
            new_angles.append(angle)
            continue

        mu_aa, sigma_aa = _stats(aa_vals, "angle")
        mu_cg, sigma_cg = _stats(cg_vals, "angle")
        updated = list(angle)

        delta = mu_aa - mu_cg
        theta0_old = float(updated[4])
        theta0_new = float(theta0_old + delta)
        theta0_new = float(np.clip(theta0_new, 0.0, 180.0))
        updated[4] = theta0_new

        k_new = _k_rescale(
            float(updated[5]),
            sigma_target=sigma_aa,
            sigma_current=sigma_cg,
        )
        updated[5] = max(float(k_new), CFG.angle_k_cutoff)

        new_angles.append(updated)
        n_angles_updated += 1

    topo.angles = new_angles

    return n_angles_updated, 0


def update_dihedrals(topo, aa_internal: InternalCoords, cg_internal: InternalCoords):
    """Update dihedrals by rescaling force constants (no phase shifting).

    Strategy
    --------
    - No shifting of phi0.
    - Rescale force constants by (sigma_CG / sigma_AA)^2 (harmonic approximation).
    - Clamp the scale factor to avoid extreme updates.

    Applied terms
    ------------
    - funct=9: scales k (field 6)
    - funct=11: scales kphi (field 5)
    """
    n_dihedrals_updated = 0

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
            new_dihedrals.extend(terms)
            continue

        _, sigma_aa = _stats(aa_vals, "dihedral")
        _, sigma_cg = _stats(cg_vals, "dihedral")
        scale = float((sigma_cg / sigma_aa))
        print(scale)


        for term in terms:
            updated = list(term)
            funct = int(updated[4])

            if funct == 9:
                mult = int(updated[7])
                if mult == len(terms) or mult == 1: 
                    scale = scale ** 0.7
                if len(terms) == 1 and mult == 1: 
                    scale = scale ** 2
                k_new = float(updated[6]) * scale
                k_new = max(k_new, float(CFG.dihedral_k_cutoff))
                updated[6] = float(k_new)
                n_dihedrals_updated += 1

            elif funct == 11:
                kphi_new = float(updated[5]) * scale**2
                # kphi_new = min(kphi_new, float(CFG.dihedral_k_cutoff))
                updated[5] = float(kphi_new)
                n_dihedrals_updated += 1

            new_dihedrals.append(updated)

    topo.dihedrals = new_dihedrals
    return n_dihedrals_updated, 0


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
    wdir = CFG.wdir
    outdir = CFG.mol_dir

    itp_updated = outdir / f"{molname}_updated.itp"
    in_itp = itp_updated 
    logger.info("Reading topology from %s", in_itp)
    topo = am.topology.read_itp(str(in_itp))

    unique_dihedrals = {(int(d[0]), int(d[1]), int(d[2]), int(d[3])) for d in topo.dihedrals}
    logger.info(
        "Loaded %s dihedral terms across %s unique torsions",
        len(topo.dihedrals),
        len(unique_dihedrals),
    )

    cg_dir = CFG.cg_dir
    cg_pdb = cg_dir / CFG.cg_runname / "topology.pdb"
    cg_xtc = cg_dir / CFG.cg_runname / "samples.xtc"

    logger.info("Reading AA trajectory")
    with open(wdir / "internal_coords.pkl", "rb") as f:
        aa_internal = pickle.load(f)

    logger.info("Reading CG trajectory from %s", cg_dir)
    cg_traj = read_cg_trajectory(cg_pdb, cg_xtc, start=0, stop=None, step=1)  
    cg_internal = calculate_internal_coordinates(cg_traj, topo)

    # Refine the CG topology based on CG-vs-AA distribution mismatch.
    # Writes a new file and leaves the original ITP unchanged.
    tmp_itp = outdir / f"{molname}_updated_tmp.itp"
    shutil.copy2(in_itp, tmp_itp)  # Start from existing ITP to preserve formatting and any unmapped terms
    out_refined_itp = itp_updated
    refine_topology_from_cg_vs_aa(topo, aa_internal, cg_internal, out_refined_itp)

    if sys.argv[-1] == "plot":
        plot_internal_coordinates_overlay(
            aa_internal,
            cg_internal,
            topo,
            output_file=wdir / "png" / "cg_vs_aa.png",
        )
