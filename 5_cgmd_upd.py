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


def _angle_stats_jacobian(values: np.ndarray, bins: int = 180, min_prob: float = 1e-6) -> Tuple[float, float]:
    """Estimate angle location/spread after correcting for the sin(theta) Jacobian.

    For angle distributions on [0, 180], the observed density follows
        p(theta) ~ sin(theta) * exp(-U(theta)/kT).
    To compare AA/CG underlying potentials, we use q(theta) = p(theta)/sin(theta),
    then report:
      - mu: mode of q(theta)
      - sigma: weighted std around that mode
    """
    if values is None or len(values) == 0:
        return np.nan, np.nan

    vals = np.asarray(values, dtype=float)
    vals = np.clip(vals, 0.0, 180.0)

    n_bins = int(max(24, bins))
    edges = np.linspace(0.0, 180.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    raw_density = np.histogram(vals, bins=edges, density=True)[0]
    raw_density = np.clip(raw_density, min_prob, None)

    jac = np.sin(np.deg2rad(centers))
    jac = np.clip(jac, 1e-6, None)
    corrected = np.clip(raw_density / jac, min_prob, None)
    corrected = raw_density

    idx = int(np.argmax(corrected))
    mu = float(centers[idx])

    weights = corrected / np.sum(corrected)
    sigma = float(np.sqrt(np.sum(weights * (centers - mu) ** 2)))
    mu = np.average(values)
    sigma = np.std(values)
    print(mu, sigma)
    return mu, sigma


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
        k_new = max(float(k_new), CFG.bond_lower_cutoff)
        k_new = min(float(k_new), CFG.bond_upper_cutoff)
        updated[4] = k_new
        
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
        
        print(f"Updating angle ({i}, {j}, {k}): AA samples={len(aa_vals)}, CG samples={len(cg_vals)}")
        mu_aa, sigma_aa = _angle_stats_jacobian(aa_vals)
        mu_cg, sigma_cg = _angle_stats_jacobian(cg_vals)
        if not np.isfinite(mu_aa) or not np.isfinite(sigma_aa):
            mu_aa, sigma_aa = _stats(aa_vals, "angle")
        if not np.isfinite(mu_cg) or not np.isfinite(sigma_cg):
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
        k_new = min(float(k_new), CFG.angle_k_upper_cutoff)
        k_new = max(float(k_new), CFG.angle_k_lower_cutoff)
        updated[5] = k_new

        new_angles.append(updated)
        n_angles_updated += 1

    topo.angles = new_angles

    return n_angles_updated, 0



def update_dihedrals(topo, aa_internal: InternalCoords, cg_internal: InternalCoords):
    """Update dihedrals using an IBI-style reweighting of the AA distribution.

    Strategy
    --------
    Given the AA dihedral density P_AA(phi) and the current CG potential
    U_expected(phi) from the ITP, we look for a new potential U_fitting such
    that the CG distribution reweighted by exp(-U_expected + U_fitting) matches
    P_AA:

        P_AA(phi) ~ exp( (-U_expected(phi) + U_fitting(phi)) / kT )

    Solving for U_fitting:

        U_fitting(phi) = U_expected(phi) - PMF_AA(phi) + const

    where  PMF_AA(phi) = -kT * log P_AA(phi).

    Steps for each dihedral quadruplet (i, j, k, l):
      1. Evaluate U_expected on a phi grid from the current ITP terms.
      2. Histogram the AA dihedral samples -> P_AA.
      3. Compute PMF_AA = -kT * log P_AA.
      4. Target potential: U_fitting = U_expected - PMF_AA  (shift min -> 0).
      5. Fit U_fitting with the appropriate functional form and update topology.

    Applied terms
    ------------
    - funct=9 : fit a Fourier cosine series preserving the existing multiplicities.
    - funct=11: fit a cosine-power polynomial (CBT).
    """
    kB = 0.008314462618  # kJ mol^-1 K^-1
    kT = kB * CFG.temperature

    n_dihedrals_updated = 0
    new_dihedrals = []

    dihedrals_by_key = {}
    for dihedral in topo.dihedrals:
        key = (int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3]))
        dihedrals_by_key.setdefault(key, []).append(dihedral)

    n_bins  = CFG.type9_bins
    phi_grid = np.linspace(-180.0, 180.0, n_bins, endpoint=False)

    for (i, j, k, l), terms in dihedrals_by_key.items():
        aa_vals = aa_internal.get((i, j, k, l, "dihedral"))

        if aa_vals is None or len(aa_vals) < 5:
            logger.debug("Dihedral (%s,%s,%s,%s): no AA data, skipping.", i, j, k, l)
            new_dihedrals.extend(terms)
            continue

        funct = int(terms[0][4])
        if funct not in (9, 11):
            new_dihedrals.extend(terms)
            continue

        # --- 1. U_expected from current ITP ----------------------------------
        if funct == 9:
            U_expected = _eval_type9_potential(terms, phi_grid)
        else:
            U_expected = _eval_type11_potential(terms[0], phi_grid)

        # --- 2. P_AA histogram -----------------------------------------------
        aa_hist, _ = np.histogram(
            wrap_to_180(np.asarray(aa_vals, dtype=float)),
            bins=n_bins,
            range=(-180.0, 180.0),
            density=True,
        )
        aa_density = np.clip(aa_hist, CFG.type9_min_prob, None)

        # --- 3. PMF from AA distribution ------------------------------------
        PMF_AA = -kT * np.log(aa_density)

        # --- 4. Target: U_fitting = U_expected - PMF_AA, shifted to min=0 ---
        U_fitting = U_expected - PMF_AA
        U_fitting -= U_fitting.min()

        # Weights proportional to sqrt(P_AA) so high-density regions dominate
        weights = np.sqrt(aa_density)

        logger.debug(
            "Dihedral (%s,%s,%s,%s) funct=%s: U_fitting range [%.2f, %.2f] kJ/mol",
            i, j, k, l, funct, float(U_fitting.min()), float(U_fitting.max()),
        )

        # --- 5. Fit and write new terms -------------------------------------
        comment = ""
        if funct == 9:
            if len(terms[0]) >= 9:
                comment = terms[0][8]
            harmonics = sorted({int(t[7]) for t in terms})
            fitted = _fit_type9_to_target(phi_grid, U_fitting, harmonics, weights)
            for mult, k_term, phi0 in fitted:
                k_term = float(np.clip(
                    k_term * CFG.fc_scale,
                    CFG.dihedral_k_lower_cutoff,
                    CFG.dihedral_k_upper_cutoff,
                ))
                new_dihedrals.append([i, j, k, l, 9, float(phi0), k_term, int(mult), comment])
            n_dihedrals_updated += 1

        elif funct == 11:
            if len(terms[0]) >= 12:
                comment = terms[0][11]
            k_phi, a = _fit_type11_to_target(phi_grid, U_fitting, weights)
            k_phi = float(np.clip(
                k_phi * CFG.fc_scale,
                CFG.dihedral_k_lower_cutoff,
                CFG.dihedral_k_upper_cutoff,
            ))
            new_dihedrals.append([i, j, k, l, 11, k_phi, a[0], a[1], a[2], a[3], a[4], comment])
            n_dihedrals_updated += 1

    topo.dihedrals = new_dihedrals
    return n_dihedrals_updated, 0


def update_topology_from_cg_vs_aa(
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

    in_itp = outdir / f"{molname}.itp"
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
    tmp_itp = outdir / f"{molname}_tmp.itp"
    shutil.copy2(in_itp, tmp_itp)  # Start from existing ITP to preserve formatting and any unmapped terms
    out_refined_itp = in_itp
    # update_topology_from_cg_vs_aa(topo, aa_internal, cg_internal, out_refined_itp)

    if "plot" in sys.argv:
        plot_internal_coordinates_overlay(
            aa_internal,
            cg_internal,
            topo,
            output_file=wdir / "png" / "cg_vs_aa.png",
        )