import copy
import pickle
import logging
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
import AutoMartini as am

from pathlib import Path
from typing import Dict, Tuple, Optional
from lpmath import (
    read_cg_trajectory,
    read_cog_trajectory,
    calculate_internal_coordinates,
    _eval_type9_potential,
    _eval_type11_potential,
    _fit_type9_to_target,
    _fit_type11_to_target,
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
    """Update dihedrals by fitting delta PMF and adding it to current terms.

    For each torsion, we compute:
        pmf = -kT * log(rho_AA / rho_expected)
    then fit that delta with either type-9 or type-11 form, and combine the
    fitted delta terms with the existing topology terms.
    """
    kB = 0.008314462618  # kJ mol^-1 K^-1
    kT = kB * CFG.temperature
    nbins = int(CFG.type9_bins)
    png_dir = Path(__file__).resolve().parent / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    def _save_pmf_plot(phi_grid, pmf, u_initial, u_updated, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        pmf_plot = np.asarray(pmf, dtype=float) - float(np.min(pmf))
        u0_plot = np.asarray(u_initial, dtype=float) - float(np.min(u_initial))
        u1_plot = np.asarray(u_updated, dtype=float) - float(np.min(u_updated))

        ax.plot(phi_grid, pmf_plot, lw=1.8, label="PMF")
        ax.plot(phi_grid, u0_plot, lw=1.5, ls="--", label="Initial potential")
        ax.plot(phi_grid, u1_plot, lw=1.5, ls="-.", label="Updated potential")
        ax.set_xlabel("Dihedral (deg)")
        ax.set_ylabel("PMF (kJ/mol)")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(png_dir / filename, dpi=180)
        plt.close(fig)

    def _update_type9_terms(i, j, k, l, aa_vals, cg_vals, terms):
        def _weighted_circular_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
            weights = np.asarray(weights, dtype=float)
            angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=float))
            wsum = float(np.sum(weights))
            if wsum <= 0:
                return 0.0
            sin_mean = float(np.sum(weights * np.sin(angles_rad)) / wsum)
            cos_mean = float(np.sum(weights * np.cos(angles_rad)) / wsum)
            return float(wrap_to_180(np.rad2deg(np.arctan2(sin_mean, cos_mean))))

        def _periodic_interp_deg(x_query: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
            x = np.asarray(x_grid, dtype=float)
            y = np.asarray(y_grid, dtype=float)
            xq = np.asarray(x_query, dtype=float)
            x_ext = np.concatenate([x - 360.0, x, x + 360.0])
            y_ext = np.concatenate([y, y, y])
            return np.interp(xq, x_ext, y_ext)

        aa_vals = np.asarray(aa_vals, dtype=float)
        cg_vals = np.asarray(cg_vals, dtype=float)
        shift_aa = float(circular_mean(aa_vals))
        shift_cg = float(circular_mean(cg_vals))

        # Canonical absolute grid for potential summation/extraction.
        phi_centers = np.linspace(-180.0, 180.0, nbins, endpoint=False)
        phi_absolute = phi_centers.copy()

        # Shift for potential-derived component from its own implied density.
        U_expected_abs = _eval_type9_potential(terms, phi_absolute)
        density_abs = np.exp(-U_expected_abs / kT)
        density_abs = np.clip(density_abs, CFG.type9_min_prob, None)
        density_abs /= np.sum(density_abs)
        shift_potential = _weighted_circular_mean_deg(phi_absolute, density_abs)

        # PMF used for plotting only: keep everything on the absolute grid
        # so phase is directly comparable to U_expected_abs.
        bins_abs = np.linspace(-180.0, 180.0, nbins + 1)
        aa_density_abs, _ = np.histogram(
            wrap_to_180(aa_vals),
            bins=bins_abs,
            density=True,
        )
        aa_density_abs = np.clip(aa_density_abs, CFG.type9_min_prob, None)
        aa_density_abs /= np.sum(aa_density_abs)

        cg_density_abs, _ = np.histogram(
            wrap_to_180(cg_vals),
            bins=bins_abs,
            density=True,
        )
        cg_density_abs = np.clip(cg_density_abs, CFG.type9_min_prob, None)
        cg_density_abs /= np.sum(cg_density_abs)

        pmf_plot_abs = -kT * (
            np.log(aa_density_abs)
            - np.log(cg_density_abs)
            + np.log(density_abs)
        )

        aa_centered = wrap_to_180(aa_vals - shift_aa)
        cg_centered = wrap_to_180(cg_vals - shift_cg)

        phi_abs_from_potential = wrap_to_180(phi_centers + shift_potential)
        U_expected_from_potential = _eval_type9_potential(terms, phi_abs_from_potential)
        density_from_potential = np.exp(-U_expected_from_potential / kT)
        density_from_potential = np.clip(density_from_potential, CFG.type9_min_prob, None)
        density_from_potential /= np.sum(density_from_potential)

        bins = np.linspace(-180.0, 180.0, nbins + 1)
        aa_density, _ = np.histogram(aa_centered, bins=bins, density=True)
        aa_density = np.clip(aa_density, CFG.type9_min_prob, None)
        aa_density /= np.sum(aa_density)

        cg_density, _ = np.histogram(cg_centered, bins=bins, density=True)
        cg_density = np.clip(cg_density, CFG.type9_min_prob, None)
        cg_density /= np.sum(cg_density)

        pmf_aa = -kT * np.log(aa_density)
        pmf_cg = +kT * np.log(cg_density)
        pmf_potential = -kT * np.log(density_from_potential)

        harmonics = sorted({int(t[7]) for t in terms})
        density_power = 1.0 if len(harmonics) == 1 else 0.2
        weights_aa = np.pow(aa_density, density_power)
        weights_cg = np.pow(cg_density, density_power)
        weights_pot = np.pow(density_from_potential, density_power)

        comment = ""
        if terms and len(terms[0]) >= 9:
            comment = terms[0][8]

        # Fit each PMF component separately, sum the resulting fitted
        # potentials, then extract a final consolidated set of type-9 terms.
        summed_fitted_potential = np.zeros_like(phi_centers, dtype=float)
        component_specs = (
            (pmf_aa, shift_aa, weights_aa),
            (pmf_cg, shift_cg, weights_cg),
            (pmf_potential, shift_potential, weights_pot),
        )
        for pmf_component, shift_component, weights_component in component_specs:
            component_fit = _fit_type9_to_target(
                pmf_component,
                shift=shift_component,
                harmonics=harmonics,
                weights=weights_component,
                phi_grid=phi_centers,
            )
            if not component_fit:
                continue
            component_terms = []
            for mult, k_term_single, phi0_single in component_fit:
                component_terms.append(
                    [
                        i,
                        j,
                        k,
                        l,
                        9,
                        float(phi0_single),
                        float(k_term_single),
                        int(mult),
                        comment,
                    ]
                )
            summed_fitted_potential += _eval_type9_potential(component_terms, phi_absolute)

        summed_fitted_potential -= float(np.min(summed_fitted_potential))
        density_summed = np.exp(-summed_fitted_potential / kT)
        density_summed = np.clip(density_summed, CFG.type9_min_prob, None)
        density_summed /= np.sum(density_summed)
        shift_summed = _weighted_circular_mean_deg(phi_absolute, density_summed)

        phi_abs_summed = wrap_to_180(phi_centers + shift_summed)
        summed_on_shifted_grid = _periodic_interp_deg(
            phi_abs_summed,
            phi_absolute,
            summed_fitted_potential,
        )

        fitted_terms = _fit_type9_to_target(
            summed_on_shifted_grid,
            shift=shift_summed,
            harmonics=harmonics,
            weights=np.pow(density_summed, density_power),
            phi_grid=phi_centers,
        )
        updated_terms = []
        for mult, k_term, phi0 in fitted_terms:
            updated_terms.append(
                [i, j, k, l, 9, float(phi0), float(k_term), int(mult), comment]
            )

        U_updated = _eval_type9_potential(updated_terms, phi_absolute)
        _save_pmf_plot(
            phi_centers,
            pmf_plot_abs,
            u_initial=U_expected_abs,
            u_updated=U_updated,
            title=f"PMF type9 ({i},{j},{k},{l})",
            filename=f"{i}{j}{k}{l}.png",
        )
        return updated_terms

    def _update_type11_terms(terms, aa_vals, cg_vals):
        aa_vals = np.asarray(aa_vals, dtype=float)
        cg_vals = np.asarray(cg_vals, dtype=float)
        phi_grid = np.linspace(-180.0, 180.0, nbins, endpoint=False)

        U_expected = _eval_type11_potential(terms[0], phi_grid)
        density_from_potential = np.exp(-U_expected / kT)
        density_from_potential = np.clip(density_from_potential, CFG.type9_min_prob, None)
        density_from_potential /= np.sum(density_from_potential)

        aa_hist, _ = np.histogram(
            wrap_to_180(aa_vals),
            bins=nbins,
            range=(-180.0, 180.0),
            density=True,
        )
        aa_density = np.clip(aa_hist, CFG.type9_min_prob, None)
        aa_density /= np.sum(aa_density)

        cg_density, _ = np.histogram(
            wrap_to_180(cg_vals),
            bins=nbins,
            range=(-180.0, 180.0),
            density=True,
        )
        cg_density = np.clip(cg_density, CFG.type9_min_prob, None)
        cg_density /= np.sum(cg_density)

        # pmf = -kT * (np.log(aa_density) - np.log(density_from_potential))
        pmf = -kT * (np.log(aa_density) - np.log(cg_density) + np.log(density_from_potential))
        # pmf = -kT * (np.log(aa_density))
        pmf -= np.min(pmf)
        weights = np.pow(aa_density, 0.30)
        i, j, k, l = (int(terms[0][0]), int(terms[0][1]), int(terms[0][2]), int(terms[0][3]))
        k_phi_new, a_new = _fit_type11_to_target(
            pmf,
            weights=weights,
            phi_grid=phi_grid,
        )
        base = list(terms[0])
        base[5] = k_phi_new
        for n in range(5):
            base[6 + n] = float(a_new[n])

        U_updated = _eval_type11_potential(base, phi_grid)
        _save_pmf_plot(
            phi_grid,
            pmf,
            u_initial=U_expected,
            u_updated=U_updated,
            title=f"PMF type11 ({i},{j},{k},{l})",
            filename=f"pmf_type11_{i}_{j}_{k}_{l}.png",
        )
        return [base]

        base = list(terms[0])
        k_phi_0 = float(base[5])
        a0 = np.asarray([float(base[6 + n]) for n in range(5)], dtype=float)
        c0 = k_phi_0 * a0
        c_delta = k_phi_delta * np.asarray(a_delta, dtype=float)
        c_new = c0 + c_delta

        k_phi_new = float(np.max(np.abs(c_new)))
        if k_phi_new < 1e-12:
            k_phi_new = float(CFG.dihedral_k_lower_cutoff)
            a_new = np.zeros(5, dtype=float)
        else:
            a_new = c_new / k_phi_new

        k_phi_new = float(np.clip(k_phi_new, CFG.dihedral_k_lower_cutoff, CFG.dihedral_k_upper_cutoff))
        base[5] = k_phi_new
        for n in range(5):
            base[6 + n] = float(a_new[n])
        return [base]

    n_dihedrals_updated = 0
    new_dihedrals = []

    dihedrals_by_key = {}
    for dihedral in topo.dihedrals:
        key = (int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3]))
        dihedrals_by_key.setdefault(key, []).append(dihedral)

    for (i, j, k, l), terms in dihedrals_by_key.items():
        aa_vals = aa_internal.get((i, j, k, l, "dihedral"))
        cg_vals = cg_internal.get((i, j, k, l, "dihedral"))

        funct = int(terms[0][4])
        if funct == 9:
            updated_terms = _update_type9_terms(i, j, k, l, aa_vals, cg_vals, terms)
            new_dihedrals.extend(updated_terms)
            n_dihedrals_updated += 1
        elif funct == 11:
            updated_terms = _update_type11_terms(terms, aa_vals, cg_vals)
            new_dihedrals.extend(updated_terms)
            n_dihedrals_updated += 1
        else:
            new_dihedrals.extend(terms)

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
    topo = am.topology.read_itp(str(in_itp))
    logger.info("Reading topology from %s", in_itp)
    # DEBUG
    ref_itp = outdir / f"{molname}_ref.itp"
    topo = am.topology.read_itp(str(ref_itp))

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
    update_topology_from_cg_vs_aa(topo, aa_internal, cg_internal, out_refined_itp)

    if "plot" in sys.argv:
        plot_internal_coordinates_overlay(
            aa_internal,
            cg_internal,
            topo,
            output_file=wdir / "png" / "cg_vs_aa.png",
        )