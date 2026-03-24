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
from AutoMartini.lpmath import (
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
from AutoMartini.plots import plot_internal_coordinates_overlay
from config import CFG

logger = logging.getLogger(__name__)

InternalCoords = Dict[Tuple[int, ...], np.ndarray]


def _stats(values: np.ndarray, value_type: str) -> Tuple[float, float]:
    """Compute mean and spread for linear or circular internal coordinates.

    Parameters
    ----------
    values : numpy.ndarray
        Sampled coordinate values.
    value_type : str
        Coordinate type. ``"dihedral"`` is treated as periodic and uses
        circular centering; all other values are treated as linear.

    Returns
    -------
    tuple[float, float]
        ``(mu, sigma)`` where ``mu`` is the location estimate and ``sigma`` is
        the standard deviation in the same units as ``values``.
    """
    if value_type == "dihedral":
        mu = float(circular_mean(values))
        centered = wrap_to_180(values - mu)
        sigma = float(np.std(centered))
        return mu, sigma

    mu = float(np.mean(values))
    sigma = float(np.std(values))
    return mu, sigma

def _pair_mode_centers(aa_centers, cg_centers):
    """Pair two AA and two CG mode centers to minimize circular mismatch.

    Returns
    -------
    dict or None
        Mapping from each CG center to its paired AA center, or ``None`` when
        either input does not contain exactly two centers.
    """
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
    """Rescale a harmonic force constant to match a target fluctuation width.

    Uses ``k_new = k_old * (sigma_current / sigma_target)^2`` and returns the
    original value when inputs are non-finite or non-positive.
    """
    if not np.isfinite(k_old) or k_old <= 0:
        return k_old
    if not np.isfinite(sigma_target) or not np.isfinite(sigma_current):
        return k_old
    if sigma_target <= 0 or sigma_current <= 0:
        return k_old
    scale = (sigma_current / sigma_target) ** 2
    return float(k_old * scale)


def _angle_stats_jacobian(values: np.ndarray, bins: int = 180, min_prob: float = 1e-6) -> Tuple[float, float]:
    """Estimate angle location/spread for fitting updates.

    For angle distributions on [0, 180], the observed density follows
        p(theta) ~ sin(theta) * exp(-U(theta)/kT).
    To compare AA/CG underlying potentials, we use q(theta) = p(theta)/sin(theta),
    then report:
      - mu: mode of q(theta)
      - sigma: weighted std around that mode

    Notes
    -----
    The current implementation builds Jacobian-corrected intermediates but
    ultimately returns arithmetic mean/std of the raw samples.
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

    Returns
    -------
    tuple[int, int]
        Number of updated bond terms and updated constraint terms.
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
        alpha = CFG.alpha_max
        updated[3] = float(updated[3]) + alpha * delta

        k_new = _k_rescale(
            float(updated[4]),
            sigma_target=sigma_aa,
            sigma_current=sigma_cg,
        )
        k_new = max(float(k_new), CFG.bond_k_lower)
        k_new = min(float(k_new), CFG.bond_k_upper)
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

    Returns
    -------
    tuple[int, int]
        Number of updated angles and number of removed angles (currently 0).
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
        alpha = CFG.alpha_max
        theta0_old = float(updated[4])
        theta0_new = float(theta0_old + alpha * delta)
        theta0_new = float(np.clip(theta0_new, 0.0, 180.0))
        updated[4] = theta0_new

        k_new = _k_rescale(
            float(updated[5]),
            sigma_target=sigma_aa,
            sigma_current=sigma_cg,
        )
        k_new = min(float(k_new), CFG.angle_k_upper)
        k_new = max(float(k_new), CFG.angle_k_lower)
        updated[5] = k_new

        new_angles.append(updated)
        n_angles_updated += 1

    topo.angles = new_angles

    return n_angles_updated, 0


def update_dihedrals(topo, aa_internal: InternalCoords, cg_internal: InternalCoords):
    """Update dihedrals by refitting PMF corrections onto existing functional forms.

    For each torsion, we compute:
        pmf = -kT * log(rho_AA / rho_expected)
    then fit that delta with either type-9 or type-11 form, and combine the
    fitted delta terms with the existing topology terms.

    Returns
    -------
    tuple[int, int]
        Number of updated dihedral keys and number removed (currently 0).

    Type-9 vs type-11 behavior
    --------------------------
    The updater groups terms by torsion key ``(i,j,k,l)`` and dispatches by
    the function type of the first existing term:

    - ``funct=9`` -> calls ``_update_type9_terms``
    - ``funct=11`` -> calls ``_update_type11_terms``
    - other funct values are passed through unchanged.

    Type-9 branch (Fourier periodic torsions)
    ----------------------------------------
    1. Center AA and CG samples around AA circular mean.
    2. Build AA and CG histograms on a fixed ``[-180, 180)`` grid.
    3. Evaluate current type-9 potential on an absolute grid and convert it to
       a Boltzmann reference density ``rho_expected``.
    4. Build correction PMF from AA/CG mismatch and current potential
       contribution.
    5. Fit selected harmonics via ``_fit_type9_to_target`` and emit updated
       type-9 terms ``[i,j,k,l,9,phi0,k,mult,comment]``.

    Type-11 branch (CBT)
    --------------------
    1. Evaluate current type-11 potential from the first term.
    2. Form AA/CG densities and construct corrected PMF.
    3. Fit CBT coefficients ``(k_phi, a0..a4)`` with
       ``_fit_type11_to_target``.
    4. Replace the type-11 parameter block in the base term and keep indices,
       function id, and comment.

    Notes
    -----
    The routine is intentionally local: each torsion key is updated from its
    own AA/CG distributions without global coupling across torsions.
    """
    kB = 0.008314462618  # kJ mol^-1 K^-1
    kT = kB * CFG.temperature
    nbins = int(CFG.nbins)
    png_dir = Path(__file__).resolve().parent / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    def _save_pmf_plot(phi_grid, pmf, u_initial, u_updated, title: str, filename: str):
        """Save PMF/initial/updated potential overlay for one torsion key."""
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
        """Refit a type-9 torsion key from AA/CG mismatch on a fixed dihedral grid.

        Parameters
        ----------
        i, j, k, l : int
            Bead indices defining the torsion.
        aa_vals, cg_vals : array-like
            Sampled AA and CG dihedral values in degrees.
        terms : list
            Existing type-9 terms for this key; multiplicities are reused.

        Returns
        -------
        list[list]
            Updated topology rows in type-9 layout
            ``[i,j,k,l,9,phi0,k,mult,comment]``.

        Notes
        -----
        - Histograms are normalized and clipped by ``CFG.min_prob`` for
          numerical stability.
        - ``alpha`` controls PMF update strength and is bounded by
          ``CFG.alpha_min``/``CFG.alpha_max`` (with an extra cap when multiple
          terms already exist).
        """
        aa_vals = np.asarray(aa_vals, dtype=float)
        cg_vals = np.asarray(cg_vals, dtype=float)
        shift_aa = float(circular_mean(aa_vals))
        shift_cg = float(circular_mean(cg_vals))

        # Canonical absolute grid for potential summation/extraction.
        phi_centers = np.linspace(-180.0, 180.0, nbins, endpoint=False)

        aa_centered = wrap_to_180(aa_vals - shift_aa)
        cg_centered = wrap_to_180(cg_vals - shift_aa)

        phi_from_potential = wrap_to_180(phi_centers + shift_aa)
        U_expected_from_potential = _eval_type9_potential(terms, phi_from_potential)
        pot_density = np.exp(-U_expected_from_potential / kT)
        pot_density = np.clip(pot_density, CFG.min_prob, None)
        pot_density /= np.sum(pot_density)

        bins = np.linspace(-180.0, 180.0, nbins + 1)
        aa_density, _ = np.histogram(aa_centered, bins=bins, density=True)
        aa_density = np.clip(aa_density, CFG.min_prob, None)
        aa_density /= np.sum(aa_density)

        cg_density, _ = np.histogram(cg_centered, bins=bins, density=True)
        cg_density = np.clip(cg_density, CFG.min_prob, None)
        cg_density /= np.sum(cg_density)

        overlap = np.sum(aa_density * cg_density)
        aa_mean = np.average(aa_centered)
        aa_std = np.std(aa_centered)
        cg_mean = np.average(cg_centered)
        cg_std = np.std(cg_centered)
        delta_shift = float(np.abs(aa_mean - cg_mean))
        inv_rel_shift = np.sqrt(aa_std * cg_std) / (delta_shift + 1e-6)
        print(f"Dihedral ({i},{j},{k},{l}): AA-CG density overlap = {overlap:.4f}, inv rel shift = {inv_rel_shift:.4f}")
        alpha = overlap
        alpha = min(alpha, CFG.alpha_max)
        alpha = max(alpha, CFG.alpha_min)
        if len(terms) > 1:
            alpha = min(alpha, 0.02)  # Be more conservative when multiple terms already exist to avoid overfitting
        print(alpha)
        pmf_aa = -alpha * kT * np.log(aa_density)
        pmf_cg = -alpha * kT * np.log(cg_density)
        pmf_pot = -kT * np.log(pot_density)
        pmf = pmf_aa - pmf_cg + pmf_pot

        harmonics = sorted({int(t[7]) for t in terms})
        density_power = 1.0 if len(harmonics) == 1 else 0.0
        density = np.sqrt(aa_density * cg_density)
        weights_aa = np.pow(aa_density, density_power)
        weights_cg = np.pow(aa_density, density_power)

        comment = ""
        if terms and len(terms[0]) >= 9:
            comment = terms[0][8]

        # Fit each PMF component separately, sum the resulting fitted
        # potentials, then extract a final consolidated set of type-9 terms.
        summed_fitted_potential = np.zeros_like(phi_centers, dtype=float)
        component_specs = (
            (pmf, shift_aa, weights_aa),
        )
        for pmf_component, shift_component, weights_component in component_specs:
            component_fit = _fit_type9_to_target(
                pmf_component,
                shift=shift_component,
                harmonics=harmonics,
                weights=weights_component,
                phi_grid=phi_centers,
            )
            component_terms = []
            for mult, k_term_single, phi0_single in component_fit:
                component_terms.append(
                    [i, j, k, l, 9, float(phi0_single), float(k_term_single), int(mult), comment]
                )
        return component_terms

    def _update_type11_terms(terms, aa_vals, cg_vals):
        """Refit a type-11 (CBT) torsion key from AA/CG dihedral distributions.

        Parameters
        ----------
        terms : list
            Existing type-11 terms for one torsion key.
        aa_vals, cg_vals : array-like
            Sampled AA and CG dihedral values in degrees.

        Returns
        -------
        list[list]
            Single updated type-11 topology row with replaced
            ``k_phi, a0..a4`` coefficients.
        """
        aa_vals = np.asarray(aa_vals, dtype=float)
        cg_vals = np.asarray(cg_vals, dtype=float)
        phi_grid = np.linspace(-180.0, 180.0, nbins, endpoint=False)

        U_expected = _eval_type11_potential(terms[0], phi_grid)
        pot_density = np.exp(-U_expected / kT)
        pot_density = np.clip(pot_density, CFG.min_prob, None)
        pot_density /= np.sum(pot_density)

        aa_hist, _ = np.histogram(
            wrap_to_180(aa_vals),
            bins=nbins,
            range=(-180.0, 180.0),
            density=True,
        )
        aa_density = np.clip(aa_hist, CFG.min_prob, None)
        aa_density /= np.sum(aa_density)

        cg_density, _ = np.histogram(
            wrap_to_180(cg_vals),
            bins=nbins,
            range=(-180.0, 180.0),
            density=True,
        )
        cg_density = np.clip(cg_density, CFG.min_prob, None)
        cg_density /= np.sum(cg_density)

        alpha = CFG.alpha_max
        pmf = -kT * (alpha * (np.log(aa_density) - np.log(cg_density)) + np.log(pot_density))
        pmf -= np.min(pmf)
        weights = np.pow(aa_density, 0.20)
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
            k_phi_new = float(CFG.dihedral_k_lower)
            a_new = np.zeros(5, dtype=float)
        else:
            a_new = c_new / k_phi_new

        k_phi_new = float(np.clip(k_phi_new, CFG.dihedral_k_lower, CFG.dihedral_k_upper))
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
    """Refine a CG topology by iteratively updating bonded terms from AA-vs-CG mismatch.

    Strategy
    --------
    - Equilibrium values: shift by (mu_AA - mu_CG)
    - Force constants: rescale by (sigma_CG / sigma_AA)^2 (harmonic approximation)

    Notes
    -----
    - Bonds/constraints use linear mean/std.
    - Dihedrals use circular mean + std of wrapped residuals.
    - Writes a NEW ITP and returns the updated topology.

    Returns
    -------
    object
        Deep-copied topology after bond, angle, and dihedral updates and write-out.
    """
    updated = copy.deepcopy(topo)

    n_bonds_updated, n_constraints_updated = update_bonds(updated, aa_internal, cg_internal)
    n_angles_updated, n_angles_removed = update_angles(updated, aa_internal, cg_internal)
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
    # # DEBUG
    # ref_itp = outdir / f"{molname}_ref.itp"
    # topo = am.topology.read_itp(str(ref_itp))

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
    # tmp_itp = outdir / f"{molname}_tmp.itp"
    # shutil.copy2(in_itp, tmp_itp)  # Start from existing ITP to preserve formatting and any unmapped terms
    out_refined_itp = in_itp
    update_topology_from_cg_vs_aa(topo, aa_internal, cg_internal, out_refined_itp)

    if "plot" in sys.argv:
        plot_internal_coordinates_overlay(
            aa_internal,
            cg_internal,
            topo,
            output_file=wdir / "png" / "cg_vs_aa.png",
        )