import copy
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from lpmath import circular_mean_deg, wrap_to_180, fit_type9_dihedral
from ligpar_config import CFG

logger = logging.getLogger(__name__)


InternalCoords = Dict[Tuple[int, ...], np.ndarray]


@dataclass
class RefineSettings:
    angle_k_min: Optional[float] = CFG.angle_k_cutoff
    dihedral_k_min: Optional[float] = CFG.dihedral_k_cutoff

    # Optional guards against extreme updates
    max_k_scale: float = CFG.refine_max_k_scale

    # Under-relaxation for dihedral equilibrium updates (helps stability)
    dihedral_shift_scale: float = CFG.refine_dihedral_shift_scale


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


def refine_topology_from_cg_vs_aa(
    topo,
    aa_internal: InternalCoords,
    cg_internal: InternalCoords,
    output_itp,
    settings: Optional[RefineSettings] = None,
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

    if settings is None:
        settings = RefineSettings()

    updated = copy.deepcopy(topo)

    n_bonds_updated = 0
    n_constraints_updated = 0
    n_angles_updated = 0
    n_angles_removed = 0
    n_dihedrals_updated = 0
    n_dihedrals_removed = 0

    # Bonds
    new_bonds = []
    for bond in updated.bonds:
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
            k_new = _k_rescale(k_old, sigma_target=sigma_aa, sigma_current=sigma_cg, max_scale=settings.max_k_scale)
            new_bonds.append([i, j, bond[2], r0_new, k_new])
        else:
            new_bonds.append([i, j, bond[2], r0_new])

        n_bonds_updated += 1

    updated.bonds = new_bonds

    # Constraints (length only)
    new_constraints = []
    for constraint in updated.constraints:
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

    updated.constraints = new_constraints

    # Angles
    new_angles = []
    for angle in updated.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        key = (i, j, k, "angle")
        aa_vals = aa_internal.get(key)
        cg_vals = cg_internal.get(key)
        if aa_vals is None or cg_vals is None:
            # Optional pruning based on existing k
            if settings.angle_k_min is not None and len(angle) >= 6 and float(angle[5]) < settings.angle_k_min:
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
            k_new = _k_rescale(k_old, sigma_target=sigma_aa, sigma_current=sigma_cg, max_scale=settings.max_k_scale)
            if settings.angle_k_min is not None and k_new < settings.angle_k_min:
                n_angles_removed += 1
                continue
            new_angles.append([i, j, k, angle[3], theta0_new, k_new])
        else:
            new_angles.append([i, j, k, angle[3], theta0_new])

        n_angles_updated += 1

    updated.angles = new_angles

    # Dihedrals (type-9 Fourier terms)
    new_dihedrals = []
    dihedrals_by_key = {}
    for dihedral in updated.dihedrals:
        key = (int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3]))
        dihedrals_by_key.setdefault(key, []).append(dihedral)

    for (i, j, k, l), terms in dihedrals_by_key.items():
        key = (i, j, k, l, "dihedral")
        aa_vals = aa_internal.get(key)
        if aa_vals is None:
            for term in terms:
                if settings.dihedral_k_min is not None and len(term) >= 7 and abs(float(term[6])) < settings.dihedral_k_min:
                    n_dihedrals_removed += 1
                    continue
                new_dihedrals.append(term)
            continue

        fit_terms = fit_type9_dihedral(
            aa_vals,
            temperature=CFG.type9_temperature,
            max_n=CFG.type9_max_n,
            bins=CFG.type9_bins,
            min_prob=CFG.type9_min_prob,
            fit_mode=CFG.type9_fit_mode,
        )
        if not fit_terms:
            # Mirror Boltzmann fitting behavior: if the fit fails/returns nothing,
            # keep the existing terms (subject to the same |k| cutoff).
            for term in terms:
                if (
                    settings.dihedral_k_min is not None
                    and len(term) >= 7
                    and abs(float(term[6])) < settings.dihedral_k_min
                ):
                    n_dihedrals_removed += 1
                    continue
                new_dihedrals.append(term)
            continue

        kept_terms = []
        for term in fit_terms:
            if len(term) == 2:
                mult, k_term = term
                phi0 = 0.0
            else:
                mult, k_term, phi0 = term

            k_new = float(k_term)

            if settings.dihedral_k_min is not None and abs(k_new) < settings.dihedral_k_min:
                n_dihedrals_removed += 1
                continue

            kept_terms.append([i, j, k, l, 9, float(phi0), k_new, int(mult)])

        if not kept_terms and fit_terms:
            best = max(fit_terms, key=lambda t: abs(t[1]))
            if len(best) == 2:
                best_mult, best_k = best
                best_phi0 = 0.0
            else:
                best_mult, best_k, best_phi0 = best
            kept_terms.append([i, j, k, l, 9, float(best_phi0), float(best_k), int(best_mult)])

        new_dihedrals.extend(kept_terms)
        n_dihedrals_updated += len(kept_terms)

    updated.dihedrals = new_dihedrals

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
