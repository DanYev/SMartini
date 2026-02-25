import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lpmath import (
    boltzmann_inversion_angle,
    boltzmann_inversion_bond,
    boltzmann_inversion_improper,
    circular_mean,
    wrap_to_180,
    fit_gmm_1d_best,
    gmm_pdf_1d,
)

logger = logging.getLogger(__name__)


def _pair_key(i: int, j: int):
    i = int(i)
    j = int(j)
    return (i, j) if i <= j else (j, i)


def _distance_series_by_pair(internal_coords):
    """Map unordered atom pairs -> distance series.

    Internal-coordinate keys may be tagged as either 'bond' or 'constraint' depending
    on which topology was used when sampling. For plotting, the distance series is the
    same regardless of the tag, so we collapse both.
    """
    out = {}
    for k, v in internal_coords.items():
        if len(k) != 3:
            continue
        i, j, kind = k
        if kind not in ("bond", "constraint"):
            continue
        out[_pair_key(i, j)] = v
    return out


def _build_bonds_data_for_topo(internal_coords, topo):
    """Return bonds_data keyed by (i,j,type) matching the provided topology."""
    series_by_pair = _distance_series_by_pair(internal_coords)
    bonds_data = {}

    for b in topo.bonds:
        i, j = int(b[0]), int(b[1])
        distances = series_by_pair.get(_pair_key(i, j))
        if distances is not None:
            bonds_data[(i, j, "bond")] = distances

    for c in topo.constraints:
        i, j = int(c[0]), int(c[1])
        distances = series_by_pair.get(_pair_key(i, j))
        if distances is not None:
            bonds_data[(i, j, "constraint")] = distances

    # Fallback: if topo has no bonded terms, just plot whatever series exist.
    if not bonds_data:
        for k, v in internal_coords.items():
            if len(k) == 3 and k[2] in ("bond", "constraint"):
                bonds_data[k] = v

    return bonds_data


def _kT_kjmol(temperature: float) -> float:
    kB = 0.008314462618  # kJ/mol/K
    return float(kB * float(temperature))


def _normalize_density(x_grid, unnorm):
    x_grid = np.asarray(x_grid, dtype=float)
    unnorm = np.asarray(unnorm, dtype=float)
    area = float(np.trapz(unnorm, x_grid))
    if not np.isfinite(area) or area <= 0.0:
        return None
    return unnorm / area


def _boltzmann_density_from_U(x_grid, U, temperature: float, prefactor=None):
    """Return normalized density p(x) ~ prefactor(x) * exp(-U(x)/kT).

    Notes
    -----
    For internal coordinates, the Jacobian matters:
    - bond length: prefactor(r) = r^2
    - bond angle: prefactor(theta) = sin(theta)
    - dihedral: prefactor(phi) = 1
    """
    kT = _kT_kjmol(temperature)
    U = np.asarray(U, dtype=float)
    # Numerical stability: subtract min(U) before exponentiating.
    U0 = float(np.nanmin(U)) if U.size else 0.0
    unnorm = np.exp(-(U - U0) / kT)
    if prefactor is not None:
        pref = np.asarray(prefactor, dtype=float)
        unnorm = unnorm * pref
    return _normalize_density(x_grid, unnorm)


def _bond_U_harmonic(r_nm, r0_nm: float, k_kjmol_nm2: float):
    r_nm = np.asarray(r_nm, dtype=float)
    return 0.5 * float(k_kjmol_nm2) * (r_nm - float(r0_nm)) ** 2


def _angle_U_harmonic(theta_deg, theta0_deg: float, k_kjmol_rad2: float):
    theta_deg = np.asarray(theta_deg, dtype=float)
    dtheta_rad = np.deg2rad(theta_deg - float(theta0_deg))
    return 0.5 * float(k_kjmol_rad2) * dtheta_rad**2


def _dihedral_U_type9(phi_deg, terms):
    """Gromacs proper dihedral type-9: V = sum k * (1 + cos(n*phi - phi0))."""
    phi_deg = np.asarray(phi_deg, dtype=float)
    phi_rad = np.deg2rad(phi_deg)
    U = np.zeros_like(phi_rad, dtype=float)
    for t in terms:
        if t.get("k") is None or t.get("mult") is None or t.get("phi0") is None:
            continue
        k = float(t["k"])
        mult = int(t["mult"])
        phi0_rad = np.deg2rad(float(t["phi0"]))
        U += k * (1.0 + np.cos(mult * phi_rad - phi0_rad))
    return U


def _dihedral_U_type11(phi_deg, terms):
    """Gromacs dihedral funct=11 (combined bending-torsion, CBT).

    We evaluate the 1D form used by LigPar fitting (theta1=theta2=90 deg):
        V(phi) = kphi * sum_{n=0..4} a_n * cos(phi)^n

    Parameters are (kphi, a0..a4). If multiple funct=11 entries exist for the same
    dihedral, their contributions are summed.
    """
    phi_deg = np.asarray(phi_deg, dtype=float)
    phi_rad = np.deg2rad(phi_deg)
    c = np.cos(phi_rad)
    U = np.zeros_like(c, dtype=float)
    for t in terms:
        if t.get("kphi") is None or t.get("a") is None:
            continue
        kphi = float(t["kphi"])
        a = t["a"]
        if len(a) < 5:
            continue
        poly = (
            float(a[0])
            + float(a[1]) * c
            + float(a[2]) * (c**2)
            + float(a[3]) * (c**3)
            + float(a[4]) * (c**4)
        )
        U += kphi * poly
    return U


def plot_internal_coordinates(
    internal_coords,
    topo,
    output_file=None,
    max_gaussians=3,
    temperature: float = 300.0,
):
    """Plot histograms of internal coordinates (bonds, angles, dihedrals).

    Each subplot overlays the topology-implied Boltzmann density
    $p(x) \propto e^{-U(x)/kT}$ for the corresponding bonded potential.
    """
    # Bonds/constraints may have changed type (bond -> constraint) after fitting.
    # Always plot using the *current* topology list, but re-use whatever distance
    # series was sampled for that atom pair.
    bonds_data = _build_bonds_data_for_topo(internal_coords, topo)
    angles_data = {k: v for k, v in internal_coords.items() if k[-1] == "angle"}
    dihedrals_data = {k: v for k, v in internal_coords.items() if k[-1] == "dihedral"}

    if bonds_data:
        _plot_bonds(bonds_data, topo, output_file, max_gaussians, temperature)
    if angles_data:
        _plot_angles(angles_data, topo, output_file, max_gaussians, temperature)
    if dihedrals_data:
        _plot_dihedrals(dihedrals_data, topo, output_file, max_gaussians, temperature)


def _plot_gmm(ax, x_grid, gmm):
    weights, means, variances = gmm
    total = gmm_pdf_1d(x_grid, weights, means, variances)
    ax.plot(x_grid, total, color="C1", linewidth=1.6, )
    for idx, (w, mu, var) in enumerate(zip(weights, means, variances)):
        comp = w * np.exp(-0.5 * (x_grid - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)
        ax.plot(x_grid, comp, color="C1", alpha=0.5, linestyle="--", linewidth=1.0)


def _dihedral_terms(topo):
    terms = {}
    for d in topo.dihedrals:
        if len(d) < 6:
            continue
        # Only use Gromacs type-9 (proper dihedral) terms for Boltzmann density overlays.
        try:
            funct = int(d[4])
        except Exception:
            funct = None
        if funct is not None and funct != 9:
            continue
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        phi0 = float(d[5]) if len(d) >= 6 else None
        k = float(d[6]) if len(d) >= 7 else None
        mult = int(d[7]) if len(d) >= 8 else None
        terms.setdefault(key, []).append({"phi0": phi0, "k": k, "mult": mult})
    return terms


def _dihedral_terms_type11(topo):
    terms = {}
    for d in topo.dihedrals:
        if len(d) < 11:
            continue
        try:
            funct = int(d[4])
        except Exception:
            continue
        if funct != 11:
            continue
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        kphi = float(d[5])
        a = [float(d[6]), float(d[7]), float(d[8]), float(d[9]), float(d[10])]
        terms.setdefault(key, []).append({"kphi": kphi, "a": a})
    return terms


def _plot_bonds(bonds_data, topo, output_file, max_gaussians, temperature: float):
    logger.info("Plotting %s bonds/constraints", len(bonds_data))

    bond_params = {_pair_key(int(b[0]), int(b[1])): b for b in topo.bonds}
    constraint_params = {_pair_key(int(c[0]), int(c[1])): c for c in topo.constraints}

    n_plots = len(bonds_data)
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (key, distances) in enumerate(bonds_data.items()):
        ax = axes[idx]
        i, j, bond_type = key
        pair = _pair_key(i, j)

        if distances is None or len(distances) == 0:
            # Keep a placeholder axis so the bond/constraint still shows up.
            ax.text(0.5, 0.5, "No samples", transform=ax.transAxes, ha="center", va="center")
            distances = np.asarray([], dtype=float)

        if len(distances) > 0:
            ax.hist(distances, bins=30, alpha=0.7, edgecolor="black", density=True)
            gmm = fit_gmm_1d_best(distances, max_components=max_gaussians)
            if gmm is not None:
                x_grid = np.linspace(np.min(distances), np.max(distances), 300)
                _plot_gmm(ax, x_grid, gmm)

        # Overlay topology-implied density p(r) ~ exp(-U(r)/kT)
        if bond_type == "bond" and pair in bond_params:
            b = bond_params[pair]
            if len(b) >= 5:
                r0 = float(b[3])
                k = float(b[4])
                if len(distances) > 0:
                    x_grid = np.linspace(np.min(distances), np.max(distances), 400)
                else:
                    x_grid = np.linspace(r0 - 0.02, r0 + 0.02, 400)
                U = _bond_U_harmonic(x_grid, r0, k)
                p = _boltzmann_density_from_U(x_grid, U, temperature, prefactor=x_grid**2)
                if p is not None:
                    ax.plot(x_grid, p, color="black", linewidth=1.6, label=r"$p\propto e^{-U/kT}$")
        elif bond_type == "constraint" and pair in constraint_params:
            c = constraint_params[pair]
            if len(c) >= 4:
                r0 = float(c[3])
                # Constraints are delta-like; approximate with a very stiff harmonic.
                if len(distances) > 0:
                    x_grid = np.linspace(np.min(distances), np.max(distances), 400)
                    if float(np.ptp(x_grid)) <= 0:
                        x_grid = np.linspace(r0 - 1e-3, r0 + 1e-3, 400)
                else:
                    x_grid = np.linspace(r0 - 1e-3, r0 + 1e-3, 400)
                k_eff = 1.0e6  # kJ/mol/nm^2 (visualization-only)
                U = _bond_U_harmonic(x_grid, r0, k_eff)
                p = _boltzmann_density_from_U(x_grid, U, temperature, prefactor=x_grid**2)
                if p is not None:
                    ax.plot(x_grid, p, color="black", linewidth=1.6, label=r"$p\propto e^{-U/kT}$")

        ax.set_xlabel("Distance (nm)", fontsize=9)
        ax.set_title(f"{bond_type.capitalize()}: {i+1}-{j+1}", fontsize=10)
        # ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        if len(distances) > 1:
            _, k_calc = boltzmann_inversion_bond(distances)
        else:
            k_calc = float("nan")

        param_k = None
        if bond_type == "bond" and pair in bond_params and len(bond_params[pair]) >= 5:
            param_k = float(bond_params[pair][4])

        if len(distances) > 0:
            mean_dist = float(np.mean(distances))
            std_dist = float(np.std(distances))
            stats_text = f"mu={mean_dist:.3f}\nsigma={std_dist:.3f}\n"
        else:
            stats_text = "(no samples)\n"

        if np.isfinite(k_calc):
            k_rounded = round(k_calc / 1000) * 1000
            stats_text += f"k={int(k_rounded/1000)}e3"
        if param_k is not None:
            param_k_rounded = round(param_k / 1000) * 1000
            stats_text += f"\nparam k={int(param_k_rounded/1000)}e3"

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    _save_or_show(output_file, "bonds")


def _plot_angles(angles_data, topo, output_file, max_gaussians, temperature: float):
    logger.info("Plotting %s angles", len(angles_data))

    n_plots = len(angles_data)
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (key, angles) in enumerate(angles_data.items()):
        ax = axes[idx]
        i, j, k, angle_type = key

        vmin = float(np.min(angles))
        vmax = float(np.max(angles))
        if vmin == vmax:
            vmin -= 1e-3
            vmax += 1e-3

        ax.hist(angles, bins=30, range=(vmin, vmax), alpha=0.7, edgecolor="black", density=True)
        gmm = fit_gmm_1d_best(angles, max_components=max_gaussians)
        if gmm is not None:
            x_grid = np.linspace(vmin, vmax, 300)
            _plot_gmm(ax, x_grid, gmm)

        # Overlay topology-implied density p(theta) ~ exp(-U(theta)/kT)
        for angle in topo.angles:
            if int(angle[0]) == i and int(angle[1]) == j and int(angle[2]) == k and len(angle) >= 6:
                theta0 = float(angle[4])
                k_param = float(angle[5])
                x_grid = np.linspace(vmin, vmax, 400)
                U = _angle_U_harmonic(x_grid, theta0, k_param)
                jac = np.clip(np.sin(np.deg2rad(x_grid)), 0.0, None)
                p = _boltzmann_density_from_U(x_grid, U, temperature, prefactor=jac)
                if p is not None:
                    ax.plot(x_grid, p, color="black", linewidth=1.6, label=r"$p\propto e^{-U/kT}$")
                break

        ax.set_xlabel("Angle (degrees)", fontsize=9)
        ax.set_title(f"Angle: {i+1}-{j+1}-{k+1}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.set_xlim(vmin, vmax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        _, k_calc = boltzmann_inversion_angle(angles)

        param_k = None
        for angle in topo.angles:
            if int(angle[0]) == i and int(angle[1]) == j and int(angle[2]) == k and len(angle) >= 6:
                param_k = float(angle[5])
                break

        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        k_rounded = round(k_calc / 10) * 10
        stats_text = f"mu={mean_angle:.1f} deg\nsigma={std_angle:.1f} deg\n"
        stats_text += f"k={int(k_rounded)}"
        if param_k is not None:
            param_k_rounded = round(param_k / 10) * 10
            stats_text += f"\nparam k={int(param_k_rounded)}"

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    _save_or_show(output_file, "angles")


def _plot_dihedrals(dihedrals_data, topo, output_file, max_gaussians, temperature: float):
    logger.info("Plotting %s dihedrals", len(dihedrals_data))

    dihedral_terms = _dihedral_terms(topo)
    dihedral_terms_11 = _dihedral_terms_type11(topo)

    n_plots = len(dihedrals_data)
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (key, dihedrals) in enumerate(dihedrals_data.items()):
        ax = axes[idx]
        i, j, k, l, dihedral_type = key

        # Plot in absolute wrapped coordinates so we match the potential definition.
        # dihedrals_wrapped = wrap_to_180(np.asarray(dihedrals, dtype=float))
        shift = float(circular_mean(dihedrals))
        dihedrals_wrapped = wrap_to_180(dihedrals - shift)
        vmin, vmax = -180.0, 180.0

        ax.hist(
            dihedrals_wrapped,
            bins=72,
            range=(vmin, vmax),
            alpha=0.7,
            edgecolor="black",
            density=True,
        )
        gmm = fit_gmm_1d_best(dihedrals_wrapped, max_components=max_gaussians)
        if gmm is not None:
            x_grid = np.linspace(vmin, vmax, 600)
            _plot_gmm(ax, x_grid, gmm)

        # Overlay topology-implied density p(phi) ~ exp(-U(phi)/kT)
        terms11 = dihedral_terms_11.get((i, j, k, l), [])
        terms9 = dihedral_terms.get((i, j, k, l), [])

        x_shifted = x_grid + shift
        if terms11:
            U = _dihedral_U_type11(x_shifted, terms11)
            p = _boltzmann_density_from_U(x_shifted, U, temperature)
            ax.plot(
                x_grid,
                p,
                color="black",
                linewidth=1.6,
                label=r"funct=11: $p\propto e^{-U/kT}$",
            )
        elif terms9:
            U = _dihedral_U_type9(x_shifted, terms9)
            p = _boltzmann_density_from_U(x_shifted, U, temperature)
            ax.plot(
                x_grid,
                p,
                color="black",
                linewidth=1.6,
                label=r"funct=9: $p\propto e^{-U/kT}$",
            )

        ax.set_xlabel(f"Dihedral (deg) - shift={shift:.1f}", fontsize=9)
        ax.set_title(f"Dihedral: {i+1}-{j+1}-{k+1}-{l+1}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.set_xlim(vmin, vmax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        ax.legend(fontsize=8)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    _save_or_show(output_file, "dihedrals")


def plot_internal_coordinates_overlay(aa_coords, cg_coords, topo, output_file=None):
    """Plot AA and CG histograms for bonds, angles, and dihedrals."""
    bonds_aa = _build_bonds_data_for_topo(aa_coords, topo)
    angles_aa = {k: v for k, v in aa_coords.items() if k[-1] == "angle"}
    dihedrals_aa = {k: v for k, v in aa_coords.items() if k[-1] == "dihedral"}

    bonds_cg = _build_bonds_data_for_topo(cg_coords, topo)
    angles_cg = {k: v for k, v in cg_coords.items() if k[-1] == "angle"}
    dihedrals_cg = {k: v for k, v in cg_coords.items() if k[-1] == "dihedral"}

    if bonds_aa or bonds_cg:
        _plot_bonds_overlay(bonds_aa, bonds_cg, topo, output_file)
    if angles_aa or angles_cg:
        _plot_angles_overlay(angles_aa, angles_cg, topo, output_file)
    if dihedrals_aa or dihedrals_cg:
        _plot_dihedrals_overlay(dihedrals_aa, dihedrals_cg, topo, output_file)


def _resolve_keys(bonds_aa, bonds_cg, topo):
    keys = []
    for bond in topo.bonds:
        keys.append((int(bond[0]), int(bond[1]), "bond"))
    for constraint in topo.constraints:
        keys.append((int(constraint[0]), int(constraint[1]), "constraint"))
    if not keys:
        keys = list(set(bonds_aa.keys()) | set(bonds_cg.keys()))
    return keys


def _plot_bonds_overlay(bonds_aa, bonds_cg, topo, output_file):
    logger.info("Plotting %s bonds/constraints", len(set(bonds_aa) | set(bonds_cg)))

    bond_ref = {(int(b[0]), int(b[1])): b[3] for b in topo.bonds}
    constraint_ref = {(int(c[0]), int(c[1])): c[3] for c in topo.constraints}

    keys = _resolve_keys(bonds_aa, bonds_cg, topo)
    n_plots = len(keys)
    if n_plots == 0:
        return

    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        i, j, bond_type = key
        aa_vals = bonds_aa.get(key)
        cg_vals = bonds_cg.get(key)
        bins = _common_bins(aa_vals, cg_vals, bins=30)
        hist_range = _preferred_range(aa_vals, cg_vals)

        _plot_hist_pair(ax, aa_vals, cg_vals, bins=bins, hist_range=hist_range)

        if bond_type == "bond" and (i, j) in bond_ref:
            ref_length = bond_ref[(i, j)]
            ax.axvline(ref_length, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_length:.3f}")
        elif bond_type == "constraint" and (i, j) in constraint_ref:
            ref_length = constraint_ref[(i, j)]
            ax.axvline(ref_length, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_length:.3f}")

        ax.set_xlabel("Distance (nm)", fontsize=9)
        ax.set_title(f"{bond_type.capitalize()}: {i+1}-{j+1}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        if hist_range is not None:
            ax.set_xlim(hist_range)

        _add_stats_box(ax, aa_vals, cg_vals, value_type="bond")
        ax.legend(fontsize=8)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    _save_or_show(output_file, "bonds")


def _plot_angles_overlay(angles_aa, angles_cg, topo, output_file):
    logger.info("Plotting %s angles", len(set(angles_aa) | set(angles_cg)))

    angle_ref = {(int(a[0]), int(a[1]), int(a[2])): a[4] for a in topo.angles}
    keys = [(int(a[0]), int(a[1]), int(a[2]), "angle") for a in topo.angles]
    if not keys:
        keys = list(set(angles_aa.keys()) | set(angles_cg.keys()))
    keys = [k for k in keys if k in angles_aa or k in angles_cg]

    n_plots = len(keys)
    if n_plots == 0:
        return

    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        i, j, k, angle_type = key
        aa_vals = angles_aa.get(key)
        cg_vals = angles_cg.get(key)

        bins = _common_bins(aa_vals, cg_vals, bins=30)
        hist_range = _preferred_range(aa_vals, cg_vals)

        _plot_hist_pair(ax, aa_vals, cg_vals, bins=bins, hist_range=hist_range)

        if (i, j, k) in angle_ref:
            ref_angle = angle_ref[(i, j, k)]
            ax.axvline(ref_angle, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_angle:.1f} deg")

        ax.set_xlabel("Angle (degrees)", fontsize=9)
        ax.set_title(f"Angle: {i+1}-{j+1}-{k+1}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        if hist_range is not None:
            ax.set_xlim(hist_range)

        _add_stats_box(ax, aa_vals, cg_vals, value_type="angle")
        ax.legend(fontsize=8)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    _save_or_show(output_file, "angles")


def _plot_dihedrals_overlay(dihedrals_aa, dihedrals_cg, topo, output_file):
    logger.info("Plotting %s dihedrals", len(set(dihedrals_aa) | set(dihedrals_cg)))

    dihedral_terms = _dihedral_terms(topo)
    # topo.dihedrals may contain multiple terms per (i,j,k,l) (e.g., multiple multiplicities)
    # but we want exactly one axis per dihedral definition.
    keys = []
    seen = set()
    for d in topo.dihedrals:
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]), "dihedral")
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    if not keys:
        keys = list(set(dihedrals_aa.keys()) | set(dihedrals_cg.keys()))
    keys = [k for k in keys if k in dihedrals_aa or k in dihedrals_cg]

    n_plots = len(keys)
    if n_plots == 0:
        return

    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        i, j, k, l, _ = key
        # aa_vals = wrap_to_180(dihedrals_aa.get(key))
        # cg_vals = wrap_to_180(dihedrals_cg.get(key))
        aa_vals = dihedrals_aa.get(key)
        cg_vals = dihedrals_cg.get(key)

        aa_shift = circular_mean(aa_vals)
        cg_shift = circular_mean(cg_vals)
        aa_shifted = wrap_to_180(aa_vals - aa_shift)
        cg_shifted = wrap_to_180(cg_vals - aa_shift)

        _plot_hist_pair(ax, aa_shifted, cg_shifted, bins=30)

        ax.set_xlabel(f"Dihedral - {aa_shift:.1f} deg", fontsize=9)
        ax.set_title(f"Dihedral: {i+1}-{j+1}-{k+1}-{l+1}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        _add_stats_box(ax, aa_vals, cg_vals, value_type="dihedral", dihedral_center=aa_shift)
        ax.legend(fontsize=8)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    _save_or_show(output_file, "dihedrals")


def _common_bins(aa_vals, cg_vals, bins=30):
    all_vals = []
    if aa_vals is not None:
        all_vals.append(aa_vals)
    if cg_vals is not None:
        all_vals.append(cg_vals)
    if not all_vals:
        return bins
    combined = np.concatenate(all_vals)
    vmin = float(np.min(combined))
    vmax = float(np.max(combined))
    if vmin == vmax:
        vmin -= 1e-3
        vmax += 1e-3
    return np.linspace(vmin, vmax, bins + 1)


def _plot_hist_pair(ax, aa_vals, cg_vals, bins=30, hist_range=None):
    if aa_vals is not None:
        ax.hist(
            aa_vals,
            bins=bins,
            range=hist_range,
            density=True,
            alpha=0.55,
            color="tab:blue",
            edgecolor="black",
            label="AA",
        )
    if cg_vals is not None:
        ax.hist(
            cg_vals,
            bins=bins,
            range=hist_range,
            density=True,
            alpha=0.55,
            color="tab:orange",
            edgecolor="black",
            label="CG",
        )


def _preferred_range(aa_vals, cg_vals):
    all_vals = []
    if aa_vals is not None and len(aa_vals) > 0:
        all_vals.append(aa_vals)
    if cg_vals is not None and len(cg_vals) > 0:
        all_vals.append(cg_vals)
    if not all_vals:
        return None
    combined = np.concatenate(all_vals)

    vmin = float(np.min(combined))
    vmax = float(np.max(combined))
    if vmin == vmax:
        vmin -= 1e-3
        vmax += 1e-3
    return (vmin, vmax)


def _add_stats_box(ax, aa_vals, cg_vals, value_type, dihedral_center=None):
    lines = []
    if aa_vals is not None:
        mu, sigma, mu_shifted = _compute_stats(aa_vals, value_type, dihedral_center=dihedral_center)
        if value_type == "dihedral" and mu_shifted is not None:
            lines.append(f"AA mu_shift={mu_shifted:.3f} sigma={sigma:.3f}")
        else:
            lines.append(f"AA mu={mu:.3f} sigma={sigma:.3f}")
    if cg_vals is not None:
        mu, sigma, mu_shifted = _compute_stats(cg_vals, value_type, dihedral_center=dihedral_center)
        if value_type == "dihedral" and mu_shifted is not None:
            lines.append(f"CG mu_shift={mu_shifted:.3f} sigma={sigma:.3f}")
        else:
            lines.append(f"CG mu={mu:.3f} sigma={sigma:.3f}")
    if not lines:
        return

    ax.text(
        0.98,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def _compute_stats(values, value_type, dihedral_center=None):
    mu_shifted = None
    if value_type == "dihedral":
        mean_val = circular_mean(values)
        centered = wrap_to_180(values - mean_val)
        std_val = float(np.std(centered))
        if dihedral_center is not None:
            mu_shifted = float(wrap_to_180(mean_val - dihedral_center))
    else:
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
    return float(mean_val), std_val, mu_shifted


def _save_or_show(output_file, suffix):
    if output_file:
        base = Path(output_file).stem if isinstance(output_file, (str, Path)) else "internal_coords"
        out_path = Path(output_file).parent / f"{base}_{suffix}.png" if isinstance(output_file, (str, Path)) else f"{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving %s plot to %s", suffix, out_path)
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
