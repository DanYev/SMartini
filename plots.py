import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lpmath import (
    boltzmann_inversion_angle,
    boltzmann_inversion_bond,
    boltzmann_inversion_dihedral,
    circular_mean_deg,
    wrap_to_180,
)

logger = logging.getLogger(__name__)


def plot_internal_coordinates(internal_coords, topo, output_file=None):
    """Plot histograms of internal coordinates (bonds, angles, dihedrals)."""
    bonds_data = {k: v for k, v in internal_coords.items() if k[-1] in ["bond", "constraint"]}
    angles_data = {k: v for k, v in internal_coords.items() if k[-1] == "angle"}
    dihedrals_data = {k: v for k, v in internal_coords.items() if k[-1] == "dihedral"}

    if bonds_data:
        _plot_bonds(bonds_data, topo, output_file)
    if angles_data:
        _plot_angles(angles_data, topo, output_file)
    if dihedrals_data:
        _plot_dihedrals(dihedrals_data, topo, output_file)


def _plot_bonds(bonds_data, topo, output_file):
    logger.info("Plotting %s bonds/constraints", len(bonds_data))

    bond_ref = {(int(b[0]), int(b[1])): b[3] for b in topo.bonds}
    constraint_ref = {(int(c[0]), int(c[1])): c[3] for c in topo.constraints}

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

        ax.hist(distances, bins=30, alpha=0.7, edgecolor="black")

        if bond_type == "bond" and (i, j) in bond_ref:
            ref_length = bond_ref[(i, j)]
            ax.axvline(ref_length, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_length:.3f}")
        elif bond_type == "constraint" and (i, j) in constraint_ref:
            ref_length = constraint_ref[(i, j)]
            ax.axvline(ref_length, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_length:.3f}")

        ax.set_xlabel("Distance (nm)", fontsize=9)
        ax.set_title(f"{bond_type.capitalize()}: {i+1}-{j+1}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        _, k_calc = boltzmann_inversion_bond(distances)

        ref_fc = None
        if bond_type == "bond":
            for bond in topo.bonds:
                if int(bond[0]) == i and int(bond[1]) == j and len(bond) >= 5:
                    ref_fc = bond[4]
                    break

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        k_rounded = round(k_calc / 1000) * 1000
        stats_text = f"mu={mean_dist:.3f}\nsigma={std_dist:.3f}\n"
        stats_text += f"k={int(k_rounded/1000)}e3"
        if ref_fc is not None:
            ref_k_rounded = round(ref_fc / 1000) * 1000
            stats_text += f"\nITP k={int(ref_k_rounded/1000)}e3"

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


def _plot_angles(angles_data, topo, output_file):
    logger.info("Plotting %s angles", len(angles_data))

    angle_ref = {(int(a[0]), int(a[1]), int(a[2])): a[4] for a in topo.angles}

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

        ax.hist(angles, bins=30, range=(vmin, vmax), alpha=0.7, edgecolor="black")

        if (i, j, k) in angle_ref:
            ref_angle = angle_ref[(i, j, k)]
            ax.axvline(ref_angle, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_angle:.1f} deg")

        ax.set_xlabel("Angle (degrees)", fontsize=9)
        ax.set_title(f"Angle: {i+1}-{j+1}-{k+1}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.set_xlim(vmin, vmax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        _, k_calc = boltzmann_inversion_angle(angles)

        ref_fc = None
        for angle in topo.angles:
            if int(angle[0]) == i and int(angle[1]) == j and int(angle[2]) == k and len(angle) >= 6:
                ref_fc = angle[5]
                break

        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        k_rounded = round(k_calc / 10) * 10
        stats_text = f"mu={mean_angle:.1f} deg\nsigma={std_angle:.1f} deg\n"
        stats_text += f"k={int(k_rounded)}"
        if ref_fc is not None:
            ref_k_rounded = round(ref_fc / 10) * 10
            stats_text += f"\nITP k={int(ref_k_rounded)}"

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


def _plot_dihedrals(dihedrals_data, topo, output_file):
    logger.info("Plotting %s dihedrals", len(dihedrals_data))

    dihedral_ref = {(int(d[0]), int(d[1]), int(d[2]), int(d[3])): d[5] for d in topo.dihedrals}

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

        circ_mean = circular_mean_deg(dihedrals)
        dihedrals_shifted = wrap_to_180(dihedrals - circ_mean)

        ax.hist(dihedrals_shifted, bins=30, range=(-180, 180), alpha=0.7, edgecolor="black")

        if (i, j, k, l) in dihedral_ref:
            ref_dihedral = dihedral_ref[(i, j, k, l)]
            ref_dihedral_shifted = wrap_to_180(ref_dihedral - circ_mean)
            ax.axvline(ref_dihedral_shifted, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_dihedral:.1f} deg")

        ax.set_xlabel(f"Dihedral - {circ_mean:.1f} deg", fontsize=9)
        ax.set_title(f"Dihedral: {i+1}-{j+1}-{k+1}-{l+1}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.set_xlim(-180, 180)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        _, k_calc = boltzmann_inversion_dihedral(dihedrals)

        ax.legend(fontsize=8)

        ref_fc = None
        for dihedral in topo.dihedrals:
            if int(dihedral[0]) == i and int(dihedral[1]) == j and int(dihedral[2]) == k and int(dihedral[3]) == l and len(dihedral) >= 7:
                ref_fc = dihedral[6]
                break

        mean_dihedral = circular_mean_deg(dihedrals)
        dihedrals_centered = wrap_to_180(dihedrals - mean_dihedral)
        std_dihedral = np.std(dihedrals_centered)
        k_rounded = round(k_calc / 10) * 10
        stats_text = f"mu={mean_dihedral:.1f} deg\nsigma={std_dihedral:.1f} deg\n"
        stats_text += f"k={int(k_rounded)}"
        if ref_fc is not None:
            ref_k_rounded = round(ref_fc / 10) * 10
            stats_text += f"\nITP k={int(ref_k_rounded)}"

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
    _save_or_show(output_file, "dihedrals")


def plot_internal_coordinates_overlay(aa_coords, cg_coords, topo, output_file=None):
    """Plot AA and CG histograms for bonds, angles, and dihedrals."""
    bonds_aa = {k: v for k, v in aa_coords.items() if k[-1] in ["bond", "constraint"]}
    angles_aa = {k: v for k, v in aa_coords.items() if k[-1] == "angle"}
    dihedrals_aa = {k: v for k, v in aa_coords.items() if k[-1] == "dihedral"}

    bonds_cg = {k: v for k, v in cg_coords.items() if k[-1] in ["bond", "constraint"]}
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
    keys = [k for k in keys if k in bonds_aa or k in bonds_cg]
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

    dihedral_ref = {(int(d[0]), int(d[1]), int(d[2]), int(d[3])): d[5] for d in topo.dihedrals}
    keys = [(int(d[0]), int(d[1]), int(d[2]), int(d[3]), "dihedral") for d in topo.dihedrals]
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
        aa_vals = dihedrals_aa.get(key)
        cg_vals = dihedrals_cg.get(key)

        circ_mean = _reference_circ_mean(aa_vals, cg_vals)
        aa_shifted = _shift_dihedrals(aa_vals, circ_mean)
        cg_shifted = _shift_dihedrals(cg_vals, circ_mean)

        _plot_hist_pair(ax, aa_shifted, cg_shifted, bins=30)

        if (i, j, k, l) in dihedral_ref:
            ref_dihedral = dihedral_ref[(i, j, k, l)]
            ref_shifted = wrap_to_180(ref_dihedral - circ_mean)
            ax.axvline(ref_shifted, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_dihedral:.1f} deg")

        ax.set_xlabel(f"Dihedral - {circ_mean:.1f} deg", fontsize=9)
        ax.set_title(f"Dihedral: {i+1}-{j+1}-{k+1}-{l+1}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        _add_stats_box(ax, aa_vals, cg_vals, value_type="dihedral", dihedral_center=circ_mean)
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
        mean_val = circular_mean_deg(values)
        centered = wrap_to_180(values - mean_val)
        std_val = float(np.std(centered))
        if dihedral_center is not None:
            mu_shifted = float(wrap_to_180(mean_val - dihedral_center))
    else:
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
    return float(mean_val), std_val, mu_shifted


def _reference_circ_mean(aa_vals, cg_vals):
    if aa_vals is not None and len(aa_vals) > 0:
        return circular_mean_deg(aa_vals)
    if cg_vals is not None and len(cg_vals) > 0:
        return circular_mean_deg(cg_vals)
    return 0.0


def _shift_dihedrals(values, center):
    if values is None:
        return None
    return wrap_to_180(values - center)


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
