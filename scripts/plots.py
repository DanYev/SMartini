"""Plots for SASA, RMSD, and Wasserstein analysis.

Provides configurable plotting functions with consistent styling suitable
for journal publication.  All functions accept keyword arguments for
fine-tuning colours, sizes, and output format.

Usage::

    from scripts.plots import plot_sasa_rmsd, plot_wasserstein

    plot_sasa_rmsd(all_results, output_dir)
    plot_wasserstein(all_results, output_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default style – tweak these constants or override via rc_context in callers.
# ---------------------------------------------------------------------------

DEFAULT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colour palette
PALETTE = {
    "aa": "#4C72B0",        # muted blue
    "cg": "#DD8452",        # muted orange
    "bond": "#55A868",      # green
    "angle": "#C44E52",     # red
    "dihedral": "#937860",  # brown
}


def _or_zero(v):
    """Return *v* if not None, else 0."""
    return v if v is not None else 0


def _or_nan(v):
    """Return *v* if not None, else ``float('nan')``."""
    return v if v is not None else float("nan")


# ---------------------------------------------------------------------------
# SASA / RMSD grouped bar chart
# ---------------------------------------------------------------------------

def _draw_err(ax, x, y, yerr_lo, yerr_hi, kw):
    """Draw error bars on *ax* with high zorder so they sit above bars."""
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi],
                fmt="none", **kw)


def plot_sasa_rmsd(
    all_results: list[dict],
    out_dir: str | Path,
    *,
    figsize: tuple[float, float] = (11.0, 3.2),
    dpi: int = 300,
    palette: dict | None = None,
    aa_label: str = "AA",
    cg_label: str = "CG",
    sasa_ylabel: str = "SASA / nm²",
    rmsd_ylabel: str = "RMSD / Å",
    rg_ylabel: str = "Rg / Å",
    bar_width: float = 0.30,
    err_capsize: float = 3.0,
    png_name: str = "sasa_rmsd_bars.png",
    **kwargs,
) -> Path:
    """Grouped bar chart comparing per-ligand mean SASA, RMSD, and Rg (AA vs CG).

    Draws 95% bootstrap confidence intervals as error bars when available.

    Parameters
    ----------
    all_results : list[dict]
        Each dict must contain at least ``"ligand"``.  Optional keys:
        ``"aa_sasa_mean"``, ``"cg_sasa_mean"``, ``"aa_rmsd_mean"``,
        ``"cg_rmsd_mean"``, ``"aa_rg_mean"``, ``"cg_rg_mean"``.
        Bootstrap CI keys: ``"aa_sasa_ci_lo"``, ``"aa_sasa_ci_hi"``, etc.
    out_dir : str or Path
        Directory in which the PNG is saved.
    figsize : (float, float)
        Figure size in inches.
    dpi : int
        Output resolution.
    palette : dict or None
        Colour overrides.  Keys: ``"aa"``, ``"cg"``.
    aa_label, cg_label : str
        Legend labels for all-atom and coarse-grained bars.
    sasa_ylabel, rmsd_ylabel, rg_ylabel : str
        Axis labels.
    bar_width : float
        Width of each bar in data units.
    err_capsize : float
        Cap size for error bars in points.
    png_name : str
        Output filename.
    **kwargs
        Passed through to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    colors = {**PALETTE, **(palette or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ligands = [r["ligand"] for r in all_results]
    n = len(ligands)
    x = np.arange(n)

    # --- data ---
    aa_sasa = [_or_zero(r.get("aa_sasa_mean")) for r in all_results]
    cg_sasa = [_or_zero(r.get("cg_sasa_mean")) for r in all_results]
    aa_rmsd = [_or_zero(r.get("aa_rmsd_mean")) * 10 for r in all_results]
    cg_rmsd = [_or_zero(r.get("cg_rmsd_mean")) * 10 for r in all_results]
    aa_rg = [_or_zero(r.get("aa_rg_mean")) * 10 for r in all_results]
    cg_rg = [_or_zero(r.get("cg_rg_mean")) * 10 for r in all_results]

    # --- error bars (bootstrap CI) ---
    def _err_lo_hi(results, prefix):
        """Build (n, 2) error arrays: [mean - lo, hi - mean]."""
        means = np.array([_or_zero(r.get(f"{prefix}_mean")) for r in results])
        los = np.array([_or_nan(r.get(f"{prefix}_ci_lo")) for r in results])
        his = np.array([_or_nan(r.get(f"{prefix}_ci_hi")) for r in results])
        los = np.where(np.isnan(los), means, los)
        his = np.where(np.isnan(his), means, his)
        return np.abs(means - los), np.abs(his - means)

    aa_sasa_lo, aa_sasa_hi = _err_lo_hi(all_results, "aa_sasa")
    cg_sasa_lo, cg_sasa_hi = _err_lo_hi(all_results, "cg_sasa")
    aa_rmsd_lo, aa_rmsd_hi = _err_lo_hi(all_results, "aa_rmsd")
    cg_rmsd_lo, cg_rmsd_hi = _err_lo_hi(all_results, "cg_rmsd")
    aa_rg_lo, aa_rg_hi = _err_lo_hi(all_results, "aa_rg")
    cg_rg_lo, cg_rg_hi = _err_lo_hi(all_results, "cg_rg")
    # Scale error bars nm → Å
    aa_rmsd_lo *= 10; aa_rmsd_hi *= 10
    cg_rmsd_lo *= 10; cg_rmsd_hi *= 10
    aa_rg_lo *= 10; aa_rg_hi *= 10
    cg_rg_lo *= 10; cg_rg_hi *= 10

    err_kw = dict(elinewidth=0.8, capsize=err_capsize, capthick=0.8,
                  ecolor="dimgray", zorder=10)

    with plt.rc_context(DEFAULT_STYLE):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, **kwargs)

        # -- SASA --
        ax1.bar(x - bar_width / 2, aa_sasa, bar_width,
                label=aa_label, color=colors["aa"], edgecolor="white",
                linewidth=0.3, zorder=1)
        ax1.bar(x + bar_width / 2, cg_sasa, bar_width,
                label=cg_label, color=colors["cg"], edgecolor="white",
                linewidth=0.3, zorder=1)
        _draw_err(ax1, x - bar_width / 2, aa_sasa, aa_sasa_lo, aa_sasa_hi, err_kw)
        _draw_err(ax1, x + bar_width / 2, cg_sasa, cg_sasa_lo, cg_sasa_hi, err_kw)
        ax1.set_xticks(x)
        ax1.set_xticklabels(ligands)
        ax1.set_ylabel(sasa_ylabel)
        ax1.legend(frameon=False, loc="upper right")
        ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax1.text(0.02, 0.97, "a", transform=ax1.transAxes,
                 fontsize=11, fontweight="bold", va="top")

        # -- RMSD --
        ax2.bar(x - bar_width / 2, aa_rmsd, bar_width,
                label=aa_label, color=colors["aa"], edgecolor="white",
                linewidth=0.3, zorder=1)
        ax2.bar(x + bar_width / 2, cg_rmsd, bar_width,
                label=cg_label, color=colors["cg"], edgecolor="white",
                linewidth=0.3, zorder=1)
        _draw_err(ax2, x - bar_width / 2, aa_rmsd, aa_rmsd_lo, aa_rmsd_hi, err_kw)
        _draw_err(ax2, x + bar_width / 2, cg_rmsd, cg_rmsd_lo, cg_rmsd_hi, err_kw)
        ax2.set_xticks(x)
        ax2.set_xticklabels(ligands)
        ax2.set_ylabel(rmsd_ylabel)
        ax2.legend(frameon=False, loc="upper right")
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax2.text(0.02, 0.97, "b", transform=ax2.transAxes,
                 fontsize=11, fontweight="bold", va="top")

        # -- Rg --
        ax3.bar(x - bar_width / 2, aa_rg, bar_width,
                label=aa_label, color=colors["aa"], edgecolor="white",
                linewidth=0.3, zorder=1)
        ax3.bar(x + bar_width / 2, cg_rg, bar_width,
                label=cg_label, color=colors["cg"], edgecolor="white",
                linewidth=0.3, zorder=1)
        _draw_err(ax3, x - bar_width / 2, aa_rg, aa_rg_lo, aa_rg_hi, err_kw)
        _draw_err(ax3, x + bar_width / 2, cg_rg, cg_rg_lo, cg_rg_hi, err_kw)
        ax3.set_xticks(x)
        ax3.set_xticklabels(ligands)
        ax3.set_ylabel(rg_ylabel)
        ax3.legend(frameon=False, loc="upper right")
        ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax3.text(0.02, 0.97, "c", transform=ax3.transAxes,
                 fontsize=11, fontweight="bold", va="top")

        fig.tight_layout()
        png_path = out_dir / png_name
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)

    logger.info("SASA/RMSD chart saved to %s", png_path)
    return png_path


# ---------------------------------------------------------------------------
# Wasserstein distance bar chart
# ---------------------------------------------------------------------------

def plot_wasserstein(
    all_results: list[dict],
    out_dir: str | Path,
    *,
    figsize: tuple[float, float] = (8.0, 4.0),
    dpi: int = 300,
    palette: dict | None = None,
    bond_label: str = "Bonds / Constraints",
    angle_label: str = "Angles",
    dihedral_label: str = "Dihedrals",
    ylabel: str = "Normalised Wasserstein distance",
    bar_width: float = 0.25,
    png_name: str = "internal_wasserstein_bars.png",
    **kwargs,
) -> Path | None:
    """Grouped bar chart of per-ligand mean Wasserstein distances.

    Parameters
    ----------
    all_results : list[dict]
        Each dict must contain at least ``"ligand"``.  Optional keys:
        ``"bond_wass_mean"``, ``"angle_wass_mean"``, ``"dihedral_wass_mean"``.
    out_dir : str or Path
        Directory in which the PNG is saved.
    figsize : (float, float)
        Figure size in inches.
    dpi : int
        Output resolution.
    palette : dict or None
        Colour overrides.  Keys: ``"bond"``, ``"angle"``, ``"dihedral"``.
    bond_label, angle_label, dihedral_label : str
        Legend labels.
    ylabel : str
        Y-axis label.
    bar_width : float
        Width of each bar group.
    png_name : str
        Output filename.
    **kwargs
        Passed through to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    Path or None
        Absolute path to the saved PNG, or *None* if there was no data.
    """
    colors = {**PALETTE, **(palette or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ligands = [r["ligand"] for r in all_results]
    n = len(ligands)
    x = np.arange(n)

    bond_vals = [_or_nan(r.get("bond_wass_mean")) for r in all_results]
    angle_vals = [_or_nan(r.get("angle_wass_mean")) for r in all_results]
    dihedral_vals = [_or_nan(r.get("dihedral_wass_mean")) for r in all_results]

    if all(np.isnan(v) for v in bond_vals + angle_vals + dihedral_vals):
        logger.info("No Wasserstein data to plot; skipping chart.")
        return None

    with plt.rc_context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=figsize, **kwargs)

        ax.bar(x - bar_width, bond_vals, bar_width,
               label=bond_label, color=colors["bond"], edgecolor="white", linewidth=0.3)
        ax.bar(x, angle_vals, bar_width,
               label=angle_label, color=colors["angle"], edgecolor="white", linewidth=0.3)
        ax.bar(x + bar_width, dihedral_vals, bar_width,
               label=dihedral_label, color=colors["dihedral"], edgecolor="white", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(ligands)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="upper left", ncol=3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        # Light horizontal grid
        ax.yaxis.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()
        png_path = out_dir / png_name
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)

    logger.info("Wasserstein chart saved to %s", png_path)
    return png_path


# ---------------------------------------------------------------------------
# Contact Frequency Comparison  (CG / AA side-by-side heatmaps)
# ---------------------------------------------------------------------------

def plot_contact_frequency_comparison(
    freq_cg: np.ndarray | None,
    freq_aa: np.ndarray | None,
    out_dir: str | Path,
    *,
    lig_labels: list[str] | None = None,
    prot_labels: list[str] | None = None,
    title: str = "Ligand–Protein Contact Frequency",
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] = (12.0, 5.5),
    dpi: int = 300,
    label_fontsize: int = 7,
    cbar_label: str = "Contact frequency",
    png_name: str = "contact_freq_comparison.png",
    add_suptitle: bool = False,
    **kwargs,
) -> Path | None:
    """Side-by-side contact-frequency heatmaps for CG and AA with identical axes.

    The y-axis (ligand beads) and x-axis (protein residues) are shared
    between the two panels so that the binding interface can be compared
    directly between representations.

    Parameters
    ----------
    freq_cg : (n_lig, n_prot) or None
        CG contact frequency matrix ∈ [0, 1].
    freq_aa : (n_lig, n_prot) or None
        AA contact frequency matrix ∈ [0, 1].
    out_dir : str or Path
        Directory in which the PNG is saved.
    lig_labels : list of str, optional
        Labels for ligand beads (y-axis).
    prot_labels : list of str, optional
        Labels for protein residues (x-axis).
    title : str
        Suptitle for the figure.
    cmap : str
        Matplotlib colormap name (default: ``"viridis"``).
    figsize : (float, float)
        Figure size in inches.
    dpi : int
        Output resolution.
    label_fontsize : int
        Font size for axis tick labels.
    cbar_label : str
        Label for the shared colour bar.
    png_name : str
        Output filename.
    **kwargs
        Passed through to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    Path or None
        Absolute path to the saved PNG, or *None* if neither matrix was given.
    """
    from matplotlib.colors import Normalize

    panels = []
    if freq_cg is not None:
        panels.append(("CG", freq_cg))
    if freq_aa is not None:
        panels.append(("AA", freq_aa))
    if not panels:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_panels = len(panels)

    with plt.rc_context(DEFAULT_STYLE):
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False,
                                 constrained_layout=True, **kwargs)
        axes = axes[0]

        for ax, (label, freq) in zip(axes, panels):
            n_lig, n_prot = freq.shape
            im = ax.imshow(freq, aspect="auto", origin="upper",
                           cmap=cmap, norm=Normalize(0, 1),
                           interpolation="nearest")
            ax.set_title(f"{label}\n({n_lig} beads × {n_prot} residues)")
            ax.set_xlabel("Protein residues")
            if ax is axes[0]:
                ax.set_ylabel("Ligand beads")
            if prot_labels is not None and n_prot <= 40:
                ax.set_xticks(range(n_prot))
                ax.set_xticklabels(prot_labels, rotation=90,
                                   fontsize=label_fontsize)
            if lig_labels is not None and n_lig <= 30:
                ax.set_yticks(range(n_lig))
                ax.set_yticklabels(lig_labels, fontsize=label_fontsize)

        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85,
                            label=cbar_label)
        cbar.ax.tick_params(labelsize=8)

        if add_suptitle:
            fig.suptitle(title, fontsize=10, y=1.01)
        png_path = out_dir / png_name
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    logger.info("Contact frequency comparison saved to %s", png_path)
    return png_path


# ---------------------------------------------------------------------------
# Fraction of Native Contacts Q(t)
# ---------------------------------------------------------------------------

def plot_Q_time_series(
    Q_data: dict[str, np.ndarray | None],
    out_dir: str | Path,
    *,
    dt_ps: float = 200.0,
    title: str = "Fraction of Native Contacts  $Q(t)$",
    figsize: tuple[float, float] = (7.0, 3.5),
    dpi: int = 300,
    palette: dict | None = None,
    xlabel: str = "Time / ns",
    ylabel: str = "$Q(t)$",
    ylim: tuple[float, float] = (0.5, 0.9),
    png_name: str = "Q_vs_time.png",
    inset_freq_cg: np.ndarray | None = None,
    inset_freq_aa: np.ndarray | None = None,
    inset_cmap: str = "YlOrRd",
    add_title: bool = False,
    **kwargs,
) -> Path | None:
    """Plot Q(t) for one or more trajectory types on the same axes.

    All series are truncated to the length of the shortest one, so that
    the time axes are aligned.

    Parameters
    ----------
    Q_data : dict
        Mapping ``{label: Q_array}``, e.g. ``{"CG": Q_cg, "AA": Q_aa}``.
        Values may be *None* (skipped).
    out_dir : str or Path
        Directory in which the PNG is saved.
    dt_ps : float
        Time step between consecutive frames in ps.
    title : str
        Plot title.
    figsize : (float, float)
        Figure size in inches.
    dpi : int
        Output resolution.
    palette : dict or None
        Colour overrides.  Keys: ``"AA"``, ``"CG"``.
    xlabel, ylabel : str
        Axis labels.
    png_name : str
        Output filename.
    **kwargs
        Passed through to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    Path or None
        Absolute path to the saved PNG, or *None* if there was no data.
    """
    colors = {**PALETTE, **(palette or {})}

    # Map standard keys to palette keys (lowercase)
    palette_map = {"AA": "aa", "CG": "cg"}

    # Truncate all series to the shortest length so axes align
    q_arrays = [Q for Q in Q_data.values() if Q is not None]
    if not q_arrays:
        logger.info("No Q data to plot; skipping.")
        return None
    min_len = min(len(q) for q in q_arrays)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, **kwargs)

        for label, Q in Q_data.items():
            if Q is None:
                continue
            Q = Q[:min_len]
            n = len(Q)
            time_ns = np.arange(n) * dt_ps / 1000.0
            color_key = palette_map.get(label, label.lower())
            ax.plot(time_ns, Q, linewidth=1.0, alpha=0.9,
                    color=colors.get(color_key, None), label=label)
            mean_q = float(np.mean(Q))
            ax.axhline(mean_q, color=colors.get(color_key, "#888888"),
                       linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(*ylim)

        # --- inset: mini contact maps (bottom-right, side-by-side CG | AA) ---
        panels = []
        labels = []
        if inset_freq_cg is not None:
            panels.append(inset_freq_cg)
            labels.append("CG")
        if inset_freq_aa is not None:
            panels.append(inset_freq_aa)
            labels.append("AA")
        if panels:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            from matplotlib.colors import Normalize as Norm
            n_panels = len(panels)
            inset_w = "30%" if n_panels == 1 else "40%"
            iax = inset_axes(ax, width=inset_w, height="38%",
                             loc="lower right",
                             bbox_to_anchor=(0.0, 0.02, 0.965, 1),
                             bbox_transform=ax.transAxes)

            if n_panels == 2:
                # insert a white gap column between the two maps
                n_lig, n_prot = panels[0].shape
                gap_cols = max(2, n_prot // 8)  # ~proportional gap
                gap = np.full((n_lig, gap_cols), np.nan)
                combined = np.hstack([panels[0], gap, panels[1]])
                # vertical separator line at centre of gap
                sep_x = panels[0].shape[1] + gap_cols / 2 + 1.5
                iax.axvline(sep_x, color="black", linewidth=0.5)
            else:
                combined = panels[0]

            iax.imshow(combined, aspect="auto", origin="upper",
                       cmap=inset_cmap, norm=Norm(0, 1), interpolation="nearest")
            iax.set_xticks([])
            iax.set_yticks([])
            iax.set_xlabel("Protein residues", fontsize=6, labelpad=1)
            if n_panels == 2:
                # left edge: "CG"  —  right edge: "AA"
                iax.text(-0.02, 0.5, "CG ligand beads", transform=iax.transAxes,
                         fontsize=6, rotation=90, va="center", ha="center")
                iax.text(0.5, 0.5, "AA ligand beads", transform=iax.transAxes,
                         fontsize=6, rotation=90, va="center", ha="center")
            else:
                iax.set_ylabel("Ligand beads", fontsize=6, labelpad=1)
            for spine in iax.spines.values():
                spine.set_linewidth(0.5)

        if add_title:
            ax.set_title(title)
        if len(Q_data) > 1:
            ax.legend(frameon=False)

        png_path = out_dir / png_name
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)

    logger.info("Q(t) plot saved to %s", png_path)
    return png_path

def _load_results(csv_path: str | Path) -> list[dict]:
    """Parse the summary CSV into a list of dicts with typed values."""
    import csv

    csv_path = Path(csv_path)
    rows: list[dict] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            r: dict = {"ligand": row["ligand"]}
            for key, converter in _CSV_CONVERTERS.items():
                raw = row.get(key, "")
                if raw == "" or raw is None:
                    r[key] = None
                else:
                    try:
                        r[key] = converter(raw)
                    except (ValueError, TypeError):
                        r[key] = None
            rows.append(r)
    return rows


_CSV_CONVERTERS: dict[str, callable] = {
    "aa_n_frames": int,
    "aa_n_atoms": int,
    "aa_sasa_mean": float,
    "aa_sasa_ci_lo": float,
    "aa_sasa_ci_hi": float,
    "aa_rmsd_mean": float,
    "aa_rmsd_ci_lo": float,
    "aa_rmsd_ci_hi": float,
    "aa_rg_mean": float,
    "aa_rg_ci_lo": float,
    "aa_rg_ci_hi": float,
    "cg_n_frames": int,
    "cg_n_beads": int,
    "cg_sasa_mean": float,
    "cg_sasa_ci_lo": float,
    "cg_sasa_ci_hi": float,
    "cg_rmsd_mean": float,
    "cg_rmsd_ci_lo": float,
    "cg_rmsd_ci_hi": float,
    "cg_rg_mean": float,
    "cg_rg_ci_lo": float,
    "cg_rg_ci_hi": float,
    "bond_wass_mean": float,
    "angle_wass_mean": float,
    "dihedral_wass_mean": float,
}


if __name__ == "__main__":
    import sys
    import pickle

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # --- Paths ---
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "analysis/sasa_rmsd_summary.csv"
    mda_dir  = sys.argv[2] if len(sys.argv) > 2 else "../media"
    sysname  = sys.argv[3] if len(sys.argv) > 3 else "1TQN"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    results = _load_results(csv_path)
    logger.info("Loaded %d ligand(s) from %s", len(results), csv_path)

    # ── Per-ligand summary plots ──────────────────────────────────────
    plot_sasa_rmsd(results, mda_dir)
    plot_wasserstein(results, mda_dir)

    # ── Protein–ligand contact analysis ───────────────────────────────
    contacts_pkl = Path(f"analysis/{sysname}_contacts.pkl")
    if contacts_pkl.exists():
        with open(contacts_pkl, "rb") as f:
            cr = pickle.load(f)
        logger.info("Loaded contact data from %s", contacts_pkl)

        freq_cg = cr.get("cg_contact_freq")
        freq_aa = cr.get("aa_contact_freq")
        if freq_cg is not None or freq_aa is not None:
            plot_contact_frequency_comparison(
                freq_cg, freq_aa,
                mda_dir,
                lig_labels  = cr.get("lig_bead_names"),
                prot_labels = cr.get("unified_prot_labels"),
                png_name    = f"{sysname}_contact_freq_comparison.png",
                cmap     = "Greys",
                # figsize  = (12.0, 5.5),
                # add_suptitle = True,
            )

        Q_data = {}
        for mode in ("cg", "aa"):
            Q = cr.get(f"{mode}_Q")
            if Q is not None:
                Q_data[mode.upper()] = Q
        if Q_data:
            plot_Q_time_series(
                Q_data,
                mda_dir,
                png_name       = f"{sysname}_Q_vs_time.png",
                inset_freq_cg  = freq_cg,
                inset_freq_aa  = freq_aa,
                inset_cmap     = "Greys",
                ylim      = (0.55, 0.90),
                # add_title = True,
            )

    logger.info("All plots saved to %s/", mda_dir)
