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


# ---------------------------------------------------------------------------
# SASA / RMSD grouped bar chart
# ---------------------------------------------------------------------------

def plot_sasa_rmsd(
    all_results: list[dict],
    out_dir: str | Path,
    *,
    figsize: tuple[float, float] = (8.0, 3.5),
    dpi: int = 300,
    palette: dict | None = None,
    aa_label: str = "AA",
    cg_label: str = "CG",
    sasa_ylabel: str = "SASA / nm²",
    rmsd_ylabel: str = "RMSD / nm",
    bar_width: float = 0.35,
    png_name: str = "sasa_rmsd_bars.png",
    **kwargs,
) -> Path:
    """Grouped bar chart comparing per-ligand mean SASA and RMSD (AA vs CG).

    Parameters
    ----------
    all_results : list[dict]
        Each dict must contain at least ``"ligand"``.  Optional keys:
        ``"aa_sasa_mean"``, ``"cg_sasa_mean"``, ``"aa_rmsd_mean"``,
        ``"cg_rmsd_mean"``.
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
    sasa_ylabel, rmsd_ylabel : str
        Axis labels.
    bar_width : float
        Width of each bar in data units.
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

    aa_sasa = [r.get("aa_sasa_mean", 0) for r in all_results]
    cg_sasa = [r.get("cg_sasa_mean", 0) for r in all_results]
    aa_rmsd = [r.get("aa_rmsd_mean", 0) for r in all_results]
    cg_rmsd = [r.get("cg_rmsd_mean", 0) for r in all_results]

    with plt.rc_context(DEFAULT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, **kwargs)

        # -- SASA --
        ax1.bar(x - bar_width / 2, aa_sasa, bar_width,
                label=aa_label, color=colors["aa"], edgecolor="white", linewidth=0.3)
        ax1.bar(x + bar_width / 2, cg_sasa, bar_width,
                label=cg_label, color=colors["cg"], edgecolor="white", linewidth=0.3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(ligands)
        ax1.set_ylabel(sasa_ylabel)
        ax1.set_title("Mean SASA")
        ax1.legend(frameon=False, loc="upper right")
        ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

        # -- RMSD --
        ax2.bar(x - bar_width / 2, aa_rmsd, bar_width,
                label=aa_label, color=colors["aa"], edgecolor="white", linewidth=0.3)
        ax2.bar(x + bar_width / 2, cg_rmsd, bar_width,
                label=cg_label, color=colors["cg"], edgecolor="white", linewidth=0.3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(ligands)
        ax2.set_ylabel(rmsd_ylabel)
        ax2.set_title("Mean RMSD")
        ax2.legend(frameon=False, loc="upper right")
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))

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
    figsize: tuple[float, float] = (10.0, 4.0),
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

    bond_vals = [r.get("bond_wass_mean", float("nan")) for r in all_results]
    angle_vals = [r.get("angle_wass_mean", float("nan")) for r in all_results]
    dihedral_vals = [r.get("dihedral_wass_mean", float("nan")) for r in all_results]

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
        ax.set_title("AA / CG Internal-Coordinate Wasserstein Distance")
        ax.legend(frameon=False, loc="upper right", ncol=3)
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
