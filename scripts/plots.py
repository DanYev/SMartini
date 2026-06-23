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
    figsize: tuple[float, float] = (8.0, 3.5),
    dpi: int = 300,
    palette: dict | None = None,
    aa_label: str = "AA",
    cg_label: str = "CG",
    sasa_ylabel: str = "SASA / nm²",
    rmsd_ylabel: str = "RMSD / Å",
    bar_width: float = 0.35,
    err_capsize: float = 3.0,
    png_name: str = "sasa_rmsd_bars.png",
    **kwargs,
) -> Path:
    """Grouped bar chart comparing per-ligand mean SASA and RMSD (AA vs CG).

    Draws 95% bootstrap confidence intervals as error bars when available.

    Parameters
    ----------
    all_results : list[dict]
        Each dict must contain at least ``"ligand"``.  Optional keys:
        ``"aa_sasa_mean"``, ``"cg_sasa_mean"``, ``"aa_rmsd_mean"``,
        ``"cg_rmsd_mean"``.
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
    sasa_ylabel, rmsd_ylabel : str
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

    # --- error bars (bootstrap CI) ---
    def _err_lo_hi(results, prefix):
        """Build (n, 2) error arrays: [mean - lo, hi - mean]."""
        means = np.array([_or_zero(r.get(f"{prefix}_mean")) for r in results])
        los = np.array([_or_nan(r.get(f"{prefix}_ci_lo")) for r in results])
        his = np.array([_or_nan(r.get(f"{prefix}_ci_hi")) for r in results])
        # replace NaN with mean (zero-height error bar)
        los = np.where(np.isnan(los), means, los)
        his = np.where(np.isnan(his), means, his)
        return np.abs(means - los), np.abs(his - means)

    aa_sasa_lo, aa_sasa_hi = _err_lo_hi(all_results, "aa_sasa")
    cg_sasa_lo, cg_sasa_hi = _err_lo_hi(all_results, "cg_sasa")
    aa_rmsd_lo, aa_rmsd_hi = _err_lo_hi(all_results, "aa_rmsd")
    cg_rmsd_lo, cg_rmsd_hi = _err_lo_hi(all_results, "cg_rmsd")
    # Scale RMSD error bars nm → Å
    aa_rmsd_lo *= 10; aa_rmsd_hi *= 10
    cg_rmsd_lo *= 10; cg_rmsd_hi *= 10

    err_kw = dict(elinewidth=0.8, capsize=err_capsize, capthick=0.8,
                  ecolor="dimgray", zorder=10)

    with plt.rc_context(DEFAULT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, **kwargs)

        # -- SASA --
        b1 = ax1.bar(x - bar_width / 2, aa_sasa, bar_width,
                     label=aa_label, color=colors["aa"], edgecolor="white",
                     linewidth=0.3, zorder=1)
        b2 = ax1.bar(x + bar_width / 2, cg_sasa, bar_width,
                     label=cg_label, color=colors["cg"], edgecolor="white",
                     linewidth=0.3, zorder=1)
        _draw_err(ax1, x - bar_width / 2, aa_sasa, aa_sasa_lo, aa_sasa_hi, err_kw)
        _draw_err(ax1, x + bar_width / 2, cg_sasa, cg_sasa_lo, cg_sasa_hi, err_kw)
        ax1.set_xticks(x)
        ax1.set_xticklabels(ligands)
        ax1.set_ylabel(sasa_ylabel)
        ax1.set_title("Mean SASA  (95% CI)")
        ax1.legend(frameon=False, loc="upper right")
        ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

        # -- RMSD --
        b3 = ax2.bar(x - bar_width / 2, aa_rmsd, bar_width,
                     label=aa_label, color=colors["aa"], edgecolor="white",
                     linewidth=0.3, zorder=1)
        b4 = ax2.bar(x + bar_width / 2, cg_rmsd, bar_width,
                     label=cg_label, color=colors["cg"], edgecolor="white",
                     linewidth=0.3, zorder=1)
        _draw_err(ax2, x - bar_width / 2, aa_rmsd, aa_rmsd_lo, aa_rmsd_hi, err_kw)
        _draw_err(ax2, x + bar_width / 2, cg_rmsd, cg_rmsd_lo, cg_rmsd_hi, err_kw)
        ax2.set_xticks(x)
        ax2.set_xticklabels(ligands)
        ax2.set_ylabel(rmsd_ylabel)
        ax2.set_title("Mean RMSD  (95% CI)")
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
        ax.set_title("AA / CG Internal-Coordinate Wasserstein Distance")
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
# CLI  (python scripts/plots.py [csv_path] [out_dir])
# ---------------------------------------------------------------------------

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
    "aa_sasa_std": float,
    "aa_rmsd_mean": float,
    "aa_rmsd_std": float,
    "cg_n_frames": int,
    "cg_n_beads": int,
    "cg_sasa_mean": float,
    "cg_sasa_std": float,
    "cg_rmsd_mean": float,
    "cg_rmsd_std": float,
    "bond_wass_mean": float,
    "angle_wass_mean": float,
    "dihedral_wass_mean": float,
}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "analysis/sasa_rmsd_summary.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "../media"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    results = _load_results(csv_path)
    logger.info("Loaded %d ligand(s) from %s", len(results), csv_path)

    plot_sasa_rmsd(results, out_dir)
    plot_wasserstein(results, out_dir)
    logger.info("All plots saved to %s/", out_dir)
