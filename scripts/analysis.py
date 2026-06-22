"""SASA and RMSD analysis for CG and AA ligands in examples.

Computes per-frame SASA and RMSD for all ligands with both AA and CG
trajectories, then saves summary CSV files and PNG plots.

Usage::

    python scripts/analysis.py              # all ligands
    python scripts/analysis.py ANP CLA      # specific ligands
"""

import logging
from pathlib import Path

import numpy as np
import mdtraj as md

import smartini
from smartini.config import CFG

logger = logging.getLogger(__name__)
smartini.setup_logging(level=logging.INFO)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
EXAMPLES_DIR = Path("examples").resolve()
OUTPUT_DIR = Path("analysis").resolve()

# Martini bead radii for SASA (nm) — mapped by element symbol in CG PDB.
# P-beads ~0.26, Q-beads ~0.24, N-beads ~0.22, C-beads ~0.23, D (dummy) ~0.26.
CG_RADII_MAP = {"P": 0.26, "VS": 0.24, "N": 0.22, "C": 0.23, "D": 0.26}


# ---------------------------------------------------------------------------
# SASA  –  mdtraj.shrake_rupley  (element radii for AA, custom for CG)
# ---------------------------------------------------------------------------

def compute_sasa(pdb_path: Path, xtc_path: Path, is_cg: bool = False) -> np.ndarray:
    """Per-frame total SASA (nm²) via MDTraj Shrake-Rupley."""
    traj = md.load(str(xtc_path), top=str(pdb_path))
    if is_cg:
        sasa_per_atom = md.shrake_rupley(traj, mode="atom",
                                         change_radii=CG_RADII_MAP)
    else:
        sasa_per_atom = md.shrake_rupley(traj, mode="atom")
    return sasa_per_atom.sum(axis=1)  # (n_frames, n_atoms) → (n_frames,)


# ---------------------------------------------------------------------------
# RMSD  –  mdtraj.rmsd  (auto-aligns to frame 0)
# ---------------------------------------------------------------------------

def compute_rmsd(pdb_path: Path, xtc_path: Path) -> np.ndarray:
    """Per-frame RMSD (nm) vs frame 0, after superposition."""
    traj = md.load(str(xtc_path), top=str(pdb_path))
    traj.superpose(reference=traj, frame=0)
    return md.rmsd(traj, traj, frame=0)  # (n_frames,)


# ---------------------------------------------------------------------------
# Per-ligand analysis
# ---------------------------------------------------------------------------

def analyze_ligand(ligand_name: str) -> dict | None:
    """Run SASA + RMSD for both AA and CG trajectories of a single ligand.

    Returns a dict of results, or None if data is missing.
    """
    aa_dir = EXAMPLES_DIR / ligand_name / "aa_md"
    cg_dir = EXAMPLES_DIR / ligand_name / "cg_md" / CFG.cg_runname

    aa_top = aa_dir / "topology.pdb"
    aa_trj = aa_dir / "samples.xtc"
    cg_top = cg_dir / "topology.pdb"
    cg_trj = cg_dir / "samples.xtc"

    results = {"ligand": ligand_name}

    # --- AA ---
    if aa_top.exists() and aa_trj.exists():
        logger.info("[%s] AA: %s", ligand_name, aa_trj)
        traj = md.load(str(aa_trj), top=str(aa_top))
        results["aa_n_frames"] = traj.n_frames
        results["aa_n_atoms"] = traj.n_atoms

        sasa_aa = compute_sasa(aa_top, aa_trj, is_cg=False)
        results["aa_sasa_mean"] = float(np.mean(sasa_aa))
        results["aa_sasa_std"] = float(np.std(sasa_aa))

        rmsd_aa = compute_rmsd(aa_top, aa_trj)
        results["aa_rmsd_mean"] = float(np.mean(rmsd_aa))
        results["aa_rmsd_std"] = float(np.std(rmsd_aa))
    else:
        logger.warning("[%s] AA data missing, skipping AA analysis.", ligand_name)

    # --- CG ---
    if cg_top.exists() and cg_trj.exists():
        logger.info("[%s] CG: %s", ligand_name, cg_trj)
        traj = md.load(str(cg_trj), top=str(cg_top))
        results["cg_n_frames"] = traj.n_frames
        results["cg_n_beads"] = traj.n_atoms

        sasa_cg = compute_sasa(cg_top, cg_trj, is_cg=True)
        results["cg_sasa_mean"] = float(np.mean(sasa_cg))
        results["cg_sasa_std"] = float(np.std(sasa_cg))

        rmsd_cg = compute_rmsd(cg_top, cg_trj)
        results["cg_rmsd_mean"] = float(np.mean(rmsd_cg))
        results["cg_rmsd_std"] = float(np.std(rmsd_cg))
    else:
        logger.warning("[%s] CG data missing, skipping CG analysis.", ligand_name)

    return results


# ---------------------------------------------------------------------------
# Summary & plotting
# ---------------------------------------------------------------------------

def save_results(all_results: list[dict], out_dir: Path):
    """Save a summary CSV and a bar-chart PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sasa_rmsd_summary.csv"
    png_path = out_dir / "sasa_rmsd_bars.png"

    # --- CSV ---
    columns = [
        "ligand",
        "aa_n_frames", "aa_n_atoms",
        "aa_sasa_mean", "aa_sasa_std",
        "aa_rmsd_mean", "aa_rmsd_std",
        "cg_n_frames", "cg_n_beads",
        "cg_sasa_mean", "cg_sasa_std",
        "cg_rmsd_mean", "cg_rmsd_std",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for r in all_results:
            f.write(",".join(str(r.get(c, "")) for c in columns) + "\n")
    logger.info("CSV summary saved to %s", csv_path)

    # --- Bar chart ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping bar chart.")
        return

    ligands = [r["ligand"] for r in all_results]
    n = len(ligands)
    x = np.arange(n)
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # SASA bars
    aa_sasa = [r.get("aa_sasa_mean", 0) for r in all_results]
    cg_sasa = [r.get("cg_sasa_mean", 0) for r in all_results]
    ax1.bar(x - w / 2, aa_sasa, w, label="AA", color="tab:blue")
    ax1.bar(x + w / 2, cg_sasa, w, label="CG", color="tab:orange")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ligands)
    ax1.set_ylabel("SASA (nm²)")
    ax1.set_title("Mean SASA")
    ax1.legend()

    # RMSD bars
    aa_rmsd = [r.get("aa_rmsd_mean", 0) for r in all_results]
    cg_rmsd = [r.get("cg_rmsd_mean", 0) for r in all_results]
    ax2.bar(x - w / 2, aa_rmsd, w, label="AA", color="tab:blue")
    ax2.bar(x + w / 2, cg_rmsd, w, label="CG", color="tab:orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ligands)
    ax2.set_ylabel("RMSD (nm)")
    ax2.set_title("Mean RMSD")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    logger.info("Bar chart saved to %s", png_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _discover_ligands() -> list[str]:
    """Return sorted list of ligand names that have both AA and CG data."""
    ligands = []
    for d in sorted(EXAMPLES_DIR.iterdir()):
        if not d.is_dir():
            continue
        aa_top = d / "aa_md" / "topology.pdb"
        aa_trj = d / "aa_md" / "samples.xtc"
        cg_top = d / "cg_md" / CFG.cg_runname / "topology.pdb"
        cg_trj = d / "cg_md" / CFG.cg_runname / "samples.xtc"
        if (aa_top.exists() and aa_trj.exists() and
                cg_top.exists() and cg_trj.exists()):
            ligands.append(d.name)
    return ligands


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] if len(sys.argv) > 1 else _discover_ligands()
    if not names:
        logger.error("No ligands with complete AA+CG data found in examples/")
        sys.exit(1)

    logger.info("Analysing %d ligand(s): %s", len(names), ", ".join(names))
    all_results = []
    for name in names:
        res = analyze_ligand(name)
        if res:
            all_results.append(res)

    save_results(all_results, OUTPUT_DIR)
    logger.info("Done.")
