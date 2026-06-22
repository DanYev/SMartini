"""SASA and RMSD analysis for CG and AA ligands in examples.

Computes per-frame SASA and RMSD for all ligands with both AA and CG
trajectories, then saves summary CSV files and PNG plots.

Usage::

    python scripts/analysis.py              # all ligands
    python scripts/analysis.py ANP CLA      # specific ligands
"""

import logging
import re
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

# Shrake-Rupley sphere points (Fibonacci lattice, 960 points)
_SR_N = 960
_SR_PHI = (1 + np.sqrt(5)) / 2
_SR_POINTS = np.empty((_SR_N, 3))
for _i in range(_SR_N):
    _y = 1 - (_i / (_SR_N - 1)) * 2
    _r = np.sqrt(1 - _y * _y)
    _theta = 2 * np.pi * _i / _SR_PHI
    _SR_POINTS[_i] = [np.cos(_theta) * _r, _y, np.sin(_theta) * _r]


# ---------------------------------------------------------------------------
# ITP parsing
# ---------------------------------------------------------------------------

def _parse_itp(itp_path: Path) -> tuple[list[list[int]], list[str]]:
    """Parse ``LIGAND.itp`` → (mapping, bead_types).

    mapping : list of lists of 0-based AA atom indices per bead.
    bead_types : list of Martini bead type strings (e.g. ``'TN6a'``).
    """
    text = itp_path.read_text()

    # --- mapping from header comment ---
    m = re.search(r"; Mapping:\s*(\[\[.*?\]\])", text)
    if not m:
        raise ValueError(f"No '; Mapping:' found in {itp_path}")
    mapping_raw = m.group(1)
    mapping = [list(map(int, grp.strip("[] ").split(",")))
               for grp in re.findall(r"\[([^\]]+)\]", mapping_raw)]

    # --- bead types from [ atoms ] section ---
    in_atoms = False
    bead_types = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_atoms = (s == "[ atoms ]" or s == "[atoms]")
            continue
        if in_atoms and s and not s.startswith(";") and not s.startswith("#"):
            parts = s.split()
            if len(parts) >= 2:
                bead_types.append(parts[1])  # column 2 = bead type
    return mapping, bead_types


def _bead_radius(bead_type: str) -> float:
    """Martini bead radius (nm): T* → 0.17, S* → 0.205, else → 0.235."""
    if bead_type.startswith("T"):
        return 0.17
    if bead_type.startswith("S"):
        return 0.205
    return 0.235


# ---------------------------------------------------------------------------
# SASA
# ---------------------------------------------------------------------------

def compute_sasa(pdb_path: Path, xtc_path: Path,
                 is_cg: bool = False, bead_types: list[str] | None = None) -> np.ndarray:
    """Per-frame total SASA (nm²).  AA uses MDTraj element radii;
    CG uses a custom Shrake-Rupley with bead-type-based radii."""
    if is_cg:
        if bead_types is None:
            raise ValueError("bead_types required for CG SASA")
        traj = md.load(str(xtc_path), top=str(pdb_path))
        radii = np.array([_bead_radius(bt) for bt in bead_types])
        return _sr_sasa(traj.xyz, radii)
    else:
        traj = md.load(str(xtc_path), top=str(pdb_path))
        sasa_per_atom = md.shrake_rupley(traj, mode="atom")
        return sasa_per_atom.sum(axis=1)


def _sr_sasa(xyz: np.ndarray, radii: np.ndarray, probe: float = 0.14) -> np.ndarray:
    """Shrake-Rupley SASA for CG: per-frame total SASA (nm²).

    Parameters
    ----------
    xyz : (n_frames, n_atoms, 3)
    radii : (n_atoms,)  bead radii in nm
    probe : float  probe radius in nm
    """
    n_frames, n_atoms = xyz.shape[:2]
    effective = radii + probe  # (n_atoms,)
    points = _SR_POINTS       # (N, 3)

    sasa_total = np.empty(n_frames)
    for f in range(n_frames):
        pos = xyz[f]  # (n_atoms, 3)
        total = 0.0
        for i in range(n_atoms):
            pi = pos[i] + effective[i] * points  # (N, 3)
            exposed = np.ones(_SR_N, dtype=bool)
            for j in range(n_atoms):
                if i == j:
                    continue
                d2 = np.sum((pi - pos[j]) ** 2, axis=1)
                exposed &= d2 > effective[j] ** 2
            total += 4 * np.pi * effective[i] ** 2 * exposed.mean()
        sasa_total[f] = total
    return sasa_total


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

def compute_cg_rmsd(pdb_path: Path, xtc_path: Path) -> np.ndarray:
    """Per-frame CG RMSD (nm) vs frame 0, after superposition."""
    traj = md.load(str(xtc_path), top=str(pdb_path))
    traj.superpose(reference=traj, frame=0)
    return md.rmsd(traj, traj, frame=0)


def compute_aa_cog_rmsd(pdb_path: Path, xtc_path: Path,
                        mapping: list[list[int]]) -> np.ndarray:
    """AA RMSD using center-of-geometry of bead-mapped groups.

    Groups AA atoms by their CG bead assignment, computes COG per group,
    then RMSD of those COGs vs frame 0 after optimal superposition.
    """
    traj = md.load(str(xtc_path), top=str(pdb_path))
    n_frames = traj.n_frames
    n_beads = len(mapping)

    # COG per frame per bead group
    cog = np.empty((n_frames, n_beads, 3))
    for f in range(n_frames):
        xyz = traj.xyz[f]
        for b, indices in enumerate(mapping):
            cog[f, b] = xyz[indices].mean(axis=0)

    # Superpose COG to frame 0 via Kabsch
    ref = cog[0].copy()
    ref_cm = ref.mean(axis=0)
    ref_centered = ref - ref_cm
    for f in range(1, n_frames):
        cm = cog[f].mean(axis=0)
        centered = cog[f] - cm
        H = centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        cog[f] = centered @ R + ref_cm

    diff = cog - ref
    return np.sqrt((diff ** 2).sum(axis=(1, 2)) / n_beads)


# ---------------------------------------------------------------------------
# Per-ligand analysis
# ---------------------------------------------------------------------------

def analyze_ligand(ligand_name: str) -> dict | None:
    """Run SASA + RMSD for both AA and CG trajectories of a single ligand.

    Returns a dict of results, or None if data is missing.
    """
    aa_dir = EXAMPLES_DIR / ligand_name / "aa_md"
    cg_dir = EXAMPLES_DIR / ligand_name / "cg_md" / CFG.cg_runname
    itp_path = EXAMPLES_DIR / ligand_name / f"{ligand_name}.itp"

    aa_top = aa_dir / "topology.pdb"
    aa_trj = aa_dir / "samples.xtc"
    cg_top = cg_dir / "topology.pdb"
    cg_trj = cg_dir / "samples.xtc"

    results = {"ligand": ligand_name}

    # Parse ITP for bead mapping and bead types
    mapping, bead_types = _parse_itp(itp_path)
    logger.info("[%s] %d beads, types: %s", ligand_name, len(bead_types),
                ", ".join(bead_types))

    # --- AA ---
    if aa_top.exists() and aa_trj.exists():
        logger.info("[%s] AA: %s", ligand_name, aa_trj)
        traj = md.load(str(aa_trj), top=str(aa_top))
        results["aa_n_frames"] = traj.n_frames
        results["aa_n_atoms"] = traj.n_atoms

        sasa_aa = compute_sasa(aa_top, aa_trj, is_cg=False)
        results["aa_sasa_mean"] = float(np.mean(sasa_aa))
        results["aa_sasa_std"] = float(np.std(sasa_aa))

        rmsd_aa = compute_aa_cog_rmsd(aa_top, aa_trj, mapping)
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

        sasa_cg = compute_sasa(cg_top, cg_trj, is_cg=True, bead_types=bead_types)
        results["cg_sasa_mean"] = float(np.mean(sasa_cg))
        results["cg_sasa_std"] = float(np.std(sasa_cg))

        rmsd_cg = compute_cg_rmsd(cg_top, cg_trj)
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
