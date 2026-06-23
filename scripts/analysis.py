"""SASA and RMSD analysis for CG and AA ligands in examples.

Computes per-frame SASA and RMSD for all ligands with both AA and CG
trajectories, then saves summary CSV files and PNG plots.

Usage::

    python scripts/analysis.py              # all ligands
    python scripts/analysis.py ANP CLA      # specific ligands
"""

import logging
import pickle
import re
from pathlib import Path

import numpy as np
import mdtraj as md

import smartini
from smartini.config import CFG
from smartini.lpmath import (
    read_cg_trajectory,
    calculate_internal_coordinates,
    circular_mean,
    wrap_to_180,
)

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

# Bootstrap defaults
_BOOTSTRAP_N = 1000
_BOOTSTRAP_CI = 95


def _bootstrap_ci(data: np.ndarray, n_boot: int = _BOOTSTRAP_N,
                  ci: float = _BOOTSTRAP_CI) -> tuple[float, float, float]:
    """Bootstrap 95% confidence interval for the mean of *data*.

    Returns ``(mean, ci_lo, ci_hi)``.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 10:
        m = float(np.mean(data))
        return m, m, m
    rng = np.random.default_rng()
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = float(np.mean(rng.choice(data, size=n, replace=True)))
    alpha = (100 - ci) / 2
    ci_lo, ci_hi = np.percentile(boot_means, [alpha, 100 - alpha])
    return float(np.mean(data)), float(ci_lo), float(ci_hi)


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
# Radius of Gyration
# ---------------------------------------------------------------------------

def compute_cg_rg(pdb_path: Path, xtc_path: Path) -> np.ndarray:
    """Per-frame CG radius of gyration (nm)."""
    traj = md.load(str(xtc_path), top=str(pdb_path))
    xyz = traj.xyz  # (n_frames, n_atoms, 3)
    cog = xyz.mean(axis=1, keepdims=True)  # (n_frames, 1, 3)
    return np.sqrt(((xyz - cog) ** 2).sum(axis=(1, 2)) / xyz.shape[1])


def compute_aa_cog_rg(pdb_path: Path, xtc_path: Path,
                       mapping: list[list[int]]) -> np.ndarray:
    """AA radius of gyration using center-of-geometry of bead-mapped groups.

    Groups AA atoms by CG bead assignment, computes COG per group,
    then Rg of those COGs.
    """
    traj = md.load(str(xtc_path), top=str(pdb_path))
    n_frames = traj.n_frames
    n_beads = len(mapping)

    cog = np.empty((n_frames, n_beads, 3))
    for f in range(n_frames):
        xyz = traj.xyz[f]
        for b, indices in enumerate(mapping):
            cog[f, b] = xyz[indices].mean(axis=0)

    center = cog.mean(axis=1, keepdims=True)  # (n_frames, 1, 3)
    return np.sqrt(((cog - center) ** 2).sum(axis=(1, 2)) / n_beads)


# ---------------------------------------------------------------------------
# Internal-coordinate AA/CG overlap
# ---------------------------------------------------------------------------

def _wasserstein_1d(u_values: np.ndarray, v_values: np.ndarray) -> float:
    """Compute 1D Wasserstein (Earth Mover's) distance between two samples.

    Uses the CDF formulation: :math:`W_1 = \\int |F_U(x) - F_V(x)|\\,dx`.
    """
    u = np.sort(np.asarray(u_values, dtype=float))
    v = np.sort(np.asarray(v_values, dtype=float))

    # Combine all breakpoints where either CDF changes
    all_vals = np.sort(np.concatenate([u, v]))

    u_cdf = np.searchsorted(u, all_vals, side='right').astype(float) / len(u)
    v_cdf = np.searchsorted(v, all_vals, side='right').astype(float) / len(v)

    dx = np.diff(all_vals)
    cdf_diff = np.abs(u_cdf[:-1] - v_cdf[:-1])

    return float(np.sum(cdf_diff * dx))


def _wasserstein_distance(aa_vals: np.ndarray, cg_vals: np.ndarray,
                          value_type: str = "bond") -> float:
    """Compute normalised Wasserstein distance (W₁ / range) between AA and CG.

    All return values lie in [0, 1].  For dihedrals the samples are circularly
    centred on the AA mean first.
    """
    aa_vals = np.asarray(aa_vals, dtype=float)
    cg_vals = np.asarray(cg_vals, dtype=float)

    if value_type == "dihedral":
        shift = float(circular_mean(aa_vals))
        aa_vals = wrap_to_180(aa_vals - shift)
        cg_vals = wrap_to_180(cg_vals - shift)
        w = _wasserstein_1d(aa_vals, cg_vals)
        return w / 180.0  # range = 360°, but centred so effective range ≈ 180°

    if value_type == "angle":
        w = _wasserstein_1d(aa_vals, cg_vals)
        return w / 180.0

    # bond / constraint: normalise by combined data range
    w = _wasserstein_1d(aa_vals, cg_vals)
    data_range = float(max(aa_vals.max(), cg_vals.max()) - min(aa_vals.min(), cg_vals.min()))
    if data_range > 1e-9:
        return w / data_range
    return 0.0


def compute_internal_wasserstein(aa_internal: dict, cg_internal: dict,
                                 topo) -> dict:
    """Compute per-term and aggregate AA/CG Wasserstein distances.

    Returns
    -------
    dict
        Keys: ``"bond_dists"``, ``"constraint_dists"``, ``"angle_dists"``,
        ``"dihedral_dists"`` (lists of per-term distances), plus
        ``"bond_mean"``, ``"angle_mean"``, ``"dihedral_mean"`` (aggregate
        means; lower = better).
    """
    result: dict = {
        "bond_dists": [],
        "constraint_dists": [],
        "angle_dists": [],
        "dihedral_dists": [],
    }

    for bond in topo.bonds:
        i, j = int(bond[0]), int(bond[1])
        aa_vals = aa_internal.get((i, j, "constraint"))
        cg_vals = cg_internal.get((i, j, "bond"))
        if aa_vals is not None and cg_vals is not None:
            result["bond_dists"].append(
                _wasserstein_distance(aa_vals, cg_vals, "bond"))

    for constraint in topo.constraints:
        i, j = int(constraint[0]), int(constraint[1])
        aa_vals = aa_internal.get((i, j, "constraint"))
        cg_vals = cg_internal.get((i, j, "constraint"))
        if aa_vals is not None and cg_vals is not None:
            result["constraint_dists"].append(
                _wasserstein_distance(aa_vals, cg_vals, "constraint"))

    for angle in topo.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        aa_vals = aa_internal.get((i, j, k, "angle"))
        cg_vals = cg_internal.get((i, j, k, "angle"))
        if aa_vals is not None and cg_vals is not None:
            result["angle_dists"].append(
                _wasserstein_distance(aa_vals, cg_vals, "angle"))

    for dihedral in topo.dihedrals:
        i, j, k, l = int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3])
        aa_vals = aa_internal.get((i, j, k, l, "dihedral"))
        cg_vals = cg_internal.get((i, j, k, l, "dihedral"))
        if aa_vals is not None and cg_vals is not None:
            result["dihedral_dists"].append(
                _wasserstein_distance(aa_vals, cg_vals, "dihedral"))

    # Aggregate means (lower = better match)
    result["bond_mean"] = float(np.mean(result["bond_dists"])) if result["bond_dists"] else float("nan")
    result["angle_mean"] = float(np.mean(result["angle_dists"])) if result["angle_dists"] else float("nan")
    result["dihedral_mean"] = float(np.mean(result["dihedral_dists"])) if result["dihedral_dists"] else float("nan")

    return result


# ---------------------------------------------------------------------------
# Per-ligand analysis
# ---------------------------------------------------------------------------

def analyze_ligand(ligand_name: str, modes: set[str] | None = None) -> dict | None:
    """Run SASA, RMSD and/or Wasserstein analysis for a single ligand.

    Parameters
    ----------
    ligand_name : str
    modes : set of {"sasa", "rmsd", "wass"} or None
        Which analyses to run.  ``None`` means all three.

    Returns a dict of results, or None if data is missing.
    """
    if modes is None:
        modes = {"sasa", "rmsd", "rg", "wass"}
    do_sasa = "sasa" in modes
    do_rmsd = "rmsd" in modes
    do_rg = "rg" in modes
    do_wass = "wass" in modes
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
        traj = md.load(str(aa_trj), top=str(aa_top))
        results["aa_n_frames"] = traj.n_frames
        results["aa_n_atoms"] = traj.n_atoms

        if do_sasa:
            logger.info("[%s] AA SASA: %s", ligand_name, aa_trj)
            sasa_aa = compute_sasa(aa_top, aa_trj, is_cg=False)
            mean, lo, hi = _bootstrap_ci(sasa_aa)
            results["aa_sasa_mean"] = mean
            results["aa_sasa_ci_lo"] = lo
            results["aa_sasa_ci_hi"] = hi

        if do_rmsd:
            logger.info("[%s] AA RMSD: %s", ligand_name, aa_trj)
            rmsd_aa = compute_aa_cog_rmsd(aa_top, aa_trj, mapping)
            mean, lo, hi = _bootstrap_ci(rmsd_aa)
            results["aa_rmsd_mean"] = mean
            results["aa_rmsd_ci_lo"] = lo
            results["aa_rmsd_ci_hi"] = hi

        if do_rg:
            logger.info("[%s] AA Rg: %s", ligand_name, aa_trj)
            rg_aa = compute_aa_cog_rg(aa_top, aa_trj, mapping)
            mean, lo, hi = _bootstrap_ci(rg_aa)
            results["aa_rg_mean"] = mean
            results["aa_rg_ci_lo"] = lo
            results["aa_rg_ci_hi"] = hi
    else:
        logger.warning("[%s] AA data missing, skipping AA analysis.", ligand_name)

    # --- CG ---
    if cg_top.exists() and cg_trj.exists():
        traj = md.load(str(cg_trj), top=str(cg_top))
        results["cg_n_frames"] = traj.n_frames
        results["cg_n_beads"] = traj.n_atoms

        if do_sasa:
            logger.info("[%s] CG SASA: %s", ligand_name, cg_trj)
            sasa_cg = compute_sasa(cg_top, cg_trj, is_cg=True, bead_types=bead_types)
            mean, lo, hi = _bootstrap_ci(sasa_cg)
            results["cg_sasa_mean"] = mean
            results["cg_sasa_ci_lo"] = lo
            results["cg_sasa_ci_hi"] = hi

        if do_rmsd:
            logger.info("[%s] CG RMSD: %s", ligand_name, cg_trj)
            rmsd_cg = compute_cg_rmsd(cg_top, cg_trj)
            mean, lo, hi = _bootstrap_ci(rmsd_cg)
            results["cg_rmsd_mean"] = mean
            results["cg_rmsd_ci_lo"] = lo
            results["cg_rmsd_ci_hi"] = hi

        if do_rg:
            logger.info("[%s] CG Rg: %s", ligand_name, cg_trj)
            rg_cg = compute_cg_rg(cg_top, cg_trj)
            mean, lo, hi = _bootstrap_ci(rg_cg)
            results["cg_rg_mean"] = mean
            results["cg_rg_ci_lo"] = lo
            results["cg_rg_ci_hi"] = hi
    else:
        logger.warning("[%s] CG data missing, skipping CG analysis.", ligand_name)

    # --- AA/CG internal-coordinate Wasserstein distances ---
    if do_wass:
        _analyze_ligand_wasserstein(ligand_name, cg_top, cg_trj, itp_path, results)

    return results


def _analyze_ligand_wasserstein(ligand_name: str, cg_top: Path, cg_trj: Path,
                                itp_path: Path, results: dict) -> None:
    """Compute AA/CG Wasserstein distances for *ligand_name*; mutate *results*."""
    pkl_path = EXAMPLES_DIR / ligand_name / "internal_coords.pkl"
    if not (pkl_path.exists() and cg_top.exists() and cg_trj.exists() and itp_path.exists()):
        logger.warning(
            "[%s] Missing data for Wasserstein "
            "(pkl=%s, cg_top=%s, cg_trj=%s, itp=%s)",
            ligand_name,
            pkl_path.exists(),
            cg_top.exists(),
            cg_trj.exists(),
            itp_path.exists(),
        )
        return

    logger.info("[%s] Computing AA/CG Wasserstein distances", ligand_name)
    try:
        with open(pkl_path, "rb") as f:
            aa_internal = pickle.load(f)

        topo = smartini.topology.read_itp(str(itp_path))

        cg_traj = read_cg_trajectory(cg_top, cg_trj, start=0, stop=None, step=1)
        cg_internal = calculate_internal_coordinates(cg_traj, topo)

        dists = compute_internal_wasserstein(aa_internal, cg_internal, topo)
        results["bond_wass_mean"] = dists["bond_mean"]
        results["angle_wass_mean"] = dists["angle_mean"]
        results["dihedral_wass_mean"] = dists["dihedral_mean"]

        # Save per-term distance arrays as pkl
        dist_pkl = OUTPUT_DIR / f"{ligand_name}_wasserstein.pkl"
        dist_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(dist_pkl, "wb") as f:
            pickle.dump(dists, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[%s] Wasserstein pkl saved to %s", ligand_name, dist_pkl)

        logger.info(
            "[%s] Mean Wasserstein — bonds: %.4f  angles: %.4f  dihedrals: %.4f",
            ligand_name,
            dists["bond_mean"],
            dists["angle_mean"],
            dists["dihedral_mean"],
        )
    except Exception:
        logger.warning("[%s] Wasserstein computation failed, skipping.", ligand_name, exc_info=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def save_results(all_results: list[dict], out_dir: Path):
    """Save a summary CSV (merging with existing), then delegate plotting."""
    import csv as _csv

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sasa_rmsd_summary.csv"

    columns = [
        "ligand",
        "aa_n_frames", "aa_n_atoms",
        "aa_sasa_mean", "aa_sasa_ci_lo", "aa_sasa_ci_hi",
        "aa_rmsd_mean", "aa_rmsd_ci_lo", "aa_rmsd_ci_hi",
        "aa_rg_mean", "aa_rg_ci_lo", "aa_rg_ci_hi",
        "cg_n_frames", "cg_n_beads",
        "cg_sasa_mean", "cg_sasa_ci_lo", "cg_sasa_ci_hi",
        "cg_rmsd_mean", "cg_rmsd_ci_lo", "cg_rmsd_ci_hi",
        "cg_rg_mean", "cg_rg_ci_lo", "cg_rg_ci_hi",
        "bond_wass_mean", "angle_wass_mean", "dihedral_wass_mean",
    ]

    # Load existing rows and merge new data by ligand name
    existing: dict[str, dict] = {}
    if csv_path.exists():
        with csv_path.open(newline="") as fh:
            for row in _csv.DictReader(fh):
                existing[row["ligand"]] = dict(row)

    for r in all_results:
        name = r["ligand"]
        merged = existing.get(name, {})
        # Only overwrite keys that the new result actually computed
        for k, v in r.items():
            if v is not None:
                merged[k] = str(v)
        existing[name] = merged

    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for name in sorted(existing):
            w.writerow(existing[name])
    logger.info("CSV summary saved to %s", csv_path)

    # --- Plots (delegated to scripts/plots.py) ---
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.plots import plot_sasa_rmsd, plot_wasserstein

    plot_sasa_rmsd(all_results, out_dir)
    plot_wasserstein(all_results, out_dir)


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
    import argparse

    parser = argparse.ArgumentParser(description="SASA / RMSD / Wasserstein analysis")
    parser.add_argument("ligands", nargs="*", help="Ligand names (default: all with AA+CG data)")
    parser.add_argument("--sasa", action="store_true", help="Compute SASA")
    parser.add_argument("--rmsd", action="store_true", help="Compute RMSD")
    parser.add_argument("--rg", action="store_true", help="Compute radius of gyration")
    parser.add_argument("--wass", action="store_true", help="Compute Wasserstein distances")
    args = parser.parse_args()

    # If no flags are given, run all analyses
    if not (args.sasa or args.rmsd or args.rg or args.wass):
        args.sasa = args.rmsd = args.rg = args.wass = True

    modes = set()
    if args.sasa:
        modes.add("sasa")
    if args.rmsd:
        modes.add("rmsd")
    if args.rg:
        modes.add("rg")
    if args.wass:
        modes.add("wass")

    names = args.ligands if args.ligands else _discover_ligands()
    if not names:
        logger.error("No ligands with complete AA+CG data found in examples/")
        sys.exit(1)

    logger.info("Modes: %s", ", ".join(sorted(modes)))
    logger.info("Analysing %d ligand(s): %s", len(names), ", ".join(names))
    all_results = []
    for name in names:
        res = analyze_ligand(name, modes=modes)
        if res:
            all_results.append(res)

    save_results(all_results, OUTPUT_DIR)
    logger.info("Done.")
