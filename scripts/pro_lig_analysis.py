"""Protein-Ligand Contact Analysis: Intermolecular Contact Maps and
Fraction of Native Contacts (Q).

Computes bead-to-residue contact maps between a ligand and its protein
environment, derives binary contact maps, and tracks the fraction of
native contacts Q(t) over the course of a simulation.

Key design choices
------------------
* **Ligand coordinates** – CG beads directly; AA atoms are reduced to
  centre-of-geometry per CG bead via the ITP mapping.
* **Protein coordinates** – always centre-of-geometry per residue (both
  CG and AA), so the x-axis (residues) is identical across modalities.
* **Spatial filter** – only residues within *proximity_cutoff* of any
  ligand bead in the reference frame are retained, focusing the analysis
  on the binding pocket.
* **Unified axes** – CG and AA contact maps share the same ligand bead
  labels (y-axis) and protein residue labels (x-axis).

Usage::

    python scripts/pro_lig_analysis.py                     # defaults (1TQN)
    python scripts/pro_lig_analysis.py --sysname 1TQN      # explicit system
    python scripts/pro_lig_analysis.py --sysname 1TQN --modes aa cg
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import mdtraj as md
import MDAnalysis as mda

import smartini

logger = logging.getLogger(__name__)
smartini.setup_logging(level=logging.INFO)

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
SYSTEMS_DIR = Path("protein_systems").resolve()
OUTPUT_DIR = Path("analysis").resolve()
DEFAULT_SYSNAME = "1TQN"
DEFAULT_LIGAND_RESNAME = "HEM"

# Contact cutoff (nm) – same for CG and AA since both are at bead/residue-COG level
CONTACT_CUTOFF = 0.8
PROXIMITY_CUTOFF = 0.8  # nm


# ---------------------------------------------------------------------------
# ITP parsing (self-contained copy from analysis.py)
# ---------------------------------------------------------------------------

def _parse_itp(itp_path: Path) -> tuple[list[list[int]], list[str]]:
    """Parse a ligand ITP → (mapping, bead_types).

    mapping : list of lists of 0-based AA atom indices per CG bead.
    bead_types : list of Martini bead type strings (e.g. ``'TN6a'``).
    """
    text = itp_path.read_text()
    m = re.search(r"; Mapping:\s*(\[\[.*?\]\])", text)
    if not m:
        raise ValueError(f"No '; Mapping:' found in {itp_path}")
    mapping_raw = m.group(1)
    mapping = [list(map(int, grp.strip("[] ").split(",")))
               for grp in re.findall(r"\[([^\]]+)\]", mapping_raw)]
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
                bead_types.append(parts[1])
    return mapping, bead_types


# ---------------------------------------------------------------------------
# Core contact analysis
# ---------------------------------------------------------------------------

def compute_distance_matrix(
    ligand_xyz: np.ndarray,
    protein_xyz: np.ndarray,
) -> np.ndarray:
    """Pairwise Euclidean distances between ligand beads and protein residues.

    Parameters
    ----------
    ligand_xyz : (n_lig, 3)   ligand bead coordinates for one frame (nm).
    protein_xyz : (n_prot, 3)  protein residue-COG coordinates (nm).

    Returns
    -------
    dist : (n_lig, n_prot)  D[i,j] = ‖lig_i − prot_j‖ (nm).
    """
    lig_sq = np.sum(ligand_xyz ** 2, axis=1, keepdims=True)
    prot_sq = np.sum(protein_xyz ** 2, axis=1)
    cross = ligand_xyz @ protein_xyz.T
    d2 = lig_sq + prot_sq - 2 * cross
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def compute_contact_map(
    ligand_xyz: np.ndarray,
    protein_xyz: np.ndarray,
    cutoff: float = CONTACT_CUTOFF,
) -> np.ndarray:
    """Binary contact map: 1 if distance ≤ cutoff, else 0."""
    return compute_distance_matrix(ligand_xyz, protein_xyz) <= cutoff


def compute_native_contact_set(
    ligand_xyz: np.ndarray,
    protein_xyz: np.ndarray,
    cutoff: float = CONTACT_CUTOFF,
) -> set[tuple[int, int]]:
    """Set of (lig_idx, prot_idx) pairs in contact in the reference frame."""
    cmap = compute_contact_map(ligand_xyz, protein_xyz, cutoff)
    rows, cols = np.where(cmap)
    return set(zip(rows.tolist(), cols.tolist()))


def compute_Q_time_series(
    ligand_traj: np.ndarray,
    protein_traj: np.ndarray,
    cutoff: float = CONTACT_CUTOFF,
    ref_frame: int = 0,
    native_set: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of native contacts Q(t) per frame.

    Q(t) = N(t) / N(0), where N(t) counts native contacts still present.
    """
    n_frames = len(ligand_traj)
    if native_set is None:
        native_set = compute_native_contact_set(
            ligand_traj[ref_frame], protein_traj[ref_frame], cutoff
        )
    N0 = len(native_set)
    if N0 == 0:
        logger.warning("No native contacts found at reference frame %d", ref_frame)
        return np.zeros(n_frames), np.zeros(n_frames, dtype=int)

    pairs = np.array(sorted(native_set))
    lig_idx = pairs[:, 0]
    prot_idx = pairs[:, 1]

    Q = np.empty(n_frames)
    n_contacts = np.empty(n_frames, dtype=int)
    for t in range(n_frames):
        d = np.linalg.norm(
            ligand_traj[t][lig_idx] - protein_traj[t][prot_idx], axis=1
        )
        n_contacts[t] = np.sum(d <= cutoff)
        Q[t] = n_contacts[t] / N0

    return Q, n_contacts


def compute_contact_frequency(
    ligand_traj: np.ndarray,
    protein_traj: np.ndarray,
    cutoff: float = CONTACT_CUTOFF,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> np.ndarray:
    """Per-pair contact frequency over a trajectory ∈ [0, 1]."""
    if stop is None:
        stop = len(ligand_traj)
    frames = range(start, stop, step)
    n_frames = len(frames)
    n_lig = ligand_traj.shape[1]
    n_prot = protein_traj.shape[1]

    freq = np.zeros((n_lig, n_prot), dtype=np.float64)
    for t in frames:
        freq += compute_contact_map(ligand_traj[t], protein_traj[t], cutoff)
    freq /= n_frames
    return freq


# ---------------------------------------------------------------------------
# Residue COG helpers
# ---------------------------------------------------------------------------

def _residue_cog_from_mdanalysis(
    atom_group: mda.AtomGroup,
    atom_xyz: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Residue-COG trajectory from an MDAnalysis AtomGroup.

    Parameters
    ----------
    atom_group : MDAnalysis AtomGroup
        The atoms to reduce to residue COGs (e.g. ``u.select_atoms("protein")``).
    atom_xyz : (n_frames, n_atoms, 3) in nm.
        Must match the order and count of *atom_group*.

    Returns
    -------
    cog_xyz : (n_frames, n_unique_res, 3)
    res_labels : list[str]  e.g. ``["ALA12", "GLY27", ...]``.
    """
    n_atoms = len(atom_group)
    residue_ids = np.array([a.residue.resindex for a in atom_group], dtype=int)
    residue_names = np.array([a.residue.resname for a in atom_group])
    residue_nums = np.array([a.residue.resid for a in atom_group])

    unique_ids = np.unique(residue_ids)
    n_res = len(unique_ids)
    n_frames = atom_xyz.shape[0]
    cog = np.empty((n_frames, n_res, 3), dtype=np.float64)
    labels = []
    for i, rid in enumerate(unique_ids):
        mask = residue_ids == rid
        first = np.where(mask)[0][0]
        labels.append(f"{residue_names[first]}{residue_nums[first]}")
        for t in range(n_frames):
            cog[t, i] = atom_xyz[t, mask].mean(axis=0)
    return cog, labels


def _residue_cog_from_mdtraj(
    traj: md.Trajectory,
    atom_indices: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Residue-COG trajectory from mdtraj atom subset.

    Parameters
    ----------
    traj : mdtraj.Trajectory
    atom_indices : (n_atoms,) int  – which atoms to include.

    Returns
    -------
    cog_xyz : (n_frames, n_unique_res, 3)
    res_labels : list[str]
    """
    table, _ = traj.topology.to_dataframe()
    sel_table = table.iloc[atom_indices]
    residue_names = sel_table["resName"].values
    residue_nums = sel_table["resSeq"].values.astype(int)

    # Build a contiguous residue index from (resName, resSeq) pairs
    res_keys = [f"{rn}{rs}" for rn, rs in zip(residue_names, residue_nums)]
    unique_keys = list(dict.fromkeys(res_keys))  # preserve order, deduplicate
    key_to_id = {k: i for i, k in enumerate(unique_keys)}
    residue_ids = np.array([key_to_id[k] for k in res_keys], dtype=int)

    n_res = len(unique_keys)
    n_frames = traj.n_frames
    cog = np.empty((n_frames, n_res, 3), dtype=np.float64)
    for i in range(n_res):
        mask = residue_ids == i
        for t in range(n_frames):
            cog[t, i] = traj.xyz[t, atom_indices[mask], :].mean(axis=0)
    return cog, unique_keys


# ---------------------------------------------------------------------------
# Proximity filter
# ---------------------------------------------------------------------------

def _select_proximal_residues(
    lig_ref_xyz: np.ndarray,
    prot_cog_xyz: np.ndarray,
    prot_labels: list[str],
    proximity_cutoff: float = PROXIMITY_CUTOFF,
) -> tuple[np.ndarray, list[str]]:
    """Return protein residues whose COG is within *proximity_cutoff* of any
    ligand bead in the reference frame.

    Parameters
    ----------
    lig_ref_xyz : (n_lig, 3)
    prot_cog_xyz : (n_prot, 3)
    prot_labels : list[str]
    proximity_cutoff : float  nm.

    Returns
    -------
    mask : (n_prot,) bool
    filtered_labels : list[str]
    """
    dist = compute_distance_matrix(lig_ref_xyz, prot_cog_xyz)
    close = np.any(dist <= proximity_cutoff, axis=0)
    labels = [prot_labels[i] for i in range(len(prot_labels)) if close[i]]
    return close, labels


# ---------------------------------------------------------------------------
# AA ligand → bead-COG via ITP mapping
# ---------------------------------------------------------------------------

def _aa_ligand_to_bead_cog(
    traj: md.Trajectory,
    lig_indices: np.ndarray,
    mapping: list[list[int]],
) -> np.ndarray:
    """Map AA ligand atoms to CG-bead centres of geometry.

    Parameters
    ----------
    traj : mdtraj.Trajectory
    lig_indices : (n_lig_atoms,) int  – indices of ligand atoms in traj.
    mapping : list[list[int]]  – ITP mapping: inner lists are 0-based indices
        into the ligand atom list (not into the full trajectory).

    Returns
    -------
    bead_xyz : (n_frames, n_beads, 3)
    """
    n_beads = len(mapping)
    n_frames = traj.n_frames
    bead_xyz = np.empty((n_frames, n_beads, 3), dtype=np.float64)
    for t in range(n_frames):
        for b, atom_indices in enumerate(mapping):
            traj_indices = lig_indices[np.array(atom_indices, dtype=int)]
            bead_xyz[t, b] = traj.xyz[t, traj_indices, :].mean(axis=0)
    logger.info("AA ligand: %d atoms → %d beads via ITP mapping",
                len(lig_indices), n_beads)
    return bead_xyz


# ---------------------------------------------------------------------------
# System-level analysis
# ---------------------------------------------------------------------------

def analyze_system(
    sysname: str = DEFAULT_SYSNAME,
    systems_dir: Path = SYSTEMS_DIR,
    modes: set[str] | None = None,
    lig_resname: str = DEFAULT_LIGAND_RESNAME,
    cg_runname: str | None = None,
    aa_runname: str = "mdrun_1",
    lig_itp_name: str | None = None,
    cg_lig_selection: str | None = None,
    aa_lig_selection: str | None = None,
    contact_cutoff: float = CONTACT_CUTOFF,
    proximity_cutoff: float = PROXIMITY_CUTOFF,
    ref_frame: int = 0,
) -> dict:
    """Run contact-map and Q analysis for a protein-ligand system.

    Parameters
    ----------
    sysname : str
        System name (e.g. ``"1TQN"``).
    systems_dir : Path
        Root directory containing system subdirectories.
    modes : set of {"aa", "cg"} or None
    lig_resname : str
        Residue name of the ligand.
    cg_runname : str, optional
        CG run subdirectory.  Default: ``"mdrun_2"``.
    aa_runname : str
        AA run subdirectory.  Default: ``"mdrun_1"``.
    lig_itp_name : str, optional
        ITP filename stem.  Default: same as *lig_resname*.
    contact_cutoff : float
        Contact distance threshold (nm), same for CG and AA.
    proximity_cutoff : float
        Only keep protein residues within this distance of the ligand (nm).
    ref_frame : int
        Reference frame index for native contacts and proximity filter.

    Returns
    -------
    dict
    """
    if modes is None:
        modes = {"aa", "cg"}
    if cg_runname is None:
        cg_runname = "mdrun_2"
    if lig_itp_name is None:
        lig_itp_name = lig_resname
    if cg_lig_selection is None:
        cg_lig_selection = f"resname {lig_resname}"
    if aa_lig_selection is None:
        aa_lig_selection = f"resname {lig_resname} or resname UNK"

    sys_dir = systems_dir / sysname
    results: dict = {
        "sysname": sysname,
        "_lig_resname": lig_resname,
        "contact_cutoff": contact_cutoff,
        "proximity_cutoff": proximity_cutoff,
    }

    # --- Parse ITP for ligand bead names (shared between CG and AA) ---
    lig_bead_names = None
    for itp_path in [
        sys_dir / "ligands" / lig_itp_name / f"{lig_itp_name}.itp",
        Path("examples") / lig_itp_name / f"{lig_itp_name}.itp",
    ]:
        if itp_path.exists():
            mapping, bead_types = _parse_itp(itp_path)
            lig_bead_names = [f"{bt[0]}{i + 1:02d}" for i, bt in enumerate(bead_types)]
            logger.info("Parsed ITP: %d beads from %s", len(mapping), itp_path)
            break
    if lig_bead_names is None:
        logger.warning("No ITP found for ligand '%s' – bead labels won't be available.",
                       lig_itp_name)
    results["lig_bead_names"] = lig_bead_names

    cg_prot_labels_all: list[str] | None = None
    aa_prot_labels_all: list[str] | None = None

    # --- CG analysis ---
    if "cg" in modes:
        cg_dir = sys_dir / "mdruns" / cg_runname
        cg_top = cg_dir / "topology.pdb"
        cg_trj = cg_dir / "samples.xtc"
        if cg_top.exists() and cg_trj.exists():
            logger.info("[%s] CG data: %s", sysname, cg_dir)
            try:
                _cg_analyze(
                    results, cg_top, cg_trj,
                    lig_selection=cg_lig_selection,
                    contact_cutoff=contact_cutoff,
                    proximity_cutoff=proximity_cutoff,
                    ref_frame=ref_frame,
                )
                cg_prot_labels_all = results.get("cg_prot_labels_all")
            except Exception:
                logger.warning("[%s] CG analysis failed", sysname, exc_info=True)
        else:
            logger.warning("[%s] No CG trajectory in %s", sysname, cg_dir)

    # --- AA analysis ---
    if "aa" in modes:
        aa_dir = systems_dir / f"{sysname}_aa" / "mdruns" / aa_runname
        aa_top = aa_dir / "topology.pdb"
        aa_trj = aa_dir / "samples.xtc"
        if aa_top.exists() and aa_trj.exists():
            logger.info("[%s] AA data: %s", sysname, aa_dir)
            try:
                _aa_analyze(
                    results, aa_top, aa_trj,
                    lig_selection=aa_lig_selection,
                    contact_cutoff=contact_cutoff,
                    proximity_cutoff=proximity_cutoff,
                    ref_frame=ref_frame,
                )
                aa_prot_labels_all = results.get("aa_prot_labels_all")
            except Exception:
                logger.warning("[%s] AA analysis failed", sysname, exc_info=True)
        else:
            logger.warning("[%s] No AA trajectory in %s", sysname, aa_dir)

    # --- Unify residue labels for identical axes ---
    _unify_residue_selection(results, cg_prot_labels_all, aa_prot_labels_all)

    return results


# ---------------------------------------------------------------------------
# Per-mode analysis subroutines
# ---------------------------------------------------------------------------

def _cg_analyze(
    results: dict,
    top_path: Path,
    trj_path: Path,
    lig_selection: str,
    contact_cutoff: float,
    proximity_cutoff: float,
    ref_frame: int,
) -> None:
    """CG analysis: ligand beads directly; protein → residue COG."""
    u = mda.Universe(str(top_path), str(trj_path))
    lig_atoms = u.select_atoms(lig_selection)
    prot_atoms = u.select_atoms("protein")
    if len(lig_atoms) == 0:
        raise ValueError(f"CG: no atoms matched '{lig_selection}'")
    if len(prot_atoms) == 0:
        raise ValueError("CG: no protein atoms found")
    n_frames = len(u.trajectory)

    # Ligand: bead positions directly
    lig_xyz = np.empty((n_frames, len(lig_atoms), 3), dtype=np.float64)
    prot_atom_xyz = np.empty((n_frames, len(prot_atoms), 3), dtype=np.float64)
    for t, _ in enumerate(u.trajectory):
        lig_xyz[t] = lig_atoms.positions / 10.0
        prot_atom_xyz[t] = prot_atoms.positions / 10.0

    # Protein: COG per residue
    prot_cog_all, prot_labels_all = _residue_cog_from_mdanalysis(
        prot_atoms, prot_atom_xyz
    )

    logger.info("CG: %d ligand beads, %d protein residues, %d frames",
                lig_xyz.shape[1], prot_cog_all.shape[1], n_frames)

    # Proximity filter
    prox_mask, prot_labels_filt = _select_proximal_residues(
        lig_xyz[ref_frame], prot_cog_all[ref_frame],
        prot_labels_all, proximity_cutoff,
    )
    prot_cog = prot_cog_all[:, prox_mask, :]
    n_res = prot_cog.shape[1]
    logger.info("CG: %d/%d residues within %.1f nm of ligand",
                n_res, prot_cog_all.shape[1], proximity_cutoff)

    # Native contacts & Q
    native = compute_native_contact_set(
        lig_xyz[ref_frame], prot_cog[ref_frame], contact_cutoff
    )
    Q, n_contacts = compute_Q_time_series(
        lig_xyz, prot_cog, contact_cutoff, ref_frame, native
    )
    freq = compute_contact_frequency(lig_xyz, prot_cog, contact_cutoff)

    results["cg_Q"] = Q
    results["cg_n_contacts"] = n_contacts
    results["cg_contact_freq"] = freq
    results["cg_lig_labels"] = results.get("lig_bead_names") or _build_labels(lig_xyz.shape[1], "L")
    results["cg_prot_labels"] = prot_labels_filt
    results["cg_prot_labels_all"] = prot_labels_all
    results["cg_prox_mask"] = prox_mask
    results["native_set_cg"] = native
    results["cg_n_frames"] = n_frames
    results["cg_n_lig"] = lig_xyz.shape[1]
    results["cg_n_prot"] = n_res
    logger.info("[CG] %d native contacts, ⟨Q⟩ = %.3f", len(native), float(np.mean(Q)))


def _aa_analyze(
    results: dict,
    top_path: Path,
    trj_path: Path,
    lig_selection: str,
    contact_cutoff: float,
    proximity_cutoff: float,
    ref_frame: int,
) -> None:
    """AA analysis: ligand → COG per CG bead; protein → COG per residue."""
    traj = md.load(str(trj_path), top=str(top_path))

    prot_indices = traj.topology.select("protein")
    lig_indices = traj.topology.select(lig_selection)
    if len(lig_indices) == 0:
        raise ValueError(
            f"AA: no atoms matched '{lig_selection}'. "
            f"Residues: {set(r.name for r in traj.topology.residues)}"
        )
    if len(prot_indices) == 0:
        raise ValueError("AA: no protein atoms found")

    # Protein: COG per residue
    prot_cog_all, prot_labels_all = _residue_cog_from_mdtraj(traj, prot_indices)

    # AA ligand → COG per CG bead using ITP mapping
    lig_bead_names = results.get("lig_bead_names")
    lig_resname = results.get("_lig_resname", DEFAULT_LIGAND_RESNAME)
    sys_dir = SYSTEMS_DIR / results.get("sysname", DEFAULT_SYSNAME)

    mapping = None
    for itp_path in [
        sys_dir / "ligands" / lig_resname / f"{lig_resname}.itp",
        Path("examples") / lig_resname / f"{lig_resname}.itp",
    ]:
        if itp_path.exists():
            mapping, _ = _parse_itp(itp_path)
            break

    if mapping is not None:
        lig_bead_xyz = _aa_ligand_to_bead_cog(traj, lig_indices, mapping)
    else:
        logger.warning("No ITP mapping found; using per-atom coords for AA ligand.")
        lig_bead_xyz = traj.xyz[:, lig_indices, :]
        if lig_bead_names is None:
            lig_bead_names = _build_labels(lig_bead_xyz.shape[1], "L")

    logger.info("AA: %d ligand beads, %d protein residues, %d frames",
                lig_bead_xyz.shape[1], prot_cog_all.shape[1], traj.n_frames)

    # Proximity filter
    prox_mask, prot_labels_filt = _select_proximal_residues(
        lig_bead_xyz[ref_frame], prot_cog_all[ref_frame],
        prot_labels_all, proximity_cutoff,
    )
    prot_cog = prot_cog_all[:, prox_mask, :]
    n_res = prot_cog.shape[1]
    logger.info("AA: %d/%d residues within %.1f nm of ligand",
                n_res, prot_cog_all.shape[1], proximity_cutoff)

    # Native contacts & Q
    native = compute_native_contact_set(
        lig_bead_xyz[ref_frame], prot_cog[ref_frame], contact_cutoff
    )
    Q, n_contacts = compute_Q_time_series(
        lig_bead_xyz, prot_cog, contact_cutoff, ref_frame, native
    )
    freq = compute_contact_frequency(lig_bead_xyz, prot_cog, contact_cutoff)

    results["aa_Q"] = Q
    results["aa_n_contacts"] = n_contacts
    results["aa_contact_freq"] = freq
    results["aa_lig_labels"] = lig_bead_names or _build_labels(lig_bead_xyz.shape[1], "L")
    results["aa_prot_labels"] = prot_labels_filt
    results["aa_prot_labels_all"] = prot_labels_all
    results["aa_prox_mask"] = prox_mask
    results["native_set_aa"] = native
    results["aa_n_frames"] = traj.n_frames
    results["aa_n_lig"] = lig_bead_xyz.shape[1]
    results["aa_n_prot"] = n_res
    logger.info("[AA] %d native contacts, ⟨Q⟩ = %.3f", len(native), float(np.mean(Q)))


# ---------------------------------------------------------------------------
# Unify residue selection for identical axes
# ---------------------------------------------------------------------------

def _unify_residue_selection(
    results: dict,
    cg_labels_all: list[str] | None,
    aa_labels_all: list[str] | None,
) -> None:
    """Ensure CG and AA contact frequency matrices share the same residue
    columns (x-axis) by intersecting the sets of proximal residues.

    When both CG and AA are present, the intersection is used so that
    the heatmaps have identical residue columns.
    """
    cg_freq = results.get("cg_contact_freq")
    aa_freq = results.get("aa_contact_freq")

    if cg_freq is None or aa_freq is None:
        return

    cg_labels = results.get("cg_prot_labels")
    aa_labels = results.get("aa_prot_labels")
    if cg_labels is None or aa_labels is None:
        return

    # Find common residues by label
    common = sorted(set(cg_labels) & set(aa_labels), key=lambda x: (x,))
    if not common:
        logger.warning("No common protein residues between CG and AA proximal sets.")
        return

    cg_idx = [cg_labels.index(l) for l in common]
    aa_idx = [aa_labels.index(l) for l in common]

    results["cg_contact_freq"] = cg_freq[:, cg_idx]
    results["aa_contact_freq"] = aa_freq[:, aa_idx]
    results["cg_prot_labels"] = common
    results["aa_prot_labels"] = common
    results["unified_prot_labels"] = common
    logger.info("Unified residue axes: %d common residues.", len(common))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_labels(n: int, prefix: str = "L") -> list[str]:
    return [f"{prefix}{i + 1}" for i in range(n)]


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def save_results(results: dict, out_dir: Path, sysname: str | None = None):
    """Save analysis results to CSV and pickle files."""
    sysname = sysname or results.get("sysname", "system")
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / f"{sysname}_contacts.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Full results saved to %s", pkl_path)

    csv_path = out_dir / f"{sysname}_Q_summary.csv"
    rows = []
    for mode in ("cg", "aa"):
        Q = results.get(f"{mode}_Q")
        n_contacts = results.get(f"{mode}_n_contacts")
        if Q is not None:
            for t in range(len(Q)):
                rows.append({
                    "mode": mode,
                    "frame": t,
                    "Q": f"{Q[t]:.6f}",
                    "n_native_contacts": str(n_contacts[t]) if n_contacts is not None else "",
                })
    if rows:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode", "frame", "Q", "n_native_contacts"])
            w.writeheader()
            w.writerows(rows)
        logger.info("Q summary CSV saved to %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Protein-ligand contact map and native-contact analysis"
    )
    parser.add_argument("--sysname", default=DEFAULT_SYSNAME,
                        help=f"System name (default: {DEFAULT_SYSNAME})")
    parser.add_argument("--systems-dir", type=Path, default=SYSTEMS_DIR,
                        help="Root directory for protein systems")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for results and plots")
    parser.add_argument("--lig-resname", default=DEFAULT_LIGAND_RESNAME,
                        help="Ligand residue name")
    parser.add_argument("--lig-itp-name", default=None,
                        help="ITP filename stem (default: same as --lig-resname)")
    parser.add_argument("--modes", nargs="+", choices=["aa", "cg"],
                        default=["aa", "cg"],
                        help="Which trajectory modes to analyse (default: both)")
    parser.add_argument("--contact-cutoff", type=float, default=CONTACT_CUTOFF,
                        help=f"Contact cutoff in nm (default: {CONTACT_CUTOFF})")
    parser.add_argument("--proximity-cutoff", type=float, default=PROXIMITY_CUTOFF,
                        help=f"Proximity filter cutoff in nm (default: {PROXIMITY_CUTOFF})")
    parser.add_argument("--ref-frame", type=int, default=0,
                        help="Reference frame for native contacts (default: 0)")
    parser.add_argument("--cg-runname", default=None,
                        help="CG run directory name (default: mdrun_1)")
    parser.add_argument("--aa-runname", default="mdrun_1",
                        help="AA run directory name (default: mdrun_1)")
    args = parser.parse_args()

    modes = set(args.modes)
    logger.info("Analysing system '%s' with modes: %s", args.sysname, sorted(modes))

    results = analyze_system(
        sysname=args.sysname,
        systems_dir=args.systems_dir,
        modes=modes,
        lig_resname=args.lig_resname,
        cg_runname=args.cg_runname,
        aa_runname=args.aa_runname,
        lig_itp_name=args.lig_itp_name,
        contact_cutoff=args.contact_cutoff,
        proximity_cutoff=args.proximity_cutoff,
        ref_frame=args.ref_frame,
    )

    save_results(results, args.out_dir, args.sysname)

    # --- Plots (delegated to scripts/plots.py) ---
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.plots import plot_contact_frequency_comparison, plot_Q_time_series

    # --- Contact frequency comparison plot (CG + AA side-by-side, same axes) ---
    freq_cg = results.get("cg_contact_freq")
    freq_aa = results.get("aa_contact_freq")
    if freq_cg is not None or freq_aa is not None:
        plot_contact_frequency_comparison(
            freq_cg, freq_aa,
            args.out_dir,
            lig_labels=results.get("lig_bead_names"),
            prot_labels=results.get("unified_prot_labels"),
            png_name=f"{args.sysname}_contact_freq_comparison.png",
        )

    # --- Q(t) combined plot (with inset contact map) ---
    Q_data = {}
    for mode in modes:
        Q = results.get(f"{mode}_Q")
        if Q is not None:
            Q_data[mode.upper()] = Q
    if Q_data:
        # Build unified contact freq for inset
        inset = None
        if freq_cg is not None and freq_aa is not None:
            inset = (freq_cg + freq_aa) / 2.0
        elif freq_cg is not None:
            inset = freq_cg
        elif freq_aa is not None:
            inset = freq_aa

        plot_Q_time_series(
            Q_data,
            args.out_dir,
            inset_freq=inset,
            png_name=f"{args.sysname}_Q_vs_time.png",
        )

    logger.info("Done.")
