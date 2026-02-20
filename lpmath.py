import logging
from typing import Dict, Tuple

import numpy as np
from MDAnalysis import Universe

logger = logging.getLogger(__name__)


def read_cog_trajectory(in_pdb, in_xtc, partitioning, start=0, stop=5000):
    """Read AA trajectory and calculate COG trajectory for CG beads.

    Parameters
    ----------
    in_pdb : str or Path
        Path to atomistic PDB file
    in_xtc : str or Path
        Path to atomistic XTC trajectory
    partitioning : dict
        Mapping of atom indices to bead indices {atom_idx: bead_idx}
    start : int
        Starting frame index to read from the trajectory
    stop : int
        Ending frame index to read from the trajectory

    Returns
    -------
    numpy.ndarray
        CG trajectory array with shape (n_frames, n_beads, 3)
    """
    logger.info("Reading AA trajectory: %s, %s", in_pdb, in_xtc)

    u = Universe(str(in_pdb), str(in_xtc))
    n_frames = stop - start

    n_beads = max(partitioning.values()) + 1
    bead_to_atoms = {i: [] for i in range(n_beads)}
    for atom_idx, bead_idx in partitioning.items():
        bead_to_atoms[bead_idx].append(atom_idx)

    cg_trajectory = np.zeros((n_frames, n_beads, 3))

    for frame_idx, _ in enumerate(u.trajectory[start:stop]):
        for bead_idx in range(n_beads):
            atom_indices = bead_to_atoms[bead_idx]
            if atom_indices:
                positions = u.atoms[atom_indices].positions
                cg_trajectory[frame_idx, bead_idx] = positions.mean(axis=0) / 10.0

    logger.info("COG trajectory computed: %s frames, %s beads", cg_trajectory.shape[0], n_beads)
    return cg_trajectory


def read_cg_trajectory(in_pdb, in_xtc, start=0, stop=5000):
    """Read CG trajectory and return positions in nm.

    Parameters
    ----------
    in_pdb : str or Path
        Path to CG PDB file
    in_xtc : str or Path
        Path to CG XTC trajectory

    Returns
    -------
    numpy.ndarray
        CG trajectory array with shape (n_frames, n_beads, 3) in nm
    """
    logger.info("Reading CG trajectory: %s, %s", in_pdb, in_xtc)
    u = Universe(str(in_pdb), str(in_xtc))
    n_frames = stop - start
    n_beads = len(u.atoms)
    cg_trajectory = np.zeros((n_frames, n_beads, 3))

    for frame_idx, _ in enumerate(u.trajectory[start:stop]):
        cg_trajectory[frame_idx] = u.atoms.positions / 10.0

    logger.info("Loaded CG trajectory: %s frames, %s beads", n_frames, n_beads)
    return cg_trajectory


def calculate_internal_coordinates(cg_trajectory, topo):
    """Calculate internal coordinates (bonds, angles, dihedrals) from CG trajectory."""
    n_frames = cg_trajectory.shape[0]
    internal_coords = {}

    for bond in topo.bonds:
        i, j = int(bond[0]), int(bond[1])
        distances = np.zeros(n_frames)
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            distances[frame_idx] = np.linalg.norm(pos_i - pos_j)
        internal_coords[(i, j, "bond")] = distances

    for constraint in topo.constraints:
        i, j = int(constraint[0]), int(constraint[1])
        distances = np.zeros(n_frames)
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            distances[frame_idx] = np.linalg.norm(pos_i - pos_j)
        internal_coords[(i, j, "constraint")] = distances

    for angle in topo.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        angles = np.zeros(n_frames)
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            pos_k = cg_trajectory[frame_idx, k]
            v1 = pos_i - pos_j
            v2 = pos_k - pos_j
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles[frame_idx] = np.degrees(np.arccos(cos_angle))
        internal_coords[(i, j, k, "angle")] = angles

    for dihedral in topo.dihedrals:
        i, j, k, l = int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3])
        dihedrals = np.zeros(n_frames)
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            pos_k = cg_trajectory[frame_idx, k]
            pos_l = cg_trajectory[frame_idx, l]
            b1 = pos_j - pos_i
            b2 = pos_k - pos_j
            b3 = pos_l - pos_k
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            b2_norm = b2 / np.linalg.norm(b2)
            x = np.dot(n1, n2)
            y = np.dot(np.cross(n1, b2_norm), n2)
            dihedrals[frame_idx] = np.degrees(np.arctan2(y, x))
        internal_coords[(i, j, k, l, "dihedral")] = dihedrals

    return internal_coords


def boltzmann_inversion_bond(distances, temperature=300.0):
    """Estimate harmonic bond parameters from samples.

    This is a *mean-based harmonic approximation* (not PMF-minimum based):
    - Equilibrium value r0 is the sample mean.
    - Force constant k is computed from fluctuations: k = kT / var(r).
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    distances = np.asarray(distances, dtype=float)
    r0 = float(np.mean(distances))

    variance = float(np.var(distances))
    k = float(kT / variance)

    return r0, k


def boltzmann_inversion_angle(angles, temperature=300.0):
    """Estimate harmonic angle parameters from samples.

    Mean-based harmonic approximation:
    - Equilibrium value theta0 is the sample mean (degrees).
    - Force constant k is computed from fluctuations in radians:
      k = kT / var(theta_rad).
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    angles = np.asarray(angles, dtype=float)
    theta0 = float(np.mean(angles))

    residual_rad = np.deg2rad(angles - theta0)
    variance_rad = float(np.var(residual_rad))
    k = float(kT / variance_rad)

    return theta0, k


def circular_mean_deg(angles):
    """Calculate circular mean of angles in degrees."""
    angles_rad = np.deg2rad(angles)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_mean, cos_mean)
    return np.rad2deg(mean_rad)


def wrap_to_180(angles):
    """Wrap angles to [-180, 180] range."""
    return (angles + 180) % 360 - 180 


def boltzmann_inversion_dihedral(dihedrals, temperature=300.0):
    """Estimate harmonic dihedral parameters from samples.

    Mean-based harmonic approximation for periodic angles:
    - Equilibrium value phi0 is the circular mean (degrees, wrapped to [-180, 180]).
    - Force constant k is computed from wrapped residual fluctuations in radians:
      k = kT / var(phi_rad).
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    dihedrals = np.asarray(dihedrals, dtype=float)
    phi0 = float(wrap_to_180(circular_mean_deg(dihedrals)))

    residual_deg = wrap_to_180(dihedrals - phi0)
    residual_rad = np.deg2rad(residual_deg)
    variance_rad = float(np.var(residual_rad))
    k = float(kT / variance_rad)

    return phi0, k


def fit_type9_dihedral(
    dihedrals,
    temperature=300.0,
    max_n=3,
    bins=360,
    min_prob=1e-6,
    fit_mode="sum",
):
    r"""Fit Gromacs type-9 dihedral terms from a distribution.

    We estimate a PMF from the histogram and fit Fourier terms of the form:

        U(\phi) = \sum_n k_n \left(1 + \cos(n\phi - \phi_n)\right)

    Since
        k\cos(n\phi-\phi_n) = (k\cos\phi_n)\cos(n\phi) + (k\sin\phi_n)\sin(n\phi),
    we can fit this robustly via (weighted) linear least squares in the basis
    \{1, \cos(n\phi), \sin(n\phi)\}.

    Parameters
    ----------
    fit_mode : {"sum", "best1"}
        - "sum": pick best single harmonic, then refit with it plus one additional harmonic.
        - "best1": fit each harmonic individually and return the single best n.

    Returns
    -------
    list[tuple[int, float, float]]
        List of (multiplicity n, k_n, phi_n_deg) terms.
        The phase angle is in degrees (wrapped to [-180, 180]).
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    values = dihedrals
    shift = circular_mean_deg(values)
    values -= shift
    values = wrap_to_180(values)
    min_val = np.min(values)
    max_val = np.max(values)

    hist, edges = np.histogram(values, bins=bins, range=(min_val, max_val), density=True)
    # hist, edges = np.histogram(values, bins=bins, range=(-180, 180), density=True)
    hist = np.clip(hist, min_prob, None)

    phi_centers = 0.5 * (edges[:-1] + edges[1:])
    phi_rad = np.deg2rad(phi_centers)

    pmf = -kT * np.log(hist)
    pmf = pmf - float(np.min(pmf))

    if max_n <= 0:
        return []

    # weights = np.sqrt(hist)
    weights = np.ones_like(hist)

    def _solve_weighted(A, y):
        Aw = A * weights[:, None]
        yw = y * weights
        coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
        return coeffs

    def _k_phi_from_ab(a, b, n: int):
        k = float(np.hypot(a, b))
        if k < 1e-12:
            return 0.0, 0.0
        phi = float(np.rad2deg(np.arctan2(b, a)))
        # We fitted on shifted angles: phi' = phi - shift.
        # For k*cos(n*phi - phi_n), shifting phi by `shift` shifts the phase by n*shift.
        phi += float(n) * float(shift)
        phi = (360 - phi) % 360
        phi = float(wrap_to_180(phi))
        return k, phi

    fit_mode = str(fit_mode).lower().strip()
    if fit_mode not in {"sum", "best1"}:
        raise ValueError(f"fit_mode must be 'sum' or 'best1' (got {fit_mode!r})")

    def _fit_terms_for_ns(ns):
        cols = [np.ones_like(phi_rad)]
        for n in ns:
            cols.append(np.cos(n * phi_rad))
            cols.append(np.sin(n * phi_rad))
        A = np.column_stack(cols)
        coeffs = _solve_weighted(A, pmf)
        pred = A @ coeffs
        resid = pmf - pred
        sse = float(np.sum((resid * weights) ** 2))
        terms = []
        for idx, n in enumerate(ns):
            a = float(coeffs[1 + 2 * idx])
            b = float(coeffs[1 + 2 * idx + 1])
            k, phi = _k_phi_from_ab(a, b, n)
            terms.append((int(n), float(k), float(phi)))
        return sse, terms

    if fit_mode == "best1":
        best_sse = None
        best_terms = []
        for n in range(1, max_n + 1):
            sse, terms = _fit_terms_for_ns([n])
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_terms = terms

        return best_terms

    # fit_mode == "sum": iterative best1 + one additional term
    best_sse = None
    best_terms = []
    best_n = None
    for n in range(1, max_n + 1):
        sse, terms = _fit_terms_for_ns([n])
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_terms = terms
            best_n = n

    if best_n is None or max_n <= 1:
        return best_terms

    best_pair_sse = None
    best_pair_terms = None
    for n in range(1, max_n + 1):
        if n == best_n:
            continue
        sse, terms = _fit_terms_for_ns([best_n, n])
        if best_pair_sse is None or sse < best_pair_sse:
            best_pair_sse = sse
            best_pair_terms = terms

    return best_pair_terms if best_pair_terms is not None else best_terms
