import logging
from typing import Dict, Tuple

import numpy as np
from MDAnalysis import Universe

logger = logging.getLogger(__name__)


def read_cog_trajectory(in_pdb, in_xtc, partitioning, start=0, stop=-2, selection="all"):
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
    selection : str, optional
        MDAnalysis selection string (default: "all")
        Examples: "all", "protein", "resname LIG", etc.

    Returns
    -------
    numpy.ndarray
        CG trajectory array with shape (n_frames, n_beads, 3)
    """
    logger.info("Reading AA trajectory: %s, %s, selection='%s'", in_pdb, in_xtc, selection)

    u = Universe(str(in_pdb), str(in_xtc))
    
    # Select atoms based on selection string
    atom_group = u.select_atoms(selection)
    logger.info("Selected %s atoms with selection '%s'", len(atom_group), selection)

    n_beads = max(partitioning.values()) + 1
    bead_to_atoms = {i: [] for i in range(n_beads)}
    for atom_idx, bead_idx in partitioning.items():
        bead_to_atoms[bead_idx].append(atom_idx)

    # Read frames into list first to avoid zeros from frame mismatch
    frames = []
    for ts in u.trajectory[start:stop]:
        frame_beads = np.zeros((n_beads, 3))
        for bead_idx in range(n_beads):
            atom_indices = bead_to_atoms[bead_idx]
            if atom_indices:
                positions = atom_group[atom_indices].positions
                frame_beads[bead_idx] = positions.mean(axis=0) / 10.0
        frames.append(frame_beads)
    
    # Convert list to numpy array
    cg_trajectory = np.array(frames)
    n_frames = len(frames)

    logger.info("COG trajectory computed: %s frames, %s beads", n_frames, n_beads)
    return cg_trajectory


def read_cg_trajectory(in_pdb, in_xtc, start=0, stop=5000, selection="all"):
    """Read CG trajectory and return positions in nm.

    Parameters
    ----------
    in_pdb : str or Path
        Path to CG PDB file
    in_xtc : str or Path
        Path to CG XTC trajectory
    start : int, optional
        Starting frame index (default: 0)
    stop : int, optional
        Stopping frame index (default: 5000)
    selection : str, optional
        MDAnalysis selection string (default: "all")
        Examples: "all", "name BB", "resname LIG", "protein", etc.

    Returns
    -------
    numpy.ndarray
        CG trajectory array with shape (n_frames, n_beads, 3) in nm
    """
    logger.info("Reading CG trajectory: %s, %s, selection='%s'", in_pdb, in_xtc, selection)
    u = Universe(str(in_pdb), str(in_xtc))
    
    # Select atoms based on selection string
    atom_group = u.select_atoms(selection)
    n_beads = len(atom_group)
    logger.info("Selected %s atoms with selection '%s'", n_beads, selection)
    
    # Read frames into list first to avoid zeros from frame mismatch
    frames = []
    for ts in u.trajectory[start:stop]:
        frames.append(atom_group.positions / 10.0)
    
    # Convert list to numpy array
    cg_trajectory = np.array(frames)
    n_frames = len(frames)
    
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
    return ((angles + 180) % 360) - 180


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
    phi0 = wrap_to_180(circular_mean_deg(dihedrals))

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
):
    r"""Fit Gromacs type-9 dihedral terms from a distribution.

    We estimate a PMF from the histogram and fit Fourier terms of the form:

        U(\phi) = \sum_n k_n \left(1 + \cos(n\phi - \phi_n)\right)

    Since
        k\cos(n\phi-\phi_n) = (k\cos\phi_n)\cos(n\phi) + (k\sin\phi_n)\sin(n\phi),
    we can fit this robustly via (weighted) linear least squares in the basis
    \{1, \cos(n\phi), \sin(n\phi)\}.


    Returns
    -------
    list[tuple[int, float, float]]
        List of (multiplicity n, k_n, phi_n_deg) terms.
        The phase angle is in degrees (wrapped to [-180, 180]).
    """
    if max_n <= 0:
        return []

    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    values = np.asarray(dihedrals, dtype=float)
    shift = circular_mean_deg(values)
    values = ((values - shift + 180.0) % 360.0) - 180.0

    # hist, edges = np.histogram(values, bins=bins, range=(-180.0, 180.0), density=True)
    hist, edges = np.histogram(values, bins=bins, range=(np.min(values), np.max(values)), density=True)
    hist = np.clip(hist, min_prob, None)

    phi_centers = 0.5 * (edges[:-1] + edges[1:])
    phi_rad = np.deg2rad(phi_centers)

    def _gmm_pdf_1d(x, weights, means, variances):
        x = x[:, None]
        norm = np.sqrt(2.0 * np.pi * variances)[None, :]
        exps = np.exp(-0.5 * (x - means) ** 2 / variances)
        return np.sum(weights * exps / norm, axis=1)

    def _fit_gmm_1d(data, n_components, max_iter=200, tol=1e-6, var_floor=1e-4):
        n_samples = data.size
        if n_samples < n_components:
            return None

        percentiles = np.linspace(0.0, 100.0, n_components + 2)[1:-1]
        means = np.percentile(data, percentiles)
        variances = np.full(n_components, np.var(data) + var_floor)
        weights = np.full(n_components, 1.0 / n_components)

        prev_ll = None
        for _ in range(max_iter):
            pdf = _gmm_pdf_1d(data, weights, means, variances)
            pdf = np.clip(pdf, 1e-12, None)
            resp = (weights * np.exp(-0.5 * (data[:, None] - means) ** 2 / variances)
                    / np.sqrt(2.0 * np.pi * variances))
            resp = resp / np.clip(resp.sum(axis=1, keepdims=True), 1e-12, None)

            Nk = resp.sum(axis=0)
            weights = Nk / n_samples
            means = (resp * data[:, None]).sum(axis=0) / np.clip(Nk, 1e-12, None)
            variances = (resp * (data[:, None] - means) ** 2).sum(axis=0) / np.clip(Nk, 1e-12, None)
            variances = np.clip(variances, var_floor, None)

            ll = np.sum(np.log(pdf))
            if prev_ll is not None and abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        p = 3 * n_components - 1
        bic = -2.0 * prev_ll + p * np.log(n_samples)
        return weights, means, variances, bic

    best_gmm = None
    best_bic = None
    for n_components in range(1, max_n + 1):
        fit = _fit_gmm_1d(values, n_components)
        if fit is None:
            continue
        weights, means, variances, bic = fit
        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_gmm = (weights, means, variances)

    if best_gmm is not None:
        gmm_density = _gmm_pdf_1d(phi_centers, *best_gmm)
        gmm_density = np.clip(gmm_density, min_prob, None)
        density = gmm_density
    else:
        density = hist

    pmf = -kT * np.log(density)
    pmf = pmf - float(np.min(pmf))

    # weights = np.sqrt(hist)
    weights = np.ones_like(hist)

    def _solve_weighted(A, y):
        Aw = A * weights[:, None]
        yw = y * weights
        coeffs, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
        return coeffs

    def _k_phi_from_ab(a, b, n: int):
        k = np.hypot(a, b)
        if k < 1e-12:
            return 0.0, 0.0
        phi = np.rad2deg(np.arctan2(b, a))
        # We fitted on shifted angles: phi' = phi - shift.
        # For k*cos(n*phi - phi_n), shifting phi by `shift` shifts the phase by n*shift.
        phi += n * shift
        phi = (360 - phi) % 360
        phi = wrap_to_180(phi)
        return k, phi

    terms = []
    cols = [np.ones_like(phi_rad)]
    for n in range(1, max_n + 1):
        cols.append(np.cos(n * phi_rad))
        cols.append(np.sin(n * phi_rad))
    A = np.column_stack(cols)
    coeffs = _solve_weighted(A, pmf)

    for n in range(1, max_n + 1):
        a = coeffs[1 + 2 * (n - 1)]
        b = coeffs[1 + 2 * (n - 1) + 1]
        k, phi = _k_phi_from_ab(a, b, n)
        terms.append((int(n), float(k), float(phi)))

    return terms