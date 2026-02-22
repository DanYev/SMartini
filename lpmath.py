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
            b1 = pos_i - pos_j
            b2 = pos_j - pos_k
            b3 = pos_k - pos_l
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


def circular_mean(angles):
    """Calculate circular mean of angles in degrees."""
    angles_rad = np.deg2rad(angles)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_mean, cos_mean)
    return np.rad2deg(mean_rad)


def wrap_to_180(angles):
    """Wrap angles to [-180, 180] range."""
    return (angles + 180) % 360 - 180


def gmm_pdf_1d(x, weights, means, variances):
    """Compute 1D Gaussian mixture PDF."""
    x = x[:, None]
    norm = np.sqrt(2.0 * np.pi * variances)[None, :]
    exps = np.exp(-0.5 * (x - means) ** 2 / variances)
    return np.sum(weights * exps / norm, axis=1)


def fit_gmm_1d_best(data, max_components=3, max_iter=200, tol=1e-6, var_floor=1e-4, 
                    min_weight=0.1, min_spacing_std=6.0):
    """Fit 1D Gaussian mixture with AIC selection + penalties for low weights and overlap.
    
    Parameters
    ----------
    min_weight : float
        Penalty weight for components with weight < this threshold.
    min_spacing_std : float
        Penalty if two components' means are < this many std devs apart.
    
    Returns
    -------
    tuple or None
        (weights, means, variances) if successful, None otherwise.
    """
    data = np.asarray(data, dtype=float)
    if data.size < 2:
        return None

    max_components = int(max(1, min(max_components, data.size)))
    best = None
    best_aic_penalized = None

    for n_components in range(1, max_components + 1):
        percentiles = np.linspace(0.0, 100.0, n_components + 2)[1:-1]
        means = np.percentile(data, percentiles)
        variances = np.full(n_components, np.var(data) + var_floor)
        weights = np.full(n_components, 1.0 / n_components)

        prev_ll = None
        for _ in range(max_iter):
            pdf = gmm_pdf_1d(data, weights, means, variances)
            pdf = np.clip(pdf, 1e-12, None)
            resp = (weights * np.exp(-0.5 * (data[:, None] - means) ** 2 / variances)
                    / np.sqrt(2.0 * np.pi * variances))
            resp = resp / np.clip(resp.sum(axis=1, keepdims=True), 1e-12, None)

            Nk = resp.sum(axis=0)
            weights = Nk / data.size
            means = (resp * data[:, None]).sum(axis=0) / np.clip(Nk, 1e-12, None)
            variances = (resp * (data[:, None] - means) ** 2).sum(axis=0) / np.clip(Nk, 1e-12, None)
            variances = np.clip(variances, var_floor, None)

            ll = np.sum(np.log(pdf))
            if prev_ll is not None and abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        # Hard cutoff: reject if any component has weight < min_weight
        if np.any(weights < min_weight):
            continue
        
        # Hard cutoff: reject if any two components are too close
        if n_components > 1:
            too_close = False
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    # Distance between means in units of std dev
                    std_i = np.sqrt(variances[i])
                    std_j = np.sqrt(variances[j])
                    avg_std = 0.5 * (std_i + std_j)
                    spacing = abs(means[j] - means[i]) / avg_std
                    if spacing < min_spacing_std:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue
        
        # AIC
        p = 3 * n_components - 1
        aic = 2 * p - 2 * ll
        aic_penalized = aic
        
        if best_aic_penalized is None or aic_penalized < best_aic_penalized:
            best_aic_penalized = aic_penalized
            best = (weights, means, variances)

    return best


def boltzmann_inversion_improper(dihedrals, temperature=300.0):
    """Estimate harmonic improper dihedral parameters from samples.

    Mean-based harmonic approximation for periodic angles:
    - Equilibrium value phi0 is the circular mean (degrees, wrapped to [-180, 180]).
    - Force constant k is computed from wrapped residual fluctuations in radians:
      k = kT / var(phi_rad).
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    dihedrals = np.asarray(dihedrals, dtype=float)
    phi0 = wrap_to_180(circular_mean(dihedrals))

    residual_deg = wrap_to_180(dihedrals - phi0)
    residual_rad = np.deg2rad(residual_deg)
    variance_rad = float(np.var(residual_rad))
    k = float(kT / variance_rad)

    return phi0, k


def fit_type9_dihedral(
    dihedrals,
    temperature=300.0,
    max_n=6,
    bins=100,
    min_prob=1e-6,
):
    r"""Fit Gromacs type-9 dihedral terms from a Gaussian mixture model.

    1. Fit a GMM to the dihedral distribution (1 to max_n components, BIC selection).
    2. Determine optimal n from the spacing between modes: n = 180 / mean_spacing.
    3. Fit Fourier terms: always include n=1 (to stabilize), plus harmonics up to optimal n.
    4. Estimate PMF and fit via weighted least squares.

    Returns
    -------
    list[tuple[int, float, float]]
        List of (multiplicity n, k_n, phi_n_deg) terms.
    """
    if max_n <= 0:
        return []

    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    values = np.asarray(dihedrals, dtype=float)
    shift = circular_mean(values)
    values = wrap_to_180(values - shift) 

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    # data_min = float(-180)
    # data_max = float(180)
    lim  = max(abs(data_min), abs(data_max))

    phi_centers = np.linspace(-lim, lim, int(bins))
    phi_rad = np.deg2rad(phi_centers)

    # Fit GMM with BIC selection (using module-level function)
    best_gmm = fit_gmm_1d_best(values, max_components=int(max_n))
    best_means = best_gmm[1] if best_gmm is not None else None

    if best_gmm is None:
        raise ValueError("Type-9 dihedral fit failed: Gaussian mixture could not be fit.")

    # Determine optimal n from mode spacing
    if best_means is not None and len(best_means) > 1:
        sorted_means = np.sort(best_means)
        spacings = []
        for i in range(len(sorted_means)):
            for j in range(i + 1, len(sorted_means)):
                spacing = abs(sorted_means[j] - sorted_means[i])
                # Use smallest spacing (since angles wrap)
                if spacing > 180:
                    spacing = 360 - spacing
                spacings.append(spacing)
        if spacings:
            mean_spacing = np.mean(spacings)
            optimal_n = max(1, int(np.round(360.0 / mean_spacing)))
        else:
            optimal_n = 1
    else:
        optimal_n = 1

    # Fit PMF from GMM density
    gmm_density = gmm_pdf_1d(phi_centers, *best_gmm)
    density = gmm_density
    # density = np.clip(gmm_density, min_prob, None)
    pmf = -kT * np.log(density)

    # Solve for Fourier coefficients: fit only n=1 and n=optimal_n
    optimal_n = int(min(optimal_n, max_n))
    harmonics_to_fit = [1]
    if int(optimal_n) > 1:
        harmonics_to_fit.append(optimal_n)
    
    cols = [np.ones_like(phi_rad)]
    for n in harmonics_to_fit:
        cols.append(np.cos(n * phi_rad))
        cols.append(np.sin(n * phi_rad))

    A = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(A, pmf, rcond=None)

    def _k_phi_from_ab(a, b, n: int):
        k = np.hypot(a, b)
        if k < 1e-12:
            return 0.0, 0.0
        phi = np.rad2deg(np.arctan2(b, a))
        phi += n * shift
        phi = wrap_to_180(phi)
        test_val = (360 + n * shift - phi) % 360 - 180
        return k, phi

    # Extract and output the fitted terms
    terms = []
    for idx, n in enumerate(harmonics_to_fit):
        a = coeffs[1 + 2 * idx]
        b = coeffs[1 + 2 * idx + 1]
        k, phi = _k_phi_from_ab(a, b, n)
        terms.append((int(n), float(k), float(phi)))

    return terms