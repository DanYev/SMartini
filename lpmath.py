import logging
from typing import Dict, Tuple

import numpy as np
from MDAnalysis import Universe

import ligpar_cy

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
    cg_trajectory = np.ascontiguousarray(cg_trajectory, dtype=np.float64)

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
    cg_trajectory = np.ascontiguousarray(cg_trajectory, dtype=np.float64)
    
    logger.info("Loaded CG trajectory: %s frames, %s beads", n_frames, n_beads)
    return cg_trajectory


def calculate_internal_coordinates(cg_trajectory, topo):
    """Calculate internal coordinates (bonds, angles, dihedrals) from CG trajectory."""
    n_frames = cg_trajectory.shape[0]
    internal_coords = {}

    for bond in topo.bonds:
        i, j = int(bond[0]), int(bond[1])
        internal_coords[(i, j, "bond")] = ligpar_cy.bond_series(cg_trajectory, i, j)

    for constraint in topo.constraints:
        i, j = int(constraint[0]), int(constraint[1])
        internal_coords[(i, j, "constraint")] = ligpar_cy.bond_series(cg_trajectory, i, j)

    for angle in topo.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        internal_coords[(i, j, k, "angle")] = ligpar_cy.angle_series(cg_trajectory, i, j, k)

    for dihedral in topo.dihedrals:
        i, j, k, l = int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3])
        internal_coords[(i, j, k, l, "dihedral")] = ligpar_cy.dihedral_series(
            cg_trajectory, i, j, k, l
        )

        # Auxiliary adjacent angles for dihedral linearity checks.
        # These are geometric properties of the trajectory and should be available
        # even if the topology does not include an explicit [angles] entry.
        # Use a distinct key tag to avoid changing angle plotting/fitting behavior.
        if (i, j, k, "angle") not in internal_coords and (i, j, k, "adj_angle") not in internal_coords:
            internal_coords[(i, j, k, "adj_angle")] = ligpar_cy.angle_series(cg_trajectory, i, j, k)
        if (j, k, l, "angle") not in internal_coords and (j, k, l, "adj_angle") not in internal_coords:
            internal_coords[(j, k, l, "adj_angle")] = ligpar_cy.angle_series(cg_trajectory, j, k, l)

    return internal_coords


def boltzmann_inversion_bond(distances, temperature=300.0, fc_scale=1.0):
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
    k = float(fc_scale * kT / variance)

    return r0, k


def boltzmann_inversion_angle(angles, temperature=300.0, fc_scale=1.0):
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
    k = float(fc_scale * kT / variance_rad)

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


def fit_gmm_1d_best(data, max_components=1, max_iter=100, tol=1e-4, var_floor=1e-12, 
                    min_weight=0.05, min_spacing_std=2.0, min_prob=1e-3):
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
            pdf = np.clip(pdf, min_prob, None)
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


def boltzmann_inversion_improper(dihedrals, temperature=300.0, fc_scale=1.0):
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
    k = float(fc_scale * kT / variance_rad)

    return phi0, k


def fit_type9_dihedral(
    dihedrals,
    temperature=300.0,
    max_n=6,
    bins=360,
    min_prob=1e-2,
    return_score: bool = False,
    fc_scale: float = 1.0,
):
    r"""Fit Gromacs type-9 dihedral terms from a Gaussian mixture model.

    1. Fit a GMM to the dihedral distribution (1 to max_n components, BIC selection).
    2. Determine optimal n from the spacing between modes: n = 180 / mean_spacing.
        3. Fit Fourier terms: always include n=1, plus the optimal harmonic n=optimal_n.
        4. Estimate free energy F(\phi) = -kT ln p(\phi) and fit via *density-weighted* least squares,
             focusing the fit on the support (high-probability region).

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
    if values.size < 2:
        raise ValueError("Type-9 dihedral fit failed: not enough samples")

    shift = float(circular_mean(values))
    values = wrap_to_180(values - shift)

    # Always fit/evaluate on a fixed dihedral grid.
    # endpoint=False avoids duplicating -180 and +180 (same angle).
    phi_centers = np.linspace(-180.0, 180.0, int(max(24, bins)), endpoint=False)
    phi_rad = np.deg2rad(phi_centers)

    # Fit GMM with BIC selection (using module-level function)
    best_gmm = fit_gmm_1d_best(values, max_components=3)
    best_means = best_gmm[1] if best_gmm is not None else None

    if best_gmm is None:
        raise ValueError("Type-9 dihedral fit failed: Gaussian mixture could not be fit.")

    # Determine optimal n from the spacing between modes (circularly).
    # For two modes separated by ~180 deg, this yields n≈2; for ~120 deg, n≈3, etc.
    if best_means is None or len(best_means) <= 1:
        optimal_n = 1
    else:
        means = np.sort(np.asarray(best_means, dtype=float))
        if means.size == 2:
            d = float(abs(means[1] - means[0]))
            spacing = float(min(d, 360.0 - d))
        else:
            diffs = np.diff(means)
            wrap_diff = (means[0] + 360.0) - means[-1]
            spacings = np.concatenate([diffs, [wrap_diff]])
            spacing = float(np.median(spacings)) if spacings.size else 360.0

        optimal_n = int(np.clip(int(np.floor(360.0 / spacing)), 1, int(max_n)))

    # Fit free energy from GMM density
    gmm_density = gmm_pdf_1d(phi_centers, *best_gmm)
    # density = gmm_density
    density = np.clip(gmm_density, min_prob, None)
    pmf = -kT * np.log(density)

    dmax = float(np.max(density))
    if not np.isfinite(dmax) or dmax <= 0.0:
        raise ValueError("Type-9 dihedral fit failed: invalid density")

    optimal_n = int(max(1, min(int(optimal_n), int(max_n))))
    harmonics_to_fit = [] 
    for n in range(1, optimal_n + 1):
        harmonics_to_fit.append(n)

    cols = [np.ones_like(phi_rad)]
    for n in harmonics_to_fit:
        cols.append(np.cos(n * phi_rad))
        cols.append(np.sin(n * phi_rad))

    # Weighted least squares with weights ~ sqrt(p) so high-probability regions dominate.
    # No masking/fallbacks: low-density regions naturally get near-zero weight.
    # w = np.sqrt(density)
    if len(harmonics_to_fit) == 1:
        w = np.pow(density, 1.0)
    else:
        w = np.pow(density, 0.20)

    A = np.column_stack(cols)
    Aw = A * w[:, None]
    bw = pmf * w
    coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    resid = Aw @ coeffs - bw
    score = float(np.mean(resid**2))

    def _k_phi_from_ab(a, b, n: int):
        k = np.hypot(a, b)
        if k < 1e-12:
            return 0.0, 0.0
        phi = np.rad2deg(np.arctan2(b, a))
        # We fit in the centered coordinate x = wrap(phi - shift).
        # Convert phase back to the absolute coordinate expected by type-9: cos(n*phi - phi0).
        phi += n * shift
        phi = wrap_to_180(phi)
        return k, phi

    # Extract and output the fitted terms
    terms = []
    for idx, n in enumerate(harmonics_to_fit):
        a = coeffs[1 + 2 * idx]
        b = coeffs[1 + 2 * idx + 1]
        k, phi = _k_phi_from_ab(a, b, n)
        if not (n == 1 and len(harmonics_to_fit) > 1):
            # if not n == len(harmonics_to_fit):
            k *= fc_scale
        terms.append((int(n), float(k), float(phi)))

    if return_score:
        return terms, score
    return terms


def fit_type11_cbt_dihedral(
    dihedrals,
    temperature=300.0,
    bins=360,
    min_prob=1e-3,
    cos_power_max: int = 4,
    return_score: bool = False,
    fc_scale: float = 1.0,
):
    r"""Fit GROMACS dihedral funct=11 (combined bending-torsion, CBT).

    The CBT functional form (GROMACS manual Eq. 204) is:

    $$
    V(\theta_1, \theta_2, \phi) = k_\phi \sin^3\theta_1\,\sin^3\theta_2\,\sum_{n=0}^{4} a_n \cos^n\phi
    $$

    Parameters in the topology are: k_phi (kJ/mol), then a0..a4 (dimensionless).

    In principle, CBT depends on \phi and the adjacent bending angles \theta_1 and \theta_2.
    In LigPar we use CBT specifically as a numerically stable replacement for ill-defined
    torsions; for fitting we assume \theta_1=\theta_2=90° so the prefactor

        \sin^3\theta_1\sin^3\theta_2 = 1.

    Under this approximation we fit a 1D PMF in \phi:

        PMF(\phi) \approx \sum_n c_n \cos^n\phi,

    then rescale c_n into (k_phi, a_n) with c_n = k_phi * a_n.

    Returns
    -------
    tuple[float, list[float]]
        (k_phi, [a0, a1, a2, a3, a4])
    """
    if cos_power_max != 4:
        raise ValueError("Only cos_power_max=4 is supported for funct=11")

    kB = 0.008314462618  # kJ/mol/K
    kT = kB * float(temperature)

    phi = np.asarray(dihedrals, dtype=float)
    if phi.size < 2:
        raise ValueError("Type-11 CBT fit failed: not enough samples")

    phi = wrap_to_180(phi)
    phi_centers = np.linspace(-180.0, 180.0, int(max(24, bins)), endpoint=False)
    phi_rad = np.deg2rad(phi_centers)

    best_gmm = fit_gmm_1d_best(phi, max_components=3)
    if best_gmm is None:
        raise ValueError("Type-11 CBT fit failed: Gaussian mixture could not be fit.")

    density = np.clip(gmm_pdf_1d(phi_centers, *best_gmm), float(min_prob), None)
    pmf = -kT * np.log(density)

    # Build weighted least squares system: PMF(phi) ~ sum_n c_n * cos^n(phi)
    cols = []
    cos_phi = np.cos(phi_rad)
    cos_pow = np.ones_like(cos_phi)
    for n in range(0, 5):
        if n == 0:
            cos_pow = np.ones_like(cos_phi)
        elif n == 1:
            cos_pow = cos_phi
        else:
            cos_pow = cos_pow * cos_phi
        cols.append(cos_pow)

    A = np.column_stack(cols)  # shape (nbins, 5)
    # Weights: emphasize well-sampled/high-probability regions
    w = np.power(density, 0.20)
    Aw = A * w[:, None]
    bw = pmf * w
    coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    coeffs = np.asarray(coeffs, dtype=float)

    resid = Aw @ coeffs - bw
    score = float(np.mean(resid**2))

    scale = float(np.max(np.abs(coeffs)))
    if not np.isfinite(scale) or scale < 1e-12:
        if return_score:
            return (0.0, [0.0, 0.0, 0.0, 0.0, 0.0]), score
        return 0.0, [0.0, 0.0, 0.0, 0.0, 0.0]

    k_phi = scale
    a = (coeffs / scale).tolist()
    # Ensure length exactly 5
    a = [float(x) for x in a[:5]]
    result = (float(k_phi), a)
    if return_score:
        return result, score
    return result