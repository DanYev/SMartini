import logging
from typing import Dict, Tuple

import numpy as np
from MDAnalysis import Universe

import ligpar_cy

logger = logging.getLogger(__name__)

################################################################################
### Helper functions ###
################################################################################

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


def flat_set(lst):
    """Flatten a list of lists into a set of unique elements."""
    if not lst:
        return set()
    aset = set(item for sublist in lst for item in sublist) 
    return aset

################################################################################
### Trajectory Reading ###
################################################################################

def read_cog_trajectory(in_pdb, in_xtc, mapping, start=0, stop=-2, step=1, selection="all"):
    """Read AA trajectory and calculate COG trajectory for CG beads.

    Parameters
    ----------
    in_pdb : str or Path
        Path to atomistic PDB file
    in_xtc : str or Path
        Path to atomistic XTC trajectory
    mapping : list[list[int]]
        A list of lists, where each inner list contains the 0-based atom indices
        belonging to a CG bead.
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

    if not mapping:
        raise ValueError("Input 'mapping' is empty. Cannot compute COG trajectory for 0 beads.")

    u = Universe(str(in_pdb), str(in_xtc))
    
    # Select atoms based on selection string
    atom_group = u.select_atoms(selection)
    logger.info("Selected %s atoms with selection '%s'", len(atom_group), selection)

    n_beads = len(mapping)
    
    # Read frames into list first to avoid zeros from frame mismatch
    frames = []
    for ts in u.trajectory[start:stop:step]:
        frame_beads = np.zeros((n_beads, 3))
        for bead_idx, atom_indices in enumerate(mapping):
            if atom_indices:
                # MDAnalysis atom indices are 0-based, which matches our mapping
                positions = atom_group[atom_indices].positions
                frame_beads[bead_idx] = positions.mean(axis=0) / 10.0
        frames.append(frame_beads)
    
    # Convert list to numpy array
    cg_trajectory = np.array(frames)
    n_frames = len(frames)
    cg_trajectory = np.ascontiguousarray(cg_trajectory, dtype=np.float64)

    logger.info("COG trajectory computed: %s frames, %s beads", n_frames, n_beads)
    return cg_trajectory


def read_cg_trajectory(in_pdb, in_xtc, start=0, stop=5000, step=1,selection="all"):
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
    for ts in u.trajectory[start:stop:step]:
        frames.append(atom_group.positions / 10.0)
    
    # Convert list to numpy array
    cg_trajectory = np.array(frames)
    n_frames = len(frames)
    cg_trajectory = np.ascontiguousarray(cg_trajectory, dtype=np.float64)
    
    logger.info("Loaded CG trajectory: %s frames, %s beads", n_frames, n_beads)
    return cg_trajectory

################################################################################
### Internal Coordinates ###
################################################################################

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

################################################################################
### Fitting ###
################################################################################

def gmm_pdf_1d(x, weights, means, variances):
    """Compute 1D Gaussian mixture PDF."""
    x = x[:, None]
    norm = np.sqrt(2.0 * np.pi * variances)[None, :]
    exps = np.exp(-0.5 * (x - means) ** 2 / variances)
    return np.sum(weights * exps / norm, axis=1)


def fit_gmm_1d_best(data, max_components=1, max_iter=100, tol=1e-4, var_floor=1e-6, 
                    min_weight=0.1, min_spacing_std=2.0, min_prob=1e-3):
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


def boltzmann_inversion_bond(distances, temperature=300.0, fc_scale=1.0, max_components=1):
    """Estimate harmonic bond parameters from samples.

    This is a *mean-based harmonic approximation* (not PMF-minimum based):
    - Equilibrium value r0 is the sample mean.
    - Force constant k is computed from fluctuations: k = kT / var(r).
    
    Returns:
        r0 (float), k (float), gmm (tuple or None)
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    distances = np.asarray(distances, dtype=float)
    r0 = float(np.mean(distances))

    variance = float(np.var(distances))
    k = float(fc_scale * kT / variance)

    gmm = fit_gmm_1d_best(distances, max_components=max_components)
    
    density = None
    if gmm is not None:
        min_prob = 1e-3
        x_centers = np.linspace(np.percentile(distances, 1), np.percentile(distances, 99), 100)
        gmm_density = gmm_pdf_1d(x_centers, *gmm)
        density = np.clip(gmm_density, min_prob, None)

    return r0, k, density


def boltzmann_inversion_angle(angles, temperature=300.0, fc_scale=1.0, max_components=1):
    """Estimate harmonic angle parameters from samples.

    Mean-based harmonic approximation:
    - Equilibrium value theta0 is the sample mean (degrees).
    - Force constant k is computed from fluctuations in radians:
      k = kT / var(theta_rad).

    Returns:
        theta0 (float), k (float), gmm (tuple or None)
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    angles = np.asarray(angles, dtype=float)
    theta0 = float(np.mean(angles))

    residual_rad = np.deg2rad(angles - theta0)
    variance_rad = float(np.var(residual_rad))
    k = float(fc_scale * kT / variance_rad)

    gmm = fit_gmm_1d_best(angles, max_components=max_components)
    
    density = None
    if gmm is not None:
        min_prob = 1e-3
        x_centers = np.linspace(np.percentile(angles, 1), np.percentile(angles, 99), 100)
        gmm_density = gmm_pdf_1d(x_centers, *gmm)
        density = np.clip(gmm_density, min_prob, None)

    return theta0, k, density


def fit_type2_angle(angles, temperature=300.0, fc_scale=1.0, bins=180, min_prob=1e-3):
    """Fit GROMACS angle type-2 parameters from sampled angles.

    Type-2 form:
        U(theta) = 0.5 * k * (cos(theta) - cos(theta0))^2

    For 0-180 degree angles, the geometric Jacobian contributes sin(theta):
        p(theta) ~ sin(theta) * exp(-U(theta)/kT)
    so we fit PMF from the Jacobian-corrected density p(theta) / sin(theta).

    Returns
    -------
    tuple
        (theta0_deg, k_kjmol), fit_density
        where fit_density is the Jacobian-corrected histogram density on bin centers.
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * float(temperature)

    angles = np.asarray(angles, dtype=float)
    angles = np.clip(angles, 0.0, 180.0)

    n_bins = int(max(24, bins))
    theta_edges = np.linspace(0.0, 180.0, n_bins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_rad = np.deg2rad(theta_centers)

    raw_density = np.histogram(angles, bins=theta_edges, density=True)[0]
    raw_density = np.clip(raw_density, min_prob, None)

    jac = np.sin(theta_rad)
    jac = np.clip(jac, 1e-6, None)
    fit_density = raw_density / jac
    fit_density = np.clip(fit_density, min_prob, None)
    pmf = -kT * np.log(fit_density)

    c = np.cos(theta_rad)
    A = np.column_stack([c * c, c, np.ones_like(c)])
    w = np.power(raw_density, 0.30)
    Aw = A * w[:, None]
    bw = pmf * w

    coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    alpha, beta, _ = [float(x) for x in coeffs]

    # U(theta) = alpha*c^2 + beta*c + const = 0.5*k*(c-c0)^2 + const
    k_param = max(1e-8, 2.0 * alpha)
    cos_theta0 = np.clip(-beta / k_param, -1.0, 1.0)
    theta0 = float(np.rad2deg(np.arccos(cos_theta0)))
    k_param = float(fc_scale * k_param)

    return (theta0, k_param), fit_density


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
    shift = float(circular_mean(values))
    values = wrap_to_180(values - shift)

    # Always fit/evaluate on a fixed dihedral grid.
    # endpoint=False avoids duplicating -180 and +180 (same angle).
    phi_centers = np.linspace(-180.0, 180.0, int(max(24, bins)), endpoint=False)
    phi_rad = np.deg2rad(phi_centers)

    # Fit GMM with BIC selection (using module-level function)
    best_gmm = fit_gmm_1d_best(values, max_components=3)
    best_means = best_gmm[1] if best_gmm is not None else None
    
    # Fit free energy from GMM density
    gmm_density = gmm_pdf_1d(phi_centers, *best_gmm)
    density = np.clip(gmm_density, min_prob, None)
    pmf = -kT * np.log(density)

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

    optimal_n = int(max(1, min(int(optimal_n), int(max_n))))
    harmonics_to_fit = [] 
    # optimal_n = max_n
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
        w = np.pow(density, 0.2)

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

    return terms, density


def fit_type11_dihedral(
    dihedrals,
    temperature=300.0,
    bins=180,
    min_prob=1e-2,
    cos_power_max: int = 4,
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

    dihs = np.asarray(dihedrals, dtype=float)
    dihs = wrap_to_180(dihs)
    n_bins = int(max(24, bins))
    phi_edges = np.linspace(-180.0, 180.0, n_bins + 1)
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    phi_rad = np.deg2rad(phi_centers)

    # Fit free energy from the distribution density
    density = np.histogram(dihs, bins=phi_edges, density=True)[0]
    density = np.clip(density, min_prob, None)
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

    w = np.pow(density, 0.30)
    A = np.column_stack(cols)
    Aw = A * w[:, None]
    bw = pmf * w
    coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    resid = Aw @ coeffs - bw

    scale = float(np.max(np.abs(coeffs)))
    k_phi = scale
    a = (coeffs / scale).tolist()
    # Ensure length exactly 5
    a = [float(x) for x in a[:5]]
    result = (float(k_phi), a)
    return result, density