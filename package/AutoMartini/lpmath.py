"""Numerical utilities for trajectory processing and Boltzmann inversion.

This module provides helpers to:
- read atomistic/coarse-grained trajectories into bead coordinates,
- compute internal coordinates from trajectories,
- fit bonded terms (bonds, angles, dihedrals) from sampled distributions.

Angles are expressed in degrees unless otherwise noted.
Coordinate arrays are expected to be in nm in fitting routines.
"""

import logging
import numpy as np
from . import ligpar_cy
from MDAnalysis import Universe
from sklearn.mixture import GaussianMixture
from typing import Dict, Tuple


logger = logging.getLogger(__name__)

################################################################################
### Helper functions ###
################################################################################

def circular_mean(angles):
    """Compute the circular mean of angular samples in degrees.

    Parameters
    ----------
    angles : array-like
        Angle values in degrees.

    Returns
    -------
    float
        Circular mean angle in degrees, in the range ``[-180, 180]``.
    """
    angles_rad = np.deg2rad(angles)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_mean, cos_mean)
    return np.rad2deg(mean_rad)


def wrap_to_180(angles):
    """Wrap angle values to the interval ``[-180, 180)``.

    Parameters
    ----------
    angles : array-like or float
        Angle values in degrees.

    Returns
    -------
    numpy.ndarray or float
        Wrapped angles with the same broadcasted shape as the input.
    """
    return (angles + 180) % 360 - 180


def flat_set(lst):
    """Flatten a nested list into a set of unique values.

    Parameters
    ----------
    lst : list[list[Any]]
        Nested iterable where each inner list contributes elements.

    Returns
    -------
    set
        Unique elements across all inner lists.
    """
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
    """Compute bonded internal-coordinate time series from a CG trajectory.

    Parameters
    ----------
    cg_trajectory : numpy.ndarray
        Trajectory array of shape ``(n_frames, n_beads, 3)`` in nm.
    topo : object
        Topology-like object exposing ``bonds``, ``constraints``, ``angles``,
        and ``dihedrals`` iterables with bead indices.

    Returns
    -------
    dict
        Mapping from tuple keys to 1D numpy arrays of sampled values:
        - ``(i, j, "bond")``
        - ``(i, j, "constraint")``
        - ``(i, j, k, "angle")``
        - ``(i, j, k, l, "dihedral")``
        - optional ``(i, j, k, "adj_angle")`` for dihedral-adjacent angles.
    """
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
    """Evaluate a 1D Gaussian-mixture probability density function.

    Parameters
    ----------
    x : numpy.ndarray
        Evaluation points, shape ``(n_samples,)``.
    weights : numpy.ndarray
        Mixture weights, shape ``(n_components,)``.
    means : numpy.ndarray
        Component means, shape ``(n_components,)``.
    variances : numpy.ndarray
        Component variances, shape ``(n_components,)``.

    Returns
    -------
    numpy.ndarray
        Density values at ``x``, shape ``(n_samples,)``.
    """
    x = x[:, None]
    norm = np.sqrt(2.0 * np.pi * variances)[None, :]
    exps = np.exp(-0.5 * (x - means) ** 2 / variances)
    return np.sum(weights * exps / norm, axis=1)


def fit_gmm_1d_best_old(data, max_components=1, max_iter=100, tol=1e-4, var_floor=1e-6, 
                    min_weight=0.1, min_spacing_std=2.0, min_prob=1e-3):
    """Fit a 1D Gaussian mixture and choose model order by AIC.

    Candidate models from 1 to ``max_components`` are fit via EM.
    Models are rejected when any component has too-low weight or when
    component means are too close (in standard-deviation units).

    Parameters
    ----------
    data : array-like
        Input samples.
    max_components : int, optional
        Maximum number of mixture components to test.
    max_iter : int, optional
        Maximum EM iterations per candidate.
    tol : float, optional
        Absolute log-likelihood convergence threshold.
    var_floor : float, optional
        Minimum variance used to stabilize EM updates.
    min_weight : float, optional
        Minimum allowed component weight; otherwise candidate is discarded.
    min_spacing_std : float, optional
        Minimum separation between component means measured in average
        component standard deviations.
    min_prob : float, optional
        Lower bound used when clipping densities in EM computations.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] or None
        Best ``(weights, means, variances)`` by AIC among non-rejected models,
        or ``None`` when fitting is not possible.
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
            resp = resp / np.clip(resp.sum(axis=1, keepdims=True), min_prob, None)

            Nk = resp.sum(axis=0)
            weights = Nk / data.size
            means = (resp * data[:, None]).sum(axis=0) / np.clip(Nk, min_prob, None)
            variances = (resp * (data[:, None] - means) ** 2).sum(axis=0) / np.clip(Nk, min_prob, None)
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


def fit_gmm_1d_best(
    data,
    max_components=6,
    max_iter=100,
    tol=1e-4,
    var_floor=1e-6,
    min_weight=0.1,
    min_spacing_std=2.0,
    min_prob=1e-3,
):
    """Fit a 1D Gaussian mixture and choose model order by AIC.

    Uses ``sklearn.mixture.GaussianMixture`` and preserves the historical
    return shape and filtering heuristics used in LigPar.

    Parameters
    ----------
    data : array-like
        Input samples.
    max_components : int, optional
        Maximum number of mixture components to test.
    max_iter : int, optional
        Maximum EM iterations per candidate.
    tol : float, optional
        EM convergence threshold for sklearn.
    var_floor : float, optional
        Minimum variance floor (mapped to sklearn ``reg_covar``).
    min_weight : float, optional
        Minimum allowed component weight; otherwise candidate is discarded.
    min_spacing_std : float, optional
        Minimum separation between component means measured in average
        component standard deviations.
    min_prob : float, optional
        Kept for API compatibility (unused in sklearn path).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] or None
        Best ``(weights, means, variances)`` by AIC among non-rejected models,
        or ``None`` when fitting is not possible.
    """
    data = np.asarray(data, dtype=float)
    if data.size < 2:
        return None
    max_components = int(max(1, int(max_components)))

    best = None
    best_aic = None

    X = data.reshape(-1, 1)
    for n_components in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                max_iter=int(max_iter),
                tol=float(tol),
                reg_covar=float(var_floor),
                n_init=5,
                random_state=0,
            )
            gmm.fit(X)
        except Exception:
            continue

        weights = np.asarray(gmm.weights_, dtype=float)
        means = np.asarray(gmm.means_.reshape(-1), dtype=float)
        variances = np.asarray(gmm.covariances_.reshape(-1), dtype=float)
        variances = np.clip(variances, var_floor, None)

        order = np.argsort(means)
        weights = weights[order]
        means = means[order]
        variances = variances[order]

        if np.any(weights < min_weight):
            continue

        if n_components > 1:
            too_close = False
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    std_i = np.sqrt(variances[i])
                    std_j = np.sqrt(variances[j])
                    avg_std = 0.5 * (std_i + std_j)
                    spacing = abs(means[j] - means[i]) / max(avg_std, 1e-12)
                    if spacing < min_spacing_std:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue

        aic = float(gmm.aic(X))
        if best_aic is None or aic < best_aic:
            best_aic = aic
            best = (weights, means, variances)

    return best


def boltzmann_inversion_bond(distances, temperature=300.0, fc_scale=1.0, max_components=1):
    """Estimate harmonic bond parameters from sampled bond lengths.

    Uses a mean/variance harmonic approximation:
    ``r0 = mean(r)`` and ``k = fc_scale * kT / var(r)``.

    Parameters
    ----------
    distances : array-like
        Bond-length samples (typically in nm).
    temperature : float, optional
        Temperature in K.
    fc_scale : float, optional
        Multiplicative scaling factor applied to the fitted force constant.
    max_components : int, optional
        Maximum number of components for an auxiliary GMM estimate.

    Returns
    -------
    tuple[float, float, numpy.ndarray or None]
        ``(r0, k, density)``, where ``density`` is the clipped GMM density on
        an internal support grid, or ``None`` if GMM fitting fails.
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


def fit_type10_angle(
    angles,
    temperature=300.0,
    fc_scale=1.0,
    bins=180,
    min_prob=1e-3,
):
    """Fit GROMACS angle funct=10 (restricted bending) from sampled angles.

    Functional form:
        U(theta) = 0.5 * k * (cos(theta) - cos(theta0))^2 / sin(theta)^2

    For 0-180 degree angles, p(theta) includes the geometric Jacobian sin(theta):
        p(theta) ~ sin(theta) * exp(-U(theta)/kT)
    We therefore estimate PMF from p(theta)/sin(theta).

    Returns
    -------
    tuple
        theta0_deg, k_kjmol, fit_density
        where fit_density is the Jacobian-corrected density on bin centers.
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * float(temperature)

    angles = np.asarray(angles, dtype=float)
    angles = np.clip(angles, 0.0, 180.0)

    n_bins = int(max(24, bins))
    # theta_min, theta_max = angles.min(), angles.max()
    theta_min, theta_max = 0.0, 180.0
    theta_edges = np.linspace(theta_min, theta_max, n_bins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_rad = np.deg2rad(theta_centers)

    raw_density = np.histogram(angles, bins=theta_edges, density=True)[0]
    raw_density = np.clip(raw_density, min_prob, None)

    sin_theta = np.sin(theta_rad)
    sin_theta = np.clip(sin_theta, 1e-6, None)
    fit_density = raw_density / sin_theta
    fit_density = np.clip(fit_density, min_prob, None)
    pmf = -kT * np.log(fit_density)

    cos_theta = np.cos(theta_rad)
    sin2_theta = np.clip(sin_theta * sin_theta, 1e-6, None)
    w = np.power(raw_density, 0.30)

    theta0_grid = np.linspace(theta_min, theta_max, max(181, n_bins))
    best = None

    for theta0_candidate in theta0_grid:
        c0 = float(np.cos(np.deg2rad(theta0_candidate)))
        basis = 0.5 * (cos_theta - c0) ** 2 / sin2_theta

        A = np.column_stack([basis, np.ones_like(basis)])
        Aw = A * w[:, None]
        bw = pmf * w
        sol, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
        k_candidate = float(sol[0])

        if not np.isfinite(k_candidate) or k_candidate <= 0.0:
            continue

        resid = Aw @ sol - bw
        score = float(np.mean(resid**2))

        if best is None or score < best[0]:
            best = (score, float(theta0_candidate), k_candidate)

    if best is None:
        theta0 = float(np.mean(angles))
        residual_rad = np.deg2rad(angles - theta0)
        variance_rad = float(np.var(residual_rad))
        k_param = float(fc_scale * kT / max(variance_rad, 1e-12))
        return theta0, k_param, raw_density

    _, theta0, k_param = best
    k_param = float(fc_scale * k_param)
    return float(theta0), float(k_param), raw_density


def fit_type1_angle(angles, temperature=300.0, fc_scale=1.0, bins=180, min_prob=1e-6):
    """Fit GROMACS angle funct=1 parameters from sampled angles.

    Type-1 form:
        U(theta) = 0.5 * k * (theta - theta0)^2

    Returns
    -------
    tuple
        (theta0_deg, k_kjmol), fit_density
        where fit_density is the histogram density on bin centers.
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * float(temperature)

    values = np.asarray(angles, dtype=float)
    values = np.clip(values, 0.0, 180.0)

    theta0 = float(np.mean(values))
    residual_rad = np.deg2rad(values - theta0)
    variance_rad = float(np.var(residual_rad))
    k_param = float(fc_scale * kT / max(variance_rad, 1e-12))

    n_bins = int(max(24, bins))
    theta_edges = np.linspace(0.0, 180.0, n_bins + 1)
    raw_density = np.histogram(values, bins=theta_edges, density=True)[0]
    raw_density = np.clip(raw_density, min_prob, None)

    return (theta0, k_param), raw_density


def fit_type2_angle(angles, temperature=300.0, fc_scale=1.0, bins=180, min_prob=1e-6):
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
    theta_min, theta_max = angles.min(), angles.max()
    # theta_min, theta_max = 0.0, 180.0
    theta_edges = np.linspace(theta_min, theta_max, n_bins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_rad = np.deg2rad(theta_centers)

    raw_density = np.histogram(angles, bins=theta_edges, density=True)[0]
    raw_density = np.clip(raw_density, min_prob, None)

    jac = np.sin(theta_rad)
    jac = np.clip(jac, 1e-6, None)
    jac = np.power(jac, 0.0)
    fit_density = raw_density / jac
    fit_density = np.clip(fit_density, min_prob, None)
    pmf = -kT * np.log(fit_density)

    c = np.cos(theta_rad)
    A = np.column_stack([c * c, c, np.ones_like(c)])
    w = np.power(raw_density, 1.0)
    Aw = A * w[:, None]
    bw = pmf * w

    coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    alpha, beta, _ = [float(x) for x in coeffs]

    # U(theta) = alpha*c^2 + beta*c + const = 0.5*k*(c-c0)^2 + const
    k_param = max(1e-8, 2.0 * alpha)
    cos_theta0 = np.clip(-beta / k_param, -1.0, 1.0)
    theta0 = float(np.rad2deg(np.arccos(cos_theta0)))
    k_param = float(fc_scale * k_param)

    return (theta0, k_param), raw_density


def boltzmann_inversion_improper(dihedrals, temperature=300.0, fc_scale=1.0):
    """Estimate harmonic improper-dihedral parameters from sampled angles.

    Parameters
    ----------
    dihedrals : array-like
        Improper angle samples in degrees.
    temperature : float, optional
        Temperature in K.
    fc_scale : float, optional
        Multiplicative scaling factor applied to the fitted force constant.

    Returns
    -------
    tuple[float, float]
        ``(phi0, k)`` where ``phi0`` is the wrapped circular mean in degrees
        and ``k = fc_scale * kT / var(residual_rad)`` in kJ/mol.
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

################################################################################
### DIHEDRALS ###
################################################################################   

def _eval_type9_potential(terms, phi_deg: np.ndarray) -> np.ndarray:
    """Evaluate the sum of type-9 dihedral energies on a phi grid (degrees).

    GROMACS funct=9: U(phi) = sum_n  k_n * (1 + cos(n*phi - phi0_n))
    term layout: [i, j, k, l, funct=9, phi0_deg, k, mult, ...]
    """
    phi_rad = np.deg2rad(phi_deg)
    U = np.zeros_like(phi_deg, dtype=float)
    for term in terms:
        phi0_rad = np.deg2rad(float(term[5]))
        k_term   = float(term[6])
        mult     = int(term[7])
        U += k_term * (1.0 + np.cos(mult * phi_rad - phi0_rad))
    return U


def _eval_type11_potential(term, phi_deg: np.ndarray) -> np.ndarray:
    """Evaluate a type-11 (CBT) dihedral potential on a phi grid (degrees).

    GROMACS funct=11:  U(phi) = k_phi * sum_{n=0}^{4} a_n * cos^n(phi)
    (sin^3*sin^3 prefactor assumed = 1, consistent with fitting in lpmath)
    term layout: [i, j, k, l, funct=11, k_phi, a0, a1, a2, a3, a4, ...]
    """
    cos_phi = np.cos(np.deg2rad(phi_deg))
    kphi = float(term[5])
    a    = [float(term[6 + n]) for n in range(5)]
    U = kphi * sum(a[n] * cos_phi ** n for n in range(5))
    return U

# ---------------------------------------------------------------------------
# Fitting helpers: fit a functional form to a target potential on a grid
# ---------------------------------------------------------------------------

def _fit_type9_to_target(
    pmf: np.ndarray,
    shift: float,
    harmonics: list,
    weights: np.ndarray = None,
    phi_grid: np.ndarray = None,
    nbins: int = 360,
) -> list:
    """Fit GROMACS type-9 Fourier terms to a target potential on a phi grid.

    U(phi) = const + sum_n [ a_n*cos(n*phi) + b_n*sin(n*phi) ]
    converted to (mult, k, phi0) via  k = hypot(a, b),  phi0 = atan2(b, a).

    Parameters
    ----------
    pmf : numpy.ndarray
        Target potential values evaluated on ``phi_grid``.
    shift : float
        Angular shift (degrees) used during fitting; phases are shifted back
        before returning type-9 parameters.
    harmonics : list[int]
        Harmonic multiplicities to include.
    weights : numpy.ndarray, optional
        Per-grid-point weights for weighted least squares.
    phi_grid : numpy.ndarray, optional
        Grid of dihedral angles in degrees.
    nbins : int, optional
        Number of bins if ``phi_grid`` is not provided.

    Returns
    -------
    list[tuple[int, float, float]]
        List of ``(multiplicity, k, phi0_deg)`` terms.
    """

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

    if phi_grid is None:
        phi_grid = np.linspace(-180.0, 180.0, nbins + 1)
    phi_rad = np.deg2rad(phi_grid)

    cols = [np.ones_like(phi_rad)]
    for n in harmonics:
        cols.append(np.cos(n * phi_rad))
        cols.append(np.sin(n * phi_rad))

    w = weights if weights is not None else np.ones_like(phi_rad)
    A = np.column_stack(cols)
    Aw = A * w[:, None]
    bw = pmf * w
    coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    resid = Aw @ coeffs - bw

    # Extract and output the fitted terms
    terms = []
    for idx, n in enumerate(harmonics):
        a = coeffs[1 + 2 * idx]
        b = coeffs[1 + 2 * idx + 1]
        k, phi = _k_phi_from_ab(a, b, n)
        terms.append((int(n), float(k), float(phi)))
    return terms


def _fit_type11_to_target(
    pmf: np.ndarray,
    weights: np.ndarray = None,
    phi_grid: np.ndarray = None,
    nbins: int = 360,
):
    """Fit a type-11 (CBT) polynomial potential on a dihedral grid.

    Parameters
    ----------
    pmf : numpy.ndarray
        Target potential values on ``phi_grid``.
    weights : numpy.ndarray, optional
        Per-grid-point weights for weighted least squares.
    phi_grid : numpy.ndarray, optional
        Grid of dihedral angles in degrees.
    nbins : int, optional
        Number of bins if ``phi_grid`` is not provided.

    Returns
    -------
    tuple[float, list[float]]
        ``(k_phi, [a0, a1, a2, a3, a4])`` with normalized polynomial
        coefficients such that ``k_phi * a_n`` recovers fitted amplitudes.
    """
    if phi_grid is None:
        phi_grid = np.linspace(-180.0, 180.0, nbins + 1)
    phi_rad = np.deg2rad(phi_grid)
    cos_phi = np.cos(np.deg2rad(phi_grid))
    cols = [cos_phi ** n for n in range(5)]
    A  = np.column_stack(cols)
    Aw = A * (weights[:, None] if weights is not None else 1.0)
    bw = pmf * (weights if weights is not None else 1.0)
    coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    coeffs5 = coeffs[:5]
    k_phi = float(np.linalg.norm(coeffs5))
    if k_phi > 0.0:
        a = [float(c / k_phi) for c in coeffs5]
    else:
        a = [0.0] * 5
    return k_phi, a


def fit_type9_dihedral(
    dihedrals,
    temperature=300.0,
    max_n=6,
    nbins=360,
    min_prob=1e-6,
    fc_scale: float = 1.0,
):
    r"""Fit GROMACS type-9 dihedral terms from sampled dihedral angles.

    1. Fit a GMM to the dihedral distribution (1 to max_n components, BIC selection).
    2. Determine optimal n from the spacing between modes: n = 180 / mean_spacing.
        3. Fit Fourier terms: always include n=1, plus the optimal harmonic n=optimal_n.
        4. Estimate free energy F(\phi) = -kT ln p(\phi) and fit via *density-weighted* least squares,
             focusing the fit on the support (high-probability region).

    Parameters
    ----------
    dihedrals : array-like
        Dihedral samples in degrees.
    temperature : float, optional
        Temperature in K.
    max_n : int, optional
        Maximum harmonic multiplicity considered.
    nbins : int, optional
        Number of bins/grid points for histogram and fitting.
    min_prob : float, optional
        Lower bound applied to histogram density before PMF conversion.
    fc_scale : float, optional
        Force-constant scaling factor (currently unused in this routine).

    Returns
    -------
    tuple[list[tuple[int, float, float]], numpy.ndarray]
        ``(terms, density)`` where ``terms`` are type-9 entries
        ``(n, k_n, phi_n_deg)`` and ``density`` is the clipped histogram on
        the fitting grid.
    """

    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    values = np.asarray(dihedrals, dtype=float)
    shift = float(circular_mean(values))
    values = wrap_to_180(values - shift)

    # Always fit/evaluate on a fixed dihedral grid.
    # endpoint=False avoids duplicating -180 and +180 (same angle).
    phi_centers = np.linspace(-180.0, 180.0, nbins, endpoint=False)
    phi_rad = np.deg2rad(phi_centers)

    # Fit GMM with BIC selection (using module-level function)
    best_gmm = fit_gmm_1d_best(values, max_components=3, min_prob=min_prob)
    best_means = best_gmm[1] if best_gmm is not None else None
    
    # Fit free energy from histogram density
    density = np.histogram(values, bins=np.linspace(-180.0, 180.0, nbins + 1), density=True)[0]
    density = np.clip(density, min_prob, None)
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

    # Weighted least squares with weights ~ sqrt(p) so high-probability regions dominate.
    # No masking/fallbacks: low-density regions naturally get near-zero weight.
    # w = np.sqrt(density)
    if len(harmonics_to_fit) == 1:
        w = np.pow(density, 1.0)
    else:
        w = np.pow(density, 0.2)

    terms = _fit_type9_to_target(pmf, shift, harmonics_to_fit, weights=w, phi_grid=phi_centers)

    return terms, density


def fit_type11_dihedral(
    dihedrals,
    temperature=300.0,
    nbins=360,
    min_prob=1e-6,
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
    tuple[tuple[float, list[float]], numpy.ndarray]
        ``((k_phi, [a0, a1, a2, a3, a4]), density)`` where ``density`` is the
        clipped histogram used to build the PMF.
    """
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * float(temperature)

    dihs = np.asarray(dihedrals, dtype=float)
    dihs = wrap_to_180(dihs)
    phi_edges = np.linspace(-180.0, 180.0, nbins + 1)
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    phi_rad = np.deg2rad(phi_centers)

    # Fit free energy from the distribution density
    density = np.histogram(dihs, bins=phi_edges, density=True)[0]
    density = np.clip(density, min_prob, None)
    pmf = -kT * np.log(density)
    w = np.pow(density, 0.20)
    result = _fit_type11_to_target(pmf, weights=w, phi_grid=phi_centers)
    return result, density