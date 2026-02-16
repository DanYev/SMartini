import logging
from typing import Dict, Tuple

import numpy as np
from MDAnalysis import Universe

logger = logging.getLogger(__name__)


def read_cog_trajectory(in_pdb, in_xtc, partitioning, stop=5000):
    """Read AA trajectory and calculate COG trajectory for CG beads.

    Parameters
    ----------
    in_pdb : str or Path
        Path to atomistic PDB file
    in_xtc : str or Path
        Path to atomistic XTC trajectory
    partitioning : dict
        Mapping of atom indices to bead indices {atom_idx: bead_idx}
    stop : int
        Number of frames to read from the trajectory

    Returns
    -------
    numpy.ndarray
        CG trajectory array with shape (n_frames, n_beads, 3)
    """
    logger.info("Reading AA trajectory: %s, %s", in_pdb, in_xtc)

    u = Universe(str(in_pdb), str(in_xtc))
    n_frames = len(u.trajectory)

    n_beads = max(partitioning.values()) + 1
    bead_to_atoms = {i: [] for i in range(n_beads)}
    for atom_idx, bead_idx in partitioning.items():
        bead_to_atoms[bead_idx].append(atom_idx)

    cg_trajectory = np.zeros((n_frames, n_beads, 3))

    for frame_idx, _ in enumerate(u.trajectory):
        for bead_idx in range(n_beads):
            atom_indices = bead_to_atoms[bead_idx]
            if atom_indices:
                positions = u.atoms[atom_indices].positions
                cg_trajectory[frame_idx, bead_idx] = positions.mean(axis=0) / 10.0

    if stop and stop > 0:
        cg_trajectory = cg_trajectory[:stop, :n_beads, :]

    logger.info("COG trajectory computed: %s frames, %s beads", cg_trajectory.shape[0], n_beads)
    return cg_trajectory


def read_cg_trajectory(in_pdb, in_xtc, stop=5000):
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
    n_frames = len(u.trajectory)
    n_beads = len(u.atoms)
    cg_trajectory = np.zeros((n_frames, n_beads, 3))

    for frame_idx, _ in enumerate(u.trajectory):
        cg_trajectory[frame_idx] = u.atoms.positions / 10.0

    if stop and stop > 0:
        cg_trajectory = cg_trajectory[:stop, :n_beads, :]

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
    """Calculate spring constant for bonds using Boltzmann inversion."""
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    hist, bin_edges = np.histogram(distances, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    nonzero_mask = hist > 0
    hist = hist[nonzero_mask]
    bin_centers = bin_centers[nonzero_mask]

    pmf = -kT * np.log(hist)
    pmf = pmf - np.min(pmf)

    r0 = bin_centers[np.argmin(pmf)]

    fit_mask = np.abs(bin_centers - r0) < 0.05
    if np.sum(fit_mask) >= 3:
        r_fit = bin_centers[fit_mask]
        pmf_fit = pmf[fit_mask]
        coeffs = np.polyfit(r_fit - r0, pmf_fit, 2)
        k = 2.0 * coeffs[0]
    else:
        pass
    variance = np.var(distances)
    k = kT / variance

    return r0, k


def boltzmann_inversion_angle(angles, temperature=300.0):
    """Calculate spring constant for angles using Boltzmann inversion."""
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    hist, bin_edges = np.histogram(angles, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    nonzero_mask = hist > 0
    hist = hist[nonzero_mask]
    bin_centers = bin_centers[nonzero_mask]

    pmf = -kT * np.log(hist)
    pmf = pmf - np.min(pmf)

    theta0 = bin_centers[np.argmin(pmf)]

    fit_mask = np.abs(bin_centers - theta0) < 5.0
    if np.sum(fit_mask) >= 3:
        theta_fit = bin_centers[fit_mask]
        pmf_fit = pmf[fit_mask]
        theta_fit_rad = np.deg2rad(theta_fit - theta0)
        coeffs = np.polyfit(theta_fit_rad, pmf_fit, 2)
        k = 2.0 * coeffs[0]
    else:
        pass
    variance_rad = np.var(np.deg2rad(angles))
    k = kT / variance_rad

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
    """Calculate spring constant for dihedrals using Boltzmann inversion."""
    kB = 0.008314462618  # kJ/mol/K
    kT = kB * temperature

    circ_mean = circular_mean_deg(dihedrals)
    dihedrals_shifted = wrap_to_180(dihedrals - circ_mean)

    hist, bin_edges = np.histogram(dihedrals_shifted, bins=50, range=(-180, 180), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    nonzero_mask = hist > 0
    hist = hist[nonzero_mask]
    bin_centers = bin_centers[nonzero_mask]

    pmf = -kT * np.log(hist)
    pmf = pmf - np.min(pmf)

    phi0_shifted = bin_centers[np.argmin(pmf)]
    phi0 = wrap_to_180(phi0_shifted + circ_mean)

    fit_mask = np.abs(bin_centers - phi0_shifted) < 10.0
    if np.sum(fit_mask) >= 3:
        phi_fit = bin_centers[fit_mask]
        pmf_fit = pmf[fit_mask]
        phi_fit_rad = np.deg2rad(phi_fit - phi0_shifted)
        coeffs = np.polyfit(phi_fit_rad, pmf_fit, 2)
        k = 2.0 * coeffs[0]
    else:
        pass
    variance_rad = np.var(np.deg2rad(dihedrals_shifted))
    k = kT / variance_rad

    return phi0, k
