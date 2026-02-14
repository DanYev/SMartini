import logging
from pathlib import Path
import copy
import numpy as np
from MDAnalysis import Universe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import AutoMartini as am
import rdkit
from rdkit import Chem

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    force=True,  # override any prior logging config set by imported libs
)
logging.getLogger("AutoMartini").setLevel(logging.INFO)  # or DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_cog_trajectory(in_pdb, in_xtc, partitioning):
    """Read AA trajectory and calculate COG trajectory for CG beads.
    
    Parameters
    ----------
    in_pdb : str or Path
        Path to atomistic PDB file
    in_xtc : str or Path
        Path to atomistic XTC trajectory
    partitioning : dict
        Mapping of atom indices to bead indices {atom_idx: bead_idx}
        
    Returns
    -------
    numpy.ndarray
        CG trajectory array with shape (n_frames, n_beads, 3)
    """
    logger.info(f"Reading AA trajectory: {in_pdb}, {in_xtc}")
    logger.debug(f"Partitioning: {len(partitioning)} atoms -> {max(partitioning.values())+1} beads")
    
    # Load universe
    u = Universe(str(in_pdb), str(in_xtc))
    logger.info(f"Loaded trajectory with {len(u.trajectory)} frames")
    
    # Group atoms by bead
    n_beads = max(partitioning.values()) + 1
    bead_to_atoms = {i: [] for i in range(n_beads)}
    for atom_idx, bead_idx in partitioning.items():
        bead_to_atoms[bead_idx].append(atom_idx)
    
    # Prepare output array
    n_frames = len(u.trajectory)
    cg_trajectory = np.zeros((n_frames, n_beads, 3))
    logger.info(f"Computing COG trajectory: {n_frames} frames, {n_beads} beads")
    
    # Calculate COG for each bead at each frame
    for frame_idx, ts in enumerate(u.trajectory):
        if frame_idx % 100 == 0:
            logger.debug(f"Processing frame {frame_idx}/{n_frames}")
        for bead_idx in range(n_beads):
            atom_indices = bead_to_atoms[bead_idx]
            if atom_indices:
                # Get positions of atoms in this bead (in Angstroms)
                positions = u.atoms[atom_indices].positions
                # Calculate center of geometry (mean position) and convert to nm
                cg_trajectory[frame_idx, bead_idx] = positions.mean(axis=0) / 10.0
    
    logger.info(f"COG trajectory computed successfully: shape {cg_trajectory[:-2, :n_beads, :].shape}")
    return cg_trajectory[:-2, :n_beads, :]


def calculate_internal_coordinates(cg_trajectory, topo):
    """Calculate internal coordinates (bonds, angles, dihedrals) from CG trajectory.
    
    Parameters
    ----------
    cg_trajectory : numpy.ndarray
        CG trajectory array with shape (n_frames, n_beads, 3)
    topo : Topology
        Topology object containing bonds, angles, dihedrals, constraints
        
    Returns
    -------
    dict
        Dictionary with coordinate info as keys and values as arrays
        Format: 
        - bonds: {(i, j, 'bond'): distances_array}
        - angles: {(i, j, k, 'angle'): angles_array}
        - dihedrals: {(i, j, k, l, 'dihedral'): dihedrals_array}
    """
    logger.info(f"Calculating internal coordinates from trajectory")
    logger.debug(f"Trajectory shape: {cg_trajectory.shape}")
    logger.debug(f"Number of bonds: {len(topo.bonds)}, constraints: {len(topo.constraints)}, "
                 f"angles: {len(topo.angles)}, dihedrals: {len(topo.dihedrals)}")
    
    n_frames = cg_trajectory.shape[0]
    internal_coords = {}
    
    # Calculate distances for bonds
    for bond in topo.bonds:
        i, j = int(bond[0]), int(bond[1])
        distances = np.zeros(n_frames)
        
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            distances[frame_idx] = np.linalg.norm(pos_i - pos_j)
        
        internal_coords[(i, j, 'bond')] = distances
    
    # Calculate distances for constraints
    for constraint in topo.constraints:
        i, j = int(constraint[0]), int(constraint[1])
        distances = np.zeros(n_frames)
        
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            distances[frame_idx] = np.linalg.norm(pos_i - pos_j)
        
        internal_coords[(i, j, 'constraint')] = distances
    
    # Calculate angles
    for angle in topo.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        angles = np.zeros(n_frames)
        
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            pos_k = cg_trajectory[frame_idx, k]
            
            # Vectors from j to i and j to k
            v1 = pos_i - pos_j
            v2 = pos_k - pos_j
            
            # Calculate angle using dot product
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # Clamp to avoid numerical issues
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles[frame_idx] = np.degrees(np.arccos(cos_angle))
        
        internal_coords[(i, j, k, 'angle')] = angles
    
    # Calculate dihedrals
    for dihedral in topo.dihedrals:
        i, j, k, l = int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3])
        dihedrals = np.zeros(n_frames)
        
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            pos_k = cg_trajectory[frame_idx, k]
            pos_l = cg_trajectory[frame_idx, l]
            
            # Calculate dihedral angle
            b1 = pos_j - pos_i
            b2 = pos_k - pos_j
            b3 = pos_l - pos_k
            
            # Normal vectors to planes
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            
            # Normalize b2
            b2_norm = b2 / np.linalg.norm(b2)
            
            # Calculate dihedral
            x = np.dot(n1, n2)
            y = np.dot(np.cross(n1, b2_norm), n2)
            dihedrals[frame_idx] = np.degrees(np.arctan2(y, x))
        
        internal_coords[(i, j, k, l, 'dihedral')] = dihedrals
    
    n_bonds = len([k for k in internal_coords.keys() if k[-1] in ['bond', 'constraint']])
    n_angles = len([k for k in internal_coords.keys() if k[-1] == 'angle'])
    n_dihedrals = len([k for k in internal_coords.keys() if k[-1] == 'dihedral'])
    
    logger.info(f"Calculated {n_bonds} bonds/constraints, {n_angles} angles, {n_dihedrals} dihedrals")
    
    return internal_coords


def boltzmann_inversion_bond(distances, temperature=300.0):
    """Calculate spring constant for bonds using Boltzmann inversion.
    
    Parameters
    ----------
    distances : numpy.ndarray
        Array of bond distances from trajectory (in nm)
    temperature : float
        Temperature in Kelvin (default: 300.0)
        
    Returns
    -------
    tuple
        (r0, k) where r0 is equilibrium distance (nm) and k is spring constant (kJ/mol/nm^2)
    """
    logger.debug(f"Running bond Boltzmann inversion at T={temperature} K")
    
    # Physical constants
    kB = 0.008314462618  # Boltzmann constant in kJ/mol/K
    kT = kB * temperature
    
    # Calculate histogram
    hist, bin_edges = np.histogram(distances, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Remove zero probability bins
    nonzero_mask = hist > 0
    hist = hist[nonzero_mask]
    bin_centers = bin_centers[nonzero_mask]
    
    # Calculate PMF (potential of mean force)
    # PMF(r) = -kT * ln(P(r))
    pmf = -kT * np.log(hist)
    
    # Shift PMF so minimum is at zero
    pmf = pmf - np.min(pmf)
    
    # Equilibrium distance is at minimum PMF
    r0 = bin_centers[np.argmin(pmf)]
    
    # Fit quadratic around minimum to extract spring constant
    # V(r) = 0.5 * k * (r - r0)^2
    # Near minimum, fit: PMF = a * (r - r0)^2
    # Then k = 2 * a
    
    # Use points within 0.05 nm of minimum
    fit_mask = np.abs(bin_centers - r0) < 0.05
    if np.sum(fit_mask) >= 3:
        r_fit = bin_centers[fit_mask]
        pmf_fit = pmf[fit_mask]
        
        # Fit parabola: y = a*(x-r0)^2
        # Using polyfit with shifted coordinates
        coeffs = np.polyfit(r_fit - r0, pmf_fit, 2)
        k = 2.0 * coeffs[0]  # Spring constant
    else:
        # Fallback: estimate from variance
        logger.debug("Using variance-based k estimation (insufficient points for parabolic fit)")
        variance = np.var(distances)
        k = kT / variance
    
    logger.debug(f"Bond Boltzmann inversion result: r0={r0:.4f} nm, k={k:.1f} kJ/mol/nm^2")
    return r0, k


def boltzmann_inversion_angle(angles, temperature=300.0):
    """Calculate spring constant for angles using Boltzmann inversion.
    
    Parameters
    ----------
    angles : numpy.ndarray
        Array of angles from trajectory (in degrees)
    temperature : float
        Temperature in Kelvin (default: 300.0)
        
    Returns
    -------
    tuple
        (theta0, k) where theta0 is equilibrium angle (degrees) and k is spring constant (kJ/mol/rad^2)
    """
    logger.debug(f"Running angle Boltzmann inversion at T={temperature} K")
    
    # Physical constants
    kB = 0.008314462618  # Boltzmann constant in kJ/mol/K
    kT = kB * temperature
    
    # Calculate histogram
    hist, bin_edges = np.histogram(angles, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Remove zero probability bins
    nonzero_mask = hist > 0
    hist = hist[nonzero_mask]
    bin_centers = bin_centers[nonzero_mask]
    
    # Calculate PMF
    pmf = -kT * np.log(hist)
    pmf = pmf - np.min(pmf)
    
    # Equilibrium angle is at minimum PMF
    theta0 = bin_centers[np.argmin(pmf)]
    
    # Fit quadratic around minimum
    # Use points within 5 degrees of minimum
    fit_mask = np.abs(bin_centers - theta0) < 5.0
    if np.sum(fit_mask) >= 3:
        theta_fit = bin_centers[fit_mask]
        pmf_fit = pmf[fit_mask]
        
        # Convert to radians for force constant
        theta_fit_rad = np.deg2rad(theta_fit - theta0)
        
        # Fit parabola
        coeffs = np.polyfit(theta_fit_rad, pmf_fit, 2)
        k = 2.0 * coeffs[0]  # Spring constant in kJ/mol/rad^2
    else:
        # Fallback: estimate from variance
        logger.debug("Using variance-based k estimation for angle")
        variance_rad = np.var(np.deg2rad(angles))
        k = kT / variance_rad
    
    logger.debug(f"Angle Boltzmann inversion result: theta0={theta0:.2f} deg, k={k:.1f} kJ/mol/rad^2")
    return theta0, k


def circular_mean_deg(angles):
    """Calculate circular mean of angles in degrees.
    
    Parameters
    ----------
    angles : numpy.ndarray
        Array of angles in degrees
        
    Returns
    -------
    float
        Circular mean in degrees, in range [-180, 180]
    """
    angles_rad = np.deg2rad(angles)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_rad = np.arctan2(sin_mean, cos_mean)
    return np.rad2deg(mean_rad)


def wrap_to_180(angles):
    """Wrap angles to [-180, 180] range.
    
    Parameters
    ----------
    angles : numpy.ndarray
        Array of angles in degrees
        
    Returns
    -------
    numpy.ndarray
        Wrapped angles in range [-180, 180]
    """
    return (angles + 180) % 360 - 180


def boltzmann_inversion_dihedral(dihedrals, temperature=300.0):
    """Calculate spring constant for dihedrals using Boltzmann inversion.
    
    Handles periodic boundary conditions properly for dihedrals centered near ±180°.
    
    Parameters
    ----------
    dihedrals : numpy.ndarray
        Array of dihedral angles from trajectory (in degrees)
    temperature : float
        Temperature in Kelvin (default: 300.0)
        
    Returns
    -------
    tuple
        (phi0, k) where phi0 is equilibrium dihedral (degrees) and k is spring constant (kJ/mol/rad^2)
    """
    logger.debug(f"Running dihedral Boltzmann inversion at T={temperature} K")
    
    # Physical constants
    kB = 0.008314462618  # Boltzmann constant in kJ/mol/K
    kT = kB * temperature
    
    # Calculate circular mean to find distribution center
    circ_mean = circular_mean_deg(dihedrals)
    logger.debug(f"Dihedral circular mean: {circ_mean:.2f} deg")
    
    # Shift angles so distribution is centered near 0 (away from ±180° boundary)
    dihedrals_shifted = wrap_to_180(dihedrals - circ_mean)
    
    # Calculate histogram on shifted data
    hist, bin_edges = np.histogram(dihedrals_shifted, bins=50, range=(-180, 180), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Remove zero probability bins
    nonzero_mask = hist > 0
    hist = hist[nonzero_mask]
    bin_centers = bin_centers[nonzero_mask]
    
    # Calculate PMF
    pmf = -kT * np.log(hist)
    pmf = pmf - np.min(pmf)
    
    # Equilibrium dihedral is at minimum PMF (in shifted frame)
    phi0_shifted = bin_centers[np.argmin(pmf)]
    
    # Transform back to original frame
    phi0 = wrap_to_180(phi0_shifted + circ_mean)
    
    # Fit quadratic around minimum (in shifted frame where distribution is centered)
    # Use points within 10 degrees of minimum
    fit_mask = np.abs(bin_centers - phi0_shifted) < 10.0
    if np.sum(fit_mask) >= 3:
        phi_fit = bin_centers[fit_mask]
        pmf_fit = pmf[fit_mask]
        
        # Convert to radians for force constant
        phi_fit_rad = np.deg2rad(phi_fit - phi0_shifted)
        
        # Fit parabola
        coeffs = np.polyfit(phi_fit_rad, pmf_fit, 2)
        k = 2.0 * coeffs[0]  # Spring constant in kJ/mol/rad^2
    else:
        # Fallback: estimate from circular variance
        logger.debug("Using variance-based k estimation for dihedral")
        # Circular variance in shifted frame
        variance_rad = np.var(np.deg2rad(dihedrals_shifted))
        k = kT / variance_rad
    
    logger.debug(f"Dihedral Boltzmann inversion result: phi0={phi0:.2f} deg, k={k:.1f} kJ/mol/rad^2")
    return phi0, k


def plot_internal_coordinates(internal_coords, topo, output_file=None):
    """Plot histograms of internal coordinates (bonds, angles, dihedrals) in separate figures.
    
    Parameters
    ----------
    internal_coords : dict
        Dictionary from calculate_internal_coordinates()
    topo : Topology
        Topology object for reference values
    output_file : str or Path, optional
        Base path to save the figures. If None, displays interactively.
    """
    logger.info(f"Plotting internal coordinates: {len(internal_coords)} total")
    
    # Separate by type
    bonds_data = {k: v for k, v in internal_coords.items() if k[-1] in ['bond', 'constraint']}
    angles_data = {k: v for k, v in internal_coords.items() if k[-1] == 'angle'}
    dihedrals_data = {k: v for k, v in internal_coords.items() if k[-1] == 'dihedral'}
    
    logger.info(f"Bonds: {len(bonds_data)}, Angles: {len(angles_data)}, Dihedrals: {len(dihedrals_data)}")
    
    # Plot bonds
    if bonds_data:
        _plot_bonds(bonds_data, topo, output_file)
    
    # Plot angles
    if angles_data:
        _plot_angles(angles_data, topo, output_file)
    
    # Plot dihedrals
    if dihedrals_data:
        _plot_dihedrals(dihedrals_data, topo, output_file)


def _plot_bonds(bonds_data, topo, output_file):
    """Plot bond histograms."""
    logger.info(f"Plotting {len(bonds_data)} bonds/constraints")
    
    # Create reference dictionaries
    # bonds: [i, j, funct, dist, k]
    bond_ref = {(int(b[0]), int(b[1])): b[3] for b in topo.bonds}
    # constraints: [i, j, funct, dist]
    constraint_ref = {(int(c[0]), int(c[1])): c[3] for c in topo.constraints}
    
    # Determine grid layout
    n_plots = len(bonds_data)
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, distances) in enumerate(bonds_data.items()):
        ax = axes[idx]
        i, j, bond_type = key
        
        # Plot histogram
        ax.hist(distances, bins=30, alpha=0.7, edgecolor='black')
        
        # Add reference line
        if bond_type == 'bond' and (i, j) in bond_ref:
            ref_length = bond_ref[(i, j)]
            ax.axvline(ref_length, color='red', linestyle='--', linewidth=1.5, 
                      label=f'ITP: {ref_length:.3f}')
        elif bond_type == 'constraint' and (i, j) in constraint_ref:
            ref_length = constraint_ref[(i, j)]
            ax.axvline(ref_length, color='red', linestyle='--', linewidth=1.5,
                      label=f'ITP: {ref_length:.3f}')
        
        ax.set_xlabel('Distance (nm)', fontsize=9)
        ax.set_title(f'{bond_type.capitalize()}: {i+1}-{j+1}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
        
        # Calculate via Boltzmann inversion
        r0_calc, k_calc = boltzmann_inversion_bond(distances)
        
        # Get reference force constant
        ref_fc = None
        if bond_type == 'bond':
            for bond in topo.bonds:
                # bonds: [i, j, funct, dist, k]
                if int(bond[0]) == i and int(bond[1]) == j and len(bond) >= 5:
                    ref_fc = bond[4]
                    break
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        k_rounded = round(k_calc / 1000) * 1000
        stats_text = f'μ={mean_dist:.3f}\nσ={std_dist:.3f}\n'
        stats_text += f'r₀={r0_calc:.3f}\nk={int(k_rounded/1000)}e3'
        if ref_fc is not None:
            ref_k_rounded = round(ref_fc / 1000) * 1000
            stats_text += f'\nITP k={int(ref_k_rounded/1000)}e3'
        
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_file:
        base = Path(output_file).stem if isinstance(output_file, (str, Path)) else "internal_coords"
        bonds_file = Path(output_file).parent / f"{base}_bonds.png" if isinstance(output_file, (str, Path)) else "bonds.png"
        logger.info(f"Saving bonds plot to {bonds_file}")
        plt.savefig(bonds_file, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _plot_angles(angles_data, topo, output_file):
    """Plot angle histograms."""
    logger.info(f"Plotting {len(angles_data)} angles")
    
    # Create reference dictionary
    # angles: [i, j, k, funct, angle, force_const]
    angle_ref = {(int(a[0]), int(a[1]), int(a[2])): a[4] for a in topo.angles}
    
    n_plots = len(angles_data)
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, angles) in enumerate(angles_data.items()):
        ax = axes[idx]
        i, j, k, angle_type = key
        
        ax.hist(angles, bins=30, alpha=0.7, edgecolor='black')
        
        # Add reference line
        if (i, j, k) in angle_ref:
            ref_angle = angle_ref[(i, j, k)]
            ax.axvline(ref_angle, color='red', linestyle='--', linewidth=1.5,
                      label=f'ITP: {ref_angle:.1f}°')
        
        ax.set_xlabel('Angle (degrees)', fontsize=9)
        ax.set_title(f'Angle: {i+1}-{j+1}-{k+1}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        
        # Calculate via Boltzmann inversion
        theta0_calc, k_calc = boltzmann_inversion_angle(angles)
        
        # Get reference force constant
        ref_fc = None
        for angle in topo.angles:
            # angles: [i, j, k, funct, angle, force_const]
            if int(angle[0]) == i and int(angle[1]) == j and int(angle[2]) == k and len(angle) >= 6:
                ref_fc = angle[5]
                break
        
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        k_rounded = round(k_calc / 10) * 10
        stats_text = f'μ={mean_angle:.1f}°\nσ={std_angle:.1f}°\n'
        stats_text += f'θ₀={theta0_calc:.1f}°\nk={int(k_rounded)}'
        if ref_fc is not None:
            ref_k_rounded = round(ref_fc / 10) * 10
            stats_text += f'\nITP k={int(ref_k_rounded)}'
        
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_file:
        base = Path(output_file).stem if isinstance(output_file, (str, Path)) else "internal_coords"
        angles_file = Path(output_file).parent / f"{base}_angles.png" if isinstance(output_file, (str, Path)) else "angles.png"
        logger.info(f"Saving angles plot to {angles_file}")
        plt.savefig(angles_file, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _plot_dihedrals(dihedrals_data, topo, output_file):
    """Plot dihedral histograms."""
    logger.info(f"Plotting {len(dihedrals_data)} dihedrals")
    
    # Create reference dictionary
    # dihedrals: [i, j, k, l, funct, angle, force_const]
    dihedral_ref = {(int(d[0]), int(d[1]), int(d[2]), int(d[3])): d[5] for d in topo.dihedrals}
    
    n_plots = len(dihedrals_data)
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (key, dihedrals) in enumerate(dihedrals_data.items()):
        ax = axes[idx]
        i, j, k, l, dihedral_type = key
        
        # Calculate circular mean to determine centering
        circ_mean = circular_mean_deg(dihedrals)
        
        # Shift data to center distribution (avoids splitting at ±180°)
        dihedrals_shifted = wrap_to_180(dihedrals - circ_mean)
        
        # Plot histogram centered at 0
        ax.hist(dihedrals_shifted, bins=30, range=(-180, 180), alpha=0.7, edgecolor='black')
        
        # Add reference line (shifted to same frame)
        if (i, j, k, l) in dihedral_ref:
            ref_dihedral = dihedral_ref[(i, j, k, l)]
            ref_dihedral_shifted = wrap_to_180(ref_dihedral - circ_mean)
            ax.axvline(ref_dihedral_shifted, color='red', linestyle='--', linewidth=1.5,
                      label=f'ITP: {ref_dihedral:.1f}°')
        
        ax.set_xlabel(f'Dihedral - {circ_mean:.1f}° (degrees)', fontsize=9)
        ax.set_title(f'Dihedral: {i+1}-{j+1}-{k+1}-{l+1}', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yticks([])
        ax.set_xlim(-180, 180)  # Always show full periodic range
        ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
        
        # Calculate via Boltzmann inversion
        phi0_calc, k_calc = boltzmann_inversion_dihedral(dihedrals)
        
        # Add Boltzmann-inverted equilibrium line (shifted to plotting frame)
        phi0_shifted = wrap_to_180(phi0_calc - circ_mean)
        ax.axvline(phi0_shifted, color='green', linestyle=':', linewidth=2,
                  label=f'φ₀: {phi0_calc:.1f}°')
        
        # Update legend after adding all lines
        ax.legend(fontsize=8)
        
        # Get reference force constant
        ref_fc = None
        for dihedral in topo.dihedrals:
            # dihedrals: [i, j, k, l, funct, angle, force_const]
            if int(dihedral[0]) == i and int(dihedral[1]) == j and int(dihedral[2]) == k and int(dihedral[3]) == l and len(dihedral) >= 7:
                ref_fc = dihedral[6]
                break
        
        # Use circular mean for proper averaging
        mean_dihedral = circular_mean_deg(dihedrals)
        # For std, shift to center before calculating
        dihedrals_centered = wrap_to_180(dihedrals - mean_dihedral)
        std_dihedral = np.std(dihedrals_centered)
        k_rounded = round(k_calc / 10) * 10
        stats_text = f'μ={mean_dihedral:.1f}°\nσ={std_dihedral:.1f}°\n'
        stats_text += f'φ₀={phi0_calc:.1f}°\nk={int(k_rounded)}'
        if ref_fc is not None:
            ref_k_rounded = round(ref_fc / 10) * 10
            stats_text += f'\nITP k={int(ref_k_rounded)}'
        
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_file:
        base = Path(output_file).stem if isinstance(output_file, (str, Path)) else "internal_coords"
        dihedrals_file = Path(output_file).parent / f"{base}_dihedrals.png" if isinstance(output_file, (str, Path)) else "dihedrals.png"
        logger.info(f"Saving dihedrals plot to {dihedrals_file}")
        plt.savefig(dihedrals_file, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def update_topology_with_boltzmann(topo, internal_coords, output_itp):
    """Update topology with Boltzmann-inverted parameters and write new ITP.
    
    Parameters
    ----------
    topo : Topology
        Original topology object
    internal_coords : dict
        Dictionary from calculate_internal_coordinates()
    output_itp : str or Path
        Path to write the updated ITP file
    
    Returns
    -------
    Topology
        Updated topology object
    """
    logger.info(f"Updating topology with Boltzmann-inverted parameters")
    
    # Create a copy of topology
    updated_topo = copy.deepcopy(topo)
    
    # Update bonds with Boltzmann-inverted values
    n_bonds_updated = 0
    for idx, bond in enumerate(updated_topo.bonds):
        i, j = int(bond[0]), int(bond[1])
        
        # Find corresponding distances
        if (i, j, 'bond') in internal_coords:
            distances = internal_coords[(i, j, 'bond')]
            r0_calc, k_calc = boltzmann_inversion_bond(distances)
            
            # Round k to nearest 1000
            k_rounded = round(k_calc / 1000) * 1000
            
            # Update bond: [i, j, funct, length, force_const]
            logger.debug(f"Bond {i+1}-{j+1}: r0 {bond[3]:.4f} -> {r0_calc:.4f} nm, k -> {int(k_rounded)}")
            updated_topo.bonds[idx] = [i, j, bond[2], r0_calc, k_rounded]
            n_bonds_updated += 1
    
    logger.info(f"Updated {n_bonds_updated} bonds with Boltzmann-inverted parameters")
    
    # Update angles with Boltzmann-inverted values
    n_angles_updated = 0
    for idx, angle in enumerate(updated_topo.angles):
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        
        # Find corresponding angles
        if (i, j, k, 'angle') in internal_coords:
            angles = internal_coords[(i, j, k, 'angle')]
            theta0_calc, k_calc = boltzmann_inversion_angle(angles)
            
            # Round k to nearest 10
            k_rounded = round(k_calc / 10) * 10
            
            # Update angle: [i, j, k, funct, angle, force_const]
            logger.debug(f"Angle {i+1}-{j+1}-{k+1}: θ0 {angle[4]:.2f} -> {theta0_calc:.2f} deg, k -> {int(k_rounded)}")
            updated_topo.angles[idx] = [i, j, k, angle[3], theta0_calc, k_rounded]
            n_angles_updated += 1
    
    logger.info(f"Updated {n_angles_updated} angles with Boltzmann-inverted parameters")
    
    # Update dihedrals with Boltzmann-inverted values
    n_dihedrals_updated = 0
    for idx, dihedral in enumerate(updated_topo.dihedrals):
        i, j, k, l = int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3])
        
        # Find corresponding dihedrals
        if (i, j, k, l, 'dihedral') in internal_coords:
            dihedrals = internal_coords[(i, j, k, l, 'dihedral')]
            phi0_calc, k_calc = boltzmann_inversion_dihedral(dihedrals)
            
            # Round k to nearest 10
            k_rounded = round(k_calc / 10) * 10
            
            # Update dihedral: [i, j, k, l, funct, angle, force_const]
            logger.debug(f"Dihedral {i+1}-{j+1}-{k+1}-{l+1}: φ0 {dihedral[5]:.2f} -> {phi0_calc:.2f} deg, k -> {int(k_rounded)}")
            updated_topo.dihedrals[idx] = [i, j, k, l, dihedral[4], phi0_calc, k_rounded]
            n_dihedrals_updated += 1
    
    logger.info(f"Updated {n_dihedrals_updated} dihedrals with Boltzmann-inverted parameters")
    
    # Write updated topology to ITP file using built-in method
    logger.info(f"Generating updated ITP file")
    itp_content = updated_topo.to_itp(trial=False)
    
    # Write to file
    logger.info(f"Writing updated topology to {output_itp}")
    with open(output_itp, 'w') as f:
        f.write(itp_content)
    
    return updated_topo


if __name__ == "__main__":
    molname = "FTA"
    
    logger.info(f"Starting analysis for molecule: {molname}")
    
    # CG topology from .itp file
    wdir = Path("systems") / molname
    in_itp = wdir / "mapping" / f"{molname}.itp"
    logger.info(f"Reading topology from {in_itp}")
    topo = am.topology.read_itp(str(in_itp))
    logger.info(f"Loaded topology: {len(topo.atoms)} atoms, {len(topo.bonds)} bonds, "
                f"{len(topo.constraints)} constraints, {len(topo.angles)} angles, "
                f"{len(topo.dihedrals)} dihedrals")

    # Trajectory
    mddir = wdir / "aa_md" 
    in_pdb = mddir / "md.pdb"
    in_xtc = mddir / "md.xtc"
    logger.info(f"Reading trajectory files from {mddir}")
    
    # Calculate internal coordinates
    cg_traj = read_cog_trajectory(in_pdb, in_xtc, topo.partitioning)
    internal_coords = calculate_internal_coordinates(cg_traj, topo)
    
    # Plot all internal coordinates
    plot_internal_coordinates(internal_coords, topo, output_file=wdir / "mapping" / "internal_coords.png")
    
    # Update topology with Boltzmann-inverted parameters
    out_itp = wdir / "mapping" / f"{molname}_updated.itp"
    updated_topo = update_topology_with_boltzmann(topo, internal_coords, out_itp)
    
    logger.info(f"Analysis complete!")
    logger.info(f"Updated ITP file written to: {out_itp}")