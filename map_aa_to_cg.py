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


def calculate_bond_distances(cg_trajectory, topo):
    """Calculate bond distances from CG trajectory.
    
    Parameters
    ----------
    cg_trajectory : numpy.ndarray
        CG trajectory array with shape (n_frames, n_beads, 3)
    topo : Topology
        Topology object containing bonds and constraints
        
    Returns
    -------
    dict
        Dictionary with bond/constraint info as keys and distances as arrays
        Format: {(bead_i, bead_j, 'bond'|'constraint'): distances_array}
    """
    logger.info(f"Calculating bond distances from trajectory")
    logger.debug(f"Trajectory shape: {cg_trajectory.shape}")
    logger.debug(f"Number of bonds: {len(topo.bonds)}, constraints: {len(topo.constraints)}")
    
    n_frames = cg_trajectory.shape[0]
    bond_distances = {}
    
    # Calculate distances for bonds
    for bond in topo.bonds:
        i, j = int(bond[0]), int(bond[1])
        distances = np.zeros(n_frames)
        
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            distances[frame_idx] = np.linalg.norm(pos_i - pos_j)
        
        bond_distances[(i, j, 'bond')] = distances
    
    # Calculate distances for constraints
    for constraint in topo.constraints:
        i, j = int(constraint[0]), int(constraint[1])
        distances = np.zeros(n_frames)
        
        for frame_idx in range(n_frames):
            pos_i = cg_trajectory[frame_idx, i]
            pos_j = cg_trajectory[frame_idx, j]
            distances[frame_idx] = np.linalg.norm(pos_i - pos_j)
        
        bond_distances[(i, j, 'constraint')] = distances
    
    logger.info(f"Calculated distances for {len(bond_distances)} bonds/constraints")
    for key, distances in list(bond_distances.items())[:3]:
        logger.debug(f"  {key}: mean={distances.mean():.4f} nm, std={distances.std():.4f} nm")
    
    return bond_distances


def boltzmann_inversion(distances, temperature=300.0):
    """Calculate spring constant using Boltzmann inversion.
    
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
    logger.debug(f"Running Boltzmann inversion at T={temperature} K")
    
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
    
    logger.debug(f"Boltzmann inversion result: r0={r0:.4f} nm, k={k:.1f} kJ/mol/nm^2")
    return r0, k


def plot_bond_histograms(bond_distances, topo, output_file=None):
    """Plot histograms of bond/constraint distances in a single figure.
    
    Parameters
    ----------
    bond_distances : dict
        Dictionary from calculate_bond_distances()
    topo : Topology
        Topology object for reference lengths
    output_file : str or Path, optional
        Path to save the figure. If None, displays interactively.
    """
    logger.info(f"Plotting bond histograms for {len(bond_distances)} bonds/constraints")
    
    # Create reference dictionaries for equilibrium lengths
    bond_ref = {(int(b[0]), int(b[1])): b[2] for b in topo.bonds}
    constraint_ref = {(int(c[0]), int(c[1])): c[2] for c in topo.constraints}
    
    # Determine grid layout
    n_plots = len(bond_distances)
    n_cols = min(4, n_plots)  # Max 4 columns
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each bond/constraint
    for idx, ((i, j, bond_type), distances) in enumerate(bond_distances.items()):
        ax = axes[idx]
        
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
        
        # Labels and formatting
        ax.set_xlabel('Distance (nm)', fontsize=9)
        ax.set_title(f'{bond_type.capitalize()}: {i+1}-{j+1}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Remove y-axis labels and ticks
        ax.set_yticks([])
        
        # Set x-axis ticks to 0.01 intervals
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
        
        # Calculate spring constant via Boltzmann inversion
        r0_calc, k_calc = boltzmann_inversion(distances)
        
        # Get reference force constant from ITP if available
        ref_fc = None
        if bond_type == 'bond':
            for bond in topo.bonds:
                if int(bond[0]) == i and int(bond[1]) == j:
                    if len(bond) >= 4:  # Has force constant
                        ref_fc = bond[3]
                    break
        
        # Add statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        k_rounded = round(k_calc / 1000) * 1000  # Round to nearest 1000
        stats_text = f'μ={mean_dist:.3f}\nσ={std_dist:.3f}\n'
        stats_text += f'r₀={r0_calc:.3f}\nk={int(k_rounded/1000)}e3'
        if ref_fc is not None:
            ref_k_rounded = round(ref_fc / 1000) * 1000
            stats_text += f'\nITP k={int(ref_k_rounded/1000)}e3'
        
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_file:
        logger.info(f"Saving bond histogram plot to {output_file}")
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        logger.info("Displaying bond histogram plot interactively")
        plt.show()


def update_topology_with_boltzmann(topo, bond_distances, output_itp):
    """Update topology with Boltzmann-inverted parameters and write new ITP.
    
    Parameters
    ----------
    topo : Topology
        Original topology object
    bond_distances : dict
        Dictionary from calculate_bond_distances()
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
    n_updated = 0
    for idx, bond in enumerate(updated_topo.bonds):
        i, j = int(bond[0]), int(bond[1])
        
        # Find corresponding distances
        if (i, j, 'bond') in bond_distances:
            distances = bond_distances[(i, j, 'bond')]
            r0_calc, k_calc = boltzmann_inversion(distances)
            
            # Round k to nearest 1000
            k_rounded = round(k_calc / 1000) * 1000
            
            # Update bond: [i, j, length, force_const]
            logger.debug(f"Bond {i+1}-{j+1}: r0 {bond[2]:.4f} -> {r0_calc:.4f} nm, k -> {int(k_rounded)}")
            updated_topo.bonds[idx] = [i, j, r0_calc, k_rounded]
            n_updated += 1
    
    logger.info(f"Updated {n_updated} bonds with Boltzmann-inverted parameters")
    
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
    in_itp = Path("output") / molname / f"ligand_{molname}.itp"
    logger.info(f"Reading topology from {in_itp}")
    topo = am.topology.read_itp(str(in_itp))
    logger.info(f"Loaded topology: {len(topo.atoms)} atoms, {len(topo.bonds)} bonds, {len(topo.constraints)} constraints")

    # Trajectory
    mddir = Path("systems") / molname / "mdruns" / "mdrun"
    in_pdb = mddir / "md.pdb"
    in_xtc = mddir / "md.xtc"
    logger.info(f"Reading trajectory files from {mddir}")
    
    cg_traj = read_cog_trajectory(in_pdb, in_xtc, topo.partitioning)
    bond_distances = calculate_bond_distances(cg_traj, topo)
    plot_bond_histograms(bond_distances, topo, output_file="bond_histograms.png")
    
    # Update topology with Boltzmann-inverted parameters
    out_itp = Path("output") / molname / f"ligand_{molname}_boltzmann.itp"
    updated_topo = update_topology_with_boltzmann(topo, bond_distances, out_itp)
    
    logger.info(f"Analysis complete!")
    logger.info(f"Updated ITP file written to: {out_itp}")