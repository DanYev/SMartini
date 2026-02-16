import logging
from pathlib import Path
import copy
import numpy as np
import AutoMartini as am
import rdkit
from rdkit import Chem

from lpmath import (
    read_cog_trajectory,
    calculate_internal_coordinates,
    boltzmann_inversion_bond,
    boltzmann_inversion_angle,
    boltzmann_inversion_dihedral,
)
from plots import plot_internal_coordinates

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    force=True,  # override any prior logging config set by imported libs
)
logging.getLogger("AutoMartini").setLevel(logging.INFO)  # or DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_topology_with_boltzmann(
    topo,
    internal_coords,
    output_itp,
    constraint_k_cutoff=20000,
    angle_k_cutoff=25,
    dihedral_k_cutoff=5,
):
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
            r0_calc = round(r0_calc, 3)
            
            # Round k to nearest 1000
            k_rounded = round(k_calc / 1000) * 1000
            
            # Update bond: [i, j, funct, length, force_const]
            logger.debug(f"Bond {i+1}-{j+1}: r0 {bond[3]:.4f} -> {r0_calc:.4f} nm, k -> {int(k_rounded)}")
            updated_topo.bonds[idx] = [i, j, bond[2], r0_calc, k_rounded]
            n_bonds_updated += 1
    
    logger.info(f"Updated {n_bonds_updated} bonds with Boltzmann-inverted parameters")

    # Update constraints with Boltzmann-inverted values (distance only)
    n_constraints_updated = 0
    for idx, constraint in enumerate(updated_topo.constraints):
        i, j = int(constraint[0]), int(constraint[1])

        if (i, j, 'constraint') in internal_coords:
            distances = internal_coords[(i, j, 'constraint')]
            r0_calc, _ = boltzmann_inversion_bond(distances)
            r0_calc = round(r0_calc, 3)

            logger.debug(
                "Constraint %s-%s: r0 %0.4f -> %0.4f nm",
                i + 1,
                j + 1,
                constraint[3],
                r0_calc,
            )
            updated_topo.constraints[idx] = [i, j, constraint[2], r0_calc]
            n_constraints_updated += 1

    logger.info(
        "Updated %s constraints with Boltzmann-inverted parameters",
        n_constraints_updated,
    )
    
    # Update angles with Boltzmann-inverted values
    n_angles_updated = 0
    n_angles_removed = 0
    new_angles = []
    for angle in updated_topo.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        
        # Find corresponding angles
        if (i, j, k, 'angle') in internal_coords:
            angles = internal_coords[(i, j, k, 'angle')]
            theta0_calc, k_calc = boltzmann_inversion_angle(angles)

            if angle_k_cutoff is not None and k_calc < angle_k_cutoff:
                n_angles_removed += 1
                continue

            # Round k to nearest 10
            k_rounded = round(k_calc / 10) * 10

            # Update angle: [i, j, k, funct, angle, force_const]
            logger.debug(
                f"Angle {i+1}-{j+1}-{k+1}: θ0 {angle[4]:.2f} -> {theta0_calc:.2f} deg, k -> {int(k_rounded)}"
            )
            new_angles.append([i, j, k, angle[3], theta0_calc, k_rounded])
            n_angles_updated += 1
        else:
            # No trajectory data for this angle: optionally filter based on existing force constant
            existing_k = float(angle[5]) if len(angle) >= 6 else None
            if angle_k_cutoff is not None and existing_k is not None and existing_k < angle_k_cutoff:
                n_angles_removed += 1
                continue
            new_angles.append(angle)

    updated_topo.angles = new_angles

    logger.info(
        "Updated %s angles with Boltzmann-inverted parameters (removed %s with k < %s)",
        n_angles_updated,
        n_angles_removed,
        angle_k_cutoff,
    )
    
    # Update dihedrals with Boltzmann-inverted values
    n_dihedrals_updated = 0
    n_dihedrals_removed = 0
    new_dihedrals = []
    for dihedral in updated_topo.dihedrals:
        i, j, k, l = int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3])
        
        # Find corresponding dihedrals
        if (i, j, k, l, 'dihedral') in internal_coords:
            dihedrals = internal_coords[(i, j, k, l, 'dihedral')]
            phi0_calc, k_calc = boltzmann_inversion_dihedral(dihedrals)

            if dihedral_k_cutoff is not None and k_calc < dihedral_k_cutoff:
                n_dihedrals_removed += 1
                continue

            # Round k to nearest 10
            k_rounded = round(k_calc / 10) * 10

            # Update dihedral: [i, j, k, l, funct, angle, force_const]
            logger.debug(
                f"Dihedral {i+1}-{j+1}-{k+1}-{l+1}: φ0 {dihedral[5]:.2f} -> {phi0_calc:.2f} deg, k -> {int(k_rounded)}"
            )
            new_dihedrals.append([i, j, k, l, dihedral[4], phi0_calc, k_rounded])
            n_dihedrals_updated += 1
        else:
            # No trajectory data for this dihedral: optionally filter based on existing force constant
            existing_k = float(dihedral[6]) if len(dihedral) >= 7 else None
            if dihedral_k_cutoff is not None and existing_k is not None and existing_k < dihedral_k_cutoff:
                n_dihedrals_removed += 1
                continue
            new_dihedrals.append(dihedral)

    updated_topo.dihedrals = new_dihedrals

    logger.info(
        "Updated %s dihedrals with Boltzmann-inverted parameters (removed %s with k < %s)",
        n_dihedrals_updated,
        n_dihedrals_removed,
        dihedral_k_cutoff,
    )
    
    # Move stiff bonds to constraints
    if constraint_k_cutoff is not None:
        new_bonds = []
        constraints_to_add = []
        existing_constraints = {
            (int(c[0]), int(c[1])) for c in updated_topo.constraints
        }
        moved_count = 0
        for bond in updated_topo.bonds:
            i, j, funct, dist, k = bond
            if k > constraint_k_cutoff:
                key = (int(i), int(j))
                rev_key = (int(j), int(i))
                if key not in existing_constraints and rev_key not in existing_constraints:
                    constraints_to_add.append([i, j, funct, dist])
                    existing_constraints.add(key)
                moved_count += 1
            else:
                new_bonds.append(bond)

        if moved_count:
            updated_topo.bonds = new_bonds
            updated_topo.constraints.extend(constraints_to_add)
            logger.info(
                "Moved %s bonds to constraints (k > %s)",
                moved_count,
                constraint_k_cutoff,
            )

    # Write updated topology to ITP file using built-in method
    logger.info(f"Generating updated ITP file")
    itp_content = updated_topo.to_itp(out_file=output_itp)

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
    plot_internal_coordinates(internal_coords, topo, output_file=wdir / "png" / "aa.png")
    
    # Update topology with Boltzmann-inverted parameters
    out_itp = wdir / "mapping" / f"{molname}_updated.itp"
    updated_topo = update_topology_with_boltzmann(
        topo,
        internal_coords,
        out_itp,
        constraint_k_cutoff=20000,
    )
    
    logger.info(f"Analysis complete!")
    logger.info(f"Updated ITP file written to: {out_itp}")