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


def _pair_key(i: int, j: int):
    return (int(i), int(j)) if int(i) <= int(j) else (int(j), int(i))


def _build_length_lookup(topo):
    """Return map (i,j)->length (nm) using bonds and constraints."""
    length = {}
    for bond in getattr(topo, "bonds", []):
        if len(bond) >= 4:
            i, j = int(bond[0]), int(bond[1])
            length[_pair_key(i, j)] = float(bond[3])
    for constraint in getattr(topo, "constraints", []):
        if len(constraint) >= 4:
            i, j = int(constraint[0]), int(constraint[1])
            length[_pair_key(i, j)] = float(constraint[3])
    return length


def _build_angle_lookup(topo):
    """Return map (i,j,k)->theta0 in degrees (symmetric in i/k)."""
    angle = {}
    for a in getattr(topo, "angles", []):
        if len(a) >= 5:
            i, j, k = int(a[0]), int(a[1]), int(a[2])
            theta0 = float(a[4])
            angle[(i, j, k)] = theta0
            angle[(k, j, i)] = theta0
    return angle


def _angle_from_triangle(length_lookup, i: int, j: int, k: int):
    """Infer angle i-j-k from triangle side lengths, if available.

    Uses the cosine law with:
    a = |i-j|, b = |j-k|, c = |i-k|
    angle at vertex j is arccos((a^2 + b^2 - c^2) / (2ab)).
    """
    a = length_lookup.get(_pair_key(i, j))
    b = length_lookup.get(_pair_key(j, k))
    c = length_lookup.get(_pair_key(i, k))
    if a is None or b is None or c is None:
        return None
    if a <= 0 or b <= 0:
        return None
    cos_theta = (a * a + b * b - c * c) / (2.0 * a * b)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def get_equilibrium_angle_deg(topo, i: int, j: int, k: int):
    """Get equilibrium angle (deg) from topology.

    Priority:
    1) explicit [angles] entry
    2) infer from triangle of bonds/constraints (i-j, j-k, i-k)
    """
    angle_lookup = _build_angle_lookup(topo)
    if (i, j, k) in angle_lookup:
        return angle_lookup[(i, j, k)]

    length_lookup = _build_length_lookup(topo)
    return _angle_from_triangle(length_lookup, i, j, k)


def filter_unstable_dihedrals_from_topology(
    topo,
    *,
    angle_linear_cutoff_deg: float = 170.0,
    drop_if_undefined: bool = True,
):
    """Remove dihedrals that are ill-defined due to near-linear adjacent angles.

    A dihedral i-j-k-l becomes numerically unstable when either adjacent angle
    (i-j-k or j-k-l) is close to 180 degrees.

    This filter is topology-only: it uses explicit angles, or infers angles from
    bond/constraint triangles (i-k or j-l third side) when an angle term is not
    present.
    """
    updated = copy.deepcopy(topo)

    # Build lookups once for speed
    angle_lookup = _build_angle_lookup(updated)
    length_lookup = _build_length_lookup(updated)

    def _eq_angle(i, j, k):
        if (i, j, k) in angle_lookup:
            return angle_lookup[(i, j, k)]
        return _angle_from_triangle(length_lookup, i, j, k)

    kept = []
    removed_linear = 0
    removed_undefined = 0

    for d in getattr(updated, "dihedrals", []):
        if len(d) < 6:
            kept.append(d)
            continue
        i, j, k, l = int(d[0]), int(d[1]), int(d[2]), int(d[3])

        a1 = _eq_angle(i, j, k)
        a2 = _eq_angle(j, k, l)

        if a1 is None or a2 is None:
            if drop_if_undefined:
                removed_undefined += 1
                continue
            kept.append(d)
            continue

        if a1 >= angle_linear_cutoff_deg or a2 >= angle_linear_cutoff_deg:
            removed_linear += 1
            continue

        kept.append(d)

    updated.dihedrals = kept
    logger.info(
        "Filtered dihedrals: kept=%s, removed_linear=%s (>%s deg), removed_undefined=%s",
        len(kept),
        removed_linear,
        angle_linear_cutoff_deg,
        removed_undefined,
    )
    return updated


def update_topology_with_boltzmann(
    topo,
    internal_coords,
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
    updated_topo = update_topology_with_boltzmann(
        topo,
        internal_coords,
        constraint_k_cutoff=20000,
    )

    # Filter out potentially unstable dihedrals based on topology-only criteria
    filtered_topo = filter_unstable_dihedrals_from_topology(
        updated_topo,
        angle_linear_cutoff_deg=165.0,
        drop_if_undefined=True,
    )

    out_itp = wdir / "mapping" / f"{molname}_updated.itp"
    itp_content = filtered_topo.to_itp(out_file=out_itp)

    logger.info(f"Analysis complete!")
    logger.info(f"Updated ITP file written to: {out_itp}")