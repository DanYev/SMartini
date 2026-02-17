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
    fit_type9_dihedral,
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

MOLNAME = "ANP"


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


def remove_unstable_dihedrals(
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


def boltzmann_invert_topology(
    topo,
    internal_coords,
    *,
    max_multiplicity: int = 3,
):
    """Compute Boltzmann-inverted bonded parameters from internal coordinate samples.

    This step *only* updates parameters when trajectory data is available.
    It does not remove terms by cutoffs and does not move bonds to constraints.
    """
    logger.info("Boltzmann inversion: estimating bonded parameters")

    updated_topo = copy.deepcopy(topo)

    # Bonds
    n_bonds_updated = 0
    for idx, bond in enumerate(getattr(updated_topo, "bonds", [])):
        i, j = int(bond[0]), int(bond[1])
        if (i, j, "bond") not in internal_coords:
            continue
        distances = internal_coords[(i, j, "bond")]
        r0_calc, k_calc = boltzmann_inversion_bond(distances)
        k_val = float(k_calc)
        updated_topo.bonds[idx] = [i, j, bond[2], r0_calc, k_val]
        n_bonds_updated += 1
    logger.info("Boltzmann inversion: updated %s bonds", n_bonds_updated)

    # Constraints (distance only)
    n_constraints_updated = 0
    for idx, constraint in enumerate(getattr(updated_topo, "constraints", [])):
        i, j = int(constraint[0]), int(constraint[1])
        if (i, j, "constraint") not in internal_coords:
            continue
        distances = internal_coords[(i, j, "constraint")]
        r0_calc, _ = boltzmann_inversion_bond(distances)
        updated_topo.constraints[idx] = [i, j, constraint[2], r0_calc]
        n_constraints_updated += 1
    logger.info("Boltzmann inversion: updated %s constraints", n_constraints_updated)

    # Angles
    n_angles_updated = 0
    for idx, angle in enumerate(getattr(updated_topo, "angles", [])):
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        if (i, j, k, "angle") not in internal_coords:
            continue
        samples = internal_coords[(i, j, k, "angle")]
        theta0_calc, k_calc = boltzmann_inversion_angle(samples)
        k_val = float(k_calc)
        updated_topo.angles[idx] = [i, j, k, angle[3], theta0_calc, k_val]
        n_angles_updated += 1
    logger.info("Boltzmann inversion: updated %s angles", n_angles_updated)

    # Dihedrals (type-9 Fourier terms)
    dihedrals_by_key = {}
    for dihedral in getattr(updated_topo, "dihedrals", []):
        key = (int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3]))
        dihedrals_by_key.setdefault(key, []).append(dihedral)

    new_dihedrals = []
    n_dihedral_sets_fit = 0
    for (i, j, k, l), terms in dihedrals_by_key.items():
        data = internal_coords.get((i, j, k, l, "dihedral"))
        if data is None:
            new_dihedrals.extend(terms)
            continue
        print((i, j, k, l))
        fit_terms = fit_type9_dihedral(data, max_n=max_multiplicity)
        if not fit_terms:
            new_dihedrals.extend(terms)
            continue

        for term in fit_terms:
            if len(term) == 2:
                mult, k_term = term
                phi0 = 0.0
            else:
                mult, k_term, phi0 = term
            new_dihedrals.append([i, j, k, l, 9, float(phi0), float(k_term), int(mult)])

        n_dihedral_sets_fit += 1

    updated_topo.dihedrals = new_dihedrals
    logger.info("Boltzmann inversion: fit %s dihedral sets", n_dihedral_sets_fit)
    return updated_topo


def filter_topology(
    topo,
    *,
    constraint_k_cutoff=20000,
    angle_k_cutoff=25,
    dihedral_k_cutoff=5,
    keep_best_dihedral_term: bool = True,
):
    """Filter/post-process a topology after Boltzmann inversion.

    - Drops weak angle terms (k < angle_k_cutoff)
    - Drops weak dihedral terms (|k| < dihedral_k_cutoff), optionally keeping the
      strongest term per dihedral if everything is filtered out
    - Moves stiff bonds (k > constraint_k_cutoff) to constraints
    """
    updated_topo = copy.deepcopy(topo)

    # Filter angles
    if angle_k_cutoff is not None:
        kept_angles = []
        removed_angles = 0
        for angle in getattr(updated_topo, "angles", []):
            k_val = float(angle[5]) if len(angle) >= 6 else None
            if k_val is not None and k_val < angle_k_cutoff:
                removed_angles += 1
                continue
            kept_angles.append(angle)
        updated_topo.angles = kept_angles
        logger.info(
            "Filtering: kept %s angles (removed %s with k < %s)",
            len(kept_angles),
            removed_angles,
            angle_k_cutoff,
        )

    # Filter dihedrals
    if dihedral_k_cutoff is not None:
        dihedrals_by_key = {}
        for dihedral in getattr(updated_topo, "dihedrals", []):
            key = (int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3]))
            dihedrals_by_key.setdefault(key, []).append(dihedral)

        new_dihedrals = []
        removed_terms = 0
        kept_terms = 0

        for (i, j, k, l), terms in dihedrals_by_key.items():
            # Only apply cutoff to terms that look like [i j k l 9 phi0 k mult]
            eligible = []
            ineligible = []
            for t in terms:
                if len(t) >= 8:
                    eligible.append(t)
                else:
                    ineligible.append(t)

            kept_for_key = []
            for t in eligible:
                k_val = float(t[6])
                if abs(k_val) < dihedral_k_cutoff:
                    removed_terms += 1
                    continue
                kept_for_key.append(t)

            if not kept_for_key and eligible and keep_best_dihedral_term:
                best = max(eligible, key=lambda t: abs(float(t[6])))
                kept_for_key.append(best)

            for t in kept_for_key:
                mult = int(t[7]) if len(t) >= 8 else "?"
                logger.info(
                    "Dihedral %s-%s-%s-%s (n=%s): phi0=%0.1f deg, k=%0.3f",
                    i + 1,
                    j + 1,
                    k + 1,
                    l + 1,
                    mult,
                    float(t[5]) if len(t) >= 6 else 0.0,
                    float(t[6]) if len(t) >= 7 else 0.0,
                )

            new_dihedrals.extend(ineligible)
            new_dihedrals.extend(kept_for_key)
            kept_terms += len(kept_for_key)

        updated_topo.dihedrals = new_dihedrals
        logger.info(
            "Filtering: kept %s dihedral terms (removed %s with |k| < %s)",
            kept_terms,
            removed_terms,
            dihedral_k_cutoff,
        )

    # Move stiff bonds to constraints
    if constraint_k_cutoff is not None:
        new_bonds = []
        constraints_to_add = []
        existing_constraints = {
            (int(c[0]), int(c[1])) for c in getattr(updated_topo, "constraints", [])
        }
        moved_count = 0
        for bond in getattr(updated_topo, "bonds", []):
            if len(bond) < 5:
                new_bonds.append(bond)
                continue
            i, j, funct, dist, k = bond
            if float(k) > constraint_k_cutoff:
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
                "Filtering: moved %s bonds to constraints (k > %s)",
                moved_count,
                constraint_k_cutoff,
            )

    return updated_topo


if __name__ == "__main__":
    molname = MOLNAME
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
    # cg_traj = read_cog_trajectory(in_pdb, in_xtc, topo.partitioning)
    # np.save("traj_coords.npy", cg_traj)
    cg_traj = np.load("traj_coords.npy")
    # Calculate internal coordinates
    logger.info("Calculating internal coordinates from trajectory")
    internal_coords = calculate_internal_coordinates(cg_traj, topo)

    # # Plot all internal coordinates
    # plot_internal_coordinates(internal_coords, topo, output_file=wdir / "png" / "aa.png")
    
    # Boltzmann inversion (fit parameters from trajectory)
    inverted_topo = boltzmann_invert_topology(
        topo,
        internal_coords,
        max_multiplicity=2,
    )

    # Filtering/post-processing (cutoffs + move stiff bonds to constraints)
    updated_topo = filter_topology(
        inverted_topo,
        constraint_k_cutoff=20000,
        angle_k_cutoff=25,
        dihedral_k_cutoff=5,
    )

    # Filter out potentially unstable dihedrals based on topology-only criteria
    filtered_topo = remove_unstable_dihedrals(
        updated_topo,
        angle_linear_cutoff_deg=160.0,
        drop_if_undefined=True,
    )

    out_itp = wdir / "mapping" / f"{molname}_updated.itp"
    itp_content = filtered_topo.to_itp(out_file=out_itp)

    logger.info(f"Analysis complete!")
    logger.info(f"Updated ITP file written to: {out_itp}")