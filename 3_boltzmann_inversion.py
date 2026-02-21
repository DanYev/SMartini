import copy
import logging
from pathlib import Path

import AutoMartini as am
import numpy as np

from ligpar_config import CFG, get_logger
from lpmath import (
    boltzmann_inversion_angle,
    boltzmann_inversion_bond,
    calculate_internal_coordinates,
    fit_type9_dihedral,
    read_cog_trajectory,
)
from plots import plot_internal_coordinates

logging.getLogger("AutoMartini").setLevel(logging.INFO)  # or DEBUG
logger = get_logger(__name__)
logger.setLevel(logging.INFO)


MOLNAME = CFG.molname


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
    """Infer angle i-j-k from triangle side lengths, if available."""
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


def remove_unstable_dihedrals(
    topo,
    *,
    angle_linear_cutoff_deg: float = 170.0,
    drop_if_undefined: bool = True,
):
    """Remove dihedrals ill-defined due to near-linear adjacent angles."""
    updated = copy.deepcopy(topo)

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
):
    """Compute Boltzmann-inverted bonded parameters from internal coordinate samples."""
    updated_topo = copy.deepcopy(topo)

    # Bonds
    for idx, bond in enumerate(getattr(updated_topo, "bonds", [])):
        i, j = int(bond[0]), int(bond[1])
        if (i, j, "bond") not in internal_coords:
            continue
        distances = internal_coords[(i, j, "bond")]
        r0_calc, k_calc = boltzmann_inversion_bond(distances)
        updated_topo.bonds[idx] = [i, j, bond[2], float(r0_calc), float(k_calc)]

    # Constraints (distance only)
    for idx, constraint in enumerate(getattr(updated_topo, "constraints", [])):
        i, j = int(constraint[0]), int(constraint[1])
        if (i, j, "constraint") not in internal_coords:
            continue
        distances = internal_coords[(i, j, "constraint")]
        r0_calc, _ = boltzmann_inversion_bond(distances)
        updated_topo.constraints[idx] = [i, j, constraint[2], float(r0_calc)]

    # Angles
    for idx, angle in enumerate(getattr(updated_topo, "angles", [])):
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        if (i, j, k, "angle") not in internal_coords:
            continue
        samples = internal_coords[(i, j, k, "angle")]
        theta0_calc, k_calc = boltzmann_inversion_angle(samples)
        updated_topo.angles[idx] = [i, j, k, angle[3], float(theta0_calc), float(k_calc)]

    # Dihedrals (type-9 terms)
    dihedrals_by_key = {}
    for d in getattr(updated_topo, "dihedrals", []):
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        dihedrals_by_key.setdefault(key, []).append(d)

    new_dihedrals = []
    for (i, j, k, l), existing_terms in dihedrals_by_key.items():
        data = internal_coords.get((i, j, k, l, "dihedral"))
        if data is None:
            new_dihedrals.extend(existing_terms)
            continue

        fit_terms = fit_type9_dihedral(
            data,
            temperature=CFG.temperature,
            max_n=CFG.type9_max_n,
            bins=CFG.type9_bins,
            min_prob=CFG.type9_min_prob,
        )
        if not fit_terms:
            new_dihedrals.extend(existing_terms)
            continue

        for term in fit_terms:
            if len(term) == 2:
                mult, k_term = term
                phi0 = 0.0
            else:
                mult, k_term, phi0 = term
            new_dihedrals.append([i, j, k, l, 9, float(phi0), float(k_term), int(mult)])

    updated_topo.dihedrals = new_dihedrals
    return updated_topo


def filter_topology(
    topo,
    *,
    constraint_k_cutoff=20000,
    angle_k_cutoff=25,
    dihedral_k_cutoff=5,
    keep_best_dihedral_term: bool = True,
):
    """Filter/post-process a topology after Boltzmann inversion."""
    updated_topo = copy.deepcopy(topo)

    # Angles: drop weak k
    if angle_k_cutoff is not None:
        kept_angles = []
        for angle in getattr(updated_topo, "angles", []):
            k_val = float(angle[5]) if len(angle) >= 6 else None
            if k_val is not None and k_val < angle_k_cutoff:
                continue
            kept_angles.append(angle)
        updated_topo.angles = kept_angles

    # Dihedrals: drop weak |k|, optionally keep strongest
    if dihedral_k_cutoff is not None:
        dihedrals_by_key = {}
        for d in getattr(updated_topo, "dihedrals", []):
            key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
            dihedrals_by_key.setdefault(key, []).append(d)

        new_dihedrals = []
        for key, terms in dihedrals_by_key.items():
            eligible = [t for t in terms if len(t) >= 8]
            ineligible = [t for t in terms if len(t) < 8]

            kept_for_key = [t for t in eligible if abs(float(t[6])) >= dihedral_k_cutoff]
            if not kept_for_key and eligible and keep_best_dihedral_term:
                kept_for_key = [max(eligible, key=lambda t: abs(float(t[6])))]

            new_dihedrals.extend(ineligible)
            new_dihedrals.extend(kept_for_key)

        updated_topo.dihedrals = new_dihedrals

    # Move stiff bonds to constraints
    if constraint_k_cutoff is not None:
        new_bonds = []
        constraints_to_add = []
        existing_constraints = {
            (int(c[0]), int(c[1])) for c in getattr(updated_topo, "constraints", [])
        }
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
            else:
                new_bonds.append(bond)

        updated_topo.bonds = new_bonds
        updated_topo.constraints.extend(constraints_to_add)

    return updated_topo


if __name__ == "__main__":
    molname = MOLNAME
    wdir = CFG.wdir()
    logger.info("Starting analysis for molecule: %s", molname)

    in_itp = wdir / "mapping" / f"{molname}.itp"
    logger.info("Reading topology from %s", in_itp)
    topo = am.topology.read_itp(str(in_itp))

    mddir = CFG.aa_dir()
    in_pdb = mddir / "topology.pdb"
    in_xtc = mddir / "samples.xtc"
    logger.info("Reading trajectory files from %s", mddir)

    cg_traj = read_cog_trajectory(in_pdb, in_xtc, topo.partitioning, selection="resname ANP")
    # np.save("traj_coords.npy", cg_traj)
    # cg_traj = np.load("traj_coords.npy")

    logger.info("Calculating internal coordinates from trajectory")
    internal_coords = calculate_internal_coordinates(cg_traj, topo)

    inverted_topo = boltzmann_invert_topology(topo, internal_coords)
    # plot_internal_coordinates(internal_coords, topo, output_file=wdir / "png" / "aa.png")

    updated_topo = filter_topology(
        inverted_topo,
        constraint_k_cutoff=CFG.constraint_k_cutoff,
        angle_k_cutoff=CFG.angle_k_cutoff,
        dihedral_k_cutoff=CFG.dihedral_k_cutoff,
    )

    filtered_topo = remove_unstable_dihedrals(
        updated_topo,
        angle_linear_cutoff_deg=160.0,
        drop_if_undefined=True,
    )

    out_itp = wdir / "mapping" / f"{molname}_updated.itp"
    filtered_topo.to_itp(out_file=out_itp)
    logger.info("Updated ITP file written to: %s", out_itp)
