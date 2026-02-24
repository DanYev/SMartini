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
    fit_type11_cbt_dihedral,
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
    for bond in topo.bonds:
        if len(bond) >= 4:
            i, j = int(bond[0]), int(bond[1])
            length[_pair_key(i, j)] = float(bond[3])
    for constraint in topo.constraints:
        if len(constraint) >= 4:
            i, j = int(constraint[0]), int(constraint[1])
            length[_pair_key(i, j)] = float(constraint[3])
    return length


def _build_angle_lookup(topo):
    """Return map (i,j,k)->theta0 in degrees (symmetric in i/k)."""
    angle = {}
    for a in topo.angles:
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


def _build_bead_to_rings(topo):
    """Return bead_index -> set(ring_ids) map from topo.ringbeads."""
    bead_to_rings = {}
    for ring_id, ring in enumerate(topo.ringbeads):
        try:
            beads = [int(b) for b in ring]
        except Exception:
            continue
        for bead in beads:
            bead_to_rings.setdefault(bead, set()).add(ring_id)
    return bead_to_rings


def _connects_two_different_rings(i: int, j: int, bead_to_rings) -> bool:
    """Return True if i-j connects two different rings (per topo.ringbeads).

    If `bead_to_rings` is provided, it should be a map bead_index -> set(ring_ids).
    """
    rings_i = bead_to_rings.get(int(i), set())
    rings_j = bead_to_rings.get(int(j), set())
    if not rings_i or not rings_j:
        return False
    return rings_i.isdisjoint(rings_j)


def boltzmann_invert_bonds(
    topo,
    internal_coords,
):
    updated_topo = copy.deepcopy(topo)
    updated_topo.bonds = []  
    updated_topo.constraints = []  

    # Bonds
    for bond in topo.bonds:
        i, j = int(bond[0]), int(bond[1])
        distances = internal_coords[(i, j, "bond")]
        r0_calc, k_calc = boltzmann_inversion_bond(distances)
        comment = bond[5] if len(bond) >= 6 else ""
        updated_topo.bonds.append([i, j, bond[2], float(r0_calc), float(k_calc), comment])

    # Constraints
    for bond in topo.constraints:
        i, j = int(bond[0]), int(bond[1])
        distances = internal_coords[(i, j, "constraint")]
        r0_calc, k_calc = boltzmann_inversion_bond(distances)
        comment = bond[4] if len(bond) >= 5 else ""
        updated_topo.bonds.append([i, j, bond[2], float(r0_calc), float(k_calc), comment])
    
    return updated_topo


def boltzmann_invert_angles(topo, internal_coords):
    updated_topo = copy.deepcopy(topo)

    for idx, angle in enumerate(updated_topo.angles):
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        if (i, j, k, "angle") not in internal_coords:
            continue
        samples = internal_coords[(i, j, k, "angle")]
        theta0_calc, k_calc = boltzmann_inversion_angle(samples)
        comment = angle[6] if len(angle) >= 7 else ""
        updated_topo.angles[idx] = [i, j, k, 10, float(theta0_calc), float(k_calc), comment]

    return updated_topo


def boltzmann_invert_dihedrals(topo, internal_coords):
    updated_topo = copy.deepcopy(topo)

    dihedrals_by_key = {}
    for d in updated_topo.dihedrals:
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        dihedrals_by_key.setdefault(key, []).append(d)

    new_dihedrals = []
    for (i, j, k, l), existing_terms in dihedrals_by_key.items():
        data = internal_coords.get((i, j, k, l, "dihedral"))

        comment = ""
        if existing_terms and len(existing_terms[0]) >= 9:
            comment = existing_terms[0][8]

        fit_terms = fit_type9_dihedral(
            data,
            temperature=CFG.temperature,
            max_n=CFG.type9_max_n,
            bins=CFG.type9_bins,
            min_prob=CFG.type9_min_prob,
        )

        for term in fit_terms:
            mult, k_term, phi0 = term
            new_dihedrals.append(
                [i, j, k, l, 9, float(phi0), float(k_term), int(mult), comment]
            )

    updated_topo.dihedrals = new_dihedrals

    return updated_topo


def update_bonds(
    topo,
    k_cutoff=20000,
):
    """update/post-process bond+constraint terms."""
    updated_topo = copy.deepcopy(topo)
    bead_to_rings = _build_bead_to_rings(updated_topo)
    
    # Move stiff bonds to constraints
    new_bonds = []
    new_constraints = []

    for bond in updated_topo.bonds:
        # Extract fields: [i, j, funct, dist, k, comment?]
        i, j, funct, dist, k = int(bond[0]), int(bond[1]), int(bond[2]), float(bond[3]), float(bond[4])
        comment = bond[5] if len(bond) >= 6 else ""

        # Never convert ring-ring link bonds into constraints.
        if _connects_two_different_rings(i, j, bead_to_rings):
            bond[4] = min(bond[4], k_cutoff)  # boost k to ensure it's kept as a bond
            bond[5] = "ring-ring link"  # update comment
            new_bonds.append(bond)
            continue
        
        if float(k) > k_cutoff:
            new_constraints.append([i, j, funct, dist, comment])
        else:
            new_bonds.append(bond)

    updated_topo.bonds = new_bonds
    updated_topo.constraints = new_constraints

    return updated_topo


def update_angles(
    topo,
    k_cutoff=25.0,
    theta_cutoff=165.0
):
    """update/post-process angle terms."""
    updated_topo = copy.deepcopy(topo)
    updated_topo.angles = [a for a in updated_topo.angles if float(a[5]) >= float(k_cutoff)]

    # Change the type to 1 if theta > cutoff, to avoid numerical instability in CG MD.
    for angle in updated_topo.angles:
        if float(angle[4]) > float(theta_cutoff):
            angle[3] = 1

    return updated_topo


def update_dihedrals(
    topo,
    internal_coords=None,
    k_cutoff=5,
    angle_linear_cutoff_deg: float = 170.0,
):
    """update/post-process dihedral terms.

    Notes
    -----
    - k_cutoff/symmetric scaling are applied only to funct=9 terms.
    - Dihedrals that are ill-defined due to (near-)linear adjacent angles are converted to
      funct=11 (combined bending-torsion) using trajectory-derived coefficients.
    """
    updated_topo = copy.deepcopy(topo)

    # Drop weak |k|, optionally keep strongest per (i,j,k,l)
    dihedrals_by_key = {}
    for d in updated_topo.dihedrals:
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        dihedrals_by_key.setdefault(key, []).append(d)

    # If both (i,j,k,l) and (i,k,j,l) exist, they represent a symmetric dihedral.
    # Rescale force constants to avoid double counting.
    keys = set(dihedrals_by_key)
    symmetric_keys = {
        key
        for key in keys
        if (key[0], key[2], key[1], key[3]) in keys
    }

    new_dihedrals = []
    for key, terms in dihedrals_by_key.items():
        type9 = [t for t in terms if len(t) >= 8 and int(t[4]) == 9]
        other = [t for t in terms if not (len(t) >= 8 and int(t[4]) == 9)]

        kept_for_key = [t for t in type9 if abs(float(t[6])) >= k_cutoff]
        if not kept_for_key and type9:
            kept_for_key = [max(type9, key=lambda t: abs(float(t[6])))]

        new_dihedrals.extend(other)

        scale = 0.5 if key in symmetric_keys else 1.0
        for t in kept_for_key:
            tt = t.copy()
            tt[6] = float(tt[6]) * scale
            new_dihedrals.append(tt)

    updated_topo.dihedrals = new_dihedrals

    # Convert unstable dihedrals that become ill-defined due to near-linear adjacent angles.
    angle_lookup = _build_angle_lookup(updated_topo)
    length_lookup = _build_length_lookup(updated_topo)

    def _eq_angle(i, j, k):
        if (i, j, k) in angle_lookup:
            return angle_lookup[(i, j, k)]
        return _angle_from_triangle(length_lookup, i, j, k)

    kept = []
    converted_linear = 0
    converted_undefined = 0

    for d in updated_topo.dihedrals:
        i, j, k, l = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        funct = int(d[4])

        # Only convert funct=9 terms; leave other dihedral types alone.
        if funct != 9:
            kept.append(d)
            continue

        a1 = _eq_angle(i, j, k)
        a2 = _eq_angle(j, k, l)
        if a1 is None or a2 is None:
            converted_undefined += 1
            if internal_coords is None:
                raise ValueError(
                    f"Need internal_coords to convert dihedral ({i},{j},{k},{l}) to funct=11"
                )
            phi = internal_coords.get((i, j, k, l, "dihedral"))
            if phi is None:
                raise KeyError(
                    f"Missing samples for CBT fit of dihedral ({i},{j},{k},{l}): "
                    f"phi={phi is not None}"
                )
            kphi, a = fit_type11_cbt_dihedral(
                phi,
                temperature=CFG.temperature,
                bins=max(60, int(CFG.type9_bins)),
                min_prob=float(CFG.type9_min_prob),
            )
            comment = d[-1] if (len(d) > 0 and isinstance(d[-1], str)) else ""
            d2 = [i, j, k, l, 11, float(kphi), float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), comment]
            kept.append(d2)
            continue
        if a1 >= angle_linear_cutoff_deg or a2 >= angle_linear_cutoff_deg:
            converted_linear += 1
            if internal_coords is None:
                raise ValueError(
                    f"Need internal_coords to convert dihedral ({i},{j},{k},{l}) to funct=11"
                )
            phi = internal_coords.get((i, j, k, l, "dihedral"))
            if phi is None:
                raise KeyError(
                    f"Missing samples for CBT fit of dihedral ({i},{j},{k},{l}): "
                    f"phi={phi is not None}"
                )
            kphi, a = fit_type11_cbt_dihedral(
                phi,
                temperature=CFG.temperature,
                bins=max(60, int(CFG.type9_bins)),
                min_prob=float(CFG.type9_min_prob),
            )
            comment = d[-1] if (len(d) > 0 and isinstance(d[-1], str)) else ""
            d2 = [i, j, k, l, 11, float(kphi), float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), comment]
            kept.append(d2)
            continue
        kept.append(d)

    updated_topo.dihedrals = kept
    logger.info(
        "updated dihedrals: kept=%s, converted_linear=%s (>%s deg), converted_undefined=%s",
        len(kept),
        converted_linear,
        angle_linear_cutoff_deg,
        converted_undefined,
    )

    return updated_topo


if __name__ == "__main__":
    import pickle
    molname = MOLNAME
    wdir = CFG.wdir()
    logger.info("Starting analysis for molecule: %s", molname)

    in_itp = wdir / "mapping" / f"{molname}.itp"
    logger.info("Reading topology from %s", in_itp)
    topo = am.topology.read_itp(str(in_itp))

    # mddir = CFG.aa_dir()
    # in_pdb = mddir / "topology.pdb"
    # in_xtc = mddir / "samples.xtc"
    # logger.info("Reading trajectory files from %s", mddir)
    # cg_traj = read_cog_trajectory(in_pdb, in_xtc, topo.partitioning, stop=2000)

    # logger.info("Calculating internal coordinates from trajectory")
    # internal_coords = calculate_internal_coordinates(cg_traj, topo)
    # with open("internal_coords.pkl", "wb") as f:
    #     pickle.dump(internal_coords, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("internal_coords.pkl", "rb") as f:
        internal_coords = pickle.load(f)
    topo = boltzmann_invert_bonds(topo, internal_coords)
    topo = boltzmann_invert_angles(topo, internal_coords)
    topo = boltzmann_invert_dihedrals(topo, internal_coords)

    topo = update_bonds(topo, k_cutoff=CFG.constraint_k_cutoff)
    topo = update_angles(topo, k_cutoff=CFG.angle_k_cutoff)
    topo = update_dihedrals(
        topo,
        internal_coords=internal_coords,
        k_cutoff=CFG.dihedral_k_cutoff,
        angle_linear_cutoff_deg=160.0,
    )

    out_itp = wdir / "mapping" / f"{molname}_updated.itp"
    topo.to_itp(out_file=out_itp)
    logger.info("Updated ITP file written to: %s", out_itp)

    plot_internal_coordinates(
        internal_coords,
        topo,
        output_file=wdir / "png" / "aa.png",
        temperature=CFG.temperature,
    )
