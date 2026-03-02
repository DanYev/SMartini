import copy
import logging
from pathlib import Path

import AutoMartini as am
import numpy as np

from ligpar_config import CFG
from lpmath import (
    boltzmann_inversion_angle,
    boltzmann_inversion_bond,
    calculate_internal_coordinates,
    fit_type9_dihedral,
    fit_type11_cbt_dihedral,
    read_cog_trajectory,
)
from plots import plot_internal_coordinates
from partitioning_patch import patch_topology_partitioning_from_sdf

logger = logging.getLogger("AutoMartini")
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

    angle_lookup = _build_angle_lookup(updated_topo)
    length_lookup = _build_length_lookup(updated_topo)

    def _eq_angle(i, j, k):
        if (i, j, k) in angle_lookup:
            return angle_lookup[(i, j, k)]
        # If this angle isn't explicitly defined in the topology, use the
        # geometric mean from the trajectory (if available).
        adj = internal_coords.get((i, j, k, "adj_angle"))
        if adj is not None and len(adj) > 0:
            return float(np.mean(adj))
        return _angle_from_triangle(length_lookup, i, j, k)

    dihedrals_by_key = {}
    for d in updated_topo.dihedrals:
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        dihedrals_by_key.setdefault(key, []).append(d)

    new_dihedrals = []
    for (i, j, k, l), existing_terms in dihedrals_by_key.items():
        data = internal_coords.get((i, j, k, l, "dihedral"))

        if data is None:
            raise KeyError(f"Missing dihedral samples for ({i},{j},{k},{l})")

        comment = ""
        if existing_terms and len(existing_terms[0]) >= 9:
            comment = existing_terms[0][8]

        # If the dihedral is ill-defined due to near-linear adjacent angles,
        # only fit CBT (funct=11).
        a1 = _eq_angle(i, j, k)
        a2 = _eq_angle(j, k, l)
        ill_defined = (
            a1 is None
            or a2 is None
            or float(a1) >= 160.0
            or float(a2) >= 160.0
        )

        if ill_defined:
            (kphi, a), score11 = fit_type11_cbt_dihedral(
                data,
                temperature=CFG.temperature,
                bins=CFG.type9_bins,
                min_prob=CFG.type9_min_prob,
                return_score=True,
            )
            new_dihedrals.append(
                [
                    i,
                    j,
                    k,
                    l,
                    11,
                    float(kphi),
                    float(a[0]),
                    float(a[1]),
                    float(a[2]),
                    float(a[3]),
                    float(a[4]),
                    comment,
                ]
            )
            continue

        fit_terms9, score9 = fit_type9_dihedral(
            data,
            temperature=CFG.temperature,
            max_n=CFG.type9_max_n,
            bins=CFG.type9_bins,
            min_prob=CFG.type9_min_prob,
            return_score=True,
        )
        (kphi, a), score11 = fit_type11_cbt_dihedral(
            data,
            temperature=CFG.temperature,
            bins=CFG.type9_bins,
            min_prob=CFG.type9_min_prob,
            return_score=True,
        )
        print(f"Dihedral ({i+1},{j+1},{k+1},{l+1}): score9={score9:.4f}, score11={score11:.4f}")

        if score11 < score9:
            new_dihedrals.append(
                [
                    i,
                    j,
                    k,
                    l,
                    11,
                    float(kphi),
                    float(a[0]),
                    float(a[1]),
                    float(a[2]),
                    float(a[3]),
                    float(a[4]),
                    comment,
                ]
            )
        else:
            for mult, k_term, phi0 in fit_terms9:
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
    k_cutoff=0.5,
    angle_linear_cutoff_deg: float = 170.0,
):
    """update/post-process dihedral terms.

    Notes
    -----
    - k_cutoff is applied only to funct=9 terms.
    - duplicate scaling is applied to funct=9 and funct=11 terms.
    - funct=11 dihedrals are produced during inversion; we do not convert types here.
    """
    updated_topo = copy.deepcopy(topo)

    # Drop weak |k|, optionally keep strongest per (i,j,k,l)
    dihedrals_by_key = {}
    for d in updated_topo.dihedrals:
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        dihedrals_by_key.setdefault(key, []).append(d)

    # Some topology generators may emit the same torsion multiple times under
    # different index permutations (same 4 beads). To avoid double counting,
    # rescale force constants by 1/N where N is the number of permutations
    # actually present for that 4-bead set.
    keys = set(dihedrals_by_key)
    perm_group_size = {}
    groups = {}
    for key in keys:
        group = tuple(sorted(key))
        groups.setdefault(group, set()).add(key)
    for group, gkeys in groups.items():
        perm_group_size[group] = len(gkeys)

    new_dihedrals = []
    for key, terms in dihedrals_by_key.items():
        type9 = [t for t in terms if len(t) >= 8 and int(t[4]) == 9]
        other = [t for t in terms if not (len(t) >= 8 and int(t[4]) == 9)]

        kept_for_key = [t for t in type9 if abs(float(t[6])) >= k_cutoff]
        if not kept_for_key and type9:
            kept_for_key = [max(type9, key=lambda t: abs(float(t[6])))]

        group = tuple(sorted(key))
        n_perm = int(perm_group_size.get(group, 1))
        scale = 1.0 / float(n_perm) if n_perm > 0 else 1.0

        # Apply scaling to "other" terms when supported.
        # - funct=11: scale kphi (index 5)
        # - other funct: keep as-is (unknown parameter layout)
        for t in other:
            tt = t.copy()
            try:
                funct = int(tt[4])
            except Exception:
                funct = None
            if funct == 11 and len(tt) >= 6:
                tt[5] = float(tt[5]) * scale
            new_dihedrals.append(tt)
        for t in kept_for_key:
            tt = t.copy()
            tt[6] = float(tt[6]) * scale
            new_dihedrals.append(tt)

    updated_topo.dihedrals = new_dihedrals

    return updated_topo


if __name__ == "__main__":
    import pickle
    molname = MOLNAME
    wdir = CFG.wdir
    out_dir = CFG.out_dir
    logger.info("Starting analysis for molecule: %s", molname)

    in_itp = out_dir / f"{molname}.itp"
    logger.info("Reading topology from %s", in_itp)
    topo = am.topology.read_itp(str(in_itp))

    # Patch partitioning to match the AA atom order (incl. hydrogens) used in
    # systems/<molname>/<molname>.sdf and thus in AA topology.pdb.
    sdf_file = wdir / f"{molname}.sdf"
    topo = patch_topology_partitioning_from_sdf(topo, sdf_file)

    pickle_file = wdir / "internal_coords.pkl"
    if pickle_file.exists():
        logger.info("Loading internal coordinates from %s", pickle_file)
        with open(pickle_file, "rb") as f:
            internal_coords = pickle.load(f)
    else:
        aa_dir = CFG.aa_dir
        aa_pdb = aa_dir / "topology.pdb"
        aa_xtc = aa_dir / "samples.xtc"
        logger.info("Reading AA trajectory from %s", aa_dir)
        aa_traj = read_cog_trajectory(aa_pdb, aa_xtc, topo.partitioning, selection=CFG.aa_selection)
        logger.info("Calculating internal coordinates from AA trajectory")
        internal_coords = calculate_internal_coordinates(aa_traj, topo)
        with open(pickle_file, "wb") as f:
            pickle.dump(internal_coords, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    topo = boltzmann_invert_bonds(topo, internal_coords)
    topo = boltzmann_invert_angles(topo, internal_coords)
    topo = boltzmann_invert_dihedrals(topo, internal_coords)

    topo = update_bonds(topo, k_cutoff=CFG.constraint_k_cutoff)
    topo = update_angles(topo, k_cutoff=CFG.angle_k_cutoff)
    topo = update_dihedrals(topo, k_cutoff=CFG.dihedral_k_cutoff)

    out_itp = out_dir / f"{molname}_updated.itp"
    topo.to_itp(out_file=out_itp)
    logger.info("Updated ITP file written to: %s", out_itp)

    plot_internal_coordinates(
        internal_coords,
        topo,
        output_file=wdir / "png" / "aa.png",
        temperature=CFG.temperature,
        max_gaussians=CFG.type9_max_n,
    )
