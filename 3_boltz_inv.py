import copy
import logging
import pickle
import os
import sys

import AutoMartini as am
import numpy as np

from pathlib import Path
from config import CFG
from AutoMartini.lpmath import (
    read_cog_trajectory,
    calculate_internal_coordinates,
    boltzmann_inversion_bond,
    fit_type1_angle,
    fit_type1_angle,
    fit_type2_angle,
    fit_type10_angle,
    fit_type9_dihedral,
    fit_type11_dihedral,
)
from AutoMartini.plots import plot_internal_coordinates

logger = logging.getLogger("AutoMartini")
logger.setLevel(logging.INFO)

################################################################################
### Helper Functions for Topology Analysis and Post-processing ###
################################################################################

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


def _build_bead_bond_degree(topo):
    """Return bead index -> bonded degree map using bonds and constraints."""
    bead_bond_degree = {}
    for bond in topo.bonds:
        for idx in (0, 1):
            bead = int(bond[idx])
            bead_bond_degree[bead] = bead_bond_degree.get(bead, 0) + 1
    for constraint in topo.constraints:
        for idx in (0, 1):
            bead = int(constraint[idx])
            bead_bond_degree[bead] = bead_bond_degree.get(bead, 0) + 1
    return bead_bond_degree


def _build_ring_bead_set(topo):
    """Return set of all bead indices that belong to any ring."""
    ring_beads = set()
    for ring in topo.ringbeads:
        for bead in ring:
            ring_beads.add(int(bead))
    return ring_beads


def _is_linear_fragment(i: int, j: int, k: int, bead_bond_degree, ring_beads) -> bool:
    """Return True for 3-bead linear fragments outside rings.

    Criterion: each bead in i-j-k has degree <= 2 and none is in a ring.
    """
    return (
        bead_bond_degree.get(int(i), 0) <= 2
        and bead_bond_degree.get(int(j), 0) <= 2
        and bead_bond_degree.get(int(k), 0) <= 2
        and int(i) not in ring_beads
        and int(j) not in ring_beads
        and int(k) not in ring_beads
    )


def _connects_two_different_rings(i: int, j: int, bead_to_rings) -> bool:
    """Return True if i-j connects two different rings (per topo.ringbeads).

    If `bead_to_rings` is provided, it should be a map bead_index -> set(ring_ids).
    """
    rings_i = bead_to_rings.get(int(i), set())
    rings_j = bead_to_rings.get(int(j), set())
    if not rings_i or not rings_j:
        return False
    return rings_i.isdisjoint(rings_j)


def flat_set(lst):
    """Flatten a list of lists into a set of unique elements."""
    if not lst:
        return set()
    aset = set(item for sublist in lst for item in sublist) 
    return aset

#################################################################################
### Main Inversion Logic ###
#################################################################################

def boltzmann_invert_bonds(
    topo,
    internal_coords,
):
    """Fit bond/constraint parameters from sampled bond-length distributions.

    Parameters
    ----------
    topo : object
        Input topology object containing ``bonds`` and ``constraints`` tables.
    internal_coords : dict
        Internal-coordinate time series produced by
        ``calculate_internal_coordinates``.

    Returns
    -------
    tuple
        ``(updated_topo, fit_cache)`` where:
        - ``updated_topo`` has re-fitted bond entries,
        - ``fit_cache["bonds"]`` stores fitted 1D densities for plotting.

    Notes
    -----
    Constraints are currently reinserted into ``updated_topo.bonds`` using
    bond-like harmonic parameters derived from constraint samples.
    """
    updated_topo = copy.deepcopy(topo)
    updated_topo.bonds = []  
    updated_topo.constraints = []  

    fit_cache = {"bonds": {}}

    # Bonds
    for bond in topo.bonds:
        i, j = int(bond[0]), int(bond[1])
        distances = internal_coords[(i, j, "bond")]
        r0_calc, k_calc, density = boltzmann_inversion_bond(distances, temperature=CFG.temperature)
        comment = bond[5] if len(bond) >= 6 else ""
        updated_topo.bonds.append([i, j, bond[2], float(r0_calc), float(k_calc), comment])
        if density is not None:
            fit_cache["bonds"][(i, j, "bond")] = {
                "density": list(map(float, density))
            }

    # Constraints
    for bond in topo.constraints:
        i, j = int(bond[0]), int(bond[1])
        distances = internal_coords[(i, j, "constraint")]
        r0_calc, k_calc, density = boltzmann_inversion_bond(
            distances, temperature=CFG.temperature, fc_scale=CFG.fc_scale)
        comment = bond[4] if len(bond) >= 5 else ""
        updated_topo.bonds.append([i, j, bond[2], float(r0_calc), float(k_calc), comment])
        if density is not None:
            fit_cache["bonds"][(i, j, "constraint")] = {
                "density": list(map(float, density))
            }
    
    return updated_topo, fit_cache


def boltzmann_invert_angles(topo, internal_coords):
    """Fit angle parameters and select angle functional form per triplet.

    Parameters
    ----------
    topo : object
        Input topology object containing ``angles``.
    internal_coords : dict
        Internal-coordinate time series produced by
        ``calculate_internal_coordinates``.

    Returns
    -------
    tuple
        ``(updated_topo, fit_cache)`` where ``fit_cache["angles"]`` stores
        histogram densities used during fitting.

    Notes
    -----
    Selection logic:
    - linear, non-ring 3-bead fragments are fit with type-1 harmonic angles;
    - otherwise type-10 restricted-bending is attempted;
    - fallback to type-1 occurs for near-linear or weak type-10 fits.
    """
    updated_topo = copy.deepcopy(topo)
    fit_cache = {"angles": {}}
    bead_bond_degree = _build_bead_bond_degree(updated_topo)
    ring_beads = _build_ring_bead_set(updated_topo)

    for idx, angle in enumerate(updated_topo.angles):
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        samples = internal_coords[(i, j, k, "angle")]

        if _is_linear_fragment(i, j, k, bead_bond_degree, ring_beads):
            (theta0_calc, k_calc), density = fit_type1_angle(
                samples,
                temperature=CFG.temperature,
                fc_scale=CFG.fc_scale,
            )
            angle_funct = 1
        else:
            theta0_calc, k_calc, density = fit_type10_angle(
                samples,
                temperature=CFG.temperature,
                fc_scale=CFG.fc_scale,
            )
            # Use harmonic angle fitting for terms that will be represented as funct=1.
            if float(theta0_calc) > float(CFG.ill_defined_angle_cutoff) or float(k_calc) < float(CFG.angle_k_lower):
                (theta0_calc, k_calc), density = fit_type1_angle(
                    samples,
                    temperature=CFG.temperature,
                    fc_scale=CFG.fc_scale,
                )
                angle_funct = 1
            else:
                angle_funct = 10

        comment = angle[6] if len(angle) >= 7 else ""
        k_calc = min(float(k_calc), CFG.angle_k_upper) 
        k_calc = max(float(k_calc), CFG.angle_k_lower)
        updated_topo.angles[idx] = [i, j, k, angle_funct, float(theta0_calc), float(k_calc), comment]
        
        # Store in fit cache for plotting
        if density is not None:
            ik0, ik1 = (int(i), int(k))
            if ik0 > ik1: ik0, ik1 = ik1, ik0
            fit_cache["angles"][(ik0, int(j), ik1, "angle")] = {
                "density": list(map(float, density))
            }

    return updated_topo, fit_cache


def boltzmann_invert_ill_defined_dihedrals(topo, internal_coords, ):
    """Re-fit only ill-defined torsions using GROMACS type-11 (CBT).

    This routine is intentionally conservative: it keeps existing type-9
    torsions unchanged and only rebuilds torsions that are geometrically
    ill-defined due to near-linear adjacent angles.

    Parameters
    ----------
    topo : object
        Input topology object containing ``angles`` and ``dihedrals``.
    internal_coords : dict
        Internal-coordinate trajectories including ``("dihedral")`` and,
        optionally, auxiliary ``("adj_angle")`` series.

    Returns
    -------
    tuple
        ``(updated_topo, fit_cache)`` where:
        - ``updated_topo.dihedrals`` contains only newly produced type-11 terms
          for the ill-defined torsions encountered in this pass,
        - ``fit_cache["dihedrals"]`` stores fitted dihedral densities.

    Dihedral type selection
    -----------------------
    For each torsion ``(i,j,k,l)``:
    1. Resolve adjacent equilibrium angles ``(i,j,k)`` and ``(j,k,l)`` from
       topology, then fallback to ``adj_angle`` trajectory means, then fallback
       to a bond-length triangle estimate.
    2. Mark as ill-defined when either adjacent angle cannot be inferred.
    3. Also mark as ill-defined when either adjacent angle is in
       ``linear_angle_set`` (angle funct in ``{1,2}``, treated as linear-style).
    4. If ill-defined: fit CBT (type-11) via ``fit_type11_dihedral`` and write
       one term ``[i,j,k,l,11,kphi,a0..a4,comment]``.
    5. If not ill-defined: keep existing terms as-is in this specialized mode.

    Notes
    -----
    The fitted ``kphi`` is currently down-scaled by ``0.1`` after fitting.
    """

    def get_eq_angle(i, j, k):
        if (i, j, k) in angle_lookup:
            return angle_lookup[(i, j, k)]
        # If this angle isn't explicitly defined in the topology, use the
        # geometric mean from the trajectory (if available).
        adj = internal_coords.get((i, j, k, "adj_angle"))
        if adj is not None and len(adj) > 0:
            return float(np.mean(adj))
        return _angle_from_triangle(length_lookup, i, j, k)
        
    updated_topo = copy.deepcopy(topo)
    fit_cache = {"dihedrals": {}}

    angle_lookup = _build_angle_lookup(updated_topo)
    length_lookup = _build_length_lookup(updated_topo)

    # Collect angles already marked as linear-fragment style (funct=2),
    # while also accepting funct=1 for backward compatibility.
    linear_angle_set = set()
    for a in updated_topo.angles:
        if int(a[3]) in (1, 2):
            ai, aj, ak = int(a[0]), int(a[1]), int(a[2])
            linear_angle_set.add((ai, aj, ak))
            linear_angle_set.add((ak, aj, ai))

    dihedrals_by_key = {}
    for d in updated_topo.dihedrals:
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        dihedrals_by_key.setdefault(key, []).append(d)

    new_dihedrals = []
    for (i, j, k, l), existing_terms in dihedrals_by_key.items():
        dih_type = existing_terms[0][4] 
        if dih_type == 9:  
            new_dihedrals.extend(existing_terms)
            continue
        data = internal_coords.get((i, j, k, l, "dihedral"))

        if data is None:
            raise KeyError(f"Missing dihedral samples for ({i},{j},{k},{l})")

        comment = ""
        if existing_terms and len(existing_terms[0]) >= 9:
            comment = existing_terms[0][8]

        # If the dihedral is ill-defined due to near-linear adjacent angles,
        # only fit CBT (funct=11).
        a1 = get_eq_angle(i, j, k)
        a2 = get_eq_angle(j, k, l)
        ill_defined = (a1 is None or a2 is None)
        # If either flanking angle is a linear-fragment angle (type 2),
        # the dihedral is ill-defined and must use type 11 (CBT).
        has_linear_angle = (
            (i, j, k) in linear_angle_set
            or (j, k, l) in linear_angle_set
        )
        ill_defined = ill_defined or has_linear_angle

        if ill_defined:
            (kphi, a), density = fit_type11_dihedral(
                data,
                temperature=CFG.temperature,
                bins=CFG.nbins,
                min_prob=CFG.min_prob,
            )
            theta_1 = np.radians(a1)
            theta_2 = np.radians(a2)
            # scale = max(np.sin(theta_1) ** 3 * np.sin(theta_2) ** 3, 5e-2)
            kphi *= 0.1
            new_dihedrals.append([i, j, k, l, 11, kphi, a[0], a[1], a[2], a[3], a[4], comment])
            fit_cache["dihedrals"][(i, j, k, l, "dihedral")] = {"density": list(map(float, density))}
            continue

    updated_topo.dihedrals = new_dihedrals
    return updated_topo, fit_cache


def boltzmann_invert_dihedrals(topo, 
    internal_coords, 
    angle_cutoff: float = 150.0, 
    ):
    """Fit dihedrals by routing each torsion to type-9 or type-11 fitting.

    Parameters
    ----------
    topo : object
        Input topology object containing ``angles`` and ``dihedrals``.
    internal_coords : dict
        Internal-coordinate time series with dihedral samples.
    angle_cutoff : float, optional
        Reserved threshold parameter; currently not used directly in the
        decision logic.

    Returns
    -------
    tuple
        ``(updated_topo, fit_cache)`` where ``updated_topo.dihedrals`` is
        rebuilt from fitted terms and ``fit_cache["dihedrals"]`` stores
        histogram densities used for PMF fitting.

    Type-9 vs type-11 logic
    -----------------------
    Each unique torsion key ``(i,j,k,l)`` is processed independently.

    **Step 1: infer adjacent angle context**
    - Retrieve equilibrium angles for ``(i,j,k)`` and ``(j,k,l)`` from
      topology.
    - If missing, fallback to trajectory-derived ``adj_angle`` means.
    - Final fallback is geometric inference from bond/constraint lengths.

    **Step 2: classify ill-defined torsions**
    A torsion is classified ill-defined if either:
    - an adjacent angle could not be inferred, or
    - an adjacent angle is tagged as linear-style (funct in ``{1,2}``) and
      ``CFG.use_type11_for_linear`` is enabled.

    **Step 3: fit according to class**
    - ill-defined torsions -> ``fit_type11_dihedral``:
      produces one CBT term ``[i,j,k,l,11,kphi,a0..a4,comment]``.
    - regular torsions -> ``fit_type9_dihedral``:
      produces one or more Fourier terms
      ``[i,j,k,l,9,phi0,k,multiplicity,comment]``.

    Rationale
    ---------
    Type-9 periodic Fourier terms are robust for well-defined torsional axes.
    Near linear adjacent angles make torsional azimuth poorly defined; in that
    regime type-11 (CBT polynomial form) is used as a numerically stable
    surrogate representation.
    """

    def get_eq_angle(i, j, k):
        if (i, j, k) in angle_lookup:
            return angle_lookup[(i, j, k)]
        # If this angle isn't explicitly defined in the topology, use the
        # geometric mean from the trajectory (if available).
        adj = internal_coords.get((i, j, k, "adj_angle"))
        if adj is not None and len(adj) > 0:
            return float(np.mean(adj))
        return _angle_from_triangle(length_lookup, i, j, k)
        
    updated_topo = copy.deepcopy(topo)
    fit_cache = {"dihedrals": {}}

    angle_lookup = _build_angle_lookup(updated_topo)
    length_lookup = _build_length_lookup(updated_topo)

    # Collect angles already marked as linear-fragment style (funct=2),
    # while also accepting funct=1 for backward compatibility.
    linear_angle_set = set()
    for a in updated_topo.angles:
        if int(a[3]) in (1, 2):
            ai, aj, ak = int(a[0]), int(a[1]), int(a[2])
            linear_angle_set.add((ai, aj, ak))
            linear_angle_set.add((ak, aj, ai))

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
        a1 = get_eq_angle(i, j, k)
        a2 = get_eq_angle(j, k, l)
        ill_defined = (a1 is None or a2 is None)
        # If either flanking angle is a linear-fragment angle (type 2),
        # the dihedral is ill-defined and must use type 11 (CBT).
        has_linear_angle = (
            (i, j, k) in linear_angle_set
            or (j, k, l) in linear_angle_set
        )
        ill_defined = ill_defined or (has_linear_angle and CFG.use_type11_for_linear)

        if ill_defined:
            (kphi, a), density = fit_type11_dihedral(
                data,
                temperature=CFG.temperature,
                nbins=CFG.nbins,
                min_prob=CFG.min_prob,
            )
            theta_1 = np.radians(a1)
            theta_2 = np.radians(a2)
            scale = max(np.sin(theta_1) ** 3 * np.sin(theta_2) ** 3, 0.1)
            kphi /= scale
            new_dihedrals.append([i, j, k, l, 11, kphi, a[0], a[1], a[2], a[3], a[4], comment])
            fit_cache["dihedrals"][(i, j, k, l, "dihedral")] = {"density": list(map(float, density))}
            continue

        fit_terms9, density9 = fit_type9_dihedral(
            data,
            temperature=CFG.temperature,
            max_n=CFG.type9_max_n,
            nbins=CFG.nbins,
            min_prob=CFG.min_prob,
            fc_scale=CFG.fc_scale,
        )

        for mult, k_term, phi0 in fit_terms9:
            new_dihedrals.append(
                [i, j, k, l, 9, float(phi0), float(k_term), int(mult), comment]
            )
        density_out = density9
        fit_cache["dihedrals"][(i, j, k, l, "dihedral")] = {"density": list(map(float, density_out))}

    updated_topo.dihedrals = new_dihedrals
    return updated_topo, fit_cache


def update_bonds(
    topo,
    k_cutoff=20000,
):
    """Post-process bond terms by moving very stiff bonds to constraints.

    Parameters
    ----------
    topo : object
        Topology containing ``bonds`` and ``constraints``.
    k_cutoff : float, optional
        Harmonic bond force-constant threshold above which a bond is converted
        to a constraint.

    Returns
    -------
    object
        Updated topology with filtered ``bonds`` and rebuilt ``constraints``.

    Notes
    -----
    Bonds identified as links between two different rings are never converted
    to constraints.
    """
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
    angle_cutoff=150.0
):
    """Placeholder for angle post-processing.

    Parameters
    ----------
    topo : object
        Topology containing ``angles``.
    angle_cutoff : float, optional
        Reserved threshold for potential future filtering logic.

    Returns
    -------
    object
        Currently returns a deep-copied topology unchanged.
    """
    updated_topo = copy.deepcopy(topo)

    bead_bond_degree = _build_bead_bond_degree(updated_topo)
    ring_beads = _build_ring_bead_set(updated_topo)

    return updated_topo


def update_dihedrals(
    topo,
    k_cutoff=0.5,
    angle_cutoff: float = 150.0,
):
    """update/post-process dihedral terms.

    Notes
    -----
    - k_cutoff is applied only to funct=9 terms.
    - duplicate scaling is applied to funct=9 and funct=11 terms.
    - funct=11 dihedrals are produced during inversion; we do not convert types here.
        - scaling is ``CFG.fc_scale / n_perm`` where ``n_perm`` is the number of
            key permutations sharing the same sorted bead tuple.
    """
    updated_topo = copy.deepcopy(topo)

    # Drop weak |k|, optionally keep strongest per (i,j,k,l)
    dihedrals_by_key = {}
    for d in updated_topo.dihedrals:
        key = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
        dihedrals_by_key.setdefault(key, []).append(d)

    # Some dihedrals are "duplicated"
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
        scale = CFG.fc_scale / float(n_perm)

        # Apply scaling to "other" terms when supported.
        # - funct=11: scale kphi (index 5)
        # - other funct: keep as-is (unknown parameter layout)
        for t in other:
            tt = t.copy()
            funct = int(tt[4])
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
    molname = CFG.molname
    wdir = CFG.wdir
    mol_dir = CFG.mol_dir
    logger.info("Starting analysis for molecule: %s", molname)

    in_itp = mol_dir / f"{molname}_initial.itp"
    logger.info("Reading topology from %s", in_itp)
    topo = am.topology.read_itp(str(in_itp))

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
        aa_traj = read_cog_trajectory(aa_pdb, aa_xtc, topo.aa_mapping, 
            selection=CFG.aa_selection, stop=2000)
        logger.info("Calculating internal coordinates from AA trajectory")
        internal_coords = calculate_internal_coordinates(aa_traj, topo)
        with open(pickle_file, "wb") as f:
            pickle.dump(internal_coords, f, protocol=pickle.HIGHEST_PROTOCOL)

    master_fit_cache = {"bonds": {}, "angles": {}, "dihedrals": {}}

    if "only_ill_defined" in sys.argv:
        logger.info("Running only ill-defined dihedral fitting...")
        in_itp = mol_dir / f"{molname}.itp"
        topo = am.topology.read_itp(str(in_itp))
        topo, fit_cache = boltzmann_invert_ill_defined_dihedrals(topo, internal_coords)
        master_fit_cache["dihedrals"].update(fit_cache["dihedrals"])
        out_itp = mol_dir / f"{molname}.itp"
        topo.to_itp(out_file=out_itp)
        logger.info("Updated ITP file written to: %s", out_itp)
        exit(0)

    # BOONDS        
    logger.info("Fitting bonds and constraints...")
    topo, bond_cache = boltzmann_invert_bonds(topo, internal_coords)
    master_fit_cache["bonds"].update(bond_cache["bonds"])
    topo = update_bonds(topo, k_cutoff=CFG.bond_k_upper)
    
    # ANGLES
    logger.info("Fitting angles...")
    topo, angle_cache = boltzmann_invert_angles(topo, internal_coords)
    master_fit_cache["angles"].update(angle_cache["angles"])
    topo = update_angles(topo, angle_cutoff=CFG.ill_defined_angle_cutoff)
    
    # DIHEDRALS
    logger.info("Fitting dihedrals...")
    topo, dih_cache = boltzmann_invert_dihedrals(topo, internal_coords, angle_cutoff=CFG.ill_defined_angle_cutoff)
    master_fit_cache["dihedrals"].update(dih_cache["dihedrals"])
    topo = update_dihedrals(topo, k_cutoff=CFG.dihedral_k_lower, angle_cutoff=CFG.ill_defined_angle_cutoff)

    # Save the fit cache
    fit_cache_file = wdir / "fit_cache.pkl"
    os.remove(fit_cache_file) if fit_cache_file.exists() else None
    logger.info("Saving fit cache to %s", fit_cache_file)
    with open(fit_cache_file, "wb") as f:
        pickle.dump(master_fit_cache, f)

    out_itp = mol_dir / f"{molname}.itp"
    topo.to_itp(out_file=out_itp)
    logger.info("Updated ITP file written to: %s", out_itp)

    if "plot" in sys.argv:
        plot_internal_coordinates(
            internal_coords,
            topo,
            output_file=wdir / "png" / "aa.png",
            temperature=CFG.temperature,
            cache_file=fit_cache_file,
        )
