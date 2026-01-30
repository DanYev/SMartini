# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""Cython-accelerated helpers for hot loops in `auto_martiniM3.optimization`.

This module intentionally mirrors a small subset of pure-Python functions to
speed up the CG bead trial filtering stage.

Currently implemented:
    - `check_beads_cy_np(...)`: fast acceptance check for a trial combination
    - `find_acceptable_trials_cy_np(...)`: filter many trial combinations

Notes
-----
* This implementation avoids RDKit objects and focuses on integer-heavy loops.
* This module expects NumPy arrays (or compatible typed memoryviews) as inputs.
    Type/shape normalization should be handled in Python.
"""

cimport cython
from cython.parallel cimport prange
from libc.math cimport exp

cimport numpy as cnp
import numpy as np

cnp.import_array()


ctypedef cnp.int32_t I32
ctypedef cnp.float32_t F32
ctypedef cnp.uint8_t U8


# ---------------------------------------------------------------------------
# Energy evaluation (moved from energy_cy.pyx / optimization_energy_cy.pyx)
# ---------------------------------------------------------------------------


cdef inline F32 _sigma_for_pair(
    int bead1,
    int bead2,
    const U8[::1] in_ring,
    F32 rvdw,
    F32 rvdw_aromatic,
    F32 rvdw_cross,
) nogil:
    cdef bint b1 = in_ring[bead1] != 0
    cdef bint b2 = in_ring[bead2] != 0
    if b1 and b2:
        return rvdw_aromatic
    if b1 != b2:
        return rvdw_cross
    return rvdw


cpdef F32 gaussian_overlap(
    F32 dist,
    int bead1,
    int bead2,
    const U8[::1] in_ring,
    F32 bd_bd_overlap_coeff,
    F32 rvdw,
    F32 rvdw_aromatic,
    F32 rvdw_cross,
):
    cdef F32 sigma = _sigma_for_pair(bead1, bead2, in_ring, rvdw, rvdw_aromatic, rvdw_cross)
    return bd_bd_overlap_coeff * exp(-(dist * dist) / (4.0 * sigma * sigma))


cpdef tuple atoms_in_gaussian(
    int bead_id,
    const U8[::1] in_ring,
    const F32[:, ::1] bond_dists,
    const F32[::1] masses,
    F32 at_in_bd_coeff,
    F32 rvdw,
    F32 rvdw_aromatic,
):
    cdef Py_ssize_t n = bond_dists.shape[0]
    cdef Py_ssize_t i
    cdef F32 sigma = rvdw_aromatic if in_ring[bead_id] != 0 else rvdw
    cdef F32 sigma2 = sigma * sigma
    cdef F32 weight_sum = 0.0
    lumped_atoms = []
    cdef F32 dist

    for i in range(n):
        dist = bond_dists[i, bead_id]
        if dist < sigma:
            lumped_atoms.append(i)
        weight_sum -= masses[i] * exp(-(dist * dist) / (2.0 * sigma2))

    return at_in_bd_coeff * weight_sum, lumped_atoms


cpdef F32 penalize_lonely_atoms(
    list lumped_atoms,
    const F32[::1] masses,
    F32 lonely_atom_penalize,
):
    cdef Py_ssize_t n = masses.shape[0]
    cdef Py_ssize_t i
    cdef F32 weight_sum = 0.0

    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mask = np.zeros(n, dtype=np.uint8)
    for i in lumped_atoms:
        if 0 <= i < n:
            mask[i] = 1

    for i in range(n):
        if mask[i] == 0:
            weight_sum += masses[i]

    return lonely_atom_penalize * weight_sum


cpdef F32 eval_gaussian_interac(
    const I32[::1] list_beads,
    const U8[::1] in_ring,
    const F32[:, ::1] bond_dists,
    const F32[::1] masses,
    # bead params as scalars
    F32 offset_bd_weight,
    F32 offset_bd_aromatic_weight,
    F32 lonely_atom_penalize,
    F32 bd_bd_overlap_coeff,
    F32 at_in_bd_coeff,
    F32 rvdw,
    F32 rvdw_aromatic,
    F32 rvdw_cross,
):
    cdef Py_ssize_t nb = list_beads.shape[0]
    cdef Py_ssize_t i, j
    cdef int bead1, bead2
    cdef int num_aromatics = 0
    cdef F32 weight_sum = 0.0
    cdef F32 weight_overlap = 0.0
    cdef F32 weight_at_in_bd = 0.0

    for i in range(nb):
        bead1 = <int>list_beads[i]
        if in_ring[bead1] != 0:
            num_aromatics += 1

    weight_sum += offset_bd_weight * (nb - num_aromatics) + offset_bd_aromatic_weight * num_aromatics

    for i in range(nb):
        bead1 = <int>list_beads[i]
        for j in range(i + 1, nb):
            bead2 = <int>list_beads[j]
            weight_overlap += gaussian_overlap(
                bond_dists[bead1, bead2],
                bead1,
                bead2,
                in_ring,
                bd_bd_overlap_coeff,
                rvdw,
                rvdw_aromatic,
                rvdw_cross,
            )
    weight_sum += weight_overlap

    lumped_atoms_all = []
    for i in range(nb):
        bead1 = <int>list_beads[i]
        weight, lumped = atoms_in_gaussian(
            bead1,
            in_ring,
            bond_dists,
            masses,
            at_in_bd_coeff,
            rvdw,
            rvdw_aromatic,
        )
        weight_at_in_bd += weight
        for j in lumped:
            if j not in lumped_atoms_all:
                lumped_atoms_all.append(j)

    weight_sum += weight_at_in_bd
    weight_sum += penalize_lonely_atoms(lumped_atoms_all, masses, lonely_atom_penalize)
    return weight_sum


cdef inline bint _is_bond_mv(int a, int b, const I32[:, ::1] bonds) nogil:
    """Return True if (a,b) appears in bonds (either direction)."""
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if (bonds[k, 0] == a and bonds[k, 1] == b) or (bonds[k, 0] == b and bonds[k, 1] == a):
            return True
    return False


cdef inline int _degree_in_bonds_mv(int atom, const I32[:, ::1] bonds) nogil:
    """Count occurrences of `atom` in bonds."""
    cdef int deg = 0
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if bonds[k, 0] == atom or bonds[k, 1] == atom:
            deg += 1
    return deg


cdef inline int _partner_for_terminal_mv(int terminal_atom, const I32[:, ::1] bonds) nogil:
    """Return the partner atom bonded to a terminal atom (degree==1)."""
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if bonds[k, 0] == terminal_atom:
            return <int>bonds[k, 1]
        if bonds[k, 1] == terminal_atom:
            return <int>bonds[k, 0]
    return -1


cdef inline bint _has_terminal_partner_collision_mv(const I32[::1] trial_comb, const I32[:, ::1] bonds) nogil:
    """Return True if two terminal atoms in `trial_comb` share the same partner."""
    cdef Py_ssize_t bi, bj
    cdef int ai, aj
    cdef int partneri, partnerj
    cdef int n_trial = <int>trial_comb.shape[0]
    for bi in range(n_trial):
        ai = <int>trial_comb[bi]
        if _degree_in_bonds_mv(ai, bonds) != 1:
            continue
        partneri = _partner_for_terminal_mv(ai, bonds)
        if partneri == -1:
            continue
        for bj in range(bi + 1, n_trial):
            aj = <int>trial_comb[bj]
            if _degree_in_bonds_mv(aj, bonds) != 1:
                continue
            partnerj = _partner_for_terminal_mv(aj, bonds)
            if partnerj != -1 and partneri == partnerj:
                return True
    return False


cdef bint check_beads(
    const I32[::1] trial_comb,
    const I32[:, ::1] listbonds,
    const I32[::1] ring_id_of_atom,
) nogil:
    """Fast bead-placement acceptance check (GIL-free).

    Parameters
    ----------
    trial_comb
        1D array of atom indices for bead centers.
    listbonds
        2D array (nbonds, 2) of heavy-atom bonds.
    ring_id_of_atom
        1D array indexed by atom-id. Value is ring-id (0..nrings-1) or -1.

    Returns
    -------
    bool
    """
    cdef Py_ssize_t bi, bj
    cdef int ai, aj
    cdef int n_trial = trial_comb.shape[0]
    cdef int rid_i, rid_j
    cdef int k, m
    cdef int nrings = 0
    cdef int rid
    cdef int num_bonds_in_rings = 0

    if n_trial <= 1:
        return True

    # Check for duplicates (inline, O(n²) but n is small)
    for k in range(n_trial):
        for m in range(k + 1, n_trial):
            if trial_comb[k] == trial_comb[m]:
                return False

    # Count rings to track max ring ID
    for bi in range(n_trial):
        rid = <int>ring_id_of_atom[<int>trial_comb[bi]]
        if rid >= nrings:
            nrings = rid + 1

    # Check for beads linked by chemical bond (except in rings)
    for bi in range(n_trial):
        ai = <int>trial_comb[bi]
        for bj in range(bi + 1, n_trial):
            aj = <int>trial_comb[bj]
            if _is_bond_mv(ai, aj, listbonds):
                rid_i = <int>ring_id_of_atom[ai]
                rid_j = <int>ring_id_of_atom[aj]
                if rid_i != -1 and rid_i == rid_j:
                    num_bonds_in_rings += 1
                else:
                    return False

    # Reject if any ring had bonds between beads
    if num_bonds_in_rings > 0:
        return False

    # Check for two terminal beads linked to the same atom
    if _has_terminal_partner_collision_mv(trial_comb, listbonds):
        return False

    return True


def find_acceptable_trials_tmp(
    I32[:, ::1] seq_one_beads,
    I32[:, ::1] listbonds,
    I32[::1] ring_id_of_atom,
):
    cdef Py_ssize_t i
    acceptable_trials = []
    for i in range(seq_one_beads.shape[0]):
        if check_beads(seq_one_beads[i], listbonds, ring_id_of_atom):
            acceptable_trials.append(seq_one_beads[i])
    if not acceptable_trials:
        return np.empty((0, 0), dtype=np.int32)
    return np.asarray(acceptable_trials, dtype=np.int32)


def find_acceptable_trials(
    I32[:, ::1] seq_one_beads,
    I32[:, ::1] listbonds,
    I32[::1] ring_id_of_atom,
):
    """OpenMP-parallel outer loop over trial combinations.

    Each trial is independent, but we can't append to a Python list safely
    inside `prange` without defeating parallelism. This version uses a
    2-pass strategy:

    1) parallel: compute a boolean acceptance mask
    2) serial: pack accepted rows into a dense output array

    Notes
    -----
    * This still calls `check_beads(...)`, which includes NumPy work (sorting)
      and therefore requires the GIL. So speedups may be limited unless
      `check_beads` is made GIL-free.
    * If OpenMP isn't enabled at build time, `prange` falls back to a normal
      serial loop.
    """

    cdef Py_ssize_t n_trials = seq_one_beads.shape[0]
    cdef Py_ssize_t n_beads = seq_one_beads.shape[1]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n_acc = 0

    if n_trials == 0:
        return np.empty((0, 0), dtype=np.int32)

    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mask = np.zeros(n_trials, dtype=np.uint8)

    for i in prange(n_trials, schedule='static', nogil=True):
        if check_beads(seq_one_beads[i], listbonds, ring_id_of_atom):
            mask[i] = 1

    # Count accepted (serial)
    for i in range(n_trials):
        n_acc += mask[i]

    if n_acc == 0:
        return np.empty((0, 0), dtype=np.int32)

    # Pass 2: pack accepted rows (serial)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] out = np.empty((n_acc, n_beads), dtype=np.int32)
    j = 0
    for i in range(n_trials):
        if mask[i] != 0:
            out[j, :] = seq_one_beads[i]
            j += 1

    return out
