# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""Cython-accelerated helpers for hot loops in `auto_martiniM3.optimization`.

This module intentionally mirrors a small subset of pure-Python functions to
speed up the CG bead trial filtering stage.

Currently implemented:
  - `check_beads_cy(...)`: fast acceptance check for a trial combination
  - `find_acceptable_trials_cy(...)`: filter many trial combinations

Notes
-----
* This implementation avoids RDKit objects and focuses on integer-heavy loops.
* `heavyatom_coords` is accepted for API compatibility but not used (same as the
  current `optimization.py:check_beads`).
"""

cimport cython
from libc.stdint cimport int32_t

cimport numpy as cnp
import numpy as np

cnp.import_array()


ctypedef cnp.int32_t I32


cdef inline bint _is_bond_mv(int a, int b, I32[:, ::1] bonds) nogil:
    """Return True if (a,b) appears in bonds (either direction)."""
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if (bonds[k, 0] == a and bonds[k, 1] == b) or (bonds[k, 0] == b and bonds[k, 1] == a):
            return True
    return False


cdef inline int _degree_in_bonds_mv(int atom, I32[:, ::1] bonds) nogil:
    """Count occurrences of `atom` in bonds."""
    cdef int deg = 0
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if bonds[k, 0] == atom or bonds[k, 1] == atom:
            deg += 1
    return deg


cdef inline int _partner_for_terminal_mv(int terminal_atom, I32[:, ::1] bonds) nogil:
    """Return the partner atom bonded to a terminal atom (degree==1)."""
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if bonds[k, 0] == terminal_atom:
            return <int>bonds[k, 1]
        if bonds[k, 1] == terminal_atom:
            return <int>bonds[k, 0]
    return -1


def check_beads_cy_np(
    I32[::1] trial_comb,
    I32[:, ::1] listbonds,
    I32[::1] ring_id_of_atom,
):
    """Fast bead-placement acceptance check.

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

    if n_trial <= 1:
        return True

    # Check for beads at the same place (duplicates)
    # Fast path: sort a copy and check adjacent values.
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tmp = np.asarray(trial_comb, dtype=np.int32).copy()
    tmp.sort()
    for bi in range(1, n_trial):
        if tmp[bi] == tmp[bi - 1]:
            return False

    # Track if any bond is found fully within the same ring; reject if so.
    cdef int nrings = 0
    cdef int rid
    for bi in range(n_trial):
        rid = <int>ring_id_of_atom[<int>trial_comb[bi]]
        if rid >= nrings:
            nrings = rid + 1
    cdef int32_t[::1] bonds_in_rings = np.zeros(nrings, dtype=np.int32)

    # Check for beads linked by chemical bond (except in rings)
    for bi in range(n_trial):
        ai = <int>trial_comb[bi]
        for bj in range(bi + 1, n_trial):
            aj = <int>trial_comb[bj]
            if _is_bond_mv(ai, aj, listbonds):
                rid_i = <int>ring_id_of_atom[ai]
                rid_j = <int>ring_id_of_atom[aj]
                if rid_i != -1 and rid_i == rid_j:
                    if rid_i >= 0 and rid_i < nrings:
                        bonds_in_rings[rid_i] += 1
                else:
                    return False

    for bi in range(nrings):
        if bonds_in_rings[bi] > 0:
            return False

    # Check for two terminal beads linked to the same atom
    for bi in range(n_trial):
        ai = <int>trial_comb[bi]
        if _degree_in_bonds_mv(ai, listbonds) != 1:
            continue
        for bj in range(bi + 1, n_trial):
            aj = <int>trial_comb[bj]
            if _degree_in_bonds_mv(aj, listbonds) != 1:
                continue
            partneri = _partner_for_terminal_mv(ai, listbonds)
            partnerj = _partner_for_terminal_mv(aj, listbonds)
            if partneri != -1 and partneri == partnerj:
                return False

    return True


def check_beads_cy(
    molecule,
    list_heavyatoms,
    heavyatom_coords,
    trial_comb,
    ring_atoms,
    listbonds,
):
    """Compatibility wrapper.

    Accepts the original Python objects and converts to NumPy once per call.
    For best performance, call `check_beads_cy_np` directly.
    """
    # listbonds: list[[i,j], ...] -> (nbonds,2)
    bonds = np.asarray(listbonds, dtype=np.int32)
    trial = np.asarray(trial_comb, dtype=np.int32)
    # ring_atoms: list[list[int]] -> ring_id_of_atom
    ring_id = np.full(int(np.max(trial)) + 1 if trial.size else 0, -1, dtype=np.int32)
    for rid, ring in enumerate(ring_atoms):
        ring = np.asarray(ring, dtype=np.int32)
        for a in ring:
            if a >= ring_id.shape[0]:
                ring_id = np.pad(ring_id, (0, int(a - ring_id.shape[0] + 1)), constant_values=-1)
            ring_id[a] = rid
    return check_beads_cy_np(trial, bonds, ring_id)


def find_acceptable_trials_cy_np(
    I32[:, ::1] seq_one_beads,
    I32[:, ::1] listbonds,
    I32[::1] ring_id_of_atom,
):
    """Filter acceptable trial combinations (NumPy fast path)."""
    cdef Py_ssize_t i
    acceptable_trials = []
    for i in range(seq_one_beads.shape[0]):
        if check_beads_cy_np(seq_one_beads[i], listbonds, ring_id_of_atom):
            acceptable_trials.append(seq_one_beads[i])
    return acceptable_trials


def find_acceptable_trials_cy(
    seq_one_beads,
    molecule,
    list_heavy_atoms,
    heavyatom_coords,
    ring_atoms,
    list_bonds,
    allatom_coords,
    force_map,
):
    """Compatibility wrapper matching `optimization.find_acceptable_trials`."""
    bonds = np.asarray(list_bonds, dtype=np.int32)
    seq = np.asarray(seq_one_beads, dtype=np.int32)
    # Build ring_id mapping sized to include max atom id in bonds/seq
    max_atom = -1
    if seq.size:
        max_atom = max(max_atom, int(seq.max()))
    if bonds.size:
        max_atom = max(max_atom, int(bonds.max()))
    ring_id = np.full(max_atom + 1, -1, dtype=np.int32)
    for rid, ring in enumerate(ring_atoms):
        ring = np.asarray(ring, dtype=np.int32)
        ring_id[ring] = rid
    return find_acceptable_trials_cy_np(seq, bonds, ring_id)
