# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""Cython-accelerated energy evaluation for bead placements.

This module provides NumPy-typed implementations of:
  - gaussian_overlap
  - atoms_in_gaussian
  - penalize_lonely_atoms
  - eval_gaussian_interac

It is designed to match the *numeric* refactor in `optimization.py` where
pairwise distances and masses are precomputed and passed as arrays.
"""

from libc.math cimport exp
cimport numpy as cnp
import numpy as np

ctypedef cnp.int32_t I32
ctypedef cnp.float32_t F32

cnp.import_array()


cdef inline F32 _sigma(int bead1, int bead2, const unsigned char[:] is_ring, F32 rvdw, F32 rvdw_arom, F32 rvdw_cross) nogil:
    cdef bint r1 = is_ring[bead1] != 0
    cdef bint r2 = is_ring[bead2] != 0
    if r1 and r2:
        return rvdw_arom
    if (r1 and (not r2)) or ((not r1) and r2):
        return rvdw_cross
    return rvdw


cpdef F32 gaussian_overlap_np(
    F32 dist,
    int bead1,
    int bead2,
    const unsigned char[:] is_ring,
    F32 bd_bd_overlap_coeff,
    F32 rvdw,
    F32 rvdw_arom,
    F32 rvdw_cross,
):
    cdef F32 sigma = _sigma(bead1, bead2, is_ring, rvdw, rvdw_arom, rvdw_cross)
    # bd_bd_overlap_coeff * exp(-(dist**2) / (4*sigma**2))
    return bd_bd_overlap_coeff * exp(-(dist * dist) / (4.0 * sigma * sigma))


cpdef tuple atoms_in_gaussian_np(
    int bead_id,
    const unsigned char[:] is_ring,
    const F32[:, ::1] bond_dists,
    const F32[::1] masses,
    F32 at_in_bd_coeff,
    F32 rvdw,
    F32 rvdw_arom,
):
    cdef Py_ssize_t n = bond_dists.shape[0]
    cdef Py_ssize_t i
    cdef F32 sigma = rvdw_arom if is_ring[bead_id] != 0 else rvdw
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


cpdef F32 penalize_lonely_atoms_np(
    list lumped_atoms,
    const F32[::1] masses,
    F32 lonely_atom_penalize,
):
    cdef Py_ssize_t n = masses.shape[0]
    cdef Py_ssize_t i
    cdef F32 weight_sum = 0.0

    # Mark lumped atoms in a boolean mask (O(n + k)) instead of repeated `in` checks.
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mask = np.zeros(n, dtype=np.uint8)
    for i in lumped_atoms:
        if 0 <= i < n:
            mask[i] = 1

    for i in range(n):
        if mask[i] == 0:
            weight_sum += masses[i]

    return lonely_atom_penalize * weight_sum


cpdef F32 eval_gaussian_interac_np(
    const I32[::1] list_beads,
    const unsigned char[:] is_ring,
    const F32[:, ::1] bond_dists,
    const F32[::1] masses,
    # bead params as scalars
    F32 offset_bd_weight,
    F32 offset_bd_aromatic_weight,
    F32 lonely_atom_penalize,
    F32 bd_bd_overlap_coeff,
    F32 at_in_bd_coeff,
    F32 rvdw,
    F32 rvdw_arom,
    F32 rvdw_cross,
):
    cdef Py_ssize_t nb = list_beads.shape[0]
    cdef Py_ssize_t i, j
    cdef int bead1, bead2
    cdef int num_aromatics = 0
    cdef F32 weight_sum = 0.0
    cdef F32 weight_overlap = 0.0
    cdef F32 weight_at_in_bd = 0.0

    # Count aromatic beads
    for i in range(nb):
        bead1 = <int>list_beads[i]
        if is_ring[bead1] != 0:
            num_aromatics += 1

    weight_sum += offset_bd_weight * (nb - num_aromatics) + offset_bd_aromatic_weight * num_aromatics

    # Repulsive overlap between beads
    for i in range(nb):
        bead1 = <int>list_beads[i]
        for j in range(i + 1, nb):
            bead2 = <int>list_beads[j]
            weight_overlap += gaussian_overlap_np(
                bond_dists[bead1, bead2],
                bead1,
                bead2,
                is_ring,
                bd_bd_overlap_coeff,
                rvdw,
                rvdw_arom,
                rvdw_cross,
            )
    weight_sum += weight_overlap

    # Attraction between atoms nearby to bead
    lumped_atoms_all = []
    for i in range(nb):
        bead1 = <int>list_beads[i]
        weight, lumped = atoms_in_gaussian_np(
            bead1,
            is_ring,
            bond_dists,
            masses,
            at_in_bd_coeff,
            rvdw,
            rvdw_arom,
        )
        weight_at_in_bd += weight
        # merge unique
        for j in lumped:
            if j not in lumped_atoms_all:
                lumped_atoms_all.append(j)

    weight_sum += weight_at_in_bd

    # Penalty for excluding atoms
    weight_sum += penalize_lonely_atoms_np(lumped_atoms_all, masses, lonely_atom_penalize)

    return weight_sum
