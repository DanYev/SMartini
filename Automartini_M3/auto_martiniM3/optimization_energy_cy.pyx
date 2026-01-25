# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""Cython-accelerated energy evaluation for bead placement.

This mirrors a subset of functions from `auto_martiniM3.optimization`:
  - gaussian_overlap
  - atoms_in_gaussian
  - penalize_lonely_atoms
  - eval_gaussian_interac

All functions operate on NumPy arrays / typed memoryviews only.
"""

from libc.math cimport exp
cimport numpy as cnp
import numpy as np

cnp.import_array()

ctypedef cnp.int32_t I32
ctypedef cnp.float32_t F32
ctypedef cnp.uint8_t U8


cdef inline F32 _sigma_for_pair(int bead1, int bead2, U8[::1] in_ring, F32 rvdw, F32 rvdw_aromatic, F32 rvdw_cross) nogil:
    cdef bint b1 = in_ring[bead1] != 0
    cdef bint b2 = in_ring[bead2] != 0
    if b1 and b2:
        return rvdw_aromatic
    if b1 != b2:
        return rvdw_cross
    return rvdw


cpdef F32 eval_gaussian_interac_cy_np(
    I32[::1] list_beads,
    U8[::1] in_ring,
    F32[:, ::1] bond_dists,
    F32[::1] masses,
    # bead params
    F32 rvdw,
    F32 rvdw_aromatic,
    F32 rvdw_cross,
    F32 offset_bd_weight,
    F32 offset_bd_aromatic_weight,
    F32 lonely_atom_penalize,
    F32 bd_bd_overlap_coeff,
    F32 at_in_bd_coeff,
):
    """Fast analogue of `optimization.eval_gaussian_interac`.

    Parameters are passed explicitly (no dict lookups in the hot loop).

    Returns
    -------
    float32
    """
    cdef Py_ssize_t i, j, k
    cdef int n_beads = <int>list_beads.shape[0]
    cdef int n_atoms = <int>masses.shape[0]

    cdef F32 weight_sum = 0.0
    cdef F32 weight_overlap = 0.0
    cdef F32 weight_at_in_bd = 0.0

    # Offset energy
    cdef int num_aromatics = 0
    for i in range(n_beads):
        if in_ring[<int>list_beads[i]] != 0:
            num_aromatics += 1

    weight_sum += offset_bd_weight * (n_beads - num_aromatics) + offset_bd_aromatic_weight * num_aromatics

    # Repulsive overlap between beads
    cdef int bead1, bead2
    cdef F32 dist, sigma
    for i in range(n_beads):
        bead1 = <int>list_beads[i]
        for j in range(i + 1, n_beads):
            bead2 = <int>list_beads[j]
            dist = bond_dists[bead1, bead2]
            sigma = _sigma_for_pair(bead1, bead2, in_ring, rvdw, rvdw_aromatic, rvdw_cross)
            weight_overlap += bd_bd_overlap_coeff * exp(-(dist * dist) / (4.0 * sigma * sigma))

    weight_sum += weight_overlap

    # Attraction between atoms nearby to bead + build lumped mask
    cdef U8[::1] lumped = np.zeros(n_atoms, dtype=np.uint8)
    cdef F32 dist_bd_at
    cdef F32 mass

    for i in range(n_beads):
        bead1 = <int>list_beads[i]
        sigma = rvdw_aromatic if in_ring[bead1] != 0 else rvdw
        for k in range(n_atoms):
            dist_bd_at = bond_dists[k, bead1]
            if dist_bd_at < sigma:
                lumped[k] = 1
            mass = masses[k]
            weight_at_in_bd -= mass * exp(-(dist_bd_at * dist_bd_at) / (2.0 * sigma * sigma))

    weight_sum += at_in_bd_coeff * weight_at_in_bd

    # Penalty for excluding atoms
    cdef F32 lonely_mass = 0.0
    for k in range(n_atoms):
        if lumped[k] == 0:
            lonely_mass += masses[k]

    weight_sum += lonely_atom_penalize * lonely_mass
    return weight_sum
