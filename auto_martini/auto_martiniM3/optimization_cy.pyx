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
) nogil:
    cdef F32 sigma = _sigma_for_pair(bead1, bead2, in_ring, rvdw, rvdw_aromatic, rvdw_cross)
    return bd_bd_overlap_coeff * exp(-(dist * dist) / (4.0 * sigma * sigma))


cdef F32 atoms_in_gaussian(
    int bead_id,
    const U8[::1] in_ring,
    const F32[:, ::1] bond_dists,
    const F32[::1] masses,
    F32 at_in_bd_coeff,
    F32 rvdw,
    F32 rvdw_aromatic,
    U8[::1] lumped_mask,
) nogil:
    """Compute weight and mark lumped atoms in a mask."""
    cdef Py_ssize_t n = bond_dists.shape[0]
    cdef Py_ssize_t i
    cdef F32 sigma = rvdw_aromatic if in_ring[bead_id] != 0 else rvdw
    cdef F32 sigma2 = sigma * sigma
    cdef F32 weight_sum = 0.0
    cdef F32 dist

    for i in range(n):
        dist = bond_dists[i, bead_id]
        if dist < sigma:
            lumped_mask[i] = 1
        weight_sum -= masses[i] * exp(-(dist * dist) / (2.0 * sigma2))

    return at_in_bd_coeff * weight_sum


cdef F32 penalize_lonely_atoms(
    const U8[::1] lumped_mask,
    const F32[::1] masses,
    F32 lonely_atom_penalize,
) nogil:
    """Compute penalty for atoms not in lumped_mask."""
    cdef Py_ssize_t n = masses.shape[0]
    cdef Py_ssize_t i
    cdef F32 weight_sum = 0.0

    for i in range(n):
        if lumped_mask[i] == 0:
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
    U8[::1] lumped_mask,
    U8[::1] local_mask,
) nogil:
    cdef Py_ssize_t nb = list_beads.shape[0]
    cdef Py_ssize_t i, j
    cdef int bead1, bead2
    cdef int num_aromatics = 0
    cdef F32 weight_sum = 0.0
    cdef F32 weight_overlap = 0.0
    cdef F32 weight_at_in_bd = 0.0
    cdef Py_ssize_t n_atoms = masses.shape[0]

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
    
    for i in range(nb):
        bead1 = <int>list_beads[i]
        # Reset local mask
        for j in range(n_atoms):
            local_mask[j] = 0
        
        weight_at_in_bd += atoms_in_gaussian(
            bead1,
            in_ring,
            bond_dists,
            masses,
            at_in_bd_coeff,
            rvdw,
            rvdw_aromatic,
            local_mask,
        )
        # Accumulate lumped atoms
        for j in range(n_atoms):
            if local_mask[j] != 0:
                lumped_mask[j] = 1

    weight_sum += weight_at_in_bd
    weight_sum += penalize_lonely_atoms(lumped_mask, masses, lonely_atom_penalize)
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


cpdef tuple collect_energies(
    const I32[:, ::1] acceptable_trials,
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
    F32 initial_ene_best,
):
    """Cythonized energy collection loop for all acceptable trials (with nogil support).
    
    Returns: (ene_best_trial, best_trial_comb, energies_array, trials_array)
    """
    cdef Py_ssize_t n_trials = acceptable_trials.shape[0]
    cdef Py_ssize_t n_beads
    cdef Py_ssize_t n_atoms
    cdef Py_ssize_t i, j
    
    cdef F32 trial_ene
    cdef F32 ene_best_trial = initial_ene_best
    cdef cnp.ndarray[cnp.int32_t, ndim=1] best_trial_comb_full
    cdef cnp.ndarray[cnp.float32_t, ndim=1] energies_array
    cdef cnp.ndarray[cnp.int32_t, ndim=2] trials_array
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] lumped_mask
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] local_mask
    
    # Initialize best_trial_comb_full as empty (Python will use existing value if no improvement)
    best_trial_comb_full = np.array([], dtype=np.int32)
    
    if n_trials == 0:
        energies_array = np.array([], dtype=np.float32)
        trials_array = np.zeros((0, 0), dtype=np.int32)
        return ene_best_trial, best_trial_comb_full, energies_array, trials_array
    
    n_beads = acceptable_trials.shape[1]
    n_atoms = masses.shape[0]
    
    # Pre-allocate output arrays and work arrays
    energies_array = np.zeros(n_trials, dtype=np.float32)
    trials_array = np.zeros((n_trials, n_beads), dtype=np.int32)
    lumped_mask = np.zeros(n_atoms, dtype=np.uint8)
    local_mask = np.zeros(n_atoms, dtype=np.uint8)
    
    cdef const I32[::1] trial_mv
    cdef F32[::1] energies_view = energies_array
    cdef I32[:, ::1] trials_view = trials_array
    cdef U8[::1] lumped_mask_view = lumped_mask
    cdef U8[::1] local_mask_view = local_mask
    
    # Main loop over all acceptable trials - with nogil
    with nogil:
        for i in range(n_trials):
            # Reset masks for this iteration
            for j in range(n_atoms):
                lumped_mask_view[j] = 0
                local_mask_view[j] = 0
            
            # Get memoryview of this trial directly
            trial_mv = acceptable_trials[i, :]
            
            # Evaluate energy for this trial
            trial_ene = eval_gaussian_interac(
                trial_mv,
                in_ring,
                bond_dists,
                masses,
                offset_bd_weight,
                offset_bd_aromatic_weight,
                lonely_atom_penalize,
                bd_bd_overlap_coeff,
                at_in_bd_coeff,
                rvdw,
                rvdw_aromatic,
                rvdw_cross,
                lumped_mask_view,
                local_mask_view,
            )
            
            # Store energy
            energies_view[i] = trial_ene
            
            # Copy trial to output array
            for j in range(n_beads):
                trials_view[i, j] = trial_mv[j]
            
            # Track best energy and combination
            if trial_ene < ene_best_trial:
                ene_best_trial = trial_ene
                # Store the best trial indices for sorting later
                for j in range(n_beads):
                    trials_view[i, j] = trial_mv[j]
    
    # Outside nogil block, find and sort best trial combination (requires GIL)
    cdef I32[::1] best_trial_mv
    best_trial_mv = trials_array[np.argmin(energies_array), :]
    best_trial_comb_full = np.sort(np.asarray(best_trial_mv, dtype=np.int32))
    
    return ene_best_trial, best_trial_comb_full, energies_array, trials_array
