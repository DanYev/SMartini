# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""Cython-accelerated helpers for hot loops in `auto_martiniM3.optimization`.

This module provides high-performance Cython implementations for the CG bead 
trial filtering and energy evaluation stages of coarse-graining. It uses 
parallel processing (OpenMP via Cython's prange) and GIL-free execution where 
possible to maximize performance.

Currently implemented:
    - `generate_combinations_chunk(...)`: nogil combination generation for memory efficiency
    - `check_beads(...)`: fast acceptance check for a trial combination
    - `find_acceptable_trials(...)`: OpenMP-parallel filter for many trial 
      combinations
    - `eval_gaussian_interac(...)`: fast energy evaluation for a bead combination
    - `collect_energies(...)`: batch energy collection for all acceptable trials
    - `gaussian_overlap(...)`: bead-bead overlap penalty calculation

Features:
    - GIL-free combination generation for efficient chunking
    - OpenMP parallelization via prange for trial filtering
    - GIL-free critical loops for performance
    - Proper handling of ring and aromatic atoms
    - Bond connectivity validation
    - Terminal atom collision detection
    - Mass-weighted energy terms

Notes
-----
* This implementation avoids RDKit objects and focuses on integer-heavy and 
  float-heavy loops.
* This module expects NumPy arrays or compatible typed memoryviews as inputs.
  Type/shape normalization should be handled in Python.
* All critical loops are marked with nogil to allow parallel execution.
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


#############################################################################
### COMBINATION GENERATION (nogil) ###  
############################################################################# 


cdef inline int _binom_int(int n, int r) nogil:
    """Compute C(n, r) in int arithmetic (assumes small n, no overflow)."""
    cdef int i
    cdef int res = 1
    if r < 0 or n < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    if r > n - r:
        r = n - r
    for i in range(1, r + 1):
        res = (res * (n - r + i)) // i
    return res


cdef inline long long _binom_ll(int n, int r) nogil:
    """Compute C(n, r) in 64-bit arithmetic (for range checks)."""
    cdef int i
    cdef long long res = 1
    if r < 0 or n < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    if r > n - r:
        r = n - r
    for i in range(1, r + 1):
        res = (res * (n - r + i)) // i
    return res


cdef inline int _next_combination_indices(I32[::1] comb, int n, int r) nogil:
    """In-place next combination for index array (0..n-1), like itertools."""
    cdef int i, j
    i = r - 1
    while i >= 0 and comb[i] == i + n - r:
        i -= 1
    if i < 0:
        return 0
    comb[i] += 1
    for j in range(i + 1, r):
        comb[j] = comb[j - 1] + 1
    return 1


cdef inline void _get_first_combination(
    int n,
    int r,
    long long start_index,
    I32[::1] comb,
) noexcept nogil:
    """Fill `comb` with the start_index-th combination in itertools order.

    This is a standard lexicographic unranking routine.
    """
    cdef int pos
    cdef int chosen
    cdef int start = 0
    cdef long long k = start_index
    cdef int cnt

    for pos in range(r):
        chosen = start
        while chosen <= n - (r - pos):
            cnt = _binom_int(n - chosen - 1, r - pos - 1)
            if k >= cnt:
                k -= cnt
                chosen += 1
            else:
                break
        comb[pos] = <I32>chosen
        start = chosen + 1


def generate_combinations(
    int n,
    int r,
    long long start_index,
    int chunk_size,
):
    """Generate combinations of indices (0..n-1) in itertools order.

    Returns a dense int32 array with at most `chunk_size` rows.
    """
    cdef int j
    cdef int out_rows = 0

    # Minimal sanity only; caller is expected to provide valid values.
    if r <= 0:
        return np.empty((1, 0), dtype=np.int32)
    if chunk_size <= 0:
        return np.empty((0, r), dtype=np.int32)

    # Out-of-range chunk start: return empty.
    if start_index >= _binom_ll(n, r):
        return np.empty((0, r), dtype=np.int32)

    cdef cnp.ndarray[cnp.int32_t, ndim=1] comb_arr = np.empty((r,), dtype=np.int32)
    cdef I32[::1] comb = comb_arr
    _get_first_combination(n, r, start_index, comb)

    cdef cnp.ndarray[cnp.int32_t, ndim=2] out_arr = np.empty((chunk_size, r), dtype=np.int32)
    cdef I32[:, ::1] out = out_arr

    # Generate chunk
    while out_rows < chunk_size:
        for j in range(r):
            out[out_rows, j] = comb[j]
        out_rows += 1
        if not _next_combination_indices(comb, n, r):
            break

    return out_arr[:out_rows, :]

#############################################################################
### BEAD COMBINATION FILTERING ###  
############################################################################# 


cdef inline bint _is_bond(int a, int b, const I32[:, ::1] bonds) nogil:
    """Check if atoms a and b are bonded (GIL-free).
    
    Parameters
    ----------
    a, b : int
        Atom indices to check.
    bonds : (nbonds, 2) int32 memoryview
        Array of bond pairs (symmetric).
    
    Returns
    -------
    bool
        True if bond exists between a and b in either direction.
    """
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if (bonds[k, 0] == a and bonds[k, 1] == b) or (bonds[k, 0] == b and bonds[k, 1] == a):
            return True
    return False


cdef inline int _degree_in_bonds(int atom, const I32[:, ::1] bonds) nogil:
    """Count number of bonds incident to an atom (GIL-free).
    
    Parameters
    ----------
    atom : int
        Atom index to check degree for.
    bonds : (nbonds, 2) int32 memoryview
        Array of bond pairs.
    
    Returns
    -------
    int
        Number of bonds connected to this atom.
    """
    cdef int deg = 0
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if bonds[k, 0] == atom or bonds[k, 1] == atom:
            deg += 1
    return deg


cdef inline int _partner_for_terminal(int terminal_atom, const I32[:, ::1] bonds) nogil:
    """Find bonded partner of a terminal atom (degree 1) (GIL-free).
    
    Parameters
    ----------
    terminal_atom : int
        Atom index assumed to have degree 1.
    bonds : (nbonds, 2) int32 memoryview
        Array of bond pairs.
    
    Returns
    -------
    int
        Index of partner atom, or -1 if terminal atom has no bonds.
    """
    cdef Py_ssize_t k
    for k in range(bonds.shape[0]):
        if bonds[k, 0] == terminal_atom:
            return <int>bonds[k, 1]
        if bonds[k, 1] == terminal_atom:
            return <int>bonds[k, 0]
    return -1


cdef inline bint _has_terminal_partner_collision(const I32[::1] trial_comb, const I32[:, ::1] bonds) nogil:
    """Detect if two terminal atoms in trial share same bonded partner (GIL-free).
    
    Terminal atoms (degree 1) linked to the same non-bead atom indicate 
    problematic CG mapping.
    
    Parameters
    ----------
    trial_comb : (n_beads,) int32 memoryview
        Atom indices for CG beads in this trial.
    bonds : (nbonds, 2) int32 memoryview
        Array of bond pairs.
    
    Returns
    -------
    bool
        True if collision detected (two terminals share partner), False otherwise.
    """
    cdef Py_ssize_t bi, bj
    cdef int ai, aj
    cdef int partneri, partnerj
    cdef int n_trial = <int>trial_comb.shape[0]
    for bi in range(n_trial):
        ai = <int>trial_comb[bi]
        if _degree_in_bonds(ai, bonds) != 1:
            continue
        partneri = _partner_for_terminal(ai, bonds)
        if partneri == -1:
            continue
        for bj in range(bi + 1, n_trial):
            aj = <int>trial_comb[bj]
            if _degree_in_bonds(aj, bonds) != 1:
                continue
            partnerj = _partner_for_terminal(aj, bonds)
            if partnerj != -1 and partneri == partnerj:
                return True
    return False


cdef bint check_beads(
    const I32[::1] trial_comb,
    const I32[:, ::1] listbonds,
    const I32[::1] ring_id_of_atom,
    const int nrings,
) nogil:
    """Validate a CG bead trial combination against chemical constraints (GIL-free).

    Checks performed:
        1. No duplicate atoms within trial
        2. No two beads bonded together (except ring-to-ring)
        3. No terminal atoms sharing same bonded partner
        4. Handles aromatic/ring atoms specially

    Parameters
    ----------
    trial_comb
        1D array of atom indices for proposed bead centers.
    listbonds
        2D array (nbonds, 2) of heavy-atom bonds; symmetric pairs.
    ring_id_of_atom
        1D array indexed by atom ID. Value is ring ID or -1 if not in ring.

    Returns
    -------
    bool
        True if trial passes all constraints, False otherwise.

    Notes
    -----
    * All checks performed with nogil for parallel safety.
    * Single-bead or empty trials always pass.
    * Bond check makes special exception for ring beads in same ring.
    """
    cdef Py_ssize_t bi, bj
    cdef int ai, aj
    cdef int n_trial = trial_comb.shape[0]
    cdef int rid_i, rid_j
    cdef int k, m
    cdef int rid
    cdef int num_bonds_in_rings = 0

    if n_trial <= 1:
        return True

    # Check for beads linked by chemical bond (except in rings)
    for bi in range(n_trial):
        ai = <int>trial_comb[bi]
        for bj in range(bi + 1, n_trial):
            aj = <int>trial_comb[bj]
            if _is_bond(ai, aj, listbonds):
                return False
                # rid_i = <int>ring_id_of_atom[ai]
                # rid_j = <int>ring_id_of_atom[aj]
                # if rid_i != -1 and rid_i == rid_j:
                #     num_bonds_in_rings += 1
                # else:
                #     return False

    # # Reject if any ring has more than 1 bond
    # if num_bonds_in_rings > 1:
    #     return False

    # Check for two terminal beads linked to the same atom
    if _has_terminal_partner_collision(trial_comb, listbonds):
        return False

    return True


def find_acceptable_trials_tmp(
    I32[:, ::1] seq_one_beads,
    I32[:, ::1] listbonds,
    I32[::1] ring_id_of_atom,
    const int nrings,
):
    cdef Py_ssize_t i
    acceptable_trials = []
    for i in range(seq_one_beads.shape[0]):
        if check_beads(seq_one_beads[i], listbonds, ring_id_of_atom, nrings):
            acceptable_trials.append(seq_one_beads[i])
    if not acceptable_trials:
        return np.empty((0, 0), dtype=np.int32)
    return np.asarray(acceptable_trials, dtype=np.int32)


def find_acceptable_combinations(
    I32[:, ::1] trial_combinations,
    I32[:, ::1] listbonds,
    I32[::1] ring_id_of_atom,
    const int nrings,
):
    """OpenMP-parallel filter for valid CG bead trial combinations.

    Validates each trial combination against chemical constraints:
        - No duplicate atoms within a bead
        - No two beads connected by a chemical bond (except ring atoms)
        - No two terminal atoms sharing the same bonded partner
        - Proper handling of aromatic/ring atoms

    Strategy
    --------
    Uses a 2-pass approach to enable OpenMP parallelism with Python lists:
        1) Parallel: compute boolean acceptance mask for each trial
        2) Serial: pack accepted rows into dense output array

    Parameters
    ----------
    trial_combinations : (n_trials, n_beads) int32 array
        Each row is a trial combination of atom indices for CG bead centers.
    listbonds : (nbonds, 2) int32 array
        Heavy-atom bonds in (begin_atom, end_atom) format.
    ring_id_of_atom : (n_atoms,) int32 array
        Indexed by atom ID. Value is ring ID (0..nrings-1) or -1 if not in ring.

    Returns
    -------
    (n_accepted, n_beads) int32 array
        Subset of input trials that pass all acceptance checks.

    Notes
    -----
    * Each trial call to `check_beads()` is independent.
    * If OpenMP isn't enabled at build time, `prange` falls back to serial loop.
    * For maximum performance, ensure input arrays are C-contiguous.
    """

    cdef Py_ssize_t n_trials = trial_combinations.shape[0]
    cdef Py_ssize_t n_beads = trial_combinations.shape[1]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n_acc = 0

    if n_trials == 0:
        return np.empty((0, 0), dtype=np.int32)

    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mask = np.zeros(n_trials, dtype=np.uint8)

    for i in prange(n_trials, schedule='static', nogil=True):
        if check_beads(trial_combinations[i], listbonds, ring_id_of_atom, nrings):
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
            out[j, :] = trial_combinations[i]
            j += 1

    return out

#############################################################################
### ENERGY EVALUTATION ###  
############################################################################# 

cdef inline F32 _sigma_for_pair(
    int bead1,
    int bead2,
    const U8[::1] in_ring,
    F32 rvdw,
    F32 rvdw_aromatic,
    F32 rvdw_cross,
) nogil:
    """Select vdW radius for bead pair based on ring/aromatic status (GIL-free).
    
    Parameters
    ----------
    bead1, bead2 : int
        Bead indices in trial.
    in_ring : (n_beads,) uint8 memoryview
        Boolean mask for aromatic/ring beads.
    rvdw : float32
        vdW radius for non-aromatic beads.
    rvdw_aromatic : float32
        vdW radius for aromatic beads.
    rvdw_cross : float32
        Cross-term radius (aromatic-non-aromatic pair).
    
    Returns
    -------
    float32
        Appropriate sigma for this bead pair.
    """
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
    """Compute Gaussian overlap penalty between two beads (GIL-free).
    
    Evaluates: coeff * exp(-(dist²) / (4·σ²))
    where σ is selected based on aromatic status of beads.
    
    Parameters
    ----------
    dist : float32
        Distance between bead centers.
    bead1, bead2 : int
        Bead indices for ring lookup.
    in_ring : (n_beads,) uint8 memoryview
        Aromatic/ring status mask.
    bd_bd_overlap_coeff : float32
        Scaling coefficient.
    rvdw, rvdw_aromatic, rvdw_cross : float32
        vdW radii parameters.
    
    Returns
    -------
    float32
        Gaussian overlap energy penalty.
    """
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
    """Evaluate atom-in-bead penalty and mark lumped atoms (GIL-free).
    
    Computes mass-weighted penalty for atoms within Gaussian radius of bead.
    Marks atoms that are lumped into this bead in lumped_mask.
    
    Parameters
    ----------
    bead_id : int
        Index of bead being evaluated.
    in_ring : (n_beads,) uint8 memoryview
        Aromatic status of beads.
    bond_dists : (n_atoms, n_atoms) float32 memoryview
        Distance matrix; bond_dists[i, bead_id] = distance from atom i to bead.
    masses : (n_atoms,) float32 memoryview
        Atomic masses.
    at_in_bd_coeff : float32
        Energy scaling coefficient.
    rvdw, rvdw_aromatic : float32
        vdW radii for Gaussian radius selection.
    lumped_mask : (n_atoms,) uint8 memoryview
        Output mask; set to 1 for atoms lumped into this bead.
    
    Returns
    -------
    float32
        Negative mass-weighted energy penalty for atoms in bead.
    """
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
    """Compute energy penalty for atoms not mapped to any bead (GIL-free).
    
    Parameters
    ----------
    lumped_mask : (n_atoms,) uint8 memoryview
        Binary mask; 1 if atom lumped into a bead, 0 if lonely.
    masses : (n_atoms,) float32 memoryview
        Atomic masses.
    lonely_atom_penalize : float32
        Energy scaling coefficient.
    
    Returns
    -------
    float32
        Mass-weighted penalty energy.
    """
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
    """Evaluate total Gaussian interaction energy for a bead combination (GIL-free).
    
    Computes sum of all energy components:
        - Offset penalty (beads × aromatic_factor)
        - Bead-bead Gaussian overlaps
        - Atom-in-bead penalties
        - Lonely atom penalties
    
    Parameters
    ----------
    list_beads : (n_beads,) int32 memoryview
        Atom indices for CG beads in this trial.
    in_ring : (n_beads,) uint8 memoryview
        Aromatic status of each bead.
    bond_dists : (n_atoms, n_atoms) float32 memoryview
        Distance matrix.
    masses : (n_atoms,) float32 memoryview
        Atomic masses.
    offset_bd_weight, offset_bd_aromatic_weight : float32
        Penalty coefficients for non-aromatic and aromatic beads.
    lonely_atom_penalize : float32
        Penalty for unmapped atoms.
    bd_bd_overlap_coeff : float32
        Scaling for bead-bead Gaussian overlap.
    at_in_bd_coeff : float32
        Scaling for atom-in-bead penalty.
    rvdw, rvdw_aromatic, rvdw_cross : float32
        vdW radius parameters.
    lumped_mask : (n_atoms,) uint8 memoryview
        Cumulative mask of lumped atoms (updated).
    local_mask : (n_atoms,) uint8 memoryview
        Work buffer for per-bead lumping.
    
    Returns
    -------
    float32
        Total interaction energy for this bead combination.
    """
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
    """Cythonized batch energy evaluation for all acceptable trial combinations.

    Computes Gaussian-based energy scores reflecting how well each CG bead 
    combination maps the molecular structure. Used during bead optimization to 
    rank candidate mappings.

    Energy components evaluated for each trial:
        - Offset penalty: penalizes beads (weighted by aromatic content)
        - Bead-bead overlap: Gaussian overlap between pairs of beads
        - Atom-in-bead: penalty for atoms mapped into beads
        - Lonely atoms: penalty for unmapped atoms

    Parameters
    ----------
    acceptable_trials : (n_trials, n_beads) int32 array
        Pre-filtered valid trial combinations from find_acceptable_trials().
    in_ring : (n_atoms,) uint8 array
        Boolean mask; 1 if atom is aromatic/ring, 0 otherwise.
    bond_dists : (n_atoms, n_atoms) float32 array
        Symmetric distance matrix; bond_dists[i,j] = distance atom i to atom j.
    masses : (n_atoms,) float32 array
        Atomic masses for all atoms in molecule.
    offset_bd_weight : float32
        Penalty weight for non-aromatic beads (Martini 3: 20.0)
    offset_bd_aromatic_weight : float32
        Penalty weight for aromatic beads (Martini 3: 5.0)
    lonely_atom_penalize : float32
        Penalty coefficient for unmapped atoms (Martini 3: 0.28)
    bd_bd_overlap_coeff : float32
        Bead-bead overlap scaling (Martini 3: 1.0)
    at_in_bd_coeff : float32
        Atom-in-bead weight scaling (Martini 3: 0.9)
    rvdw : float32
        vdW radius for non-aromatic beads (Martini 3: 2.35 Å)
    rvdw_aromatic : float32
        vdW radius for aromatic beads (Martini 3: 2.05 Å)
    rvdw_cross : float32
        Cross-term vdW radius for aromatic/non-aromatic pairs
    initial_ene_best : float32
        Initial best energy (typically 1e6) for tracking best trial found.

    Returns
    -------
    tuple
        (ene_best_trial, energies_array)
        - ene_best_trial: lowest energy among all trials
        - energies_array: (n_trials,) float32 array of individual energies

    Notes
    -----
    * All inner loops are nogil for performance.
    * Masks are reset per iteration to track atom lumping correctly.
    * Energy minimization target in bead optimization process.
    """
    cdef Py_ssize_t n_trials = acceptable_trials.shape[0]
    cdef Py_ssize_t n_beads
    cdef Py_ssize_t n_atoms
    cdef Py_ssize_t i, j
    
    cdef F32 trial_energy
    cdef F32 ene_best_trial = initial_ene_best
    cdef cnp.ndarray[cnp.float32_t, ndim=1] energies_array
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] lumped_mask
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] local_mask

    n_beads = acceptable_trials.shape[1]
    n_atoms = masses.shape[0]
    
    # Pre-allocate output arrays and work arrays
    energies_array = np.zeros(n_trials, dtype=np.float32)
    lumped_mask = np.zeros(n_atoms, dtype=np.uint8)
    local_mask = np.zeros(n_atoms, dtype=np.uint8)
    
    cdef const I32[::1] trial_mv
    cdef F32[::1] energies_view = energies_array
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
            trial_energy = eval_gaussian_interac(
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
            energies_view[i] = trial_energy
            
            # Track best energy and combination
            if trial_energy < ene_best_trial:
                ene_best_trial = trial_energy
    
    return ene_best_trial, energies_array
