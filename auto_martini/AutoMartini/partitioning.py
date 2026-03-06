r"""
Created on March 13, 2019 by Andrew Abi-Mansour
Updated to Martini 3 force field on January 31, 2025 by Magdalena Szczuka

This is the::
    _   _   _ _____ ___     __  __    _    ____ _____ ___ _   _ ___   __  __ _____
   / \ | | | |_   _/ _ \   |  \/  |  / \  |  _ \_   _|_ _| \ | |_ _|  |  \/  |___ /  
  / _ \| | | | | || | | |  | |\/| | / _ \ | |_) || |  | ||  \| || |   | |\/| | |_ \  
 / ___ \ |_| | | || |_| |  | |  | |/ ___ \|  _ < | |  | || |\  || |   | |  | |___) | 
/_/  _\_\___/  |_| \___/   |_|  |_/_/   \_\_| \_\|_| |___|_| \_|___|  |_|  |_|____/    
                                                

A tool for automatic MARTINI 3 force field mapping and parametrization of small organic molecules

Developers::
        Magdalena Szczuka (magdalena.szczuka at univ-tlse3.fr)
        Tristan BEREAU (bereau at mpip-mainz.mpg.de)
        Kiran Kanekal (kanekal at mpip-mainz.mpg.de)
        Andrew Abi-Mansour (andrew.gaam at gmail.com)

AUTO_MARTINI M3 is open-source, distributed under the terms of the GNU Public
License, version 2 or later. It is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
received a copy of the GNU General Public License along with PyGran.
If not, see http://www.gnu.org/licenses . See also top-level README
and LICENSE files.
"""

from sys import exit
from .common import *
from . import topology # AutoM3 change
from .utils import timeit, memprofit
from . import optimization_cy as opcy
import math
import multiprocessing as mp
import os

logger = logging.getLogger(__name__)

#############################################################################
### HELPER FUNCTIONS ###
#############################################################################

def read_bead_params():
    """Returns bead parameter dictionary
    CG Bead vdw radius (in Angstroem)"""
    bead_params = dict()
    bead_params["rvdw"] = 4.7 / 2.0     # sigma for non-ring 
    bead_params["rvdw_aromatic"] = 4.1 / 2.0 # AutoM3 change: was 4.3 / 2.0    #sigma for ring
    bead_params["rvdw_cross"] = 0.5 * ((4.7 / 2.0) + (4.3 / 2.0))
    bead_params["offset_bd_weight"] = 20.0 # AutoM3 change: was 50.0    #penalty weight for nonring beads
    bead_params["offset_bd_aromatic_weight"] = 5.0 # AutoM3 change: was 20.0    #penalty weight for ring beads
    bead_params["lonely_atom_penalize"] = 0.28  # AutoM3 change: was 0.20
    bead_params["bd_bd_overlap_coeff"] = 1.0 # AutoM3 change: was 9.0
    bead_params["at_in_bd_coeff"] = 0.9
    return bead_params


def _get_masses(molecule):
    """Return an array of atomic masses for all atoms in the molecule."""
    masses = []
    for i in range(molecule.GetNumAtoms()):
        mass = molecule.GetAtomWithIdx(i).GetMass()
        masses.append(mass)
    return np.array(masses).astype(np.float32)


def _get_bond_distances(conformer):
    """Return a symmetric (N,N) distance matrix with a 0 diagonal.

    Notes
    -----
    `Chem.rdMolTransforms.GetBondLength` is used here (even though it returns a
    pairwise distance) to preserve existing behavior.
    """
    n = conformer.GetNumAtoms()
    dists = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = Chem.rdMolTransforms.GetBondLength(conformer, i, j)
            dists[i, j] = dist
            dists[j, i] = dist
    return dists


def _get_ring_id_of_atom(atom, rings, dtype=np.int32):
    """ring_id_of_atom[atom_id] = ring index, or 0 when not in any ring."""
    for rid, ring in enumerate(rings):
        if atom in ring:
            return rid      
    return -1


def flat_set(lst):
    """Flatten a list of lists into a set of unique elements."""
    if not lst:
        return set()
    aset = set(item for sublist in lst for item in sublist) 
    alist = sorted(aset)
    return alist

#############################################################################
### INITIAL PARTITIONING ###
#############################################################################

def _get_ha_graph(molecule):
    """Extract molecule info needed for partitioning."""
    atoms = molecule.GetAtoms()
    ha_list = [a for a in atoms if a.GetAtomicNum() > 1]
    bonds = []
    for ai in ha_list:
        for aj in ha_list:
            i = ai.GetIdx()
            j = aj.GetIdx()
            if i < j and molecule.GetBondBetweenAtoms(int(i), int(j)) is not None:
                bonds.append([i, j])
    return ha_list, bonds



def split_into_fragments(molecule):
    """Split molecule into fragments based on rings and their neighbors."""

    def _fuse_rings(rings):
        # Get ring atoms (systems of joined rings)
        all_rings = molecule.GetRingInfo().AtomRings()
        ring_systems = []
        for ring in all_rings:
            ring_atoms = set(ring)
            new_systems = []
            for system in ring_systems:
                shared = len(ring_atoms.intersection(system))
                if shared:
                    ring_atoms = ring_atoms.union(system)
                else:
                    new_systems.append(system)
            new_systems.append(ring_atoms)
            ring_systems = new_systems
        rings = [list(ring) for ring in ring_systems]
        return rings

    atoms, bonds = _get_ha_graph(molecule)
    atids = [a.GetIdx() for a in atoms]
    atom_neis_list = [[na.GetIdx() for na in a.GetNeighbors() if na.GetAtomicNum() > 1] for a in atoms]
    rings = _fuse_rings(molecule.GetRingInfo().AtomRings())
    atom_ring_ids = [_get_ring_id_of_atom(idx, rings) for idx in atids]
    n_rings = len(rings)
    fragments = [[atid for atid, segid in zip(atids, atom_ring_ids) if segid == x] for x in range(n_rings)]
    linear_atoms = [atid for atid, segid in zip(atids, atom_ring_ids) if segid == -1]
    for atom in linear_atoms:
        atom_nei = atom_neis_list[atom]
        if len(atom_nei) == 1: # if a terminal atom attached to a ring, add it to the ring fragment
            nei_ring_id = atom_ring_ids[atom_nei[0]]
            if nei_ring_id != -1:
                fragments[nei_ring_id].append(atom) 
        if atom not in flat_set(fragments): # if an atom with 2+ neighbors not in any of the fragments, make a new fragmnent
            fragments.append([atom] + atom_nei) # add linear atom as its own fragment if it 2+ neighbors
    return fragments, rings


@timeit(level=logging.INFO)
def find_bead_anchors(molecule, min_beads=None, max_beads=None, dtype=np.int32):
    """Try out all possible combinations of CG beads up to threshold number of beads per atom. Find
    arrangement with best energy score. Return all possible arrangements sorted by energy score.
    """
   
    def _get_min_max_beads(fragment, atoms):
        is_aromatic = any(atoms[a].GetIsAromatic() for a in fragment)
        n_atoms = len(fragment)
        if is_aromatic:
            min_beads = n_atoms // 2
            max_beads = n_atoms // 2 + n_atoms % 2
        else:
            min_beads = (n_atoms + n_atoms % 4) // 4
            max_beads = n_atoms // 2 
        return min_beads, max_beads

    @timeit(level=logging.DEBUG)
    def _find_bead_anchors(fragment, bonds, nbeads, dtype=np.int32):
        """Find acceptable combinations of anchor atoms for a given number of beads."""
        bonds = np.asarray(bonds, dtype=dtype)
        # all_combs = opcy.generate_combinations(int(n_atoms), int(nbeads), int(start_index), int(chunk_size))
        all_combs = np.array(list(itertools.combinations(fragment, nbeads)), dtype=dtype)
        logger.debug(f"Generated {all_combs.shape[0]} combinations.")
        acc_combs = opcy.find_acceptable_combinations(all_combs, bonds)
        logger.info(f"Found {acc_combs.shape[0]} acceptable combinations")
        return acc_combs


    @timeit(level=logging.DEBUG)
    def _filter_out_bad_combinations(combs, atoms_and_neighbors, natoms, dtype=np.int32):
        """Filter out bad combinations from acceptable_trials.
        If the sum of atoms and neighbors for a trial is greater than or equal to the total number of atoms,
        then we discard that trial, as it would not be a valid partitioning (some atoms would be in multiple beads).
        """
        filtered_list = []
        for trial in combs:
            trial_atoms_and_neighbors = flat_set([atoms_and_neighbors[idx] for idx in trial])
            if len(trial_atoms_and_neighbors) < natoms:
                continue
            filtered_list.append(trial)
        return filtered_list

    atoms, bonds = _get_ha_graph(molecule)

    n_heavy_atoms = len(atoms)
    if not min_beads:
        min_beads = (n_heavy_atoms + n_heavy_atoms % 4) // 4 
    if not max_beads:
        max_beads = n_heavy_atoms // 2 

    fragments, rings = split_into_fragments(molecule)
    atids = [a.GetIdx() for a in atoms]
    ha_neis = [[n.GetIdx() for n in a.GetNeighbors()] for a in atoms]
    ha_atoms_and_neis = [[a] + ha_neis[a] for a in atids]
    print(fragments)

    all_anchors = []
    for fragment in fragments:
        fragment_anchors = []
        min_fragment_beads, max_fragment_beads = _get_min_max_beads(fragment, atoms)
        for nbeads in range(min_fragment_beads, max_fragment_beads + 1):
            logger.info(f"Finding acceptable combinations for fragment with {len(fragment)} atoms and {nbeads} beads...")
            combs = _find_bead_anchors(fragment, bonds, nbeads, dtype=dtype)
            filtered_combs = _filter_out_bad_combinations(combs, ha_atoms_and_neis, natoms=len(fragment), dtype=dtype)
            fragment_anchors.extend(filtered_combs)
        all_anchors.append(fragment_anchors)
    logger.info(f"Number of combinations for fragment anchors: {len(all_anchors)}, Sizes: {[len(r) for r in all_anchors]}")

    merged_combs = all_anchors[0]
    for i in range(1, len(all_anchors)): 
        new_combs = []
        for comb1 in merged_combs:
            for comb2 in all_anchors[i]:
                merged = set(comb1).union(set(comb2))
                if len(merged) > max_beads:
                    continue
                new_combs.append(list(merged))
        merged_combs = new_combs
    logger.info(f"Total combinations of ring anchors: {len(merged_combs)}")

    
    logger.info("Collecting Combinations And Their Energies...")
    conformer = molecule.GetConformer()
    ringatoms_flat = [a for ring in rings for a in ring]
    list_trial_comb = []
    current_lowest_energy = float("inf")
    for n in range(min_beads, max_beads + 1):
        acceptable_trials = [comb for comb in merged_combs if len(comb) == n]
        if not acceptable_trials:
            continue
        acceptable_trials = np.array(acceptable_trials, dtype=dtype)
        list_trial_comb, ene_best_trial = collect_energies_and_combs(
            molecule,
            conformer,
            acceptable_trials,
            ringatoms_flat,
            current_lowest_energy,
            list_trial_comb,
        )

        if ene_best_trial >= current_lowest_energy:
            break
        current_lowest_energy = ene_best_trial

    sorted_combs = sorted(list_trial_comb, key=itemgetter(1))
    for comb in sorted_combs[::10]:
        logger.info("Combination length: %s, Energy: %f", len(comb[0]), comb[1])
    return_list = [x[0] for x in sorted_combs]
    return return_list


@timeit(level=logging.INFO)
def find_acceptable_trials_mp(
    list_heavy_atoms,
    num_beads,
    ring_atoms,
    list_bonds,
    atoms_and_neighbors,
    dtype=np.int32,
    chunk_size=int(1e7),
    nprocs=None,
):
    """Multiprocessing version of acceptable-trial filtering (Option 2).

    Each process generates its own chunk and filters it, then returns the full
    accepted combinations array back to the parent (IPC/pickling heavy).
    """
    if chunk_size >= 2147483647:
        chunk_size = 2147483646
    bonds = np.asarray(list_bonds, dtype=dtype)

    max_atom = bonds.max()
    ring_id = np.full(max_atom + 1, -1, dtype=dtype)
    for rid, ring in enumerate(ring_atoms):
        ring = np.asarray(ring, dtype=dtype)
        ring_id[ring] = rid
    nrings = len(ring_atoms)

    n_heavy_atoms = len(list_heavy_atoms)
    total = math.comb(int(n_heavy_atoms), int(num_beads))
    n_chunks = (total + int(chunk_size) - 1) // int(chunk_size)

    if nprocs is None:
        nprocs = int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))
    nprocs = max(1, int(nprocs))

    # Avoid OpenMP oversubscription if the cython filter uses OpenMP internally.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    work = [
        (chunk_num, n_heavy_atoms, num_beads, chunk_size, bonds, ring_id, nrings)
        for chunk_num in range(int(n_chunks))
    ]

    acc_trials_list = []
    with mp.Pool(processes=nprocs) as pool:
        for chunk_acceptable in pool.imap_unordered(_filter_chunk, work, chunksize=1):
            if chunk_acceptable.size:
                acc_trials_list.append(chunk_acceptable)

    filtered_list = _filter_out_bad_combinations(acc_trials_list, atoms_and_neighbors, 
        natoms=len(atoms_and_neighbors), dtype=dtype)

    if not filtered_list:
        return np.empty((0, int(num_beads)), dtype=dtype)

    return np.vstack(filtered_list)


def _filter_chunk(args):
    """Multiprocessing worker: generate one chunk and return acceptable combos.

    Notes
    -----
    * Must be top-level for multiprocessing pickling.
    * Returns a numpy array (can be large). This is the "Option 2" approach.
    """
    chunk_num, n_heavy_atoms, num_beads, chunk_size, bonds, ring_id, nrings = args
    start_index = int(chunk_num) * int(chunk_size)
    chunk_array = opcy.generate_combinations(
        int(n_heavy_atoms), int(num_beads), int(start_index), int(chunk_size)
    )
    if chunk_array.size == 0:
        return np.empty((0, int(num_beads)), dtype=np.int32)
    return opcy.find_acceptable_combinations(chunk_array, bonds, ring_id, nrings)


def _filter_out_bad_combinations_tmp(acc_trials_list, atoms_and_neighbors, natoms, dtype=np.int32):
    """Filter out bad combinations from acceptable_trials.
    If the sum of atoms and neighbors for a trial is greater than or equal to the total number of atoms,
    then we discard that trial, as it would not be a valid partitioning (some atoms would be in multiple beads).
    """
    filtered_list = []
    for chunk in acc_trials_list:
        for trial in chunk:
            summ = 0
            for idx in trial:
                summ += len(atoms_and_neighbors[idx])
            if summ >= natoms:
                filtered_list.append(trial)
    return filtered_list


@timeit(level=logging.INFO)
def find_acceptable_trials(atoms, bonds, nbeads, dtype=np.int32, chunk_size=int(1e5)):
    """Filter acceptable trial combinations, processing in memory-efficient chunks."""
    if chunk_size >= 2147483647:
        chunk_size = 2147483646
    logger.info("Max chunk size: %d", chunk_size)
    bonds = np.asarray(list_bonds, dtype=dtype)
  
    # Process chunks 
    acceptable_trials_list = []
    n_atoms = len(atoms)    
    total = math.comb(int(n_atoms), int(nbeads))
    n_chunks = (total + int(chunk_size) - 1) // int(chunk_size)

    logger.info("Starting processing chunks")
    for chunk_num in range(n_chunks):
        start_index = int(chunk_num) * int(chunk_size)
        chunk_array = opcy.generate_combinations(int(n_atoms), int(nbeads), int(start_index), int(chunk_size))
        logger.info(f"Processing chunk {chunk_num} ({chunk_array.shape[0]} trials)")
        if chunk_array.size == 0:
            break
        chunk_acceptable = opcy.find_acceptable_combinations(chunk_array, bonds, ring_id, nrings)
        if chunk_acceptable.size > 0:
            acceptable_trials_list.append(chunk_acceptable)
    
    return np.vstack(acceptable_trials_list)


@timeit(level=logging.INFO)
def collect_energies_and_combs(
    molecule,
    conformer,
    acceptable_trials,
    ringatoms_flat,
    ene_best_trial,
    list_trial_comb,
    dtype=np.int32
):
    """Collect energies and combinations for all acceptable trials"""
    logger.debug("Entering collect_energies_and_combs()") 
    # Trial positions: any heavy atom
    bead_params = read_bead_params()
    bond_dists = _get_bond_distances(conformer)
    masses = _get_masses(molecule)

    # Precompute ring mask once
    n_atoms = bond_dists.shape[0]
    is_ring = np.zeros(n_atoms, dtype=np.uint8)
    for a in ringatoms_flat:
        ia = int(a)
        if 0 <= ia < n_atoms:
            is_ring[ia] = 1

    # Scalarize bead params once (avoid dict lookups in the inner loop)
    p_offset = float(bead_params["offset_bd_weight"])
    p_offset_ar = float(bead_params["offset_bd_aromatic_weight"])
    p_lonely = float(bead_params["lonely_atom_penalize"])
    p_overlap = float(bead_params["bd_bd_overlap_coeff"])
    p_at_in = float(bead_params["at_in_bd_coeff"])
    p_rvdw = float(bead_params["rvdw"])
    p_rvdw_ar = float(bead_params["rvdw_aromatic"])
    p_rvdw_cross = float(bead_params["rvdw_cross"])
    
    ene_best_trial, energies_array = opcy.collect_energies(
        acceptable_trials,
        is_ring,
        bond_dists,
        masses,
        p_offset,
        p_offset_ar,
        p_lonely,
        p_overlap,
        p_at_in,
        p_rvdw,
        p_rvdw_ar,
        p_rvdw_cross,
        ene_best_trial,
    )
    list_trial_comb.extend([[acceptable_trials[i], energies_array[i]] for i in range(len(energies_array))])
    
    return list_trial_comb, ene_best_trial


#############################################################################
### TO FINISH PARTITIONING ###
#############################################################################

def get_partitioning(self, trial_comb):
        """Get partitioning of atoms into beads for given trial combination"""
        atoms = self.ha_graph["atoms"]
        mapping = self._distribute_neighbors(trial_comb, atoms)
        mapping_dict = {idx: bead for idx, bead in enumerate(mapping)}
        partitioning = self.invert_mapping_dictionary(mapping_dict)
        return partitioning


def _distribute_neighbors(trial_comb, atoms):
    """Find acceptable mappings of atoms to beads for given trial combination"""

    def _single_atom_in_mapping(mapping):
        for ns in mapping:
            if len(ns) == 1:
                return True
        return False

    def _ring_beads_are_tiny(mapping, bead_is_in_ring):
        for ns, ring in zip(mapping, bead_is_in_ring):
            if ring and len(ns) > 2:
                return False
        return True

    def _ring_beads_are_together(mapping, bead_is_in_ring, atom_is_in_ring):
        for ns, ring in zip(mapping, bead_is_in_ring):
            if not ring:
                continue
            for atom in ns:
                if not atom_is_in_ring[atom]:
                    return False
        return True

    def _beads_are_big(mapping, max_bead_size=4):
        for ns in mapping:
            if len(ns) > max_bead_size:
                return True
        return False

    bead_neighbors = [atoms[i]["neighbors"] for i in trial_comb]
    bead_is_in_ring = [atoms[i]["is_in_ring"] for i in trial_comb]
    atom_is_in_ring = [a["is_in_ring"] for a in atoms]
    n_atoms = len(atoms)
    atom_ids = set(range(n_atoms))
    nei_ids = atom_ids - set(trial_comb)
    mapping = [[int(i)] for i in trial_comb]
    mappings = [mapping]

    # Distribute neighbors of trial combination atoms to beads,
    # keeping track of all possible mappings.
    for nei_idx in nei_ids:
        updated_mappings = []
        for mapping in mappings:
            for idx, bead in enumerate(mapping):
                if nei_idx not in bead_neighbors[idx]:
                    continue
                tmp_mapping = [x.copy() for x in mapping]
                tmp_mapping[idx].append(nei_idx)
                updated_mappings.append(tmp_mapping)
        mappings = updated_mappings

    # Filter out mappings with single atoms in beads
    tmp_list = []
    for mapping in mappings:
        if _single_atom_in_mapping(mapping):
            continue
        tmp_list.append(mapping)
    mappings = tmp_list
    if len(mappings) == 1:
        return mappings[0]

    # Prefer keeping ring beads small
    tmp_list = []
    for mapping in mappings:
        if not _ring_beads_are_tiny(mapping, bead_is_in_ring):
            continue
        tmp_list.append(mapping)
    if tmp_list:
        mappings = tmp_list
    if len(mappings) == 1:
        return mappings[0]

    # Prefer keeping ring beads together (no mixing ring/non-ring)
    tmp_list = []
    for mapping in mappings:
        if not _ring_beads_are_together(mapping, bead_is_in_ring, atom_is_in_ring):
            continue
        tmp_list.append(mapping)
    if tmp_list:
        mappings = tmp_list
    if len(mappings) == 1:
        return mappings[0]

    # Prefer smaller beads overall (e.g. 5+ atoms is too big for Martini)
    tmp_list = []
    for mapping in mappings:
        if _beads_are_big(mapping):
            continue
        tmp_list.append(mapping)
    if tmp_list:
        mappings = tmp_list
    if len(mappings) == 1:
        return mappings[0]

    # TODO: SYMMETRIZE MAPPINGS
    if len(mappings) == 2:
        return mappings[0]

    return mappings[0]

#############################################################################
### HELPER FUNCTIONS FOR MAPPING DICTIONARIES ###
#############################################################################

def make_mapping_dictionary(atom_partitioning):
    """Create mapping dictionary from atom_partitioning"""
    mapping_dict = {}
    for atom_idx, bead_idx in atom_partitioning.items():
        if bead_idx not in mapping_dict:
            mapping_dict[bead_idx] = []
        mapping_dict[bead_idx].append(atom_idx)
    return mapping_dict


def invert_mapping_dictionary(mapping_dict):
    """Inverse of make_mapping_dictionary(): bead_idx -> [atom_idx] to atom_idx -> bead_idx."""
    atom_partitioning = {}
    for bead_idx, atom_indices in mapping_dict.items():
        for atom_idx in atom_indices:
            if atom_idx in atom_partitioning:
                raise ValueError(f"Atom {atom_idx} appears in multiple beads")
            atom_partitioning[atom_idx] = bead_idx
    return dict(sorted(atom_partitioning.items()))


### AutoM3 change :  Including Ertl Functional Groups Finder algorithm (merge, identify_functional_groups) ###
def identify_functional_groups(mol): # AutoM3 change
    # atoms connected by non-aromatic double or triple bond to any heteroatom
    PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
    # atoms in non-aromatic carbon-carbon double or triple bonds
    PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
    # acetal carbons, i.e. sp3 carbons connected to two or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
    PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
    # all atoms in oxirane, aziridine and thiirane rings
    PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')
    # the bridge between two aromatic cycles
    PATT_BRIDGE_AROMATIC = Chem.MolFromSmarts("[x;!x2]")

    PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC, PATT_BRIDGE_AROMATIC)

    marked = set()
    # mark all heteroatoms in a molecule, including halogens
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 1):  # would we ever have hydrogen?
            marked.add(atom.GetIdx())

    # mark the four specific types of carbon atom
    for patt in PATT_TUPLE:
        for path in mol.GetSubstructMatches(patt):
            for atomindex in path:
                marked.add(atomindex)

    # merge all connected marked atoms to a single FG
    groups = []
    while marked:
        grp = set([marked.pop()])
        merge(mol, marked, grp)
        groups.append(grp)

    # extract also connected unmarked carbon atoms
    ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type', 'type_atomIds'])
    ifgs = []
    for g in groups:
        uca = set()
        for atomidx in g:
            for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        type_atoms = g.union(uca)
        ifgs.append(
            ifg(atomIds=tuple(sorted(g)),
                atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True),
                type=Chem.MolFragmentToSmiles(mol, type_atoms, canonical=True),
                type_atomIds=tuple(sorted(type_atoms)))
        )
    """for ix, fg in enumerate(ifgs):
        print(f'Functional Group {ix + 1}:')
        print(f'  Atom Indices: {fg.atomIds}')
        print(f'  Atoms (SMILES): {fg.atoms}')
        print(f'  Group Type (SMILES): {fg.type}')
        print(f'  Group Type Atom Indices: {fg.type_atomIds}')"""
    
    """
    USE:
    m = Chem.MolFromSmiles(smiles)
    fgs = identify_functional_groups(m)
    print('%2d: %d fgs' % (ix + 1, len(fgs)), fgs)
    """
    return ifgs


def merge(mol, marked, aset): # AutoM3 change
    #  Original authors: Richard Hall and Guillaume Godin
    #  This file is part of the RDKit.
    #  The contents are covered by the terms of the BSD license
    #  which is included in the file license.txt, found at the root
    #  of the RDKit source tree.
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)