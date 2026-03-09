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

def flat_set(lst):
    """Flatten a list of lists into a set of unique elements."""
    if not lst:
        return set()
    aset = set(item for sublist in lst for item in sublist) 
    # alist = sorted(aset)
    return aset

def sort_nested(lst):
    """Sort a nested list of lists."""
    return sorted([sorted(sublist) for sublist in lst])

#############################################################################
### GRAPH FUNCTIONS ###
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


def _fuse_rings(molecule):
    # Get ring atoms (systems of joined rings)
    all_rings = molecule.GetRingInfo().AtomRings()
    ring_systems = []
    for ring in all_rings:
        ring_atoms = set(ring)
        new_systems = []
        shared_atoms = []
        for system in ring_systems:
            shared = ring_atoms.intersection(system)
            if shared:
                shared_atoms.append(shared)
                ring_atoms = ring_atoms.union(system)
            else:
                new_systems.append(system)
        new_systems.append(ring_atoms)
        ring_systems = new_systems
    rings = [list(ring) for ring in ring_systems]
    return rings, shared_atoms


def _remove_shared_atoms_from_bonds(bonds, shared_atoms):
    """Remove bonds between shared atoms, since they will be part of the same fragment.
    Otherwise, the mapping of the shared atoms to beads may not be consistent across fragments.
    """
    shared_atoms_flat = flat_set(shared_atoms)
    new_bonds = []
    for bond in bonds:
        if bond[0] in shared_atoms_flat and bond[1] in shared_atoms_flat:
            continue
        new_bonds.append(bond)
    return new_bonds


def _get_ring_id_of_atom(atom, rings, dtype=np.int32):
    """ring_id_of_atom[atom_id] = ring index, or 0 when not in any ring."""
    for rid, ring in enumerate(rings):
        if atom in ring:
            return rid      
    return -1
    

def _split_into_fragments(molecule):
    """Split molecule into fragments based on rings and their neighbors."""
    atoms, bonds = _get_ha_graph(molecule)
    atids = [a.GetIdx() for a in atoms]
    atom_neis_list = [[na.GetIdx() for na in a.GetNeighbors() if na.GetAtomicNum() > 1] for a in atoms]
    rings, shared_atoms = _fuse_rings(molecule)
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
            atoms_to_add = [atom] + atom_nei
            atoms_to_add = [a for a in atoms_to_add if a not in flat_set(fragments)]
            fragments.append(atoms_to_add) # add linear atom as its own fragment if it 2+ neighbors
    return sort_nested(fragments), sort_nested(rings), shared_atoms

#############################################################################
### PARTITIONING ###
#############################################################################

@timeit(level=logging.DEBUG)
def map_fragment(fragment, atoms, bonds, dtype=np.int32):
    """Map a fragment to beads, trying out all combinations of anchor atoms for different numbers of beads."""

    def get_min_max_beads(fragment, atoms):
        is_aromatic = any(atoms[a].GetIsAromatic() for a in fragment)
        is_in_ring = any(atoms[a].IsInRing() for a in fragment)
        n_atoms = len(fragment)
        if is_aromatic:
            min_beads = n_atoms // 2
            max_beads = n_atoms // 2 + n_atoms % 2
            return min_beads, max_beads
        if is_in_ring:
            min_beads = n_atoms // 3
            if n_atoms % 3 != 0:
                min_beads += 1
            max_beads = n_atoms // 2 + n_atoms % 2
            return min_beads, max_beads
        min_beads = n_atoms // 4
        if n_atoms % 4 != 0:
            min_beads += 1
        max_beads = n_atoms // 2 + 1
        return min_beads, max_beads

    @timeit(level=logging.DEBUG)
    def find_anchors(fragment, bonds, nbeads, dtype=np.int32):
        """Find acceptable combinations of anchor atoms for a given number of beads."""
        bonds = np.asarray(bonds, dtype=dtype)
        # all_combs = opcy.generate_combinations(int(n_atoms), int(nbeads), int(start_index), int(chunk_size))
        all_combs = np.array(list(itertools.combinations(fragment, nbeads)), dtype=dtype)
        logger.debug(f"Generated {all_combs.shape[0]} combinations.")
        acc_combs = opcy.find_acceptable_combinations(all_combs, bonds)
        logger.debug(f"Found {acc_combs.shape[0]} acceptable combinations")
        return acc_combs

    def find_no_overlap_mappings(combs, fragment):
        """Map a fragment to a set of beads based on the trial combinations."""
        mappings = []
        for comb in combs:
            initial_mapping = [[int(i)] + ha_neis[int(i)] for i in comb]
            no_mappings = distribute_neis(initial_mapping)
            for mapping in no_mappings:
                if mapping in mappings:
                    continue
                mappings.append(mapping) 
        logger.debug(f"Number of mappings: {len(mappings)}")
        no_mappings = [] # non overlapping mappings
        return mappings

    def distribute_neis(mapping):
        n = len(mapping)
        mapping = [set(ns) for ns in mapping]
        mappings = [mapping]
        for i in range(n):
            for j in range(i + 1, n):
                new_mappings = []
                for mapping in mappings:
                    s1 = mapping[i]
                    s2 = mapping[j]
                    overlap = s1.intersection(s2)
                    if overlap:
                        new_bead_1 = s1 - overlap
                        if len(new_bead_1) > 1:
                            new_mapping_1 = mapping.copy()
                            new_mapping_1[i] = new_bead_1
                            new_mappings.append(new_mapping_1)
                        new_bead_2 = s2 - overlap
                        if len(new_bead_2) > 1:
                            new_mapping_2 = mapping.copy()
                            new_mapping_2[j] = new_bead_2
                            new_mappings.append(new_mapping_2)
                if new_mappings:
                    mappings = new_mappings
        unique_mappings = []
        for mapping in mappings:
            mapping = sort_nested(mapping)
            if mapping not in unique_mappings:
                unique_mappings.append(mapping)
        return unique_mappings

    ha_neis = [[n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1] for a in atoms]
    fragment_mappings = []
    min_fragment_beads, max_fragment_beads = get_min_max_beads(fragment, atoms)
    for nbeads in range(min_fragment_beads, max_fragment_beads + 1):
        logger.info(f"Finding acceptable combinations for fragment with {len(fragment)} atoms and {nbeads} beads...")
        combs = find_anchors(fragment, bonds, nbeads, dtype=dtype)
        mappings = find_no_overlap_mappings(combs, fragment)
        fragment_mappings.extend(mappings)
    return fragment_mappings


@timeit(level=logging.INFO)
def generate_mappings(molecule, min_beads=None, max_beads=None, dtype=np.int32):
    """Try out all possible combinations of CG beads up to threshold number of beads per atom. Find
    arrangement with best energy score. Return all possible arrangements sorted by energy score.
    """

    def map_connection(mapping):
        n = len(mapping)
        if len(flat_set(mapping)) < 4:
            return [[sorted(list(flat_set(mapping)))]]
        mapping = [set(ns) for ns in mapping]
        s1 = mapping[0]
        s2 = mapping[1]
        overlap = s1.intersection(s2)
        if not overlap:
            return [sort_nested(mapping)]
        # if len(overlap) > 1:
        #     return []
        mappings = []
        new_bead_1 = s1 - overlap
        if len(new_bead_1) > 1:
            new_mapping_1 = sort_nested([new_bead_1, s2])
            mappings.append(new_mapping_1)
        new_bead_2 = s2 - overlap
        if len(new_bead_2) > 1:
            new_mapping_2 = sort_nested([s1, new_bead_2])
            mappings.append(new_mapping_2)
        return mappings
    
    atoms, bonds = _get_ha_graph(molecule)
    fragments, rings, shared_atoms = _split_into_fragments(molecule)
    bonds = _remove_shared_atoms_from_bonds(bonds, shared_atoms)
    atids = [a.GetIdx() for a in atoms]
    ha_neis = [[n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1] for a in atoms]
    ha_atoms_and_neis = [[a] + ha_neis[a] for a in atids]
    # new_fragments = [fragments[-2], fragments[-1]]
    # fragments = new_fragments

    # Map each fragment to beads, and collect all the combinations of mappings for each fragment
    all_mappings = []
    for fragment in fragments:
        fragment_mappings = map_fragment(fragment, atoms, bonds)
        all_mappings.append(fragment_mappings)
    logger.info(f"Number of combinations for fragment mappings. Sizes: {[len(r) for r in all_mappings]}")

    # Fragments and their nearest neighbors
    fns = [flat_set([[a] + ha_neis[a] for a in atids]) for atids in fragments]
    fragment_connections = []
    for i in range(len(fns)):
        for j in range(i + 1, len(fns)):
            overlap = set(fns[i]).intersection(set(fns[j]))
            if len(overlap) > 1:
                fragment_connections.append((i, j, list(overlap)))

    # Merge all the fragments
    merged_mappings = all_mappings[0] 
    merged_fragment = fragments[0]
    ids_to_map = list(range(1, len(fragments)))
    for conn in fragment_connections:
        i, j = conn[0], conn[1] 
        overlap = conn[2]
        id_to_map = j if j in ids_to_map else i
        ids_to_map.remove(id_to_map)
        mappings_to_add = all_mappings[id_to_map]
        new_mappings = []
        merged_fragment += fragments[id_to_map]
        for m1 in merged_mappings:
            for m2 in mappings_to_add:
                ol_beads_1 = [bead for bead in m1 if any(a in bead for a in overlap)]
                ol_beads_2 = [bead for bead in m2 if any(a in bead for a in overlap)]
                if ol_beads_1 and ol_beads_2:
                    ol_bead_1 = ol_beads_1[0]
                    m1_copy = m1.copy()
                    m1_copy.remove(ol_bead_1)
                    ol_bead_2 = ol_beads_2[0]
                    m2_copy = m2.copy()
                    m2_copy.remove(ol_bead_2)
                    connection = [ol_bead_1, ol_bead_2]
                    mappings = map_connection(connection)
                    for mapping in mappings:
                        new_mapping = m1_copy + m2_copy + mapping
                        new_mapping = sort_nested(new_mapping)
                        mapping_flat = flat_set(new_mapping)
                        all_atoms_are_covered = set(merged_fragment).issubset(mapping_flat)
                        # if not all_atoms_are_covered:
                        #     continue
                        if new_mapping in new_mappings:
                            continue
                        new_mappings.append(new_mapping)
                else:
                    new_mapping = m1 + m2
                    mapping_flat = flat_set(new_mapping)
                    all_atoms_are_covered = set(merged_fragment).issubset(mapping_flat)
                    if not all_atoms_are_covered:
                        continue
                    if new_mapping in new_mappings:
                        continue
                    new_mappings.append(new_mapping)
        merged_mappings = new_mappings

    mappings = sorted(merged_mappings, key=lambda m: len(m), reverse=True)    
    mappings = [m for m in mappings if len(m) < 10]
    print(len(mappings))
    for mapping in mappings[:10]:
        print(len(mapping), mapping)
    return mappings

    # logger.info(f"Total combinations of fragments: {len(merged_combs)}")
 
    # logger.info("Sorting Combinations By Their Energies...")
    # conformer = molecule.GetConformer()
    # ringatoms_flat = [a for ring in rings for a in ring]
    # list_trial_comb = []
    # current_lowest_energy = float("inf")
    # for n in range(min_beads, max_beads + 1):
    #     acceptable_trials = [comb for comb in merged_combs if len(comb) == n]
    #     if not acceptable_trials:
    #         continue
    #     acceptable_trials = np.array(acceptable_trials, dtype=dtype)
    #     list_trial_comb, ene_best_trial = collect_energies_and_combs(
    #         molecule,
    #         conformer,
    #         acceptable_trials,
    #         ringatoms_flat,
    #         current_lowest_energy,
    #         list_trial_comb,
    #     )

    #     if ene_best_trial >= current_lowest_energy:
    #         break
    #     current_lowest_energy = ene_best_trial

    # sorted_combs = sorted(list_trial_comb, key=itemgetter(1))
    # return_list = [x[0] for x in sorted_combs]
    # return_list = [list(map(int, comb)) for comb in return_list]
    # logger.info(f"Final number of combinations: {len(return_list)}")
    # return return_list


@timeit(level=logging.DEBUG)
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


def _distribute_atoms(trial_comb, atom_ids, atom_neighbors, atom_ring_ids, atom_is_in_ring):
    """Find acceptable mappings of atoms to beads for given trial combination"""

    def _single_atom_in_mapping(mapping):
        for ns in mapping:
            if len(ns) == 1:
                return True
        return False

    def _ring_beads_are_tiny(mapping):
        for ns, ring in zip(mapping, bead_is_in_ring):
            if ring and len(ns) > 2:
                return False
        return True

    def _ring_beads_are_together(mapping):
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

    bead_neighbors = [atom_neighbors[i] for i in trial_comb]
    print(bead_neighbors)
    bead_is_in_ring = [atom_ring_ids[i] != -1 for i in trial_comb]
    n_atoms = len(atom_ids)
    nei_ids = set(atom_ids) - set(trial_comb)
    mapping = [set([i] + atom_neighbors[i]) for i in trial_comb]
    print(mapping)
    mappings = [mapping]
    print(mappings)
    exit()
    
    # # Distribute neighbors of trial combination atoms to beads,
    # # keeping track of all possible mappings.
    # for nei_idx in nei_ids:
    #     updated_mappings = []
    #     for mapping in mappings:
    #         for idx, bead in enumerate(mapping):
    #             if nei_idx not in bead_neighbors[idx]:
    #                 continue
    #             tmp_mapping = [x.copy() for x in mapping]
    #             tmp_mapping[idx].append(nei_idx)
    #             updated_mappings.append(tmp_mapping)
    #     mappings = updated_mappings



    # Filter out mappings with single atoms in beads
    tmp_list = []
    for mapping in mappings:
        if _single_atom_in_mapping(mapping):
            continue
        tmp_list.append(mapping)
    mappings = tmp_list
    if len(mappings) == 1:
        return mappings

    # Prefer keeping ring beads small
    tmp_list = []
    for mapping in mappings:
        if not _ring_beads_are_tiny(mapping):
            continue
        tmp_list.append(mapping)
    if tmp_list:
        mappings = tmp_list
    if len(mappings) == 1:
        return mappings

    # Prefer keeping ring beads together (no mixing ring/non-ring)
    tmp_list = []
    for mapping in mappings:
        if not _ring_beads_are_together(mapping):
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
        return mappings

    # TODO: SYMMETRIZE MAPPINGS
    if len(mappings) == 2:
        return mappings
    
    if not mappings:
        return None

    return mappings

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