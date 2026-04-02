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

from .common import *
from .utils import timeit, memprofit
from . import optimization_cy as opcy
import multiprocessing as mp
from config import CFG

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
    """Build the heavy-atom graph used by the mapping workflow.

    Returns
    -------
    tuple[list, list[list[int, int]]]
        - heavy atoms (non-hydrogen RDKit atom objects),
        - undirected heavy-atom bond list as ``[i, j]`` pairs.

    Notes
    -----
    Partitioning is done on heavy atoms only. Hydrogens are removed up front so
    all subsequent graph logic uses a stable, reduced representation.
    """
    molecule = Chem.RemoveHs(molecule)
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


def _remove_shared_atoms_from_bonds(bonds, shared_atoms):
    """Drop bonds fully inside inter-fragment overlap atoms.

    Shared atoms are intentionally handled during fragment stitching; keeping
    overlap-overlap bonds in per-fragment mapping can over-constrain local
    anchor selection and lead to inconsistent bead assignment across fragments.
    """
    shared_atoms_flat = flat_set(shared_atoms)
    new_bonds = []
    for bond in bonds:
        if bond[0] in shared_atoms_flat and bond[1] in shared_atoms_flat:
            continue
        new_bonds.append(bond)
    return new_bonds
    

def split_into_fragments(molecule):
    """Decompose the molecule into overlapping ring/linear fragments.

    Logic overview
    --------------
    1. Detect and fuse ring systems that share atoms.
    2. Build ring-centered fragments (ring atoms + nearest heavy neighbors).
    3. Build linear fragments around non-ring branching atoms.
    4. Assign leftover atoms to nearby fragments so every heavy atom is covered.
    5. Keep overlaps small (ideally one atom per connection) to simplify
       downstream fragment stitching.

    Why this decomposition is used
    ------------------------------
    Enumerating mappings on the whole molecule is combinatorially expensive.
    Fragment-first mapping keeps search tractable while preserving chemically
    important local contexts (rings and branching points), then reconciles
    overlaps during merge.

    Returns
    -------
    tuple
        ``(fragments, frag_ranks_list, rings, shared_atoms)`` where
        ``fragments`` are heavy-atom index lists used by the mapper.
    """

    def fuse_rings(molecule):
        # Get ring atoms (systems of joined rings)
        rings = molecule.GetRingInfo().AtomRings()
        rings = [set(ring) for ring in rings if len(ring) < CFG.max_ring_len] # Large rings are usually not aromatic and can be broken up into smaller fragments
        n_rings = len(rings)
        fused_rings = []
        overlaps = []
        for r1 in rings:
            for r2 in rings:
                if r1 == r2:
                    continue
                overlap = r1.intersection(r2)
                if overlap:
                    rings.append(r1.union(r2))
                    rings.remove(r1)
                    rings.remove(r2)
                    overlaps.append(overlap)
        rings = sort_nested(rings)
        overlaps = sort_nested(overlaps)
        return rings, overlaps

    molecule = Chem.RemoveHs(molecule)
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    atoms, bonds = _get_ha_graph(molecule)
    atids = [a.GetIdx() for a in atoms]
    ha_neis = [[na.GetIdx() for na in a.GetNeighbors() if na.GetAtomicNum() > 1] for a in atoms]
    rings, shared_atoms = fuse_rings(molecule)
    # Initiate ring fragmesnts as rings + their nearest neighbors,
    # and remove shared atoms from ring fragments so that the overlap between fragments is at most 1 atom per connection
    n_rings = len(rings)
    ring_fragments = [list(flat_set([ha_neis[a] + [a] for a in ring])) for ring in rings]
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            for atom in ring_fragments[i]:
                if atom in rings[j]:
                    ring_fragments[i].remove(atom)
    # Make linear fragments
    nt_linear_atoms = [a.GetIdx() for a in atoms if a.GetIdx() not in flat_set(ring_fragments) and a.GetDegree() > 1]
    ring_attached_atoms = flat_set(ring_fragments) - flat_set(rings)
    nt_ring_attached_atoms = [a.GetIdx() for a in atoms if a.GetIdx() in ring_attached_atoms and a.GetDegree() > 1]
    nt_linear_atoms = sorted(nt_linear_atoms, key=lambda a: atoms[a].GetDegree(), reverse=True) # sort by rank so that we start mapping from the most important atoms (e.g. branching points) to try to preserve them as anchors if possible
    linear_fragments = []
    for atom in nt_linear_atoms:
        if atom in flat_set(linear_fragments):
            continue
        atom_neis = ha_neis[atom]
        atom_and_neis = [atom] + atom_neis
        atoms_to_add = [nei for nei in atom_and_neis if nei not in flat_set(rings)]
        if len(atoms_to_add) > 2: # don't allow small fragments, they will constrain mapping
            linear_fragments.append(atoms_to_add) # add linear atom as its own fragment if it 2+ neighbors
    for atom in nt_ring_attached_atoms:
        for frag in linear_fragments:
            atom_neis = ha_neis[atom]
            if any(nei in frag for nei in atom_neis):
                frag.append(atom)
                break
    # Add leftover atoms to existing fragments
    fragments = ring_fragments + linear_fragments
    fragments_flat = flat_set(fragments)
    leftover_atoms = [a.GetIdx() for a in atoms if a.GetIdx() not in fragments_flat]
    for atom in leftover_atoms:
        for nei in ha_neis[atom]:
            for frag in (linear_fragments + ring_fragments):
                if nei in frag:
                    frag.append(atom)
                    break
            break
    fragments = sort_nested(ring_fragments) + sort_nested(linear_fragments)
    if not len(flat_set(fragments)) == len(atoms):
        missing_atoms = set(atids) - flat_set(fragments)
        raise ValueError(f"Error in fragment generation: {missing_atoms} atoms are missing from the fragments. Check your fragments. Fragments: {fragments}")
    frag_ranks_list = [[ranks[a] for a in frag] for frag in fragments]
    return fragments, frag_ranks_list, rings, shared_atoms 

#############################################################################
### PARTITIONING ###
#############################################################################

@timeit(level=logging.DEBUG)
def map_fragment(fragment, atoms, bonds, dtype=np.int32):
    """Enumerate feasible bead mappings for a single fragment.

    Logic overview
    --------------
    - Choose a bead-count search range from fragment chemistry
      (aromatic/ring/non-ring heuristics).
    - For each bead count, enumerate anchor-atom combinations.
    - Keep only connectivity-consistent anchor sets.
    - Expand each anchor set into provisional beads (anchor + neighbors).
    - Resolve overlapping atom assignments to produce non-overlapping mappings
      that still cover the full fragment.

    Returns
    -------
    list[list[list[int]]]
        Candidate mappings for the fragment. Each mapping is a list of beads,
        and each bead is a list of atom indices.
    """

    def get_min_max_beads(fragment, atoms):
        is_aromatic = any(atoms[a].GetIsAromatic() for a in fragment)
        is_in_ring = any(atoms[a].IsInRing() for a in fragment)
        n_atoms = len(fragment)
        if is_aromatic:
            min_beads = n_atoms // 3
            max_beads = n_atoms // 2
            return min_beads, max_beads
        if is_in_ring:
            min_beads = n_atoms // 3
            if n_atoms % 3 != 0:
                min_beads += 1
            max_beads = n_atoms // 2 
            return min_beads, max_beads
        min_beads = n_atoms // 4
        if min_beads == 0:
            min_beads = 1
        # if n_atoms % 4 != 0:
        #     min_beads += 1
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
            mapping = [[int(i)] + ha_neis[int(i)] for i in comb]
            # mapping = sort_nested(mapping)
            mapping_flat = flat_set(mapping)
            all_atoms_are_covered = set(fragment).issubset(mapping_flat)
            if not all_atoms_are_covered:
                continue
            no_mappings = distribute_neis(mapping)
            for mapping in no_mappings:
                mapping = sort_nested(mapping)
                if mapping in mappings:
                    continue
                mappings.append(mapping) 
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
                    else:
                        new_mappings.append(mapping)
                mappings = new_mappings
        return mappings

    frag_neis = [[n.GetIdx() for n in a.GetNeighbors()] for a in atoms]
    is_ring = any(atoms[a].IsInRing() for a in fragment)
    # ha_neis = frag_neis
    ha_neis = [[n for n in nei if n in fragment] for nei in frag_neis] # only consider neighbors in the fragment, since we will map each fragment separately and then merge the mappings.
    fragment_mappings = []
    min_fragment_beads, max_fragment_beads = get_min_max_beads(fragment, atoms)
    for nbeads in range(min_fragment_beads, max_fragment_beads + 1):
        logger.info(f"Finding combinations for fragment {fragment} with {nbeads} beads...")
        combs = find_anchors(fragment, bonds, nbeads, dtype=dtype)
        mappings = find_no_overlap_mappings(combs, fragment)
        fragment_mappings.extend(mappings)
    return fragment_mappings


@timeit(level=logging.INFO)
def generate_mappings(molecule, min_beads=None, max_beads=None, dtype=np.int32):
    """Generate whole-molecule mapping candidates via fragment-map-merge workflow.

    Logic overview
    --------------
    1. Build heavy-atom graph and split into overlapping fragments.
    2. Enumerate candidate mappings independently per fragment.
    3. Merge fragment mappings along overlap atoms.
    4. After each merge, filter and rank candidates to control combinatorial
       growth.
    5. Apply final molecule-level filtering and sorting.

    Design intent
    -------------
    This staged workflow trades exact global search for tractable, chemically
    guided exploration. Ring-aware fragmentation and overlap stitching preserve
    local structure while avoiding explosion in candidate count.

    Returns
    -------
    list[list[list[int]]]
        Sorted candidate mappings (best-first according to internal heuristics).
    """

    def find_overlaps(frag, other_frags):
        """Find overlaps between fragments. 
        The length of the overlap should be equal to the number of connections between the fragments.
        Meaning that we will need to stich each pair of fragments along each atom in the overlap
        """
        frag = set(frag)
        for idx in range(len(other_frags)):
            other_frag = set(other_frags[idx])
            overlaps = frag.intersection(other_frag)
            if not overlaps:
                continue
            other_frag = other_frags.pop(idx)
            logger.info(f"Overlap of {overlaps} between fragments {frag} and {other_frag}")
            return other_frag, idx, overlaps
        raise ValueError(
            f"No overlap found for fragment {frag} with any of the other fragments. "
            "Check your fragments"
        )

    def merge_fragments(m1, m2, overlaps):
        """m1 and m2 are the mappings of the two fragments to be merged along ONE connection, and overlap is ONE atom.
        Each connection will have to be merged separately, and the resulting mappings will be merged together at the end.
        """
        def distribute_lonely_atoms(mappings):
            """If there are any beads that only have 1 atom, move the atom to a neighboring bead"""
            new_mappings = []
            for mapping in mappings:
                lonely_beads = [bead for bead in mapping if len(bead) == 1]
                if not lonely_beads:
                    new_mappings.append(mapping)
                    continue
                lonely_bead = lonely_beads[0]
                lonely_atom = lonely_bead[0]
                neighboring_beads = [bead for bead in mapping if any(atom in bead for atom in ha_neis[lonely_atom])]
                if not neighboring_beads:
                    new_mappings.append(mapping)
                    continue
                for neighboring_bead in neighboring_beads:
                    new_mapping = mapping.copy()
                    new_mapping.remove(lonely_bead)
                    new_mapping.remove(neighboring_bead)
                    new_bead = neighboring_bead + lonely_bead
                    new_mapping.append(new_bead)
                    new_mapping = sort_nested(new_mapping)
                    if new_mapping not in new_mappings:
                        new_mappings.append(new_mapping)
            return new_mappings

        initial_mapping = m1 + m2
        mappings = [initial_mapping]
        for overlap in overlaps:
            stitched_mappings = []
            for mapping in mappings:
                overlapping_beads = [bead for bead in mapping if overlap in bead]
                if len(overlapping_beads) != 2:
                    raise ValueError(
                        f"Expected 2 overlapping beads for overlap {overlap} between fragments with mappings {m1} and {m2}, "
                        f"but found {len(overlapping_beads)}: {overlapping_beads}. Check your fragments and their overlaps."
                    )
                mapping_copy = mapping.copy()
                ol_bead_1 = overlapping_beads[0]
                ol_bead_2 = overlapping_beads[1]
                mapping_copy.remove(ol_bead_1)
                mapping_copy.remove(ol_bead_2)
                overlap_mappings = map_overlap(overlapping_beads, overlap)
                for mapping in overlap_mappings:
                    new_mapping = mapping_copy + mapping
                    stitched_mappings.append(new_mapping)
                stitched_mappings = distribute_lonely_atoms(stitched_mappings)
            mappings = stitched_mappings
        return mappings

    def map_overlap(beads, overlap):
        """Map overlapping beads when connecting the fragments"""

        def bead_is_connected(bead):
            return True
            for i in range(len(bead)):
                for j in range(i + 1, len(bead)):
                    if not [bead[i], bead[j]] in bonds and not [bead[j], bead[i]] in bonds:
                        return False
            return True
        
        logger.debug(f"Mapping overlap of {overlap} with beads {beads}")
        beads_set = flat_set(beads)
        mappings = []
        if len(beads_set) < 4:
            mappings.append([sorted(list(beads_set))])
        b1 = beads[0]
        b2 = beads[1]
        nb1 = b1.copy()
        nb1.remove(overlap)
        nb2 = b2.copy()
        nb2.remove(overlap)
        if len(nb1) > 0 and bead_is_connected(nb1):
            mapping_1 = sort_nested([nb1, b2])
            mappings.append(mapping_1)
        if len(nb2) > 0 and bead_is_connected(nb2):
            mapping_2 = sort_nested([b1, nb2])
            mappings.append(mapping_2)
        return mappings

    logger.info("Extracting heavy-atom graph...")
    atoms, bonds = _get_ha_graph(molecule)
    logger.info("Splitting molecule into fragments...")
    fragments, top_ranks_list, fused_rings, shared_atoms = split_into_fragments(molecule)
    frag_is_symmetric = [len(set(ranks)) < len(ranks) for ranks in top_ranks_list]
    logger.info(f"Total Number of Fragments: {len(fragments)}, Number of Rings: {len(fused_rings)}")
    bonds = _remove_shared_atoms_from_bonds(bonds, shared_atoms)
    atids = [a.GetIdx() for a in atoms]
    ha_neis = [[n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1] for a in atoms]
    ha_atoms_and_neis = [[a] + ha_neis[a] for a in atids]

    # # DEBUG
    # print(fragments)
    # print(frag_is_symmetric)
    # alist = [0, 1, 2, 3, 4, 5, 6]
    # alist = [0, 1, 2, 3, 4, 6]
    # new_fragments = [fragments[i] for i in alist]
    # fragments = new_fragments

    # Stage 1: local enumeration on each fragment.
    # We intentionally solve small local mapping problems first.
    all_mappings = []
    for fragment in fragments:
        fragment_mappings = map_fragment(fragment, atoms, bonds)
        all_mappings.append(fragment_mappings)
    logger.info(f"Number of combinations for fragment mappings. Sizes: {[len(r) for r in all_mappings]}")

    # Stage 2: progressive stitching of fragment mappings across overlaps.
    # At each merge step we filter/rank to keep the search size manageable.
    merged_mappings = all_mappings.pop(0) 
    merged_frag = fragments.pop(0)
    for x in range(len(fragments)):
        other_frag, other_index, overlaps = find_overlaps(merged_frag, fragments)
        print(f"Overlap of {overlaps} between fragments {merged_frag} and {other_frag}")
        mappings_to_add = all_mappings.pop(other_index)
        merged_frag += other_frag
        new_mappings = []
        for m1 in merged_mappings[:CFG.max_mappings_to_keep]: # only keep top mappings at each step to avoid combinatorial explosion
            for m2 in mappings_to_add:
                merged_mappings = merge_fragments(m1, m2, overlaps)
                new_mappings.extend(merged_mappings)
        merged_mappings = new_mappings
        merged_mappings = filter_mappings(merged_mappings, molecule, CFG.max_bead_size + 1, CFG.max_ring_bead_size)
        merged_mappings = sort_mappings(merged_mappings, molecule, fused_rings)
    mappings = merged_mappings
    logger.info(f"Total combinations of mappings: {len(mappings)}")

    # Stage 3: final global filtering/ranking on complete molecule mappings.
    logger.info("Filtering and sorting the mappings...")
    mappings = filter_mappings(mappings, molecule, fused_rings, CFG.max_bead_size, CFG.max_ring_bead_size)
    mappings = sort_mappings(mappings, molecule, fused_rings)
    print(len(mappings))
    for mapping in mappings[:10]:
        print(mapping)

    # tmp_mappings = []
    # count = 0
    # for mapping in mappings:
    #     if [17, 18, 20] in mapping and [25, 26, 28] in mapping and [37, 38, 39] in mapping and not [6, 7, 9] in mapping:
    #         count += 1
    #         print(len(mapping), mapping)
    #         tmp_mappings.append(mapping)
    #         if count >= 10:
    #             break
    # mappings = tmp_mappings
    return mappings


def sort_mappings(mappings, molecule, fused_rings):
    """Rank mapping candidates using coarse-graining and ring-preservation heuristics.

    Prioritization logic
    --------------------
    - favor stronger coarse graining (fewer total beads),
    - discourage terminal non-ring micro-beads,
    - discourage tiny ring-local beads that over-fragment rigid motifs.

    Returns
    -------
    list
        Sorted mapping list in preferred-first order.
    """

    def num_terminal_nonring_beads(mapping):
        count = 0
        for bead in mapping:
            is_in_ring = any(atom in flat_set(fused_rings) for atom in bead)
            is_terminal = any(atom_is_terminal[a] for a in bead)
            if not is_in_ring and is_terminal:
                count += 1
        return count

    def num_tiny_ring_beads(mapping):
        count = 0
        for ring in fused_extended_rings:
            for bead in mapping:
                all_in_ring = all(atom in ring for atom in bead)
                if all_in_ring and len(bead) == 2:
                    count += 1
        return count

    def num_whole_extended_ring_beads(mapping):
        count = 0
        for ring in fused_extended_rings:
            for bead in mapping:
                all_in_ring = all(atom in ring for atom in bead)
                if all_in_ring:
                    count += 1
        return count

    def sort_key(mapping):
        num_beads = len(mapping)
        num_terminal_nonring = num_terminal_nonring_beads(mapping)
        num_tiny_ring = num_tiny_ring_beads(mapping)
        num_whole_ring = num_whole_extended_ring_beads(mapping)
        return (num_beads, num_terminal_nonring, num_tiny_ring, )

    molecule = Chem.RemoveHs(molecule)
    atoms = molecule.GetAtoms()
    atids = [a.GetIdx() for a in atoms]
    rings = molecule.GetRingInfo().AtomRings()
    rings = [ring for ring in rings if len(ring) > 5] 
    atom_is_terminal = [a.GetDegree() == 1 for a in atoms]
    ha_terminal_neis = [[na.GetIdx() for na in a.GetNeighbors() if na.GetDegree() == 1] for a in atoms]
    fused_extended_rings = [list(flat_set([ha_terminal_neis[a] + [a] for a in ring])) for ring in fused_rings] # include the terminal neighbors
    mappings = sorted(mappings, key=lambda m: sort_key(m), reverse=True) # maximize number of beads, etc     
    return mappings


def filter_mappings(
    mappings, 
    molecule, 
    fused_rings,
    max_bead_size=CFG.max_bead_size, 
    max_ring_bead_size=CFG.max_ring_bead_size,
    keep_rings_together=CFG.keep_rings_together,
    ):
    """Filter mapping candidates by hard structural constraints.

    Filtering logic (applied progressively)
    ---------------------------------------
    1. Remove mappings containing single-atom beads.
    2. Remove mappings with oversized beads.
    3. Optionally enforce ring cohesiveness (no ring/non-ring mixing in a bead).
    4. Enforce maximum bead size for fully ring-local beads.
    5. Remove duplicate mappings after canonical sorting.

    Returns
    -------
    list
        Candidate mappings that satisfy the active constraints.
    """

    def single_atom_in_mapping(mapping):
        for bead in mapping:
            if len(bead) == 1:
                return True
        return False

    def beads_are_big(mapping):
        for bead in mapping:
            if len(bead) > max_bead_size:
                return True
        return False

    def ring_beads_are_small(mapping):
        for bead in mapping:
            is_in_ring = all(atom_is_in_ring[a] for a in bead)
            if is_in_ring and len(bead) > max_ring_bead_size:
                return False
        return True

    def ring_beads_are_together(mapping, ring):
        for bead in mapping:
            any_in_ring = any(atom in ring for atom in bead)
            all_in_ring = all(atom in ring for atom in bead)
            if any_in_ring and not all_in_ring:
                return False
        return True
    
    atoms, bonds = _get_ha_graph(molecule)
    atids = [a.GetIdx() for a in atoms]
    rings = molecule.GetRingInfo().AtomRings()
    rings = [ring for ring in rings if len(ring) > 5] 
    atom_is_aromatic = [a.GetIsAromatic() for a in atoms]
    atom_is_in_ring = [a.IsInRing() for a in atoms]

    # Filter out mappings with single atoms in beads
    tmp_list = []
    for mapping in mappings:
        if single_atom_in_mapping(mapping):
            continue
        tmp_list.append(mapping)
    mappings = tmp_list
    if len(mappings) == 1:
        return mappings

    # Prefer smaller beads overall 
    tmp_list = []
    for mapping in mappings:
        if beads_are_big(mapping):
            continue
        tmp_list.append(mapping)
    if tmp_list:
        mappings = tmp_list
    if len(mappings) == 1:
        return mappings

    if keep_rings_together:
        # Prefer keeping rings together (no mixing ring/non-ring)
        tmp_list = []
        for mapping in mappings:
            is_valid = True
            for ring in rings:
                if not ring_beads_are_together(mapping, ring):
                    is_valid = False
                    break
            if is_valid:
                tmp_list.append(mapping)
        if tmp_list:
            mappings = tmp_list
        if len(mappings) == 1:
            return mappings

    # Prefer keeping ring beads small
    tmp_list = []
    for mapping in mappings:
        if not ring_beads_are_small(mapping):
            continue
        tmp_list.append(mapping)
    if tmp_list:
        mappings = tmp_list
    if len(mappings) == 1:
        return mappings

    # Remove duplicates
    tmp_list = []
    for mapping in mappings:
        mapping = sort_nested(mapping)
        if mapping in tmp_list:
            continue
        tmp_list.append(mapping)
    if tmp_list:
        mappings = tmp_list
    return mappings

#############################################################################
### OLD STUFF ###
#############################################################################

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


def identify_functional_groups(mol): # AutoM3 change
    """AutoM3 change :  Including Ertl Functional Groups Finder algorithm (merge, identify_functional_groups)"""

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

#############################################################################
### HELPER FUNCTIONS FOR MAPPING DICTIONARIES ###
#############################################################################

def make_mapping_dictionary(atom_partitioning):
    """Convert atom->bead assignment to bead->atom-list representation."""
    mapping_dict = {}
    for atom_idx, bead_idx in atom_partitioning.items():
        if bead_idx not in mapping_dict:
            mapping_dict[bead_idx] = []
        mapping_dict[bead_idx].append(atom_idx)
    return mapping_dict


def invert_mapping_dictionary(mapping_dict):
    """Convert bead->atoms representation back to atom->bead assignment."""
    atom_partitioning = {}
    for bead_idx, atom_indices in mapping_dict.items():
        for atom_idx in atom_indices:
            if atom_idx in atom_partitioning:
                raise ValueError(f"Atom {atom_idx} appears in multiple beads")
            atom_partitioning[atom_idx] = bead_idx
    return dict(sorted(atom_partitioning.items()))