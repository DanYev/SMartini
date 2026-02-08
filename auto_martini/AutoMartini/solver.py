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

from . import output, topology
from . import optimization as optimization
from .common import *

logger = logging.getLogger(__name__)

def get_coords(conformer, sites, avg_pos, ringatoms_flat):
    """Extract coordinates of CG beads"""
    # CG beads are averaged over best trial combinations for all
    # non-aromatic atoms.
    site_coords = []
    for i in range(len(sites)):
        if sites[i] in ringatoms_flat:
            site_coords.append(
                np.array([conformer.GetAtomPosition(int(sites[i]))[j] for j in range(3)])
            )
        else:
            # Use average
            site_coords.append(np.array(avg_pos[i]))
    return site_coords


def get_heavy_atom_bonds(molecule, list_heavy_atoms):
    # List of bonds between heavy atoms
    list_bonds = []
    for i in range(len(list_heavy_atoms)):
        for j in range(i + 1, len(list_heavy_atoms)):
            if molecule.GetBondBetweenAtoms(int(list_heavy_atoms[i]), int(list_heavy_atoms[j])) is not None:
                list_bonds.append([list_heavy_atoms[i], list_heavy_atoms[j]])
    return list_bonds


def check_additivity(forcepred, beadtypes, molecule, mol_smi): #AutoM3 change : added mol_smi argument
    """Check additivity assumption between sum of free energies of CG beads
    and free energy of whole molecule"""
    logger.debug("Entering check_additivity()")
    # If there's only one bead, don't check.
    sum_frag = 0.0
    rings = False
    logger.info("; Bead types: %s" % beadtypes)
    for bead in beadtypes:
        if bead[0] == "S" or bead[0] == "T": # AutoM3 change : added bead "T"
            rings = True
        delta_f_types = topology.read_delta_f_types()
        sum_frag += delta_f_types[bead] #sum of free energies of beads in ring(s)
    # Wildman-Crippen log_p
    wc_log_p = rdMolDescriptors.CalcCrippenDescriptors(molecule)[0]
    # Get SMILES string of entire molecule

    whole_mol_dg,_ = topology.smi2alogps(forcepred, mol_smi, wc_log_p, "MOL", None, None, True) # AutoM3 change : None,None=converted_smi, real_smi not needed here
    if whole_mol_dg != 0:
        m_ad = math.fabs((whole_mol_dg - sum_frag) / whole_mol_dg)
        logger.info(
            "; Mapping additivity assumption ratio: %7.4f (whole vs sum: %7.4f vs. %7.4f)"
            % (m_ad, whole_mol_dg / (-4.184), sum_frag / (-4.184))
        )
        if len(beadtypes) == 1:
            return True
        if (not rings and m_ad < 0.5) or rings:
            return True
        else:
            return False
    else:
        return False


def _get_bead_pos(trial_comb, conformer):
    # Get bead positions
    beadpos = [[0] * 3 for l in range(len(trial_comb))]
    for l in range(len(trial_comb)):
        beadpos[l] = [
            conformer.GetAtomPosition(int(sorted(trial_comb)[l]))[m]
            for m in range(3)
        ]
    return beadpos


def get_graph(mol=None, smiles=None):
    # --- Graph representation of the molecule ---
    if not mol:
        if not smiles:
            raise ValueError("Either mol or smiles must be provided")
        mol = Chem.MolFromSmiles(smiles)
    # --- Node list (atoms) ---
    nodes = []
    for a in mol.GetAtoms():
        heavy_neighbors = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        neighbor_ids = [n.GetIdx() for n in heavy_neighbors]
        neighbor_bonds = []
        for n in heavy_neighbors:
            bond_id = (int(a.GetIdx()), int(n.GetIdx()))
            b = mol.GetBondBetweenAtoms(*bond_id)
            if b is None:
                continue
            neighbor_bonds.append(bond_id)
        nodes.append({
            "idx": a.GetIdx(),                    # 0-based
            "atomic_num": a.GetAtomicNum(),       # 6 for C, 7 for N
            "formal_charge": a.GetFormalCharge(),
            "is_aromatic": a.GetIsAromatic(),
            "is_in_ring": a.IsInRing(),
            "degree": a.GetDegree(),              # total neighbors (includes H only if explicit)
            "heavy_degree": len(heavy_neighbors),
            "neighbors": neighbor_ids,             # heavy-atom neighbors only
            "neighbor_bonds": neighbor_bonds,      # heavy-atom neighbor + bond metadata
            "num_h": a.GetTotalNumHs(),            # implicit H count (unless you add Hs)
        })

    # --- Edge list (bonds) ---
    # Undirected bond list with attributes
    edges = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        edges.append({
            "ij": (i, j),
            "bond_type": str(b.GetBondType()),     # 'SINGLE', 'DOUBLE', 'AROMATIC', ...
            "is_aromatic": b.GetIsAromatic(),
            "is_conjugated": b.GetIsConjugated(),
            "stereo": str(b.GetStereo()),          # often 'STEREONONE' for this case
        })

    # --- Terminal nodes ---
    # Common definition in chemistry ML: heavy-atom degree == 1
    terminal_atoms = [a.GetIdx() for a in mol.GetAtoms()
                    if sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() > 1) == 1]

    return {"atoms": nodes, "bonds": edges, "terminal_atoms": terminal_atoms}


def _single_atom_in_mapping(mapping):
    for ns in mapping:
        if len(ns) == 1:
            return True
    return False


def _atom_count(atom, bead_neighbors):
    count = 0
    for ns in bead_neighbors:
        if atom in ns:
            count += 1
    return count 


def _ring_beads_are_good(mapping, is_in_ring):
    for ns, ring in zip(mapping, is_in_ring):
        if ring and len(ns) > 2:
            return False
    return True


def distribute_neighbors(trial_comb, atoms):
    """Find acceptable mappings of atoms to beads for given trial combination"""
    bead_neighbors = [atoms[i]["neighbors"] for i in trial_comb]
    is_in_ring = [atoms[i]["is_in_ring"] for i in trial_comb]
    n_atoms = len(atoms)
    atom_ids = set(range(n_atoms))
    nei_ids = atom_ids - set(trial_comb)
    mapping = [[int(i)] for i in trial_comb]
    mappings = [mapping]
    print(trial_comb)
    print(bead_neighbors)
    print(nei_ids)

    # Distribute neighbors of trial combination atoms to beads, 
    # and keep track of all possible mappings
    for nei_idx in nei_ids:
        updated_mappings = []
        for mapping in mappings:
            for idx, bead in enumerate(mapping):
                if not nei_idx in bead_neighbors[idx]:
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
    
    # At this point we probably have multiple mappings with different 
    # numbers of atoms in beads with most of them being equally valid
    # Try keeping ring beads small and close together
    tmp_list = []
    for mapping in mappings:
        if not _ring_beads_are_good(mapping, is_in_ring):
            continue
        tmp_list.append(mapping)
    mappings = tmp_list

    for mapping in mappings:   
        print(mapping)    

    exit()
    return mappings


def get_partitioning(trial_comb, graph): 
    """Get partitioning of atoms into beads for given trial combination"""
    atoms = graph["atoms"]
    bonds = graph["bonds"]
    bead_bonds = [atoms[i]["neighbor_bonds"] for i in trial_comb]
    bead_neighbors = distribute_neighbors(trial_comb, atoms)
    print(bead_neighbors)
    exit()

    mapping_dict = {idx: [int(atom)] for idx, atom in enumerate(trial_comb)}
    for key, item in mapping_dict.items():
        for neighbor in bead_neighbors[key]:
            if neighbor not in trial_comb:
                mapping_dict[key].append(neighbor)
    partitioning = invert_mapping_dictionary(mapping_dict)

    return partitioning


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
    # Ensure deterministic ascending key iteration order (Python dict preserves insertion order).
    return dict(sorted(atom_partitioning.items()))


class Cg_molecule:
    """Main class to coarse-grain molecule"""

    def __init__(self, molecule, mol_smi, molname, simple_model=None, topfname=None, 
        bartenderfname=None, bartender=None, logp_file=None, forcepred=True,
        min_beads=None, max_beads=None, raw_molecule=None):
        # AutoM3 new arguments : mol_smi, simple_model, bartenderfname, bartender, logp_file

        self.heavy_atom_coords = None
        self.atom_coords = None # AutoM3 new variable 
        self.list_heavyatom_names = None
        self.atom_partitioning = None
        self.cg_bead_names = []
        self.cg_bead_coords = []
        self.topout = None
        self.bartender_out = None # AutoM3 new variable 
        self.molname = molname # AutoM3 change : for pretty GRO file (will be easier to look on a molecule in VMD with its proper name)
        self.graph = None
        force_map = False # AutoM3 new variable

        if raw_molecule:
            self.graph = get_graph(mol=raw_molecule) 
        else:
            self.graph = get_graph(smiles=mol_smi) 
        graph = self.graph


        logger.info("Starting coarse-graining for '%s' (forcepred=%s, simple_model=%s)", molname, forcepred, simple_model)
        logger.debug("Inputs: topfname=%s bartender=%s bartenderfname=%s logp_file=%s", topfname, bartender, bartenderfname, logp_file)

        # _, _, instance_coords = topology.get_heavy_atom_coords(molecule)
        # print(instance_coords)

        ## AutoM3 : MINIMIZATION with RDkit ###
        molecule = Chem.Mol(molecule)
        logger.debug("Embedding + MMFF optimization")
        AllChem.EmbedMolecule(molecule, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000, mmffVariant='MMFF94s')
        AllChem.NormalizeDepiction(molecule, scaleFactor=1.12) 

        feats = topology.extract_features(molecule)

        # Get list of heavy atoms and their coordinates
        list_heavy_atoms, self.list_heavyatom_names = topology.get_atoms(molecule)
        conf, self.heavy_atom_coords, self.atom_coords = topology.get_heavy_atom_coords(molecule)
        self.output_aa(f"{self.molname}_aa.gro") # AutoM3 change : output AA structure to .gro file (for visualization purposes)
        logger.info("Detected %d heavy atoms", len(list_heavy_atoms))

        # Identify ring-type atoms
        ring_atoms = topology.get_ring_atoms(molecule)
        is_arom, num_arom = topology.is_aromatic(molecule) # AutoM3
        logger.info("Ring atoms: %d (aromatic=%s, aromatic_count=%d)", len(list(chain.from_iterable(ring_atoms))), is_arom, num_arom)

        # Get Hbond information
        hbond_a = topology.get_hbond_a(feats)
        hbond_d = topology.get_hbond_d(feats)

        # List of bonds between heavy atoms
        list_bonds = get_heavy_atom_bonds(molecule, list_heavy_atoms)

        # Flatten list of ring atoms
        ring_atoms_flat = list(chain.from_iterable(ring_atoms))

        # Optimize coarse-grained bead positions -- keep all possibilities in case something goes
        # wrong later in the code.
        list_cg_beads = optimization.find_bead_pos(
            molecule,
            conf,
            graph,
            list_heavy_atoms,
            self.heavy_atom_coords,
            self.atom_coords,
            ring_atoms,
            ring_atoms_flat,
            force_map,  # AutoM3 new argument
            min_beads=min_beads,
            max_beads=max_beads,
        )
        logger.info("Generated %d candidate bead mappings", len(list_cg_beads))

        # Remove mappings with bead numbers less than most optimal mapping.
        filtered_cg_beads = []
        for cg_beads in list_cg_beads:
            if (
                len(cg_beads) == len(list_cg_beads[0])
                # and (len(list_heavy_atoms) - (5 * len(cg_beads))) > 3
            ):
                filtered_cg_beads.append(cg_beads)
        logger.info("Removed suboptimal candidate bead mappings with bead number < %d", len(list_cg_beads[0]))
        filtered_cg_beads = list_cg_beads

        # Loop through best 1% cg_beads and avg_pos
        # max_attempts = int(math.ceil(0.5 * len(list_cg_beads)))
        max_attempts = len(filtered_cg_beads) 
        logger.info("Max. number of attempts: %d", max_attempts)
        attempt = 0

        logger.info("Going through the candidate mappings")
        while attempt < max_attempts:

            if attempt % 1000 == 0:  # Log every 1000 attempts
                logger.info("Attempt %d/%d", attempt, max_attempts)

            cg_beads = filtered_cg_beads[attempt]

            if len(cg_beads) != 5:
                attempt += 1
                continue
            partitioning = get_partitioning(cg_beads, self.graph)
            exit()

            try:
                partitioning = get_partitioning(cg_beads, self.graph)
            except Exception:
                attempt += 1
                continue
            bead_pos = _get_bead_pos(cg_beads, conf)
            success = True

            logger.debug("Attempt %d/%d: trying %d CG beads", attempt + 1, max_attempts, len(cg_beads))

            if not all_atoms_in_beads_connected(
                    cg_beads, self.heavy_atom_coords, list_heavy_atoms, list_bonds, molecule, 
                    self.atom_coords, force_map, partitioning
                ): # AutoM3 change : Added molecule and force_map arguments
                attempt += 1
                continue
            logger.info("Connection check successful")

            # Extract position of coarse-grained beads
            logger.info("Extracting coordinates for CG beads")
            cg_bead_coords = get_coords(conf, cg_beads, bead_pos, ring_atoms_flat)

            ### AutoM3 change : different partition of atoms into coarse-grained beads, depending on the number of aromatic cycles ###
            _, num_arom = topology.is_aromatic(molecule)
            if not force_map and num_arom < 7: # AutoM3
                self.atom_partitioning, self.cg_bead_coords = voronoi_atoms_new( 
                    cg_bead_coords, self.heavy_atom_coords, self.atom_coords, molecule, partitioning
                )

            else:
                self.atom_partitioning, self.cg_bead_coords = voronoi_atoms_old(
                    cg_bead_coords, self.heavy_atom_coords, self.atom_coords, molecule, partitioning
                )
            logger.info("Partitioned atoms into %d beads", len(self.cg_bead_coords))

            # self.atom_partitioning = optimization.sanitize_rings(self.atom_partitioning, self.heavy_atom_coords, ring_atoms)
            # exit()
            
            # AutoM3 : trying mapping with at least 1 of 2 new conditions : 
            #    Max 2 aromatic atoms per bead ; 
            #    Holding Functional groups together in bead ;
            max_fails = 1
            fails = 0
            if is_arom and (num_arom % 2) == 0: # only for pair number of aromatic atoms (actual code prevents sharing/mismatch)
                if not optimization.max2arperbead(self.atom_partitioning, ring_atoms):
                    fails += 1
            if not optimization.functional_groups_ok(self.atom_partitioning, molecule, ring_atoms):
                fails += 1
            if force_map:
                if fails > max_fails:
                    success = False
                else:
                    success = True
            else:
                if fails > 0: 
                    success = False
            logger.info("Atom partitioning created (%d atoms mapped)", len(self.atom_partitioning) if self.atom_partitioning else 0)

            # cgbeads should take atom rings number if ring atom in bead 
            cg_beads_rings = cg_beads.copy()
            for i, b in enumerate(cg_beads):
                if b not in ring_atoms_flat:
                    atoms_in_b = []
                    for at,bd in self.atom_partitioning.items():
                        if bd == i : atoms_in_b.append(at)
                    for a in atoms_in_b:
                        if a in ring_atoms_flat:
                            cg_beads_rings[i] = a
            logger.info("CG beads rings updated")

            # IF AN ATOM IS IN A RING, ADD ALL ATOMS OF THIS BEADS TO THE RING ATOMS
            # for connectivity purposes
            mapping_dict = optimization.make_mapping_dictionary(self.atom_partitioning)
            for ring in ring_atoms:
                for atom_idx in ring:
                    for bead_idx, atom_indices in mapping_dict.items():
                        if atom_idx in atom_indices:
                            # Add all atoms in this bead to ring_atoms_flat
                            for at in atom_indices:
                                if at not in ring:
                                    ring.append(at)

            logger.info("Printing Atoms")
            self.cg_bead_names, bead_types, _, _ = topology.print_atoms(
                    molname,
                    forcepred,
                    cg_beads,
                    molecule,
                    hbond_a,
                    hbond_d,
                    self.atom_partitioning,
                    ring_atoms,
                    ring_atoms_flat,
                    logp_file, # AutoM3 new argument
                    True,
            )

            if not self.cg_bead_names:
                success = False
            # Check additivity between fragments and entire molecule
            if not check_additivity(forcepred, bead_types, molecule, mol_smi):
                success = False
            
            # Bond list
            try:
                bond_list, const_list , _= topology.print_bonds(
                    cg_beads,
                    cg_beads_rings,
                    molecule,
                    self.atom_partitioning,
                    self.cg_bead_coords,
                    bead_types, # AutoM3 change
                    ring_atoms,
                    trial=True,
                )
            except Exception:
                raise

            # I added errval below from the master branch ... not sure where to use this anywhere, possibly leave for debugging
            if not ring_atoms and (len(bond_list) + len(const_list)) >= len(self.cg_bead_names):
                errval = 3
                success = False
            if (len(bond_list) + len(const_list)) < len(self.cg_bead_names) - 1:
                errval = 5
                success = False
            if len(cg_beads) != len(self.cg_bead_names):
                success = False
                errval = 8
            
            if success:
                logger.info("Success mapping found on attempt %d", attempt + 1)
                header_write = topology.print_header(molname, mol_smi)
                self.cg_bead_names, bead_types, atoms_write, atoms_in_smi = topology.print_atoms( # AutoM3 new variable : atoms_in_smi
                    molname,
                    forcepred,
                    cg_beads,
                    molecule,
                    hbond_a,
                    hbond_d,
                    self.atom_partitioning,
                    ring_atoms,
                    ring_atoms_flat,
                    logp_file, # AutoM3 change
                    trial=False,
                )

                logger.info("Final CG model: %d beads", len(self.cg_bead_names))

                bond_list, const_list, bonds_write = topology.print_bonds(
                    cg_beads,
                    cg_beads_rings,
                    molecule,
                    self.atom_partitioning,
                    self.cg_bead_coords,
                    bead_types, # AutoM3 change
                    ring_atoms,
                    False,
                )

                if not simple_model: # AutoM3
                    dihedrals_write = topology.print_dihedrals(
                    cg_beads,
                    const_list,
                    ring_atoms,
                    self.cg_bead_coords,
                    bead_types # AutoM3 change
                    )

                angles_write, angle_list = topology.print_angles(
                    cg_beads,
                    molecule,
                    self.atom_partitioning,
                    self.cg_bead_coords,
                    bead_types, # AutoM3 change
                    bond_list,
                    const_list,
                    ring_atoms,
                )

                if not angles_write and len(bond_list) > 1:
                    errval = 2
                if bond_list and angle_list:
                    if (len(bond_list) + len(const_list)) < 2 and len(angle_list) > 0:
                        errval = 6
                    if (
                        not ring_atoms
                        and (len(bond_list) + len(const_list)) - len(angle_list) != 1
                    ):
                        errval = 7

                self.topout, bartender_input_info = topology.topout(header_write, atoms_write, bonds_write, angles_write) # AutoM3 change : possible simple output w/o dihedrals, virtual sites

                # check if fusion of cycles
                common = False
                if len(ring_atoms) > 1:
                    cpt = list(set.intersection(*map(set, ring_atoms)))
                    if len(cpt) > 1 : common=True
                    for i in ring_atoms:
                        if len(i) > 6 : common=True
                else:
                    if len(ring_atoms_flat)>6 : common=True

                ### AutoM3 outputs ###

                # if len(ring_atoms_flat) > 0 and not simple_model:
                #     if len(ring_atoms_flat) > 7 and common:
                #         vs_write, virtual_sites, rigid_dih  = topology.print_virtualsites(ring_atoms, self.cg_bead_coords, self.atom_partitioning, molecule)
                        
                #         self.topout, vs_bead_names, bartender_input_info  = topology.topout_vs(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, virtual_sites,vs_write,rigid_dih,simple_model)
                    
                #     else:
                #         self.topout, bartender_input_info = topology.topout_noVS(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, self.cg_bead_coords, ring_atoms, cg_beads)
                self.topout, bartender_input_info = topology.topout_noVS(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, self.cg_bead_coords, ring_atoms, cg_beads)
                
                if bartender:
                    bartender_out = topology.bartender_input(molecule, molname, atoms_in_smi, bartender_input_info)
                    with open(bartenderfname, "w") as btf:
                        btf.write(bartender_out)
                    logger.info("Wrote bartender input: %s", bartenderfname)
                
                if topfname:
                    with open(topfname, "w") as fp:
                        fp.write(self.topout)
                    logger.info("Wrote topology: %s", topfname)
                if not force_map: print("Converged to solution in {} iteration(s)".format(attempt + 1))
                if force_map: print("Converged to solution in {} iteration(s)".format(attempt + 1 + max_attempts))
                break
            else:
                attempt += 1
        
                # AutoM3 change : force mapping by old code if new code doesn't give result
                if attempt == max_attempts and not force_map:
                    force_map=True
                    attempt = 0 
                    logger.info("Retrying with force_map=True")

        if attempt == max_attempts and force_map:
            raise RuntimeError(
                "ERROR: no successful mapping found.\nTry running with the --fpred and/or --verbose options."
            )

    def output_aa(self, aa_output=None): # AutoM3 change : molname is the same as argument --mol given at the beginning
        # Optional all-atom output to GRO file
        aa_out = output.output_gro(self.heavy_atom_coords, self.list_heavyatom_names, self.molname)
        if aa_output:
            with open(aa_output, "w") as fp:
                fp.write(aa_out)
        else:
            return aa_out

    def output_cg(self, cg_output=None): # AutoM3 change : molname is the same as argument --mol given at the beginning
        # Optional coarse-grained output to GRO file
        cg_out = output.output_gro(self.cg_bead_coords, self.cg_bead_names, self.molname)
        if cg_output:
            with open(cg_output, "w") as fp:
                fp.write(cg_out)
        else:
            return cg_out


def voronoi_atoms_old(cgbead_coords, heavyatom_coords, allatom_coords, molecule, in_partitioning): #AutoM3 change
    """Partition all atoms between CG beads"""
    logger.debug("Entering voronoi_atoms_old()")

    # Initial Partitioning based on closest bead to each heavy atom
    partitioning = {}
    for j in range(len(heavyatom_coords)):
        # Voronoi to check whether atom is closest to bead
        bead_at = -1
        dist_bead_at = 1000
        for k in range(len(cgbead_coords)):
            distk = np.linalg.norm(cgbead_coords[k] - heavyatom_coords[j])
            if distk < dist_bead_at:
                dist_bead_at = distk
                bead_at = k
        partitioning[j] = bead_at

    if len(cgbead_coords) > 1:
        # Book-keeping of closest atoms to every bead
        closest_atoms = {}
        for i in range(len(cgbead_coords)):
            closest_atom = -1
            closest_dist = 10000.0
            for j in range(len(heavyatom_coords)):
                dist_bead_at = np.linalg.norm(cgbead_coords[i] - heavyatom_coords[j])
                if dist_bead_at < closest_dist:
                    closest_dist = dist_bead_at
                    closest_atom = j
            if closest_atom == -1:
                logger.warning("Error. Can't find closest atom to bead %s" % i)
                exit(1)
            closest_atoms[i] = closest_atom
        print(closest_atoms)
        # If one bead has only one heavy atom, include one more
        for i in partitioning.values():
            if sum(x == i for x in partitioning.values()) == 1:
                # Find bead
                lonely_bead = i
                # Voronoi to find closest atom
                closest_bead = -1
                closest_bead_dist = 10000.0
                for j in range(len(heavyatom_coords)):
                    if partitioning[j] != lonely_bead:
                        dist_bead_at = np.linalg.norm(
                            cgbead_coords[lonely_bead] - heavyatom_coords[j]
                        )
                        # Only consider if it's closer, not a CG bead itself, and
                        # the CG bead it belongs to has more than one other atom.
                        if (
                            dist_bead_at < closest_bead_dist
                            and j != closest_atoms[partitioning[j]]
                            and sum(x == partitioning[j] for x in partitioning.values()) > 2
                        ):
                            closest_bead = j
                            closest_bead_dist = dist_bead_at
                if closest_bead == -1:
                    logger.warning("Error. Can't find an atom close to atom $s" % lonely_bead)
                    exit(1)
                partitioning[closest_bead] = lonely_bead

    partitioning = in_partitioning
    # find all bonds between atoms in molecule
    bonds = []
    for b in range(len(molecule.GetBonds())):
        abond = molecule.GetBondWithIdx(b)
        at1 = abond.GetBeginAtomIdx()
        at2 = abond.GetEndAtomIdx()
        if f"{at1}-{at2}" not in bonds and f"{at2}-{at1}" not in bonds:
            bonds.append(f"{at1}-{at2}")

    # create partitioning including hydrogens inside beads
    aa_partitioning = partitioning.copy()
    for at in range(len(allatom_coords)):
        if at not in aa_partitioning.keys():
            hbead = None
            for b in bonds:
                bond = b.split('-')
                if str(at) in bond:
                    at1 = int(bond[0])
                    at2 = int(bond[-1])
                    if at == at1 and at2 in partitioning.keys(): 
                        hbead = partitioning[at2]
                        hydrogen = at1
                    if at == at2 and at1 in partitioning.keys():
                        hbead = partitioning[at1]
                        hydrogen = at2

                    if hbead is not None: # found hydrogen atom connected to 
                        aa_partitioning[hydrogen]=hbead

    #compute COG while taking into account hydrogens
    bead_coord = {}
    for atom in range(len(allatom_coords)):
        bead = aa_partitioning[atom]
        if bead not in bead_coord.keys(): 
            bead_coord[bead] = []
        bead_coord[bead].append(allatom_coords[atom])

    bead_cog = []
    for bead, coords in sorted(bead_coord.items()):
        cog = np.mean(coords,axis=0)
        bead_cog.append(cog)

    return partitioning, bead_cog


def voronoi_atoms_new(cgbead_coords, heavyatom_coords, allatom_coords, molecule, in_partitioning): # AutoM3
    """
    Partition all atoms between CG beads, based on headliners coordinates and distances between other atoms coordinates. 
    Headliners are atoms with cgbead_coords coordinates.
    """
    logger.debug("Entering voronoi_atoms()")
    partitioning = {}

    #Populate partitioning with atoms and atom headliners of beads
    for j in range(len(heavyatom_coords)):
        partitioning[j] = None
        for b in range(len(cgbead_coords)):
            if(heavyatom_coords[j]==cgbead_coords[b]).all():
                partitioning[j] = b

    # Find closest atoms to atom headliners of beads
    if len(cgbead_coords) > 1:
        closest_atoms = {}  # Book-keeping of closest atoms to every bead
        for i in range(len(cgbead_coords)):
            distances = {}
            for j in range(len(heavyatom_coords)):
                if (cgbead_coords[i] != heavyatom_coords[j]).all():
                    dist_bead_at = np.linalg.norm(cgbead_coords[i] - heavyatom_coords[j])
                    distances[j] = dist_bead_at  # Atom index as key, distance as value

            # Sort distances by value and keep the closest atoms
            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
            closest_atoms[i] = sorted_distances  # Dictionary of atoms and their distances for each bead

        # Populate partitioning with closest atoms
        for atom, bead in partitioning.items():

            if bead is None:
                closest_index = float('inf')  # Initialize with infinity
                closest_bead = None

                for current_bead, atoms_dict in closest_atoms.items():
                    if atom in atoms_dict:
                        index = list(atoms_dict.keys()).index(atom)  # Find the index of the atom in the sorted keys
                        if index < closest_index:
                            closest_index = index
                            closest_bead = current_bead

                if closest_bead is not None:
                    partitioning[atom] = closest_bead

        # If one bead has only one heavy atom, include one more
        for i in partitioning.values():
            if sum(x == i for x in partitioning.values()) == 1:
                # Find bead
                lonely_bead = i
                # Voronoi to find closest atom
                closest_bead = -1
                closest_bead_dist = 10000.0
                for j in range(len(heavyatom_coords)):
                    if partitioning[j] != lonely_bead:
                        dist_bead_at = np.linalg.norm(
                            cgbead_coords[lonely_bead] - heavyatom_coords[j]
                        )
                        # Only consider if it's closer, not a CG bead itself, and
                        # the CG bead it belongs to has more than one other atom. 
                        if (
                            dist_bead_at < closest_bead_dist
                            and j != closest_atoms[partitioning[j]]
                            and sum(x == partitioning[j] for x in partitioning.values()) > 2
                        ):
                            closest_bead = j
                            closest_bead_dist = dist_bead_at
                if closest_bead == -1:
                    logger.warning("Error. Can't find an atom close to atom $s" % lonely_bead)
                    exit(1)
                partitioning[closest_bead] = lonely_bead
    else:
        for j in range(len(heavyatom_coords)):
            partitioning[j] = 0 #len(cgbead_coords)

    # find all bonds between atoms in molecule
    bonds = []
    for b in range(len(molecule.GetBonds())):
        abond = molecule.GetBondWithIdx(b)
        at1 = abond.GetBeginAtomIdx()
        at2 = abond.GetEndAtomIdx()
        if f"{at1}-{at2}" not in bonds and f"{at2}-{at1}" not in bonds:
            bonds.append(f"{at1}-{at2}")

    # create partitioning including hydrogens inside beads
    aa_partitioning = partitioning.copy()
    for at in range(len(allatom_coords)):
        if at not in aa_partitioning.keys():
            hbead = None
            for b in bonds:
                bond = b.split('-')
                if str(at) in bond:
                    at1=int(bond[0])
                    at2=int(bond[-1])
                    if at==at1 and at2 in partitioning.keys(): 
                        hbead = partitioning[at2]
                        hydrogen = at1
                    if at==at2 and at1 in partitioning.keys():
                        hbead = partitioning[at1]
                        hydrogen = at2

                    if hbead is not None: # found hydrogen atom connected to 
                        aa_partitioning[hydrogen]=hbead

    partitioning = in_partitioning.copy()

    #compute COG while taking into account hydrogens
    bead_coord={}
    for atom in range(len(allatom_coords)):
        bead=aa_partitioning[atom]
        if bead not in bead_coord.keys(): 
            bead_coord[bead]=[]
        bead_coord[bead].append(allatom_coords[atom])

    bead_cog=[]
    for bead, coords in sorted(bead_coord.items()):
        cog = np.mean(coords,axis=0)
        bead_cog.append(cog)

    return partitioning, bead_cog


def sanitize_rings(atom_partitioning, atoms_xyz, ringatoms):
    mapping_dict = make_mapping_dictionary(atom_partitioning)
    print(mapping_dict)
    print(atoms_xyz)
    for bead, atoms in mapping_dict.items():
        for ring in ringatoms:
            if not set(atoms).issubset(ring):
                continue
            if len(atoms) <= 2:
                continue
            print("More than 2 atoms in bead %s are in ring %s" % (bead, ring))

    return atom_partitioning


def all_atoms_in_beads_connected(trial_comb, heavyatom_coords, 
    list_heavyatoms, bondlist, mol, allatom_coords, force_map, in_partitioning): #AutoM3 change: added mol, force_map
    """Make sure all atoms within one CG bead are connected to at least
    one other atom in that bead"""
    # Bead coordinates are given by heavy atoms themselves
    cgbead_coords = []

    for i in range(len(trial_comb)):
        cgbead_coords.append(heavyatom_coords[list_heavyatoms.index(trial_comb[i])])
    
    _, num_arom = topology.is_aromatic(mol) #AutoM3 change
    ### AutoM3 change of mapping approach to differenciate molecules with 0-1 and more cycles
    if not force_map and num_arom < 7: #AutoM3 change
        voronoi, _  = voronoi_atoms_new(cgbead_coords, heavyatom_coords, allatom_coords, mol, in_partitioning) #AutoM3 change
    else:
        voronoi, _  = voronoi_atoms_old(cgbead_coords, heavyatom_coords, allatom_coords, mol, in_partitioning) #AutoM3 change
    logger.debug("voronoi %s" % voronoi)

    for i in range(len(trial_comb)):
        cg_bead = trial_comb[i]
        num_atoms = list(voronoi.values()).count(voronoi[list_heavyatoms.index(cg_bead)])
        # sub-part of bond list that only contains atoms within CG bead
        sub_bond_list = []
        for j in range(len(bondlist)):
            if (
                voronoi[list_heavyatoms.index(bondlist[j][0])] == voronoi[list_heavyatoms.index(cg_bead)]
                and voronoi[list_heavyatoms.index(bondlist[j][1])] == voronoi[list_heavyatoms.index(cg_bead)]
            ):
                sub_bond_list.append(bondlist[j])
        num_bonds = len(sub_bond_list)
        if num_bonds < num_atoms - 1 or num_atoms == 1:
            logger.debug("Error: Not all atoms in beads connected in %s" % trial_comb)
            logger.debug("Error: %s < %s, %s" % (num_bonds, num_atoms - 1, sub_bond_list))
            return False
    return True
