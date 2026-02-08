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


class Cg_molecule:
    """Main class to coarse-grain molecule"""

    # NOTE: These helpers are static because they don't depend on instance state.
    # Keeping them on the class groups mapping logic in one place.

    @staticmethod
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

    @staticmethod
    def get_heavy_atom_bonds(molecule, list_heavy_atoms):
        # List of bonds between heavy atoms
        list_bonds = []
        for i in range(len(list_heavy_atoms)):
            for j in range(i + 1, len(list_heavy_atoms)):
                if molecule.GetBondBetweenAtoms(int(list_heavy_atoms[i]), int(list_heavy_atoms[j])) is not None:
                    list_bonds.append([list_heavy_atoms[i], list_heavy_atoms[j]])
        return list_bonds

    @staticmethod
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

    @staticmethod
    def _get_bead_pos(trial_comb, conformer):
        # Get bead positions
        beadpos = [[0] * 3 for l in range(len(trial_comb))]
        for l in range(len(trial_comb)):
            beadpos[l] = [
                conformer.GetAtomPosition(int(sorted(trial_comb)[l]))[m]
                for m in range(3)
            ]
        return beadpos
    
    @staticmethod
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
        edges = []
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            edges.append({
                "ij": (i, j),
                "bond_type": str(b.GetBondType()),     # 'SINGLE', 'DOUBLE', 'AROMATIC', ...
                "is_aromatic": b.GetIsAromatic(),
                "is_conjugated": b.GetIsConjugated(),
                "stereo": str(b.GetStereo()),
            })

        terminal_atoms = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() > 1) == 1
        ]

        return {"atoms": nodes, "bonds": edges, "terminal_atoms": terminal_atoms}

    @staticmethod
    def _single_atom_in_mapping(mapping):
        for ns in mapping:
            if len(ns) == 1:
                return True
        return False

    @staticmethod
    def _ring_beads_are_tiny(mapping, bead_is_in_ring):
        for ns, ring in zip(mapping, bead_is_in_ring):
            if ring and len(ns) > 2:
                return False
        return True

    @staticmethod
    def _ring_beads_are_together(mapping, bead_is_in_ring, atom_is_in_ring):
        for ns, ring in zip(mapping, bead_is_in_ring):
            if not ring:
                continue
            for atom in ns:
                if not atom_is_in_ring[atom]:
                    return False
        return True

    @staticmethod
    def distribute_neighbors(trial_comb, atoms):
        """Find acceptable mappings of atoms to beads for given trial combination"""
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
            if Cg_molecule._single_atom_in_mapping(mapping):
                continue
            tmp_list.append(mapping)
        mappings = tmp_list
        if len(mappings) == 1:
            return mappings[0]

        # Prefer keeping ring beads small
        tmp_list = []
        for mapping in mappings:
            if not Cg_molecule._ring_beads_are_tiny(mapping, bead_is_in_ring):
                continue
            tmp_list.append(mapping)
        if tmp_list:
            mappings = tmp_list
        if len(mappings) == 1:
            return mappings[0]

        # Prefer keeping ring beads together (no mixing ring/non-ring)
        tmp_list = []
        for mapping in mappings:
            if not Cg_molecule._ring_beads_are_together(mapping, bead_is_in_ring, atom_is_in_ring):
                continue
            tmp_list.append(mapping)
        if tmp_list:
            mappings = tmp_list
        if len(mappings) == 1:
            return mappings[0]

        if len(mappings) == 2:
            return mappings[0]

        return mappings[0]

    @staticmethod
    def get_partitioning(trial_comb, graph):
        """Get partitioning of atoms into beads for given trial combination"""
        atoms = graph["atoms"]
        mapping = Cg_molecule.distribute_neighbors(trial_comb, atoms)
        mapping_dict = {idx: bead for idx, bead in enumerate(mapping)}
        partitioning = Cg_molecule.invert_mapping_dictionary(mapping_dict)
        return partitioning

    @staticmethod
    def make_mapping_dictionary(partitioning):
        """Create mapping dictionary from partitioning"""
        mapping_dict = {}
        for atom_idx, bead_idx in partitioning.items():
            if bead_idx not in mapping_dict:
                mapping_dict[bead_idx] = []
            mapping_dict[bead_idx].append(atom_idx)
        return mapping_dict

    @staticmethod
    def invert_mapping_dictionary(mapping_dict):
        """Inverse of make_mapping_dictionary(): bead_idx -> [atom_idx] to atom_idx -> bead_idx.

        If mapping is not one-to-one, raises ValueError.
        """
        partitioning = {}
        for bead_idx, atom_indices in mapping_dict.items():
            for atom_idx in atom_indices:
                if atom_idx in partitioning:
                    raise ValueError(f"Atom {atom_idx} appears in multiple beads")
                partitioning[atom_idx] = bead_idx
        return dict(sorted(partitioning.items()))

    @staticmethod
    def get_bead_coords(partitioning, allatom_coords, molecule):
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

                        if hbead is not None:
                            aa_partitioning[hydrogen] = hbead

        # compute COG while taking into account hydrogens
        bead_coord = {}
        for atom in range(len(allatom_coords)):
            bead = aa_partitioning[atom]
            if bead not in bead_coord.keys():
                bead_coord[bead] = []
            bead_coord[bead].append(allatom_coords[atom])

        bead_cog = []
        for bead, coords in sorted(bead_coord.items()):
            cog = np.mean(coords, axis=0)
            bead_cog.append(cog)

        return bead_cog


    def _build_topology(self, cg_beads, cg_beads_rings, bead_types):
        # Build complete topology using new Topology class
        topo = topology.build_topology(
            molname=self.molname,
            mol_smi=self.smiles,
            forcepred=self.forcepred,
            cgbeads=cg_beads,
            cgbeads_ring=cg_beads_rings,
            molecule=self.molecule,
            hbonda=self.hbond_a,
            hbondd=self.hbond_d,
            partitioning=self.partitioning,
            cgbead_coords=self.cg_bead_coords,
            ringatoms=self.ring_atoms,
            ringatoms_flat=self.ring_atoms_flat,
            logp_file=self.logp_file,
            beadtypes=bead_types,
            trial=False,
            simple_model=self.simple_model
        )
        return topo


    def _finalize_topology(self, cg_beads, cg_beads_rings, bead_types, attempt):
        """Finalize topology after successful mapping - builds complete Topology object."""
        
        # Store convenience references
        self.cg_bead_names = self.topology.atomnames
        
        logger.info("Final CG model: %d beads", len(self.cg_bead_names))

        # Validation checks
        bond_list = self.topology.bonds
        const_list = self.topology.constraints
        angle_list = self.topology.angles
        
        errval = 0
        if len(bond_list) > 1 and len(angle_list) == 0:
            errval = 2
        if bond_list and angle_list:
            if (len(bond_list) + len(const_list)) < 2 and len(angle_list) > 0:
                errval = 6
            if (
                not self.ring_atoms
                and (len(bond_list) + len(const_list)) - len(angle_list) != 1
            ):
                errval = 7

        # Generate formatted outputs
        header_write = topology.format_topology_header(self.topology)
        atoms_write = topology.format_topology_atoms(self.topology.atoms, trial=False)
        bonds_write = topology.format_topology_bonds(
            self.topology.bonds, self.topology.constraints, 
            self.topology.beadtypes, self.ring_atoms, trial=False
        )
        angles_write = topology.format_topology_angles(self.topology.angles, self.topology.beadtypes)
        
        if not self.simple_model and self.topology.dihedrals:
            dihedrals_write = topology.format_topology_dihedrals(
                self.topology.dihedrals, 0, cg_beads, self.ring_atoms, 
                self.cg_bead_coords, self.topology.beadtypes
            )
        else:
            dihedrals_write = ""

        self.topout, bartender_input_info = topology.topout(header_write, atoms_write, bonds_write, angles_write)

        # Check if fusion of cycles
        common = False
        if len(self.ring_atoms) > 1:
            cpt = list(set.intersection(*map(set, self.ring_atoms)))
            if len(cpt) > 1:
                common = True
            for i in self.ring_atoms:
                if len(i) > 6:
                    common = True
        else:
            if len(self.ring_atoms_flat) > 6:
                common = True

        self.topout, bartender_input_info = topology.topout_noVS(
            header_write, atoms_write, bonds_write, angles_write, dihedrals_write, 
            self.cg_bead_coords, self.ring_atoms, cg_beads
        )
        
        if self.bartender:
            self.bartender_out = topology.bartender_input(
                self.molecule, self.molname, self.topology.atoms_in_smi_dict, bartender_input_info
            )
            with open(self.bartenderfname, "w") as btf:
                btf.write(self.bartender_out)
            logger.info("Wrote bartender input: %s", self.bartenderfname)
        
        if self.topfname:
            with open(self.topfname, "w") as fp:
                fp.write(self.topout)
            logger.info("Wrote topology: %s", self.topfname)
        if not self.force_map: 
            print("Converged to solution in {} iteration(s)".format(attempt + 1))
        if self.force_map: 
            print("Converged to solution in {} iteration(s)".format(attempt + 1 + self.max_attempts))

    def __init__(self, molecule, mol_smi, molname, simple_model=None, topfname=None, 
        bartenderfname=None, bartender=None, logp_file=None, forcepred=True,
        min_beads=None, max_beads=None, raw_molecule=None):
        # AutoM3 new arguments : mol_smi, simple_model, bartenderfname, bartender, logp_file

        # Store all arguments as instance attributes
        self.molecule = molecule
        self.smiles = mol_smi
        self.molname = molname
        self.simple_model = simple_model
        self.topfname = topfname
        self.bartenderfname = bartenderfname
        self.bartender = bartender
        self.logp_file = logp_file
        self.forcepred = forcepred
        self.min_beads = min_beads
        self.max_beads = max_beads
        self.raw_molecule = raw_molecule
        
        # Initialize state attributes
        self.heavy_atom_coords = None
        self.atom_coords = None
        self.list_heavyatom_names = None
        self.partitioning = None
        self.cg_bead_names = []
        self.cg_bead_coords = []
        self.topout = None
        self.bartender_out = None
        self.graph = None
        self.topology = None  # Will be populated after successful mapping
        self.force_map = False

        if self.raw_molecule:
            self.graph = self.get_graph(mol=self.raw_molecule)
        else:
            self.graph = self.get_graph(smiles=self.smiles)

        logger.info("Starting coarse-graining for '%s' (forcepred=%s, simple_model=%s)", self.molname, self.forcepred, self.simple_model)
        logger.debug("Inputs: topfname=%s bartender=%s bartenderfname=%s logp_file=%s", self.topfname, self.bartender, self.bartenderfname, self.logp_file)

        # _, _, instance_coords = topology.get_heavy_atom_coords(molecule)
        # print(instance_coords)

        ## AutoM3 : MINIMIZATION with RDkit ###
        self.molecule = Chem.Mol(self.molecule)
        logger.debug("Embedding + MMFF optimization")
        AllChem.EmbedMolecule(self.molecule, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(self.molecule, maxIters=1000, mmffVariant='MMFF94s')
        AllChem.NormalizeDepiction(self.molecule, scaleFactor=1.12) 

        self.feats = topology.extract_features(self.molecule)

        # Get list of heavy atoms and their coordinates
        self.list_heavy_atoms, self.list_heavyatom_names = topology.get_atoms(self.molecule)
        self.conf, self.heavy_atom_coords, self.atom_coords = topology.get_heavy_atom_coords(self.molecule)
        self.output_aa(f"{self.molname}_aa.gro") # AutoM3 change : output AA structure to .gro file (for visualization purposes)
        logger.info("Detected %d heavy atoms", len(self.list_heavy_atoms))

        # Identify ring-type atoms
        self.ring_atoms = topology.get_ring_atoms(self.molecule)
        self.is_arom, self.num_arom = topology.is_aromatic(self.molecule) # AutoM3
        logger.info("Ring atoms: %d (aromatic=%s, aromatic_count=%d)", len(list(chain.from_iterable(self.ring_atoms))), self.is_arom, self.num_arom)

        # Get Hbond information
        self.hbond_a = topology.get_hbond_a(self.feats)
        self.hbond_d = topology.get_hbond_d(self.feats)

        # List of bonds between heavy atoms
        self.list_bonds = self.get_heavy_atom_bonds(self.molecule, self.list_heavy_atoms)

        # Flatten list of ring atoms
        self.ring_atoms_flat = list(chain.from_iterable(self.ring_atoms))

        # Optimize coarse-grained bead positions -- keep all possibilities in case something goes
        # wrong later in the code.
        list_cg_beads = optimization.find_bead_pos(
            self.molecule,
            self.conf,
            self.graph,
            self.list_heavy_atoms,
            self.heavy_atom_coords,
            self.atom_coords,
            self.ring_atoms,
            self.ring_atoms_flat,
            self.force_map,  # AutoM3 new argument
            min_beads=self.min_beads,
            max_beads=self.max_beads,
        )
        logger.info("Generated %d candidate bead mappings", len(list_cg_beads))

        # Remove mappings with bead numbers less than most optimal mapping.
        self.filtered_cg_beads = []
        for cg_beads in list_cg_beads:
            if (
                len(cg_beads) == len(list_cg_beads[0])
                # and (len(self.list_heavy_atoms) - (5 * len(cg_beads))) > 3
            ):
                self.filtered_cg_beads.append(cg_beads)
        logger.info("Removed suboptimal candidate bead mappings with bead number < %d", len(list_cg_beads[0]))
        self.filtered_cg_beads = list_cg_beads

        # Loop through best 1% cg_beads and avg_pos
        # max_attempts = int(math.ceil(0.5 * len(list_cg_beads)))
        self.max_attempts = len(self.filtered_cg_beads) 
        logger.info("Max. number of attempts: %d", self.max_attempts)
        attempt = 0

        logger.info("Going through the candidate mappings")
        for attempt in range(self.max_attempts):

            if attempt % 1000 == 0:  # Log every 1000 attempts
                logger.info("Attempt %d/%d", attempt, self.max_attempts)

            cg_beads = self.filtered_cg_beads[attempt]

            # if len(cg_beads) != 11:
            #     attempt += 1
            #     continue
            # partitioning = get_partitioning(cg_beads, self.graph)
            # exit()

            try:
                partitioning = self.get_partitioning(cg_beads, self.graph)
            except Exception:
                continue
            
            bead_pos = self._get_bead_pos(cg_beads, self.conf)
            success = True
            self.partitioning = partitioning
            logger.debug("Attempt %d/%d: trying %d CG beads", attempt + 1, self.max_attempts, len(cg_beads))

            # Extract position of coarse-grained beads
            logger.info("Extracting coordinates for CG beads")
            self.cg_bead_coords = self.get_bead_coords(self.partitioning, self.atom_coords, self.molecule)
            logger.info("Partitioned atoms into %d beads", len(self.cg_bead_coords))

            # AutoM3 : trying mapping with at least 1 of 2 new conditions : 
            #    Max 2 aromatic atoms per bead ; 
            #    Holding Functional groups together in bead ;
            max_fails = 1
            fails = 0
            if self.is_arom and (self.num_arom % 2) == 0: # only for pair number of aromatic atoms (actual code prevents sharing/mismatch)
                if not optimization.max2arperbead(self.partitioning, self.ring_atoms):
                    fails += 1
            if not optimization.functional_groups_ok(self.partitioning, self.molecule, self.ring_atoms):
                fails += 1
            if self.force_map:
                if fails > max_fails:
                    success = False
                else:
                    success = True
            else:
                if fails > 0: 
                    success = False
            logger.info("Atom partitioning created (%d atoms mapped)", len(self.partitioning) if self.partitioning else 0)

            # cgbeads should take atom rings number if ring atom in bead 
            cg_beads_rings = cg_beads.copy()
            for i, b in enumerate(cg_beads):
                if b not in self.ring_atoms_flat:
                    atoms_in_b = []
                    for at,bd in self.partitioning.items():
                        if bd == i : atoms_in_b.append(at)
                    for a in atoms_in_b:
                        if a in self.ring_atoms_flat:
                            cg_beads_rings[i] = a
            logger.info("CG beads rings updated")

            # IF AN ATOM IS IN A RING, ADD ALL ATOMS OF THIS BEADS TO THE RING ATOMS
            # for connectivity purposes
            mapping_dict = make_mapping_dictionary(self.partitioning)
            for ring in self.ring_atoms:
                for atom_idx in ring:
                    for bead_idx, atom_indices in mapping_dict.items():
                        if atom_idx in atom_indices:
                            # Add all atoms in this bead to self.ring_atoms_flat
                            for at in atom_indices:
                                if at not in ring:
                                    ring.append(at)

            logger.info("Building Atoms")
            self.cg_bead_names, bead_types, _, _ = topology.build_atoms_data(
                    self.molname,
                    self.forcepred,
                    cg_beads,
                    self.molecule,
                    self.hbond_a,
                    self.hbond_d,
                    self.partitioning,
                    self.ring_atoms,
                    self.ring_atoms_flat,
                    self.logp_file, # AutoM3 new argument
                    True,
            )

            if not self.cg_bead_names:
                success = False
                continue
            # Check additivity between fragments and entire molecule
            if not self.check_additivity(self.forcepred, bead_types, self.molecule, self.smiles):
                continue
            
            logger.info("Success mapping found on attempt %d", attempt + 1)
            self.topology = self._build_topology(cg_beads, cg_beads_rings, bead_types)
            self._finalize_topology(cg_beads, cg_beads_rings, bead_types, attempt)
            break

                # # AutoM3 change : force mapping by old code if new code doesn't give result
                # attempt += 1
                # if attempt == self.max_attempts and not self.force_map:
                #     self.force_map = True
                #     attempt = 0 
                #     logger.info("Retrying with force_map=True")

        if attempt == self.max_attempts and self.force_map:
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


################
# OLD STUFF
#################

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


def sanitize_rings(partitioning, atoms_xyz, ringatoms):
    mapping_dict = Cg_molecule.make_mapping_dictionary(partitioning)
    print(mapping_dict)
    print(atoms_xyz)
    for bead, atoms in mapping_dict.items():
        for ring in ringatoms:
            if not set(atoms).issubset(ring):
                continue
            if len(atoms) <= 2:
                continue
            print("More than 2 atoms in bead %s are in ring %s" % (bead, ring))

    return partitioning


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


# --- Backwards-compatible module-level wrappers ---
# Some external code (or older notebooks/scripts) may import these helpers from this module.
def get_coords(conformer, sites, avg_pos, ringatoms_flat):
    return Cg_molecule.get_coords(conformer, sites, avg_pos, ringatoms_flat)


def get_heavy_atom_bonds(molecule, list_heavy_atoms):
    return Cg_molecule.get_heavy_atom_bonds(molecule, list_heavy_atoms)


def check_additivity(forcepred, beadtypes, molecule, mol_smi):
    return Cg_molecule.check_additivity(forcepred, beadtypes, molecule, mol_smi)


def get_graph(mol=None, smiles=None):
    return Cg_molecule.get_graph(mol=mol, smiles=smiles)


def distribute_neighbors(trial_comb, atoms):
    return Cg_molecule.distribute_neighbors(trial_comb, atoms)


def get_partitioning(trial_comb, graph):
    return Cg_molecule.get_partitioning(trial_comb, graph)


def make_mapping_dictionary(partitioning):
    return Cg_molecule.make_mapping_dictionary(partitioning)


def invert_mapping_dictionary(mapping_dict):
    return Cg_molecule.invert_mapping_dictionary(mapping_dict)


def get_bead_coords(partitioning, allatom_coords, molecule):
    return Cg_molecule.get_bead_coords(partitioning, allatom_coords, molecule)
