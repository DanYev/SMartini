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

from . import output
from . import optimization as optimization
from .topology import Topology, read_delta_f_types, smi2alogps, run_bartender
from .common import *

logger = logging.getLogger(__name__)


class Cg_molecule:
    """Main class to coarse-grain molecule"""

    # Initialize feature factory for feature extraction
    _fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    _factory = ChemicalFeatures.BuildFeatureFactory(_fdefName)

    # NOTE: These helpers are static because they don't depend on instance state.
    # Keeping them on the class groups mapping logic in one place.

    def __init__(self, molecule, mol_smi, molname, logp_file_name="logP_smi_extended.dat", 
        simple_model=None, topfname=None, bartenderfname=None, bartender=None, forcepred=True,
        min_beads=None, max_beads=None, raw_molecule=None):
        
        # NOTE _ha refers to heavy atoms, _aa refers to all atoms (including hydrogens), 

        # Store all arguments as instance attributes
        # AutoM3 new arguments : mol_smi, simple_model, bartenderfname, bartender, logp_file
        self.molecule = molecule
        self.smiles = mol_smi
        self.molname = molname
        self.simple_model = simple_model
        self.topfname = topfname
        self.bartenderfname = bartenderfname
        self.bartender = bartender
        self.logp_file = os.path.join(os.path.dirname(__file__), logp_file_name)
        self.forcepred = forcepred
        self.min_beads = min_beads
        self.max_beads = max_beads
        self.raw_molecule = raw_molecule
        
        # Initialize state attributes
        self.list_ha = None
        self.list_ha_names = None
        self.conf = None
        self.ha_coords = None
        self.aa_coords = None
        self.ring_atoms = None
        self.ring_atoms_flat = None
        self.is_arom = None
        self.num_arom = None
        self.hbond_a = None
        self.hbond_d = None
        self.list_bonds = None
        self.partitioning = None
        self.cg_bead_names = []
        self.cg_bead_coords = []
        self.topout = None
        self.bartender_out = None
        self.ga_graph = None
        self.aa_graph = None
        # Initialize topology early so it can be updated throughout
        self.topology = Topology(molname=self.molname, mol_smi=self.smiles, nrexcl=2)
        self.force_map = False

        logger.info("Starting coarse-graining for '%s' (forcepred=%s, simple_model=%s)", self.molname, self.forcepred, self.simple_model)
        logger.debug("Inputs: topfname=%s bartender=%s bartenderfname=%s logp_file=%s", self.topfname, self.bartender, self.bartenderfname, self.logp_file)

        # INITIALIZE THE AA MOLECULE
        self.ha_graph = self.build_ha_graph()  # Heavy atom graph for partitioning

        ## AutoM3 : MINIMIZATION with RDkit ###
        self.molecule = Chem.Mol(self.molecule)
        logger.debug("Embedding + MMFF optimization")
        AllChem.EmbedMolecule(self.molecule, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(self.molecule, maxIters=1000, mmffVariant='MMFF94s')
        AllChem.NormalizeDepiction(self.molecule, scaleFactor=1.12) 

        # Extract features and build all-atom graph structure
        self.feats = self.extract_features()
        self.aa_graph = self.build_aa_graph()
        
        # Populate attributes from aa_graph
        self.list_ha = self.aa_graph["list_ha"]
        self.list_ha_names = self.aa_graph["list_ha_names"]
        self.conf = self.aa_graph["conf"]
        if self.raw_molecule:
            self.ha_coords = self.ha_graph["ha_coords"]
        else:
            self.ha_coords = self.aa_graph["ha_coords"]
        self.aa_coords = self.aa_graph["aa_coords"]
        self.ring_atoms = self.aa_graph["ring_atoms"]
        self.ring_atoms_flat = self.aa_graph["ring_atoms_flat"]
        self.is_arom = self.aa_graph["is_aromatic"]
        self.num_arom = self.aa_graph["num_aromatic"]
        self.hbond_a = self.aa_graph["hbond_a"]
        self.hbond_d = self.aa_graph["hbond_d"]
        self.list_bonds = self.aa_graph["bonds"]
        
        # self.output_aa(f"{self.molname}_aa.gro") 
        logger.info("Detected %d heavy atoms", len(self.list_ha))
        logger.info("Ring atoms: %d (aromatic=%s, aromatic_count=%d)", 
                    len(self.ring_atoms_flat), 
                    self.is_arom, 
                    self.num_arom)

        # Actual mapping process
        self.process()

    def process(self):
        # Optimize coarse-grained bead positions 
        # -- keep all possibilities in case something goes wrong later in the code.
        list_cg_beads = optimization.find_bead_pos(
            self.molecule,
            self.conf,
            self.ha_graph,
            self.list_ha,
            self.ha_coords,
            self.aa_coords,
            self.ring_atoms,
            self.ring_atoms_flat,
            self.force_map,  # AutoM3 new argument
            min_beads=self.min_beads,
            max_beads=self.max_beads,
        )
        logger.info("Generated %d candidate bead mappings", len(list_cg_beads))

        # # Remove mappings with bead numbers less than most optimal mapping.
        # self.cg_beads_list = []
        # for cg_beads in list_cg_beads:
        #     if (
        #         len(cg_beads) == len(list_cg_beads[0])
        #         # and (len(self.list_ha) - (5 * len(cg_beads))) > 3
        #     )::1
        #         self.cg_beads_list.append(cg_beads)
        # logger.info("Removed suboptimal candidate bead mappings with bead number < %d", len(list_cg_beads[0]))

        # Loop through best 1% cg_beads and avg_pos
        # max_attempts = int(math.ceil(0.5 * len(list_cg_beads)))

        self.max_attempts = len(list_cg_beads) 
        logger.info("Going through the candidate mappings")
        for attempt in range(self.max_attempts):

            if attempt % 1000 == 0:  # Log every 1000 attempts
                logger.info("Attempt %d/%d", attempt, self.max_attempts)

            cg_beads = list_cg_beads[attempt]

            # if len(cg_beads) == 11:
            #     continue
            # self.partitioning = self.get_partitioning(cg_beads, self.graph)
            # exit()

            logger.info("Trying to partition the atoms between beads")
            try:
                self.partitioning = self.get_partitioning(cg_beads)
            except Exception:
                continue
            logger.info("Partitioned atoms into %d beads", len(self.cg_bead_coords))

            logger.debug("Attempt %d/%d: trying %d CG beads", attempt + 1, self.max_attempts, len(cg_beads))

            # Extract position of coarse-grained beads
            logger.info("Extracting coordinates for CG beads")
            self.cg_bead_coords = self.get_bead_coords()

            # CG beads should take atom rings number if ring atom in bead 
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
            mapping_dict = self.make_mapping_dictionary(self.partitioning)
            for ring in self.ring_atoms:
                for atom_idx in ring:
                    for bead_idx, atom_indices in mapping_dict.items():
                        if atom_idx in atom_indices:
                            # Add all atoms in this bead to self.ring_atoms_flat
                            for at in atom_indices:
                                if at not in ring:
                                    ring.append(at)

            logger.info("Building Atoms")
            # Use temporary Topology instance for trial build
            temp_topo = Topology(molname=self.molname, mol_smi=self.smiles)
            temp_topo.build_atoms(
                cgbeads=cg_beads,
                forcepred=self.forcepred,
                molecule=self.molecule,
                hbonda=self.hbond_a,
                hbondd=self.hbond_d,
                partitioning=self.partitioning,
                ringatoms=self.ring_atoms,
                ringatoms_flat=self.ring_atoms_flat,
                logp_file=self.logp_file,
                trial=True
            )
            self.cg_bead_names = temp_topo.atomnames
            bead_types = temp_topo.beadtypes

            # Check additivity between fragments and entire molecule
            if not self.check_additivity(bead_types):
                continue
            
            logger.info("Success mapping found on attempt %d", attempt)
            num_ar = self.build_topology(cg_beads, cg_beads_rings, bead_types)
            self.update_topology(cg_beads, cg_beads_rings, bead_types, attempt, num_ar)
            self.write_topology()
            break

    def extract_features(self):
        """Extract features of molecule (H-bond donors/acceptors, etc.)"""
        logger.debug("Entering extract_features()")
        features = Cg_molecule._factory.GetFeaturesForMol(self.molecule)
        return features

    def build_aa_graph(self):
        """Build all-atom graph data structure with heavy atoms, coords, rings, H-bonds, etc."""
        logger.debug("Entering build_aa_graph()")
        
        # Get list of heavy atoms and their names
        conformer = self.molecule.GetConformer()
        num_atoms = conformer.GetNumAtoms()
        list_ha = []
        list_ha_names = []
        atoms = range(num_atoms)
        for i in np.nditer(atoms):
            atom_name = self.molecule.GetAtomWithIdx(int(atoms[i])).GetSymbol()
            if atom_name != "H":
                list_ha.append(atoms[i])
                list_ha_names.append(f"{atom_name}{i+1}")
        if len(list_ha) == 0:
            print("Error. No heavy atom found.")
            exit(1)
        
        # Get coordinates - heavy atoms and all atoms
        ha_coords = []
        aa_coords = []
        for i in range(num_atoms):
            coord = np.array([conformer.GetAtomPosition(i)[j] for j in range(3)])
            if self.molecule.GetAtomWithIdx(i).GetSymbol() != "H":
                ha_coords.append(coord)
                aa_coords.append(coord)
            else:
                aa_coords.append(coord)
        
        # Get ring atoms (systems of joined rings)
        rings = self.molecule.GetRingInfo().AtomRings()
        ring_systems = []
        for ring in rings:
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
        ring_atoms = [list(ring) for ring in ring_systems]
        ring_atoms_flat = list(chain.from_iterable(ring_atoms))
        
        # Check if molecule is aromatic
        aromatic_atoms = [atom.GetIsAromatic() for atom in self.molecule.GetAtoms()]
        num_aromatic = sum(aromatic_atoms)
        is_aromatic = num_aromatic > 0
        
        # Get H-bond acceptors
        hbond_a = []
        for feat in self.feats:
            if feat.GetFamily() == "Acceptor":
                for i in feat.GetAtomIds():
                    if i not in hbond_a:
                        hbond_a.append(i)
        
        # Get H-bond donors
        hbond_d = []
        for feat in self.feats:
            if feat.GetFamily() == "Donor":
                for i in feat.GetAtomIds():
                    if i not in hbond_d:
                        hbond_d.append(i)
        
        # Get bonds between heavy atoms
        bonds = []
        for i in range(len(list_ha)):
            for j in range(i + 1, len(list_ha)):
                if self.molecule.GetBondBetweenAtoms(int(list_ha[i]), int(list_ha[j])) is not None:
                    bonds.append([list_ha[i], list_ha[j]])
        
        return {
            "list_ha": list_ha,
            "list_ha_names": list_ha_names,
            "conf": conformer,
            "ha_coords": ha_coords,
            "aa_coords": aa_coords,
            "ring_atoms": ring_atoms,
            "ring_atoms_flat": ring_atoms_flat,
            "is_aromatic": is_aromatic,
            "num_aromatic": num_aromatic,
            "hbond_a": hbond_a,
            "hbond_d": hbond_d,
            "bonds": bonds,
        }

    def get_ha_bonds(self):
        # List of bonds between heavy atoms
        list_bonds = []
        for i in range(len(self.list_ha)):
            for j in range(i + 1, len(self.list_ha)):
                if self.molecule.GetBondBetweenAtoms(int(self.list_ha[i]), int(self.list_ha[j])) is not None:
                    list_bonds.append([self.list_ha[i], self.list_ha[j]])
        return list_bonds

    def build_ha_graph(self):
        """Get graph representation of molecule based on heavy atoms only"""
        # --- Graph representation of the molecule ---
        if self.raw_molecule:
            mol = self.raw_molecule
        elif self.smiles:
            mol = Chem.MolFromSmiles(self.smiles)
        else:
            raise ValueError("Either mol or smiles must be provided")
        
        # Ensure molecule has conformer and extract coordinates
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=1)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000, mmffVariant='MMFF94s')
        
        conformer = mol.GetConformer()
        
        # Extract heavy atom coordinates
        ha_coords = []
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() != "H":
                coord = np.array([conformer.GetAtomPosition(i)[j] for j in range(3)])
                ha_coords.append(coord)
        
        # --- Node list (atoms) ---
        nodes = []
        ha_idx = 0
        for a in mol.GetAtoms():
            # Get heavy atom coordinate if this is a heavy atom
            coord = None
            if a.GetSymbol() != "H":
                coord = ha_coords[ha_idx]
                ha_idx += 1
            
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
                "coord": coord,                        # coordinates (only for heavy atoms)
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

        return {"atoms": nodes, "bonds": edges, "terminal_atoms": terminal_atoms, "ha_coords": ha_coords}

    def get_partitioning(self, trial_comb):
        """Get partitioning of atoms into beads for given trial combination"""
        atoms = self.ha_graph["atoms"]
        mapping = self._distribute_neighbors(trial_comb, atoms)
        mapping_dict = {idx: bead for idx, bead in enumerate(mapping)}
        partitioning = self.invert_mapping_dictionary(mapping_dict)
        return partitioning

    @staticmethod
    def _distribute_neighbors(trial_comb, atoms):
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

    def get_bead_coords(self):
        partitioning = self.partitioning
        # if raw_molecule is provided, use its coordinates instead of the original molecule's coordinates
        if self.raw_molecule:
            bead_coord = {}
            for atom in range(len(self.ha_coords)):
                bead = partitioning[atom]
                if bead not in bead_coord.keys():
                    bead_coord[bead] = []
                bead_coord[bead].append(self.ha_coords[atom])
            bead_cog = []
            for bead, coords in sorted(bead_coord.items()):
                cog = np.mean(coords, axis=0)
                bead_cog.append(cog)
            return bead_cog

        # compute COG while taking into account hydrogens
        # find all bonds between atoms in molecule
        bonds = []
        for b in range(len(self.molecule.GetBonds())):
            abond = self.molecule.GetBondWithIdx(b)
            at1 = abond.GetBeginAtomIdx()
            at2 = abond.GetEndAtomIdx()
            if f"{at1}-{at2}" not in bonds and f"{at2}-{at1}" not in bonds:
                bonds.append(f"{at1}-{at2}")

        # create partitioning including hydrogens inside beads
        aa_partitioning = self.partitioning.copy()
        for at in range(len(self.aa_coords)):
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
        for atom in range(len(self.aa_coords)):
            bead = aa_partitioning[atom]
            if bead not in bead_coord.keys():
                bead_coord[bead] = []
            bead_coord[bead].append(self.aa_coords[atom])

        bead_cog = []
        for bead, coords in sorted(bead_coord.items()):
            cog = np.mean(coords, axis=0)
            bead_cog.append(cog)

        return bead_cog

    def check_additivity(self, beadtypes): #AutoM3 change : added mol_smi argument
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
            delta_f_types = read_delta_f_types()
            sum_frag += delta_f_types[bead] #sum of free energies of beads in ring(s)
        # Wildman-Crippen log_p
        wc_log_p = rdMolDescriptors.CalcCrippenDescriptors(self.molecule)[0]
        # Get SMILES string of entire molecule

        whole_mol_dg,_ = smi2alogps(self.forcepred, self.smiles, wc_log_p, "MOL", None, None, True) # AutoM3 change : None,None=converted_smi, real_smi not needed here
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

    def build_topology(self, cg_beads, cg_beads_rings, bead_types):
        """Build topology data using Topology instance methods."""
        
        # Build atoms data
        self.topology.build_atoms(
            cgbeads=cg_beads,
            forcepred=self.forcepred,
            molecule=self.molecule,
            hbonda=self.hbond_a,
            hbondd=self.hbond_d,
            partitioning=self.partitioning,
            ringatoms=self.ring_atoms,
            ringatoms_flat=self.ring_atoms_flat,
            logp_file=self.logp_file,
            trial=False
        )
        # Override beadtypes if provided
        if bead_types is not None:
            self.topology.beadtypes = bead_types
        
        # Build bonds and constraints
        self.topology.build_bonds(
            cgbeads=cg_beads,
            cgbeads_ring=cg_beads_rings,
            molecule=self.molecule,
            partitioning=self.partitioning,
            cgbead_coords=self.cg_bead_coords,
            ringatoms=self.ring_atoms
        )
        
        # Build angles
        self.topology.build_angles(
            cgbeads=cg_beads,
            molecule=self.molecule,
            partitioning=self.partitioning,
            cgbead_coords=self.cg_bead_coords,
            ringatoms=self.ring_atoms
        )
        
        # Build dihedrals (unless simple model)
        num_ar = 0
        if not self.simple_model:
            num_ar = self.topology.build_dihedrals(
                cgbeads=cg_beads,
                ringatoms=self.ring_atoms,
                cgbead_coords=self.cg_bead_coords
            )
        
        # Build virtual sites if needed
        if self.ring_atoms and len(sum(self.ring_atoms, [])) > 6:
            self.topology.build_virtual_sites(
                ringatoms=self.ring_atoms,
                cg_bead_coords=self.cg_bead_coords,
                partitioning=self.partitioning,
                molecule=self.molecule
            )
        
        return num_ar


    def update_topology(self, cg_beads, cg_beads_rings, bead_types, attempt, num_ar=0):
        """Update topology with formatted output strings after successful mapping."""
        
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

        # Generate formatted outputs using topology methods
        header_write = self.topology.format_header()
        atoms_write = self.topology.format_atoms(trial=False)
        bonds_write = self.topology.format_bonds()
        angles_write = self.topology.format_angles()
        
        if not self.simple_model and self.topology.dihedrals:
            dihedrals_write = self.topology.format_dihedrals()
        else:
            dihedrals_write = ""
        
        virtual_sites_write = self.topology.format_virtual_sites()

        self.topout = self.topology.to_itp()

        # Build topology output and bartender input
        # run_bartender generates complete topology including exclusions and position_restraints
        if self.bartender and self.bartenderfname:
            self.bartender_out = run_bartender(
            header_write, atoms_write, bonds_write, angles_write, dihedrals_write,
            self.cg_bead_coords, self.ring_atoms, cg_beads,
            self.molecule, self.molname, self.topology.atoms_in_smi_dict,
            )
        
    def write_topology(self):
        """Write topology and bartender files to disk."""
        
        if self.bartender and self.bartenderfname:
            with open(self.bartenderfname, "w") as btf:
                btf.write(self.bartender_out)
            logger.info("Wrote bartender input: %s", self.bartenderfname)
        
        if self.topfname:
            with open(self.topfname, "w") as fp:
                fp.write(self.topout)
            logger.info("Wrote topology: %s", self.topfname)

    def output_aa_gro(self, aa_output=None): # AutoM3 change : molname is the same as argument --mol given at the beginning
        # Optional all-atom output to GRO file
        aa_out = output.output_gro(self.ha_coords, self.list_ha_names, self.molname)
        if aa_output:
            with open(aa_output, "w") as fp:
                fp.write(aa_out)
        else:
            return aa_out

    def output_cg_gro(self, cg_output=None): # AutoM3 change : molname is the same as argument --mol given at the beginning
        # Optional coarse-grained output to GRO file
        cg_out = output.output_gro(self.cg_bead_coords, self.cg_bead_names, self.molname)
        if cg_output:
            with open(cg_output, "w") as fp:
                fp.write(cg_out)
        else:
            return cg_out

    def output_cg_pdb(self, cg_output=None):
        """Output CG structure to PDB file with CONECT records from topology
        
        Parameters
        ----------
        cg_output : str, optional
            Path to output PDB file. If None, returns PDB string.
            
        Returns
        -------
        str or None
            PDB format string if cg_output is None, otherwise None
        """
        # Get bonds and constraints from topology
        bonds = self.topology.bonds if hasattr(self, 'topology') else None
        constraints = self.topology.constraints if hasattr(self, 'topology') else None
        # Generate PDB output with connectivity information
        cg_out = output.output_pdb(
            self.cg_bead_coords, 
            self.cg_bead_names, 
            self.molname,
            bonds=bonds,
            constraints=constraints
        )
        if cg_output:
            with open(cg_output, "w") as fp:
                fp.write(cg_out)
        else:
            return cg_out


################
# OLD STUFF
#################

def voronoi_atoms_old(cgbead_coords, ha_coords, aa_coords, molecule, in_partitioning): #AutoM3 change
    """Partition all atoms between CG beads"""
    logger.debug("Entering voronoi_atoms_old()")

    # Initial Partitioning based on closest bead to each heavy atom
    partitioning = {}
    for j in range(len(ha_coords)):
        # Voronoi to check whether atom is closest to bead
        bead_at = -1
        dist_bead_at = 1000
        for k in range(len(cgbead_coords)):
            distk = np.linalg.norm(cgbead_coords[k] - ha_coords[j])
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
            for j in range(len(ha_coords)):
                dist_bead_at = np.linalg.norm(cgbead_coords[i] - ha_coords[j])
                if dist_bead_at < closest_dist:
                    closest_dist = dist_bead_at
                    closest_atom = j
            if closest_atom == -1:
                logger.warning("Error. Can't find closest atom to bead %s" % i)
                exit(1)
            closest_atoms[i] = closest_atom

        # If one bead has only one heavy atom, include one more
        for i in partitioning.values():
            if sum(x == i for x in partitioning.values()) == 1:
                # Find bead
                lonely_bead = i
                # Voronoi to find closest atom
                closest_bead = -1
                closest_bead_dist = 10000.0
                for j in range(len(ha_coords)):
                    if partitioning[j] != lonely_bead:
                        dist_bead_at = np.linalg.norm(
                            cgbead_coords[lonely_bead] - ha_coords[j]
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
    for at in range(len(aa_coords)):
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
    for atom in range(len(aa_coords)):
        bead = aa_partitioning[atom]
        if bead not in bead_coord.keys(): 
            bead_coord[bead] = []
        bead_coord[bead].append(aa_coords[atom])

    bead_cog = []
    for bead, coords in sorted(bead_coord.items()):
        cog = np.mean(coords,axis=0)
        bead_cog.append(cog)

    return partitioning, bead_cog


def voronoi_atoms_new(cgbead_coords, ha_coords, aa_coords, molecule, in_partitioning): # AutoM3
    """
    Partition all atoms between CG beads, based on headliners coordinates and distances between other atoms coordinates. 
    Headliners are atoms with cgbead_coords coordinates.
    """
    logger.debug("Entering voronoi_atoms()")
    partitioning = {}

    #Populate partitioning with atoms and atom headliners of beads
    for j in range(len(ha_coords)):
        partitioning[j] = None
        for b in range(len(cgbead_coords)):
            if(ha_coords[j]==cgbead_coords[b]).all():
                partitioning[j] = b

    # Find closest atoms to atom headliners of beads
    if len(cgbead_coords) > 1:
        closest_atoms = {}  # Book-keeping of closest atoms to every bead
        for i in range(len(cgbead_coords)):
            distances = {}
            for j in range(len(ha_coords)):
                if (cgbead_coords[i] != ha_coords[j]).all():
                    dist_bead_at = np.linalg.norm(cgbead_coords[i] - ha_coords[j])
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
                for j in range(len(ha_coords)):
                    if partitioning[j] != lonely_bead:
                        dist_bead_at = np.linalg.norm(
                            cgbead_coords[lonely_bead] - ha_coords[j]
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
        for j in range(len(ha_coords)):
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
    for at in range(len(aa_coords)):
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
    for atom in range(len(aa_coords)):
        bead=aa_partitioning[atom]
        if bead not in bead_coord.keys(): 
            bead_coord[bead]=[]
        bead_coord[bead].append(aa_coords[atom])

    bead_cog=[]
    for bead, coords in sorted(bead_coord.items()):
        cog = np.mean(coords,axis=0)
        bead_cog.append(cog)

    return partitioning, bead_cog

def sanitize_rings(partitioning, atoms_xyz, ringatoms):
    mapping_dict = Cg_molecule.make_mapping_dictionary(partitioning)
    for bead, atoms in mapping_dict.items():
        for ring in ringatoms:
            if not set(atoms).issubset(ring):
                continue
            if len(atoms) <= 2:
                continue
            logger.warning("More than 2 atoms in bead %s are in ring %s" % (bead, ring))
    return partitioning

def all_atoms_in_beads_connected(trial_comb, ha_coords, 
    list_ha, bondlist, mol, aa_coords, force_map, in_partitioning): #AutoM3 change: added mol, force_map
    """Make sure all atoms within one CG bead are connected to at least
    one other atom in that bead"""
    # Bead coordinates are given by heavy atoms themselves
    cgbead_coords = []

    for i in range(len(trial_comb)):
        cgbead_coords.append(ha_coords[list_ha.index(trial_comb[i])])
    
    _, num_arom = topology.is_aromatic(mol) #AutoM3 change
    ### AutoM3 change of mapping approach to differenciate molecules with 0-1 and more cycles
    if not force_map and num_arom < 7: #AutoM3 change
        voronoi, _  = voronoi_atoms_new(cgbead_coords, ha_coords, aa_coords, mol, in_partitioning) #AutoM3 change
    else:
        voronoi, _  = voronoi_atoms_old(cgbead_coords, ha_coords, aa_coords, mol, in_partitioning) #AutoM3 change
    logger.debug("voronoi %s" % voronoi)

    for i in range(len(trial_comb)):
        cg_bead = trial_comb[i]
        num_atoms = list(voronoi.values()).count(voronoi[list_ha.index(cg_bead)])
        # sub-part of bond list that only contains atoms within CG bead
        sub_bond_list = []
        for j in range(len(bondlist)):
            if (
                voronoi[list_ha.index(bondlist[j][0])] == voronoi[list_ha.index(cg_bead)]
                and voronoi[list_ha.index(bondlist[j][1])] == voronoi[list_ha.index(cg_bead)]
            ):
                sub_bond_list.append(bondlist[j])
        num_bonds = len(sub_bond_list)
        if num_bonds < num_atoms - 1 or num_atoms == 1:
            logger.debug("Error: Not all atoms in beads connected in %s" % trial_comb)
            logger.debug("Error: %s < %s, %s" % (num_bonds, num_atoms - 1, sub_bond_list))
            return False
    return True
