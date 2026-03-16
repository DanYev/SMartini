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
from . import partitioning
from .topology import Topology, run_bartender
from .common import *

logger = logging.getLogger(__name__)


class Cg_molecule:
    """Main class to coarse-grain molecule"""

    # Initialize feature factory for feature extraction
    _fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    _factory = ChemicalFeatures.BuildFeatureFactory(_fdefName)

    # NOTE: These helpers are static because they don't depend on instance state.
    # Keeping them on the class groups mapping logic in one place.

    def __init__(self, molecule, mol_smi, molname, 
        specify_beads=None, min_beads=None, max_beads=None, 
        use_vsites=True, forcepred=True, raw_molecule=None, 
        bartenderfname=None, bartender=None, logp_file_name="logP_smi_extended.dat"):
        
        # NOTE _ha refers to heavy atoms, _aa refers to all atoms (including hydrogens), 

        # Store all arguments as instance attributes
        # AutoM3 new arguments : mol_smi, simple_model, bartenderfname, bartender, logp_file
        self.molecule = molecule
        self.smiles = mol_smi
        self.molname = molname
        self.specify_beads = specify_beads
        self.min_beads = min_beads
        self.max_beads = max_beads
        self.raw_molecule = raw_molecule
        self.use_vsites = use_vsites
        self.forcepred = forcepred
        self.logp_file = os.path.join(os.path.dirname(__file__), logp_file_name)
        self.bartender = bartender
        self.bartenderfname = bartenderfname
        
        # Initialize state attributes
        self.ha_list = None
        self.ha_bonds = None
        self.aa_coords = None
        self.ring_atoms = None
        self.ring_atoms_flat = None
        self.is_arom = None
        self.num_arom = None
        self.hbond_a = None
        self.hbond_d = None
        self.list_bonds = None
        self.neighbors = None
        self.mapping = None
        self.bead_names = []
        self.bead_coords = []
        self.bartender_out = None
        self.ga_graph = None
        self.aa_graph = None
        # Initialize topology early so it can be updated throughout
        self.topology = Topology(molname=self.molname, mol_smi=self.smiles, nrexcl=2)
        self.force_map = False

        logger.info("Starting coarse-graining for '%s' (forcepred=%s)", self.molname, self.forcepred)
        # INITIALIZE THE AA MOLECULE
        logger.info("Embedding the AA molecule + MMFF optimization")
        self.molecule = Chem.Mol(self.molecule)
        AllChem.EmbedMolecule(self.molecule, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(self.molecule, maxIters=1000, mmffVariant='MMFF94s')
        if not self.raw_molecule:
            self.raw_molecule = self.molecule  
        self.conformer = self.raw_molecule.GetConformer()

        # TODO: HA and AA / RAW and MOLECULE are messy now
        self.ha_list, self.ha_bonds = self.build_ha_graph()  # Heavy atom graph for partitioning
        self.ha_neighbors = [[n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1] for a in self.ha_list]

        # Extract features and build all-atom graph structure
        self.feats = self.extract_features()
        self.aa_graph = self.build_aa_graph()
        
        # Populate attributes from aa_graph
        self.list_aa = self.aa_graph["list_aa"]
        self.list_aa_names = self.aa_graph["list_aa_names"]
        self.conf = self.aa_graph["conf"]
        self.aa_coords = self.aa_graph["aa_coords"]
        self.ring_atoms = self.aa_graph["ring_atoms"]
        self.ring_atoms_flat = self.aa_graph["ring_atoms_flat"]
        self.is_arom = self.aa_graph["is_aromatic"]
        self.num_arom = self.aa_graph["num_aromatic"]
        self.hbond_a = self.aa_graph["hbond_a"]
        self.hbond_d = self.aa_graph["hbond_d"]
        self.list_bonds = self.aa_graph["bonds"]
        
        # self.output_aa(f"{self.molname}_aa.gro") 
        logger.info("Detected %d heavy atoms", len(self.ha_list))
        logger.info("Ring atoms: %d (aromatic=%s, aromatic_count=%d)", 
                    len(self.ring_atoms_flat), 
                    self.is_arom, 
                    self.num_arom)

        # Actual mapping process
        self.process()

    def process(self):
        # Find coarse-grained bead positions 
        # -- keep all possibilities in case something goes wrong later in the code.
        import pickle
        # mapping_pickle = Path(f"{self.molname}_candidate_mappings.pkl")
        # if not mapping_pickle.exists():
        #     mappings = partitioning.generate_mappings(
        #         self.molecule, 
        #         min_beads=self.min_beads,
        #         max_beads=self.max_beads,
        #     )
        #     logger.info("Generated %d candidate bead mappings", len(mappings))
        #     with open(mapping_pickle, "wb") as f:
        #         pickle.dump(mappings, f)
        # else:
        #     with open(mapping_pickle, "rb") as f:
        #         mappings = pickle.load(f)
        
        mappings = partitioning.generate_mappings(
                self.molecule, 
                min_beads=self.min_beads,
                max_beads=self.max_beads,
            )

        logger.info("Going through the candidate mappings")
        attempt = -1
        self.max_attempts = len(mappings) 
        for mapping in mappings:

            attempt += 1
            if attempt % 100 == 0:  # Log every 1000 attempts
                logger.info("Attempt %d/%d", attempt, self.max_attempts)
            print(mapping)

            # NOT NEEDED ANYMORE BUT USEFUL FOR DEBUGGING 
            try:
                mapping_dict = {idx: bead for idx, bead in enumerate(mapping)}
                self.partitioning = partitioning.invert_mapping_dictionary(mapping_dict)
            except:
                logger.warning("Failed to create partitioning dictionary for attempt %d: %s", attempt, mapping)
                # continue
            logger.debug("Attempt %d/%d: trying %d CG beads", attempt + 1, self.max_attempts, len(mapping))

            # IF AN ATOM IS IN A RING, ADD ALL ATOMS OF THIS BEADS TO THE RING ATOMS
            # for connectivity purposes
            for ring in self.ring_atoms:
                for atom_idx in ring:
                    for bead_idx, atom_indices in mapping_dict.items():
                        if atom_idx in atom_indices:
                            # Add all atoms in this bead to self.ring_atoms_flat
                            for at in atom_indices:
                                if at not in ring:
                                    ring.append(at)
            ringbeads = []
            for ring in self.ring_atoms:
                new_ring = []
                for atom_idx in ring:
                    for bead_idx, atom_indices in mapping_dict.items():
                        if atom_idx in atom_indices and bead_idx not in new_ring:
                            new_ring.append(bead_idx)
                            new_ring.sort()
                ringbeads.append(new_ring)

            # Get bead types based on mapping and features of the atoms in each bead
            try:
                bead_types, bead_smiles, bead_atomnames, charges = get_bead_types(
                    mapping=mapping,
                    molecule=self.molecule,
                    hbonda=self.hbond_a,
                    hbondd=self.hbond_d,
                )
            except:
                continue  # If get_bead_types fails for any reason, skip to the next mapping
            logger.info("Assigned bead types: %s", bead_types)

            # Extract position of coarse-grained beads
            logger.info("Extracting coordinates for CG beads")
            self.mapping = mapping
            sym_mapping = self.symmetrize_rings_in_mapping(mapping)
            # sym_mapping = mapping
            self.aa_mapping = self.get_aa_mapping(sym_mapping)  # Update mapping to include hydrogens in the same bead as their heavy atom neighbors
            self.bead_coords = self.get_bead_coords(mapping=self.aa_mapping)  # Get bead coordinates based on AA mapping
            logger.info("Partitioned atoms into %d beads", len(self.bead_coords))

            # Build the topology instance for this mapping
            logger.info("Building Atoms...")
            topo = Topology(molname=self.molname, mol_smi=self.smiles)
            topo.bead_atomnames = bead_atomnames
            topo.bead_smiles = bead_smiles
            topo.charges = charges
            topo.ringbeads = ringbeads
            topo.build_atoms(
                mapping=mapping,
                bead_types=bead_types,
                bead_coords=self.bead_coords,
                molecule=self.molecule, 
                molname=self.molname,
            )
            self.bead_names = topo.names

            # Check additivity between fragments and entire molecule
            # if not self.check_additivity(bead_types):
            #     continue
            
            logger.info("Success mapping found on attempt %d", attempt)
            self.topology = topo
            self.build_topology(mapping)
            self.update_topology(mapping, bead_types, attempt)
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
        conformer = self.conformer
        num_atoms = conformer.GetNumAtoms()
        list_aa = []
        list_aa_names = []
        atoms = range(num_atoms)
        for i in np.nditer(atoms):
            atom_name = self.molecule.GetAtomWithIdx(int(atoms[i])).GetSymbol()
            list_aa.append(atoms[i])
            list_aa_names.append(f"{atom_name}{i+1}")
        
        # Get coordinates - heavy atoms and all atoms
        aa_coords = []
        for i in range(num_atoms):
            coord = np.array([conformer.GetAtomPosition(i)[j] for j in range(3)])
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
        for i in range(len(list_aa)):
            for j in range(i + 1, len(list_aa)):
                if self.molecule.GetBondBetweenAtoms(int(list_aa[i]), int(list_aa[j])) is not None:
                    bonds.append([list_aa[i], list_aa[j]])
        
        return {
            "list_aa": list_aa,
            "list_aa_names": list_aa_names,
            "conf": conformer,
            "aa_coords": aa_coords,
            "ring_atoms": ring_atoms,
            "ring_atoms_flat": ring_atoms_flat,
            "is_aromatic": is_aromatic,
            "num_aromatic": num_aromatic,
            "hbond_a": hbond_a,
            "hbond_d": hbond_d,
            "bonds": bonds,
        }


    def build_ha_graph(self):
        """Extract molecule info needed for partitioning."""
        molecule = self.molecule
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


    def symmetrize_rings_in_mapping(self, mapping):

        def symmetrize_ring(mapping, ring, ring_beads):
            m = mapping.copy()
            for bead in ring_beads:
                if bead in m:
                    m.remove(bead)
            mapping_1 = m.copy()
            for i in range(0, len(ring), 2):
                bead = [ring[i], ring[(i + 1) % len(ring)], ring[(i + 1) % len(ring)], ring[(i + 2) % len(ring)]] 
                mapping_1.append(bead)
            mapping_1 = sort_nested(mapping_1)
            mapping_2 = m.copy()
            for i in range(1, len(ring) + 1, 2):
                bead = [ring[i], ring[(i + 1) % len(ring)], ring[(i + 1) % len(ring)], ring[(i + 2) % len(ring)]] 
                mapping_2.append(bead)
            mapping_2 = sort_nested(mapping_2)
            mappings = [mapping_1, mapping_2]
            return mappings

        molecule = self.molecule
        rings = molecule.GetRingInfo().AtomRings()
        # go thru each ring
        sym_mappings = [mapping]
        for ring in rings:
            new_mappings = []
            for mapping in sym_mappings:
                ring_beads = [bead for bead in mapping if all(atom in ring for atom in bead)]
                if flat_set(ring_beads) == set(ring) and len(ring) > 5:
                    symetrized = symmetrize_ring(mapping, ring, ring_beads)
                    new_mappings.extend(symetrized)
            sym_mappings = new_mappings if new_mappings else sym_mappings
        # If specify_beads is set, filter sym_mappings to only those that contain all specified atoms in the same bead
        if self.specify_beads:
            mappings_with_ag = sym_mappings
            for ag in self.specify_beads:
                mappings_with_ag = [m for m in mappings_with_ag if any(all(atom in bead for atom in ag) for bead in m)]
            if mappings_with_ag:
                return mappings_with_ag[0] 
            else:
                raise ValueError(f"None of the symmetrized mappings contain all specified atoms {self.specify_beads}")
        return sym_mappings[0]


    def get_aa_mapping(self, mapping):
        """Update HA partitioning to include hydrogens in the same bead as their heavy atom neighbors"""
        aa_mapping = []
        for bead in mapping:
            bead = bead.copy()
            for atom_idx in bead:
                atom = self.molecule.GetAtomWithIdx(int(atom_idx))
                if atom.GetSymbol() != "H":
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 1:  # If neighbor is hydrogen
                            neighbor_idx = neighbor.GetIdx()
                            if neighbor_idx not in bead:
                                bead.append(neighbor_idx) 
            aa_mapping.append(bead)
        return aa_mapping


    def get_bead_coords(self, mapping):
        # Extract atom coordinates
        aa_coords = []
        mol = self.raw_molecule
        conformer = self.conformer
        for i in range(mol.GetNumAtoms()):
            coord = np.array([conformer.GetAtomPosition(i)[j] for j in range(3)])
            aa_coords.append(coord)
        # Map
        bead_coords = []
        for bead in mapping:
            bead_cog = np.zeros(3)
            for atom_idx in bead:
                bead_cog += aa_coords[atom_idx]
            bead_cog /= len(bead)
            bead_coords.append(bead_cog)
        return bead_coords


    def check_additivity(self, beadtypes): #AutoM3 change : added mol_smi argument
        """Check additivity assumption between sum of free energies of CG beads
        and free energy of whole molecule"""
        logger.info("Checking LogP additivity...")
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


    def build_topology(self, beads):
        """Build topology data using Topology instance methods."""
        # Build bonds and constraints
        self.topology.build_bonds(ha_neighbors=self.ha_neighbors)
        # Build exclusions 
        # BEFORE virtual sites so that we can duplicate default nrexcl exclusions for virtual sites
        self.topology.build_exclusions()
        # Build virtual sites.
        # BEFORE angles and dihedrals so that they do not end up in any angles or dihedrals.
        if self.use_vsites:
            self.topology.build_virtual_sites()
        # Build angles
        self.topology.build_angles()
        # Build dihedrals
        self.topology.build_dihedrals()


    def update_topology(self, beads, bead_types, attempt):
        """Update topology with formatted output strings after successful mapping."""


        # Store convenience references
        self.topology.aa_mapping = self.aa_mapping
        self.bead_names = self.topology.names
        
        logger.info("Final CG model: %d beads", len(self.bead_names))

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
        atoms_write = self.topology.format_atoms()
        bonds_write = self.topology.format_bonds()
        angles_write = self.topology.format_angles()
        dihedrals_write = self.topology.format_dihedrals()
        virtual_sites_write = self.topology.format_virtual_sites()

        # Build topology output and bartender input
        # run_bartender generates complete topology including exclusions and position_restraints
        if self.bartender and self.bartenderfname:
            self.bartender_out = run_bartender(
            header_write, atoms_write, bonds_write, angles_write, dihedrals_write,
            self.bead_coords, self.ring_atoms, beads,
            self.molecule, self.molname, self.topology.atoms_in_smi_dict,
            )
        

    def to_itp(self, itp_output=None):
        """Write topology and bartender files to disk."""
        topout = self.topology.to_itp()
        
        if self.bartender and self.bartenderfname:
            with open(self.bartenderfname, "w") as btf:
                btf.write(self.bartender_out)
            logger.info("Wrote bartender input: %s", self.bartenderfname)
        
        if itp_output:
            with open(itp_output, "w") as fp:
                fp.write(topout)
            logger.info("Wrote topology: %s", itp_output)
        else:
            return topout


    def to_aa_gro(self, aa_output=None): # AutoM3 change : molname is the same as argument --mol given at the beginning
        # Optional all-atom output to GRO file
        aa_out = output.output_gro(self.ha_coords, self.list_ha_names, self.molname)
        if aa_output:
            with open(aa_output, "w") as fp:
                fp.write(aa_out)
        else:
            return aa_out


    def to_gro(self, cg_output=None): # AutoM3 change : molname is the same as argument --mol given at the beginning
        # Optional coarse-grained output to GRO file
        cg_out = output.output_gro(self.bead_coords, self.bead_names, self.molname)
        if cg_output:
            with open(cg_output, "w") as fp:
                fp.write(cg_out)
        else:
            return cg_out


    def to_pdb(self, cg_output=None):
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
            self.bead_coords, 
            self.bead_names, 
            self.molname,
            bonds=bonds,
            constraints=constraints
        )
        if cg_output:
            with open(cg_output, "w") as fp:
                fp.write(cg_out)
        else:
            return cg_out


    def output_map(self, 
        map_file: str = None, 
        to_ff: str = "martini3001"
        ):
        output.output_map(self.topology, map_file, to_ff=to_ff)


def get_bead_types(mapping, molecule, hbonda, hbondd, logp_file=None, forcepred=True):
    """Determine bead types based on smiles of the the atomistic bead fragment."""

    logger.info("Determining bead types")
    bead_types = []
    bead_smiles = []
    bead_atomnames = []
    charges = []
    at_counts = {}

    for idx, bead in enumerate(mapping):
        smi_frag, wc_log_p, charge, atoms_in_smi, converted_smi, real_smi, at_counts = substruct2smi(bead, molecule, at_counts)
        if "." in smi_frag:
            logger.info((f"Fragment SMILES contains a dot ('.'), your atoms in bead {bead}: {smi_frag} are disconnected. "
            "Skipping to the next one."))
            raise ValueError

        if charge == 0:
            alogps, logp_origin = smi2alogps(forcepred, smi_frag, wc_log_p, idx + 1, converted_smi, real_smi, logp_file)
        else:
            alogps = 0.0
            logp_origin = "; Charged fragment"

        hbond_a_flag = sum(1 for at in hbonda if at in bead)
        hbond_d_flag = sum(1 for at in hbondd if at in bead)

        ring_atoms = molecule.GetRingInfo().AtomRings()
        ring_atoms_flat = [at for ring in ring_atoms for at in ring]
        in_ring = any(at in ring_atoms_flat for at in bead)

        bead_type = determine_bead_type(alogps, charge, hbond_a_flag, hbond_d_flag, in_ring, smi_frag)

        bead_types.append(bead_type)
        bead_smiles.append(smi_frag)
        bead_atomnames.append(atoms_in_smi)
        charges.append(charge)

    return bead_types, bead_smiles, bead_atomnames, charges


def substruct2smi(bead, molecule, at_counts={}):
    """Substructure to smiles conversion; also output Wildman-Crippen log_p;
    and charge of group."""
    logger.debug("Entering substruct2smi() for bead %d", bead)
    frag = rdchem.EditableMol(molecule)

    num_atoms = molecule.GetNumAtoms()
    # First delete all hydrogens
    for i in range(num_atoms):
        if molecule.GetAtomWithIdx(i).GetSymbol() == "H":
            # find atom from coordinates
            submol = frag.GetMol()
            for j in range(submol.GetNumAtoms()):
                if (
                    molecule.GetConformer().GetAtomPosition(i)[0]
                    == submol.GetConformer().GetAtomPosition(j)[0]
                ):
                    frag.RemoveAtom(j)
    n_heavy = frag.GetMol().GetNumAtoms()
    
    # Then heavy atoms that aren't part of the CG bead #(except those
    # involved in the same ring).
    for atom_idx in range(n_heavy):
        if atom_idx not in bead: # AutoM3 change
            # find atom from coordinates
            submol = frag.GetMol()
            for j in range(submol.GetNumAtoms()):
                if (
                    molecule.GetConformer().GetAtomPosition(atom_idx)[0]
                    == submol.GetConformer().GetAtomPosition(j)[0]
                ):
                    frag.RemoveAtom(j)
    # Wildman-Crippen log_p
    wc_log_p = rdMolDescriptors.CalcCrippenDescriptors(frag.GetMol())[0]
    # Charge -- look at atoms that are only part of the bead (no ring rule)
    chg = 0
    for i in bead:
        chg += molecule.GetAtomWithIdx(i).GetFormalCharge()

    smi = Chem.MolToSmiles(Chem.rdmolops.AddHs(frag.GetMol(), addCoords=True))
    ### AutoM3 ###
    atoms_in_smi = ""
    converted_smi = False
    real_smi = None
    for at in range(n_heavy):
        if at in bead:
            at_symbol = molecule.GetAtomWithIdx(at).GetSymbol()
            if at_symbol not in at_counts.keys():
                at_counts[at_symbol] = 1
            else:                
                at_counts[at_symbol] += 1 
            atom_label = at_symbol + str(at_counts[at_symbol])
            atoms_in_smi += atom_label + " "
    if "c" in smi or "n" in smi or "s" in smi:
        converted_smi = True
        real_smi = smi
        smi = cyclic_smi_conversion(smi)
    # fragment smi: Nc1ncnn1 ---------> FAILURE! Need to fix this Andrew! For now, just a hackish soln:
    # smi = smi.lower() if smi.islower() else smi.upper()
    atoms_in_smi = atoms_in_smi
    return smi, wc_log_p, chg, atoms_in_smi, converted_smi, real_smi, at_counts


def letter_occurrences(string):
    """Count letter occurences"""
    frequencies = defaultdict(lambda: 0)
    for character in string:
        if character.isalnum():
            frequencies[character.upper()] += 1
    return frequencies


def cyclic_smi_conversion(smi): # AutoM3 function
    """ Function for converting cyclic atoms in smiles from upper case to lower for them being accepted by rdkit and not raise error : 
    rdkit.Chem.rdchem.AtomKekulizeException: non-ring atom 0 marked aromatic
    """
    smi = smi.replace("ccc","CC=C")
    smi = smi.replace("cc","C=C")
    smi = smi.replace("c","C")
    smi = smi.replace("n","N")
    smi = smi.replace("s","S")
    smi = smi.replace("o","O")
    return (smi)


def find_closest_key(dictionary, target_value): # AutoM3 function
    lst=list(dictionary.keys())
    closest_key = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-target_value))]
    return closest_key


def rearrange_until_match(input_string): # AutoM3 function
    letters = [char for char in input_string if char.isalpha()]
    random.shuffle(letters)
    result_string = '-'.join(letters)
    return result_string


def gen_molecule_smi(smi):
    """Generate mol object from smiles string"""
    logger.debug("Input SMILES: %s", smi)
    errval = 0
    if "." in smi:
        logger.warning("Error. Only one molecule may be provided.")
        logger.warning(smi)
        errval = 4
        exit(1)
    # If necessary, adjust smiles for Aromatic Ns
    # Redirect current stderr in log file
    stderr_fd = None
    stderr_save = None
    # try:
    #     stderr_fileno = sys.stderr.fileno()
    #     stderr_save = os.dup(stderr_fileno)
    #     stderr_fd = open("sanitize.log", "w")
    #     os.dup2(stderr_fd.fileno(), stderr_fileno)
    # except Exception:
    #     stderr_fileno = None
    # Get smiles without sanitization
    logger.debug("Creating RDKit Mol from SMILES (sanitize=False)")
    molecule = Chem.MolFromSmiles(smi, False)
    try:
        logger.debug("Sanitizing RDKit Mol")
        cp = Chem.Mol(molecule)
        Chem.SanitizeMol(cp)

        # # Close log file and restore old sys err
        # if stderr_fileno is not None:
        #     stderr_fd.close()
        #     os.dup2(stderr_save, stderr_fileno)
        molecule = cp
    except ValueError:
        logger.warning("Bad smiles format %s found" % smi)
        logger.debug("Attempting to adjust aromatic nitrogens")
        nm = AdjustAromaticNs(molecule)

        if nm is not None:
            Chem.SanitizeMol(nm)
            molecule = nm
            smi = Chem.MolToSmiles(nm)
            logger.warning("Fixed smiles format to %s" % smi)
        else:
            logger.warning("Smiles cannot be adjusted %s" % smi)
            errval = 1
    # Continue
    logger.debug("Adding hydrogens + embedding + UFF optimization")
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule, randomSeed=1, useRandomCoords=True)  # Set Seed for random coordinate generation = 1.
    try:
        AllChem.UFFOptimizeMolecule(molecule)
    except ValueError as e:
        logger.warning("%s" % e)
        exit(1)
    logger.debug("Successfully generated molecule from SMILES")
    return molecule, errval


def gen_molecule_sdf(sdf):
    """Generate mol object from SD file"""
    logger.debug("Entering gen_molecule_sdf()")
    logger.info("Input SDF: %s", sdf)
    suppl = Chem.SDMolSupplier(sdf)
    logger.debug("SDF supplier length: %d", len(suppl))
    molecule = suppl[0]
    logger.debug("Sanitizing RDKit Mol from SDF")
    Chem.SanitizeMol(molecule)
    logger.debug("Adding hydrogens + embedding + UFF optimization")
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule, randomSeed=1, useRandomCoords=True)  # Set Seed for random coordinate generation = 1.
    try:
        AllChem.UFFOptimizeMolecule(molecule)
    except ValueError as e:
        exit(1)
    logger.info("Successfully generated molecule from SDF")
    raw_molecule = suppl[0]
    return molecule, raw_molecule


def get_charge(molecule):
    """Get net charge of molecule"""
    return Chem.rdmolops.GetFormalCharge(molecule)


def find_closest_logPvalue(value, keyslist, in_ring): ### AutoM3 ###
    closest_key = None
    closest_diff = float('inf')
    dict=read_delta_f_types()
    for key in keyslist:
        if key in dict:
            diff = mad(key,value,in_ring)
            #diff = abs(value - dict[key])
            if diff < closest_diff:
                closest_key = key
                closest_diff = diff
    return closest_key


def determine_bead_type(delta_f, charge, hbonda, hbondd, in_ring, smi_frag): ### AutoM3 ###
    """Determine CG bead type from delta_f value, charge,
    and hbond acceptor, and donor"""
    if charge < -2 or charge > +2:
        logger.error("Charge is too large: %s" % charge)
        exit(1)
    # bead_type = None
    #smi_frag = ''.join(char for char in smi_frag if char.isalpha() and char!='H')
    if charge != 0:
        if charge == -2 or charge == -2:
            if count_letters(str(smi_frag)) == 2:
                bead_type = "TD"
            if count_letters(str(smi_frag)) == 3:
                bead_type = "SD"
            if count_letters(str(smi_frag)) > 3:
                bead_type = " D"
        else:
            # The compound has a +/- charge -> Q type
            if count_letters(str(smi_frag)) == 2:
                other_types_Q = ["TQ1", "TQ2", "TQ3", "TQ4", "TQ5",]
            if count_letters(str(smi_frag)) == 3:
                othertypes_Q = ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5",]
            if count_letters(str(smi_frag)) > 3:
                othertypes_Q = ["Q1", "Q2", "Q3", "Q4", "Q5",]
            bead_type = find_closest_logPvalue(delta_f, othertypes_Q, in_ring)

    else:
        # Neutral group
        if hbonda > 0 or hbondd > 0:
            if count_letters(str(smi_frag)) == 2:
                other_types_NPa = ["TN1a", "TN2a", "TN3a", "TN4a", "TN5a", "TN6a", "TP1a", "TP2a", "TP3a", "TP4a", "TP5a", "TP6a"]
                other_types_NPd = ["TN1d", "TN2d", "TN3d", "TN4d", "TN5d", "TN6d", "TP1d", "TP2d", "TP3d", "TP4d", "TP5d", "TP6d"]
            if count_letters(str(smi_frag)) == 3:
                other_types_NPa = ["SN1a", "SN2a", "SN3a", "SN4a", "SN5a", "SN6a", "SP1a", "SP2a", "SP3a", "SP4a", "SP5a", "SP6a"]
                other_types_NPd = ["SN1d", "SN2d", "SN3d", "SN4d", "SN5d", "SN6d", "SP1d", "SP2d", "SP3d", "SP4d", "SP5d", "SP6d"]
            if count_letters(str(smi_frag)) > 3:
                other_types_NPa = ["N1a", "N2a", "N3a", "N4a", "N5a", "N6a", "P1a", "P2a", "P3a", "P4a", "P5a", "P6a"]
                other_types_NPd = ["N1d", "N2d", "N3d", "N4d", "N5d", "N6d", "P1d", "P2d", "P3d", "P4d", "P5d", "P6d"]

            if hbonda > 0 and hbondd == 0:
                bead_type = find_closest_logPvalue(delta_f, other_types_NPa, in_ring)
            if hbonda >= 0 and hbondd > 0:
                bead_type = find_closest_logPvalue(delta_f, other_types_NPd, in_ring)

        else:
            # all other cases. Simply find the atom type that's closest in
            # free energy.
            
            if count_letters(str(smi_frag)) == 2:
                other_types = ["TP6", "TP5", "TP4", "TP3", "TP2", "TP1", "TC6", "TC5", "TC4", "TC3", "TC2", "TC1", "TN6", "TN5", "TN4", "TN3", "TN2", "TN1"]
                if not in_ring: other_types.remove("TC5")

            if count_letters(str(smi_frag)) == 3:
                other_types = ["SP6", "SP5", "SP4", "SP3", "SP2", "SP1", "SC6", "SC5", "SC4", "SC3", "SC2", "SC1", "SN6", "SN5", "SN4", "SN3", "SN2", "SN1"]

            if count_letters(str(smi_frag)) > 3:
                other_types = ["P6", "P5", "P4", "P3", "P2", "P1", "C6", "C5", "C4", "C3", "C2", "C1", "N6", "N5", "N4", "N3", "N2", "N1"]

            bead_type = find_closest_logPvalue(delta_f, other_types, in_ring)
            #logger.debug("closest type: %s; error %7.4f" % (bead_type, min_error))
    
    for hal in ["Cl", "Br", "F", "I"]:
        if hal in str(smi_frag):
            if count_letters(str(smi_frag)) == 2: other_types = ["TX4", "TX3", "TX2", "TX1"]
            if count_letters(str(smi_frag)) == 3: other_types = ["SX4", "SX3", "SX2", "SX1"]
            if count_letters(str(smi_frag)) > 3: other_types = ["X4", "X3", "X2", "X1"]
            bead_type = find_closest_logPvalue(delta_f, other_types, in_ring)

    return bead_type


def get_mass(smi): # AutoM3
    """Gets real mass of atoms in smile code"""
    smi_mass=0
    atom_mass={"C":12,"O":16,"N":14,"S":32,"Cl":35,"I":127,"F":19,"Br":80,"P":31,"Si":28,"B":11,"Be":9,"Li":1,"Mg":24,"Ca":40,"K":39}
    i = 0
    while i < len(smi):
        if i < len(smi)-1 and smi[i:i+2] in atom_mass:  # Check if the current two characters form a known atom
            smi_mass += atom_mass[smi[i:i+2]]
            i += 2
        elif smi[i] in atom_mass:  # Check if the current character forms a known atom
            smi_mass += atom_mass[smi[i]]
            i += 1
        else:  # Skip unknown characters
            i += 1
    return smi_mass


def smi2alogps(forcepred, smi, wc_log_p, bead, converted_smi, real_smi, logp_file=None, trial=False): 
    """
    Returns water/octanol partitioning free energy according to ALOGPS
    AutoM3 : Returns water/octanol partitioning free energy defined empiricaly from customized database
    """
    logger.debug("Entering smi2alogps()")
    forcepred = False

    ## AutoM3 ###
    if not logp_file:
        logp_file = os.path.join(os.path.dirname(__file__), 'logP_smi_extended.dat')
    found_smi = False
    if bead != "MOL":
        logP_data = {}
        if converted_smi:
            smi=real_smi

        # Check if logp_file is a valid file name
        if isinstance(logp_file, str) and logp_file:
            try:
                with open(logp_file) as f:
                    for line in f:
                        (key, val) = line.rstrip().split()
                        logP_data[key] = float(val)
            except Exception as e:
                logger.error(f"An error occurred while reading the logP file: {e}")
        else:
            logger.error(f"Invalid file name: {logp_file}")
        
        log_p = 0.0
        for smiles, logp in logP_data.items():
            if smiles == smi:
                log_p = float(logp)
                found_smi = True
                return (log_p, "")
                #break

    if not found_smi:
        if converted_smi:
            smi=real_smi
        req = ""
        soup = ""
        try:
            session = requests.session()
            logger.debug("Calling http://vcclab.org/web/alogps/calc?SMILES=" + str(smi))
            req = session.get(
                "http://vcclab.org/web/alogps/calc?SMILES=" + str(smi.replace("#", "%23"))
            )
        except:
            print("Error. Can't reach vcclab.org to estimate free energy.")
            exit(1)
        try:
            doc = BeautifulSoup(req.content, "lxml")
        except Exception:
            raise
        try:
            soup = doc.prettify()
        except:
            print("Error with BeautifulSoup prettify")
            exit(1)
        found_mol_1 = False
        log_p = None
        for line in soup.split("\n"):
            line = line.split()
            if "mol_1" in line:
                log_p = float(line[line.index("mol_1") + 1])
                found_mol_1 = True
                break
        if not found_mol_1:
            # If we're forcing a prediction, use Wildman-Crippen
            if forcepred:
                if trial:
                    wrn = (
                        "; Warning: bead ID "
                        + str(bead)
                        + " predicted from Wildman-Crippen. Fragment "
                        + str(smi)
                        + "\n"
                    )
                    sys.stderr.write(wrn)
                log_p = wc_log_p
            else:
                log_p = 0.0
                # print("ALOGPS can't predict fragment: %s" % smi)
                # exit(1)
        logger.debug("logp value: %7.4f" % log_p)
        return (convert_log_k(log_p),"; ALOGPS defined bead")


def convert_log_k(log_k):
    """Convert log_{10}K to free energy (in kJ/mol)"""
    val = 0.008314 * 300.0 * log_k / math.log10(math.exp(1))
    logger.debug("free energy %7.4f kJ/mol" % val)
    return val


def mad(bead_type, delta_f, in_ring=False):
    """Mean absolute difference between bead type and delta_f"""
    # logger.debug('Entering mad()')
    delta_f_types = read_delta_f_types()
    return math.fabs(delta_f_types[bead_type] - delta_f)


def count_letters(s): ### AutoM3 ###
    """ Counting atoms in SMILES code """
    count = 0
    i = 0
    while i < len(s):
        if s[i:i+2] in ["Cl", "Br"]:
            count += 1
            i += 2
        elif s[i].isalpha():
            count += 1
            i += 1
        else:
            i += 1
    return count


def read_delta_f_types():
    """
    AutoM3 : New data for Martini 3 Force Field, from SI https://doi.org/10.1038/s41592-021-01098-3
    Returns delta_f types dictionary
    """
    delta_f_types = dict()
    delta_f_types = {"C1":18.9,"C2":14.8,"C3":13.8,"C4":13.4,"C5":11.2,"C6":10.1,"N1":8.1,"N2":5.6,"N3":1.8,"N4":2.2,"N5":0.0,"N6":-1.1,"P1":-2.0,"P2":-3.8,"P3":-5.1,"P4":-7.4,"P5":-9.1,
                     "P6":-9.2,"X1":14.3,"X2":12.7,"X3":13.9,"X4":8.7,"N1d":10.7,"N1a":10.7,"N2d":7.8,"N2a":7.8,"N3d":3.8,"N3a":3.8,"N4d":4.3,"N4a":4.3,"N5d":2.2,"N5a":2.2,"N6d":1.0,
                     "N6a":1.0,"P1d":0.2,"P1a":0.2,"P2a":-1.9,"P2d":-1.9,"P3d":-3.5,"P3a":-3.5,"P4d":-5.1,"P4a":-5.1,"P5d":-7.0,"P5a":-7.0,"P6d":-7.4,"P6a":-7.4,"Q1":-10.9,"Q2":-15.1,
                     "Q3":-17.4,"Q4":-18.8,"Q5":-23.0," D":-26.8,
                     "SC1":14.2,"SC2":9.9,"SC3":9.2,"SC4":8.4,"SC5":6.3,"SC6":5.3,"SN1":3.6,"SN2":2.1,"SN3":-1.8,"SN4":-0.9,"SN5":-3.6,"SN6":-4.2,"SP1":-5.2,"SP2":-6.9,"SP3":-7.7,"SP4":-9.8,"SP5":-11.8,
                     "SP6":-12.0,"SX1":9.4,"SX2":7.2,"SX3":8.0,"SX4":4.3,"SN1d":6.0,"SN1a":6.0,"SN2d":3.8,"SN2a":3.8,"SN3d":0.2,"SN3a":0.2,"SN4d":1.1,"SN4a":1.1,"SN5d":-1.0,"SN5a":-1.0,"SN6d":-2.5,
                     "SN6a":-2.5,"SP1d":-3.7,"SP1a":-3.7,"SP2d":-5.4,"SP2a":-5.4,"SP3d":-6.1,"SP3a":-6.1,"SP4d":-7.8,"SP4a":-7.8,"SP5d":-9.5,"SP5a":-9.5,"SP6d":-9.6,"SP6a":-9.6,"SQ1":-10.6,"SQ2":-14.3,
                     "SQ3":-18.0,"SQ4":-18.2,"SQ5":-18.2,"SD":-36.4,
                     "TC1":12.0,"TC2":7.8,"TC3":6.7,"TC4":6.4,"TC5":4.5,"TC6":3.6,"TN1":2.3,"TN2":0.3,"TN3":-3.1,"TN4":-2.9,"TN5":-4.9,"TN6":-6.1,"TP1":-7.2,"TP2":-8.8,"TP3":-9.8,"TP4":-12.1,"TP5":-15.2,
                     "TP6":-14.8,"TX1":7.6,"TX2":5.2,"TX3":5.4,"TX4":2.7,"TN1d":3.9,"TN1a":3.9,"TN2d":2.3,"TN2a":2.3,"TN3d":-1.4,"TN3a":-1.4,"TN4d":-1.2,"TN4a":-1.2,"TN5d":-2.8,"TN5a":-2.8,"TN6d":-4.1,
                     "TN6a":-4.1,"TP1d":-5.0,"TP1a":-5.0,"TP2d":-6.8,"TP2a":-6.8,"TP3d":-7.8,"TP3a":-7.8,"TP4d":-9.5,"TP4a":-9.5,"TP5d":-13.2,"TP5a":-13.2,"TP6d":-12.7,"TP6a":-12.7,"TQ1":-14.2,"TQ2":-14.5,
                     "TQ3":-18.7,"TQ4":-16.3,"TQ5":-17.0,"TD":-36.8
                     }
    return delta_f_types


def sort_nested(lst):
    """Sort a nested list of lists."""
    return sorted([sorted(sublist) for sublist in lst])


def flat_set(lst):
    """Flatten a list of lists into a set of unique elements."""
    if not lst:
        return set()
    aset = set(item for sublist in lst for item in sublist) 
    # alist = sorted(aset)
    return aset
