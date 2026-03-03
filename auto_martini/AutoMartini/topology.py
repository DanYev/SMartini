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
from dataclasses import dataclass, field
from pathlib import Path
from sys import exit

from AutoMartini._version import __version__

from .common import *

logger = logging.getLogger(__name__)


@dataclass
class Topology:
    """Container for coarse-grained molecular topology data.
    
    This class holds all topology information in structured form,
    separating data from formatting logic.
    """
    # Header information
    molname: str = ""
    mol_smi: str = ""
    
    # Atoms data: list of dicts with keys: id, type, resnr, residue, atom, cgnr, charge, mass, smiles, atoms_in_smi, logp_origin
    atoms: list = field(default_factory=list)
    partitioning: dict = field(default_factory=dict)
    atomnames: list = field(default_factory=list)
    beadtypes: list = field(default_factory=list)
    atoms_in_smi_dict: dict = field(default_factory=dict)
    
    # Bonds data: list of [i, j, funct, dist, k]
    bonds: list = field(default_factory=list)
    
    # Constraints data: list of [i, j, funct, dist]
    constraints: list = field(default_factory=list)
    
    # Angles data: list of [i, j, k, funct, angle, force_const]
    angles: list = field(default_factory=list)
    
    # Dihedrals data: list of [i, j, k, l, funct, angle, force_const, multiplicity]
    dihedrals: list = field(default_factory=list)
    
    # Virtual sites (if any)
    # Stored as a dict of Gromacs virtual-site section -> list of entries.
    # Supported keys: virtual_sites2, virtual_sites3, virtual_sites4, virtual_sitesn
    virtual_sites: dict[str, list] = field(default_factory=dict)
    
    # Rigid dihedrals (for virtual sites)
    rigid_dihedrals: list = field(default_factory=list)
    
    # Metadata
    nrexcl: int = 2
    
    # Store context for formatting (needed by to_itp)
    num_ar: int = 0
    ringatoms: list = field(default_factory=list)
    cgbeads: list = field(default_factory=list)
    ringbeads: list = field(default_factory=list)
    cgbead_coords: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Exclusions data: list of [i, j] pairs
    exclusions: list = field(default_factory=list)
    
    # Build methods - update topology data
    def build_atoms(self, cgbeads, cgbead_coords, forcepred, molecule, hbonda, hbondd, mapping, partitioning, 
                    ringatoms, logp_file):
        """Build atoms data structure."""
        logger.debug("Entering Topology.build_atoms()")
        
        self.mapping = mapping
        self.partitioning = partitioning
        self.cgbeads = cgbeads
        self.cgbead_coords = cgbead_coords
        self.ringatoms = ringatoms
        
        # Identify which beads are in rings and store them in self.ringbeads
        for ring in ringatoms:
            bead_ring = []
            for atom_idx in ring:
                for bead_idx, atom_indices in mapping.items():
                    if atom_idx in atom_indices:
                        bead_ring.append(bead_idx)
            self.ringbeads.append(sorted(list(set(bead_ring))))
        
        # Create global atom name map (PDB-style: N1, C1, C2, etc.)
        atom_name_map = {}
        atom_type_counter = {}
        for at_idx in range(molecule.GetNumAtoms()):
            at_symbol = molecule.GetAtomWithIdx(at_idx).GetSymbol()
            if at_symbol not in atom_type_counter:
                atom_type_counter[at_symbol] = 0
            atom_type_counter[at_symbol] += 1
            atom_name_map[at_idx] = f"{at_symbol}{atom_type_counter[at_symbol]}"
        
        for bead in range(len(cgbeads)):
            try:
                smi_frag, wc_log_p, charge, atoms_in_smi, converted_smi, real_smi = substruct2smi(
                    molecule, partitioning, bead, atom_name_map
                )
            except Exception:
                raise
            self.atoms_in_smi_dict[bead + 1] = atoms_in_smi.replace(" ; atoms: ", "")

            atom_name = ""
            for character, count in sorted(six.iteritems(letter_occurrences(smi_frag))):
                try:
                    float(character)
                except ValueError:
                    if count == 1:
                        atom_name += "{:s}".format(character)
                    else:
                        atom_name += "{:s}{:s}".format(character, str(count))
            
            mol_frag, errval = gen_molecule_smi(smi_frag)
            charge_frag = get_charge(mol_frag)

            try:
                if charge_frag == 0:
                    alogps, logp_origin = smi2alogps(forcepred, smi_frag, wc_log_p, bead + 1, converted_smi, real_smi, logp_file)
                else:
                    alogps = 0.0
                    logp_origin = "; Charged fragment"
            except (NameError, TypeError, ValueError):
                return

            hbond_a_flag = sum(1 for at in hbonda if partitioning[at] == bead)
            hbond_d_flag = sum(1 for at in hbondd if partitioning[at] == bead)

            ringbeads_flat = [bead for ring in self.ringbeads for bead in ring]
            in_ring = bead in ringbeads_flat

            bead_type = determine_bead_type(alogps, charge, hbond_a_flag, hbond_d_flag, in_ring, smi_frag)
            atom_name = ""
            name_index = bead + 1
            atom_name = "{:1s}{:02d}".format(bead_type[1], name_index)
            self.atomnames.append(atom_name)
            
            mass = get_standard_mass(bead_type)

            atom_dict = {
                'id': bead + 1,
                'type': bead_type,
                'resnr': 1,
                'residue': self.molname[:4] if len(self.molname) > 4 else self.molname,
                'atom': atom_name,
                'cgnr': bead + 1,
                'charge': charge,
                'mass': mass,
                'smiles': smi_frag,
                'atoms_in_smi': atoms_in_smi,
                'logp_origin': logp_origin
            }
            self.atoms.append(atom_dict)
            self.beadtypes.append(bead_type)
    
    def build_bonds(self, ha_neighbors):
        """Build bonds and constraints data."""
        logger.info("Building bonds and constraints...")

        mapping = self.mapping
        cgbead_coords = self.cgbead_coords
        nbeads = len(self.cgbeads)
        if nbeads <= 1:
            return
        
        # First make list of bonds based on connectivity of the original molecule
        for i in range(nbeads):
            for j in range(i + 1, nbeads):
                found_connection = False
                for at_i in mapping[i]:
                    for at_j in mapping[j]:
                        if at_j in ha_neighbors[at_i]:
                            dist = np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                            self.constraints.append([i, j, 1, dist, ""])
                            found_connection = True
                            break
                    if found_connection:
                        break

        # Filter based on distance and add constraints for beads in the same ring
        for bond in self.constraints:
            dist = bond[3]
            if dist > 0.54:
                self.constraints.remove(bond)
            if dist < 0.134:
                raise NameError("Bond too short")

        # If we have 4 bonds corrected in a ring, we can add a constraint between 
        # the two non-bonded beads in the ring with the shortest distance. 
        # This is to help maintain ring structure during simulations.
        for ring in self.ringbeads:
            if len(ring) == 4:
                # Add constraint between the two non-bonded beads in the ring with the shortest distance
                min_dist = float('inf')
                min_pair = None
                n_bonds = 0
                for i in range(len(ring)):
                    for j in range(i + 1, len(ring)):
                        pair = (ring[i], ring[j])
                        if any((pair[0] == b[0] and pair[1] == b[1]) or (pair[0] == b[1] and pair[1] == b[0]) for b in self.constraints):
                            n_bonds += 1
                            continue
                        dist = np.linalg.norm(cgbead_coords[pair[0]] - cgbead_coords[pair[1]]) * 0.1
                        if dist < min_dist:
                            min_dist = dist
                            min_pair = pair
                if n_bonds == 4: 
                    self.constraints.append([min_pair[0], min_pair[1], 1, min_dist, "ring_diagonal"])
              
    def build_angles(self):
        """Build angles data structure."""
        logger.info("Building angles...")

        bondlist = self.bonds + self.constraints
        partitioning = self.partitioning
        cgbead_coords = self.cgbead_coords

        nbeads = len(self.cgbead_coords)
        if nbeads <= 2:
            return

        for i in range(nbeads):
            for j in range(nbeads): 
                for k in range(nbeads):     

                    # Check if all indices are different
                    if i == j or j == k or i == k:
                        continue

                    # Check if angle already exists
                    stop_iteration = False
                    for a in self.angles:
                        it, jt, kt = a[0], a[1], a[2]
                        if i == kt and j == jt and k == it:
                            stop_iteration = True
                            break
                    if stop_iteration:
                        continue

                    # Check if all of them are in one ring
                    for ring in self.ringbeads:
                        if i in ring and j in ring and k in ring:
                            stop_iteration = True
                            break
                    if stop_iteration:
                        continue

                    # Check if all are bonded
                    ij_bonded = False
                    jk_bonded = False
                    ik_bonded = False
                    ij_ring_diagonal = False
                    jk_ring_diagonal = False
                    ik_ring_diagonal = False
                    for b in bondlist:
                        connectivity = b[:2]
                        is_ring_diag = b[-1] == "ring_diagonal"
                        if i in connectivity and j in connectivity:
                            ij_bonded = True
                            if is_ring_diag:
                                ij_ring_diagonal = True
                        if j in connectivity and k in connectivity:
                            jk_bonded = True
                            if is_ring_diag:
                                jk_ring_diagonal = True
                        if i in connectivity and k in connectivity:
                            ik_bonded = True
                            if is_ring_diag:
                                ik_ring_diagonal = True

                    # Skip angles involving ring_diagonal constraints
                    if ij_ring_diagonal or jk_ring_diagonal or ik_ring_diagonal:
                        continue
                    # If all three are bonded, skip. If only ij and jk are bonded, keep.
                    if ij_bonded and jk_bonded and ik_bonded:
                        continue
                    # Skip if they do not form a chain (i-j-k or k-j-i)
                    if not (ij_bonded and jk_bonded):
                        continue

                    # Measure angle between i, j, and k.
                    angle = (
                        180.0
                        / math.pi
                        * math.acos(
                            np.dot(
                                cgbead_coords[i] - cgbead_coords[j],
                                cgbead_coords[k] - cgbead_coords[j],
                            )
                            / (
                                np.linalg.norm(cgbead_coords[i] - cgbead_coords[j])
                                * np.linalg.norm(cgbead_coords[k] - cgbead_coords[j])
                            )
                        )
                    )
                    
                    funct = 1
                    force_const = 250.0
                    self.angles.append([i, j, k, funct, angle, force_const, ""])
    
    def build_dihedrals(self):
        """Build dihedrals data structure and return num_ar."""
        logger.info("Building dihedrals...")

        def _is_ring_diagonal(entry) -> bool:
            """Return True if a bond/constraint entry is tagged as ring_diagonal."""
            # Constraints are typically [i, j, funct, dist, comment]
            # Bonds are typically [i, j, funct, dist, k, comment]
            for val in reversed(entry):
                if isinstance(val, str) and "ring_diagonal" in val:
                    return True
            return False

        cgbead_coords = self.cgbead_coords
        bondlist = self.bonds + self.constraints
        self.dihedrals = []

        nbeads = len(cgbead_coords)
        if nbeads <= 3:
            return 

        # Dihedrals
        for i in range(nbeads):
            for j in range(nbeads):
                for k in range(nbeads):
                    for l in range(nbeads):

                        # Check if all indices are different
                        if  i == j or i == k or i == l or j == k or j == l or k == l:
                            continue

                        # Check if dihedral already exists (in either direction)
                        stop_iteration = False
                        for dih in self.dihedrals:
                            if dih[0] == l and dih[1] == k and dih[2] == j and dih[3] == i:
                                stop_iteration = True
                                break
                        if stop_iteration:
                            continue

                        # Check if all are bonded (for proper dihedral chains)
                        ij_bonded = False
                        jk_bonded = False
                        kl_bonded = False
                        ik_bonded = False
                        jl_bonded = False
                        il_bonded = False
                        ij_ring_diagonal = False
                        kl_ring_diagonal = False
                        for b in bondlist:
                            connectivity = b[:2]
                            if i in connectivity and j in connectivity:
                                ij_bonded = True
                                if _is_ring_diagonal(b):
                                    ij_ring_diagonal = True
                            if j in connectivity and k in connectivity:
                                jk_bonded = True
                            if k in connectivity and l in connectivity:
                                kl_bonded = True
                                if _is_ring_diagonal(b):
                                    kl_ring_diagonal = True
                            if i in connectivity and k in connectivity:
                                ik_bonded = True
                            if j in connectivity and l in connectivity:
                                jl_bonded = True
                            if i in connectivity and l in connectivity:
                                il_bonded = True

                        # Exclude dihedrals whose terminal bond is a ring diagonal.
                        # (These constraints are added to stabilize rings and
                        # should not define torsional terms.)
                        if ij_ring_diagonal or kl_ring_diagonal:
                            continue
                        
                        # # Skip if any shortcut bonds exist (not a proper dihedral chain)
                        if il_bonded:
                            continue

                        # Skip if they do not form a chain (i-j-k-l)
                        if not (ij_bonded and jk_bonded and kl_bonded):
                            continue

                        # Measure dihedral angle between i, j, k, and l.    
                        r1 = cgbead_coords[j] - cgbead_coords[i]
                        r2 = cgbead_coords[k] - cgbead_coords[j]
                        r3 = cgbead_coords[l] - cgbead_coords[k]
                        p1 = np.cross(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                        p2 = np.cross(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
                        r2 /= np.linalg.norm(r2)
                        cosphi = np.dot(p1, p2)
                        sinphi = np.dot(r2, np.cross(p1, p2))
                        angle = 180.0 / math.pi * np.arctan2(sinphi, cosphi)                       
            
                        forc_const = 10.0
                        angle = (180 + angle) % 360
                        multiplicity = 1  # Default multiplicity
                        self.dihedrals.append([i, j, k, l, 9, angle, forc_const, multiplicity, ""])


    def build_virtual_sites(self):
        self.virtual_sites = self._init_virtual_sites_dict()
        self.virtual_sites["virtual_sites2"] = self.build_vs_2()
        self.virtual_sites["virtual_sites3"] = self.build_vs_3()
        self.virtual_sites["virtual_sites4"] = self.build_vs_4()
        self.virtual_sites["virtual_sitesn"] = self.build_vs_n()

    @staticmethod
    def _init_virtual_sites_dict() -> dict[str, list]:
        return {
            "virtual_sites2": [],
            "virtual_sites3": [],
            "virtual_sites4": [],
            "virtual_sitesn": [],
        }

    def build_vs_3(self):
        """Build type 3 virtual sites for fused rings.
        For rings of 5 or 6 beads, we select 3 beads as anchors to be connected with bonds or constaints, 
        the rest of the beads will go the type 3 virtual sites.
        """
        logger.info("Building virtual sites for fused rings...")
        atoms = self.atoms
        bonds = self.constraints
        vsites = []

        def _has_external_bond(bead, ring):
            for b in bonds:
                if bead in b[:2]:
                    other_bead = b[1] if b[0] == bead else b[0]
                    if other_bead not in ring:
                        return True
            return False

        def _find_anchors(ring): 
            anchors = [] # beads that are bonded to beads in the ring but are not in the ring themselves
            for bead in ring:
                if _has_external_bond(bead, ring):
                    anchors.append(bead)
            if not anchors:
                anchors = [ring[0]] # if no anchor beads, just pick one bead in the ring to be the anchor

            if len(anchors) < 3:
                # if we have less than 3 then the furthest bead in the ring from the anchor beads will be added as an anchor until we have 3
                while len(anchors) < 3:
                    max_dist = -1
                    furthest_bead = None
                    for bead in ring:
                        if bead in anchors:
                            continue
                        dist = min(np.linalg.norm(self.cgbead_coords[bead] - self.cgbead_coords[anchor]) for anchor in anchors)
                        if dist > max_dist:
                            max_dist = dist
                            furthest_bead = bead
                    anchors.append(furthest_bead)                
            return sorted(anchors)

        def _make_vs3_fad_entry(site: int, i: int, j: int, k: int) -> dict:
            """Create a `virtual_sites3` funct=3 (3fad) entry.

            Parameters are derived from the current `cgbead_coords` so that the
            virtual site reproduces the present position of `site` from (i, j, k).

            Returns a dict compatible with `format_virtual_sites()`.
            """
            ri = np.asarray(self.cgbead_coords[i], dtype=float)
            rj = np.asarray(self.cgbead_coords[j], dtype=float)
            rk = np.asarray(self.cgbead_coords[k], dtype=float)
            rs = np.asarray(self.cgbead_coords[site], dtype=float)

            rij = rj - ri
            n_rij = np.linalg.norm(rij)

            rjk = rk - rj
            denom = np.dot(rij, rij)
 
            r_perp = rjk - np.dot(rij, rjk) / denom * rij
            n_rperp = np.linalg.norm(r_perp)

            e1 = rij / n_rij
            e2 = r_perp / n_rperp

            v = rs - ri
            v1 = np.dot(v, e1)
            v2 = np.dot(v, e2)

            d_angstrom = np.sqrt(v1 * v1 + v2 * v2)
            theta_deg = np.degrees(np.atan2(v2, v1))
            d_nm = d_angstrom * 0.1

            return [site, i, j, k, 3, theta_deg, d_nm]
        
        def _sanitize_atoms(atoms, vsites, anchors):
            # vsites must have 0 mass
            vsites_ids = [vs[0]+1 for vs in vsites]
            anchors_ids = [a+1 for a in anchors]
            factor = len(anchors + vsites) / len(anchors)
            for atom in atoms:
                if atom['id'] in vsites_ids:  # atom ids are 1-indexed
                    atom['mass'] = int(0)
                if atom['id'] in anchors_ids:  # atom ids are 1-indexed
                    atom['mass'] = int(atom['mass'] * factor)  # double the mass of the anchor beads to help stabilize the ring structure during simulations

        def _sanitize_bonds(bonds, ring, anchors):
            # Remove any within the ring and make bonds involving the anchor beads 
            for bead_i in ring:
                for bead_j in ring:
                    if bead_i == bead_j:
                        continue
                    for bond in bonds:
                        if bead_i in bond[:2] and bead_j in bond[:2]:
                            bonds.remove(bond)
                            logger.info(f"Removed bond between {bead_i} and {bead_j} in ring {ring}")
            for i in range(len(anchors)):
                anchor = anchors[i]
                bead = anchors[(i + 1) % len(anchors)]
                dist = np.linalg.norm(self.cgbead_coords[anchor] - self.cgbead_coords[bead]) * 0.1
                bond_entry = [anchor, bead, 1, dist, "ring_anchor"]
                bonds.append(bond_entry)
                logger.info(f"Added bond between {anchor} and {bead} in ring {ring}")
        
        # For now check if we have rings with 5+ beads 
        for ring in self.ringbeads:
            if len(ring) < 5 or len(ring) > 6:
                continue
            anchors = _find_anchors(ring)
            i, j, k = anchors[0], anchors[1], anchors[2]
            for bead in ring:
                if bead in anchors:
                    continue
                vs3_entry = _make_vs3_fad_entry(bead, i, j, k)
                vsites.append(vs3_entry)
            _sanitize_bonds(bonds, ring, anchors)

        if vsites and anchors:
            _sanitize_atoms(atoms, vsites, anchors)

        return vsites

    def build_vs_2(self) -> list:
        return []

    def build_vs_4(self) -> list:
        return []

    def build_vs_n(self) -> list:
        return []

    def build_exclusions(self):
        logger.info("Building exclusions")
        bonds = self.bonds + self.constraints
        conn = [(b[:2]) for b in bonds]

        # just duplicate defauls nrexcl exclusions in case it changes later
        # e.g merging with a protein or smth
        def _remove_duplicates(lst):
            seen = []
            tmp = lst.copy()
            for item in tmp:
                if item not in seen:
                    seen.append(item)
                else:
                    for i in range(lst.count(item)):
                        lst.remove(item)

        ext_conn = [b for b in conn]
        init_ext_conn = ext_conn.copy()
        for eb in init_ext_conn:
            for n in range(1, self.nrexcl):
                for b in conn:
                    if b == eb:
                        continue
                    x = eb + b
                    _remove_duplicates(x)
                    if len(x) != 2:
                        continue
                    if x not in ext_conn:
                        ext_conn.append(sorted(x))
        for b in ext_conn:
            self.exclusions.append(b)

        # For all the beads within a ring add all of them to the exclusion list with each other
        for ring in self.ringbeads:
            for i in ring:
                for j in ring:
                    if i >= j:
                        continue
                    self.exclusions.append([i, j])

        self.exclusions = [(b[0], b[1]) for b in self.exclusions]
        self.exclusions = list(set(self.exclusions))
        self.exclusions.sort()

####################################################################################################
### TOPOLOGY ITP FORMATTING METHODS
####################################################################################################
    
    def format_header(self):
        """Format Topology header section."""
        text = "; GENERATED WITH Auto_Martini for {}\n".format(self.molname)
        info = (
            "; Developed by: Kiran Kanekal, Tristan Bereau, and Andrew Abi-Mansour\n"
            + "; updated to Martini 3 force field by Magdalena Szczuka\n"
            + "; supervised by Matthieu Chavent, Pierre Poulain and Paulo C. T. Souza \n"
            + "; SMILES code : " + self.mol_smi + "\n"
            + "; Partitioning: " + str(self.partitioning) + "\n"
            + "; Ringbeads: " + str(self.ringbeads) + "\n"
            + "\n"
            + "[moleculetype]\n"
            + "; molname       nrexcl\n"
            + "  {:5s}         {:d}\n\n".format(self.molname, self.nrexcl)
            + "[atoms]\n"
            + "; id      type   resnr residue atom    cgnr    charge  mass ;  smiles    ; atom_num"
        )
        return text + info
    
    def format_atoms(self):
        """Format atoms list into ITP text."""
        text = ""
        for atom in self.atoms:
            text += (
                "   {:<5d}   {:5s}   {:d}   {:5s}   {:7s}   {:<5d}   {:2d}   {:3d}   ;   {:8s}{:8s}{:9s}\n".format(
                    atom['id'], atom['type'], atom['resnr'], atom['residue'], atom['atom'],
                    atom['cgnr'], atom['charge'], atom['mass'], atom['smiles'],
                    atom['atoms_in_smi'], atom['logp_origin']
                )
            )
        return text
    
    def format_bonds(self):
        """Format bonds and constraints into ITP text."""
        text = "\n[bonds]\n" + ";  i  j     funct   length   force.c.\n"
        for b in self.bonds:
            # Bond data is [i, j, funct, dist, k, comment]
            comment = f" ; {b[5]}" if len(b) >= 6 and b[5] else ""
            text = text + "  {:2} {:2}       {:2}      {:<5.3f}       {:4.1f}{}\n".format(
                b[0] + 1, b[1] + 1, b[2], b[3], b[4], comment,
            )

        text = text + "\n\n[constraints]\n" + ";  i   j     funct   length\n"
        for c in self.constraints:
                # Constraint data is [i, j, funct, dist, comment]
                comment = f" ; {c[4]}" if len(c) >= 5 and c[4] else ""
                text = text + "  {:2} {:2}       {:2}      {:<5.3f}{}\n".format(
                    c[0] + 1, c[1] + 1, c[2], c[3], comment
                )
        
        return text
    
    def format_angles(self):
        """Format angles into ITP text."""
        text = ""
        if len(self.angles) > 0:
            text = text + "\n[angles]\n"
            text = text + ";  i  j  k    funct  angle  force.c.\n"
            for a in self.angles:
                # Angle data is [i, j, k, funct, angle, force_const, comment]
                comment = f" ; {a[6]}" if len(a) >= 7 and a[6] else ""
                text = text + "  {:2} {:2} {:2}       {:2}    {:<5.1f}  {:5.1f}{}\n".format(
                    a[0] + 1, a[1] + 1, a[2] + 1, a[3], a[4], a[5], comment
                )
        return text
    
    def format_dihedrals(self):
        """Format dihedrals into ITP text."""
        text = ""
        if len(self.dihedrals) > 0:
            text = text + "\n[dihedrals]\n"
            text = text + ";  i  j  k  l  funct  parameters...\n"
            for d in self.dihedrals:
                # Supported storage forms:
                # - funct 1/9/etc: [i, j, k, l, funct, angle, force_const, multiplicity, comment]
                # - funct 11 (CBT): [i, j, k, l, 11, kphi, a0, a1, a2, a3, a4, comment]
                funct = int(d[4])
                comment = ""
                if len(d) > 0 and isinstance(d[-1], str) and d[-1]:
                    comment = f" ; {d[-1]}"

                if funct == 11:
                    if len(d) < 11:
                        raise ValueError(f"Invalid funct=11 dihedral entry (too short): {d}")
                    kphi = float(d[5])
                    a0, a1, a2, a3, a4 = (float(d[6]), float(d[7]), float(d[8]), float(d[9]), float(d[10]))
                    text += (
                        "  {:2} {:2} {:2} {:2}    11   {: .6g}  {: .6g}  {: .6g}  {: .6g}  {: .6g}  {: .6g}{}\n".format(
                            d[0] + 1,
                            d[1] + 1,
                            d[2] + 1,
                            d[3] + 1,
                            kphi,
                            a0,
                            a1,
                            a2,
                            a3,
                            a4,
                            comment,
                        )
                    )
                else:
                    if len(d) < 8:
                        raise ValueError(f"Invalid dihedral entry (too short): {d}")
                    angle = float(d[5])
                    force_const = float(d[6])
                    multiplicity = int(d[7])
                    text += (
                        "  {:2} {:2} {:2} {:2}    {:1}    {:<5.1f}  {:5.1f}     {:1}{}\n".format(
                            d[0] + 1,
                            d[1] + 1,
                            d[2] + 1,
                            d[3] + 1,
                            funct,
                            angle,
                            force_const,
                            multiplicity,
                            comment,
                        )
                    )
        return text
    
    def format_virtual_sites(self):
        """Format all virtual-site sections for .itp output."""
        if not self.virtual_sites:
            return ""

        text = ""
        text += self.format_virtual_sites2()
        text += self.format_virtual_sites3()
        text += self.format_virtual_sites4()
        text += self.format_virtual_sitesn()
        return text

    def format_virtual_sites2(self):
        """Format the `[virtual_sites2]` section."""
        entries = (self.virtual_sites or {}).get("virtual_sites2", [])
        if not entries:
            return ""
        lines = [
            "",
            "[virtual_sites2]",
            "; site  i  j  funct  params...",
        ]
        for entry in entries:
            # Expected list format (0-based indices): [site, i, j, funct, <float params...>, <optional comment str>]
            comment = ""
            if len(entry) > 0 and isinstance(entry[-1], str) and entry[-1]:
                comment = entry[-1]
                entry = entry[:-1]

            if len(entry) < 4:
                continue

            site = int(entry[0]) + 1
            i = int(entry[1]) + 1
            j = int(entry[2]) + 1
            funct = int(entry[3])
            params = entry[4:]
            formatted_params = []
            for p in params:
                try:
                    formatted_params.append(f"{float(p):.3f}")
                except (TypeError, ValueError):
                    formatted_params.append(str(p))

            fields = [str(site), str(i), str(j), str(funct), *formatted_params]
            line = "  " + " ".join(fields).rstrip()
            if comment:
                line += f" ; {comment}"
            lines.append(line)
        return "\n".join(lines) + "\n"

    def format_virtual_sites3(self):
        """Format the `[virtual_sites3]` section."""
        entries = (self.virtual_sites or {}).get("virtual_sites3", [])
        if not entries:
            return ""
        lines = [
            "",
            "[virtual_sites3]",
            "; site  i  j  k  funct  params...",
        ]
        for entry in entries:
            # Expected list format (0-based indices): [site, i, j, k, funct, <float params...>, <optional comment str>]
            comment = ""
            if len(entry) > 0 and isinstance(entry[-1], str) and entry[-1]:
                comment = entry[-1]
                entry = entry[:-1]

            if len(entry) < 5:
                continue

            site = int(entry[0]) + 1
            i = int(entry[1]) + 1
            j = int(entry[2]) + 1
            k = int(entry[3]) + 1
            funct = int(entry[4])
            params = entry[5:]
            formatted_params = []
            for p in params:
                try:
                    formatted_params.append(f"{float(p):.3f}")
                except (TypeError, ValueError):
                    formatted_params.append(str(p))

            fields = [str(site), str(i), str(j), str(k), str(funct), *formatted_params]
            line = "  " + " ".join(fields).rstrip()
            if comment:
                line += f" ; {comment}"
            lines.append(line)
        return "\n".join(lines) + "\n"

    def format_virtual_sites4(self):
        """Format the `[virtual_sites4]` section."""
        entries = (self.virtual_sites or {}).get("virtual_sites4", [])
        if not entries:
            return ""
        lines = [
            "",
            "[virtual_sites4]",
            "; site  i  j  k  l  funct  params...",
        ]
        for entry in entries:
            # Expected list format (0-based indices): [site, i, j, k, l, funct, <float params...>, <optional comment str>]
            comment = ""
            if len(entry) > 0 and isinstance(entry[-1], str) and entry[-1]:
                comment = entry[-1]
                entry = entry[:-1]

            if len(entry) < 6:
                continue

            site = int(entry[0]) + 1
            i = int(entry[1]) + 1
            j = int(entry[2]) + 1
            k = int(entry[3]) + 1
            l = int(entry[4]) + 1
            funct = int(entry[5])
            params = entry[6:]
            formatted_params = []
            for p in params:
                try:
                    formatted_params.append(f"{float(p):.3f}")
                except (TypeError, ValueError):
                    formatted_params.append(str(p))

            fields = [str(site), str(i), str(j), str(k), str(l), str(funct), *formatted_params]
            line = "  " + " ".join(fields).rstrip()
            if comment:
                line += f" ; {comment}"
            lines.append(line)
        return "\n".join(lines) + "\n"

    def format_virtual_sitesn(self):
        """Format the `[virtual_sitesn]` section."""
        entries = (self.virtual_sites or {}).get("virtual_sitesn", [])
        if not entries:
            return ""
        lines = [
            "",
            "[virtual_sitesn]",
            "; site  funct  constructing atom indices",
        ]
        for entry in entries:
            # Expected list format (0-based indices): [site, funct, <atom indices...>, <optional comment str>]
            comment = ""
            if len(entry) > 0 and isinstance(entry[-1], str) and entry[-1]:
                comment = entry[-1]
                entry = entry[:-1]

            if len(entry) < 2:
                continue

            site = int(entry[0]) + 1
            funct = int(entry[1])

            rest = []
            for x in entry[2:]:
                if isinstance(x, (int, np.integer)):
                    rest.append(str(int(x) + 1))
                    continue
                try:
                    rest.append(f"{float(x):.3f}")
                except (TypeError, ValueError):
                    rest.append(str(x))

            fields = [str(site), str(funct), *rest]
            line = "  " + " ".join(fields).rstrip()
            if comment:
                line += f" ; {comment}"
            lines.append(line)
        return "\n".join(lines) + "\n"
    
    def format_exclusions(self):
        """Format `[exclusions]` section.

        `self.exclusions` is expected to be a list of pairs of 0-based bead indices,
        e.g. `[[0, 3], [1, 4]]`.
        """
        if not self.exclusions:
            return ""

        lines = ["", "[exclusions]"]
        for pair in self.exclusions:
            if pair is None or len(pair) < 2:
                continue
            i = int(pair[0]) + 1
            j = int(pair[1]) + 1
            lines.append(f"  {i} {j}")

        # If everything was malformed and we emitted no pairs, emit nothing.
        if len(lines) <= 2:
            return ""
        return "\n".join(lines) + "\n"
    
    def format_position_restraints(
        self,
        force_constant: str = "POSRES_FC",
        funct: int = 1,
        ifdef: str = "POSRES",
        include_end_if: bool = True,
    ):
        """Format position restraints section for all atoms.
        
        Parameters
        ----------
        force_constant : str
            Force constant label written for x/y/z (default: POSRES_FC).
        funct : int
            Gromacs function type (default: 1).
        ifdef : str
            Preprocessor symbol used for conditional inclusion.
        include_end_if : bool
            Whether to append a matching #endif line.
        """
        if not self.atoms:
            return ""
        
        lines = [
            "",
            "#ifndef POSRES_FC",
            "#define POSRES_FC 1000.0",
            "#endif",
            "[ position_restraints ]",
            f"#ifdef {ifdef}",
        ]
        
        for atom in self.atoms:
            atom_id = atom['id']
            lines.append(
                f"{atom_id:5d} {funct:d} {force_constant} {force_constant} {force_constant}"
            )
        
        if include_end_if:
            lines.append("#endif")
        
        return "\n".join(lines) + "\n"
    
    def to_itp(self, out_file=None, write_exclusions=True, write_posres=True):
        """Generate complete ITP file content.
        
        Parameters
        ----------
        trial : bool
            If True, skip some sections (for trial runs)
        write_exclusions : bool
            If True, include exclusions section
        write_posres : bool
            If True, include position restraints section
        """
        text = self.format_header() + "\n"
        text += self.format_atoms() + "\n"
        text += self.format_bonds() + "\n"
        text += self.format_angles() + "\n"
        text += self.format_dihedrals() + "\n"
        text += self.format_virtual_sites() + "\n"
        if write_exclusions:
            text += self.format_exclusions()
        if write_posres:
            text += self.format_position_restraints()
        if out_file:
            with open(out_file, 'w') as f:
                f.write(text)
        return text


def read_itp(itp_file):
    """Read an ITP file and create a Topology object.
    
    Parameters
    ----------
    itp_file : str
        Path to the ITP file
        
    Returns
    -------
    Topology
        Topology object populated with data from ITP file
    """
    
    itp_path = Path(itp_file)
    if not itp_path.exists():
        raise FileNotFoundError(f"ITP file not found: {itp_file}")
    
    content = itp_path.read_text()
    lines = content.splitlines()
    
    # Initialize topology object
    topo = Topology()
    
    # Track current section
    current_section = None
    
    # Parse line by line
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        # Check for section headers
        if stripped.startswith('[') and stripped.endswith(']'):
            current_section = stripped[1:-1].strip().lower()
            continue
        
        # Skip comments (except those with info we need)
        if stripped.startswith(';'):
            # Extract SMILES from header comment
            if 'SMILES code :' in line:
                topo.mol_smi = line.split('SMILES code :')[1].strip()
            # Extract Partitioning from header comment
            if 'Partitioning:' in line:
                import ast
                partitioning_str = line.split('Partitioning:')[1].strip()
                try:
                    topo.partitioning = ast.literal_eval(partitioning_str)
                except (ValueError, SyntaxError):
                    logger.warning(f"Could not parse partitioning from ITP: {partitioning_str}")
            # Extract Ringbeads from header comment
            if 'Ringbeads:' in line:
                import ast
                ringbeads_str = line.split('Ringbeads:')[1].strip()
                try:
                    topo.ringbeads = ast.literal_eval(ringbeads_str)
                except (ValueError, SyntaxError):
                    logger.warning(f"Could not parse ringbeads from ITP: {ringbeads_str}")
            continue
        
        # Parse based on current section
        if current_section == 'moleculetype':
            parts = stripped.split()
            if len(parts) >= 2:
                topo.molname = parts[0]
                topo.nrexcl = int(parts[1])
        
        elif current_section == 'atoms':
            # Parse atom line with comment
            # Format: id type resnr residue atom cgnr charge mass ; smiles ; atoms: N1, C1, ...
            pre_comment, _, comment = line.partition(';')
            parts = pre_comment.split()
            
            if len(parts) >= 8:
                atom_id = int(parts[0])
                atom_type = parts[1]
                resnr = int(parts[2])
                residue = parts[3]
                atom_name = parts[4]
                cgnr = int(parts[5])
                charge = int(parts[6])
                mass = int(parts[7])
                
                # Extract smiles and atom info from comment
                smiles_part = ""
                atoms_in_smi = ""
                logp_origin = ""
                
                if comment:
                    # Split multiple comment sections
                    comment_parts = comment.split(';')
                    if len(comment_parts) >= 1:
                        smiles_part = comment_parts[0].strip()
                    if len(comment_parts) >= 2:
                        atoms_str = comment_parts[1].strip()
                        if atoms_str.startswith('atoms:'):
                            atoms_in_smi = '; ' + atoms_str
                    if len(comment_parts) >= 3:
                        logp_origin = '; ' + comment_parts[2].strip()
                
                atom_dict = {
                    'id': atom_id,
                    'type': atom_type,
                    'resnr': resnr,
                    'residue': residue,
                    'atom': atom_name,
                    'cgnr': cgnr,
                    'charge': charge,
                    'mass': mass,
                    'smiles': smiles_part,
                    'atoms_in_smi': atoms_in_smi,
                    'logp_origin': logp_origin
                }
                
                topo.atoms.append(atom_dict)
                topo.atomnames.append(atom_name)
                topo.beadtypes.append(atom_type)
                
                # Extract atom labels for partitioning
                if atoms_in_smi:
                    atoms_label_str = atoms_in_smi.replace('; atoms:', '').strip()
                    atom_labels = [a.strip().rstrip(',') for a in atoms_label_str.split(',') if a.strip()]
                    topo.atoms_in_smi_dict[atom_id] = ', '.join(atom_labels)
        
        elif current_section == 'bonds':
            # Parse data and comment separately
            pre_comment, _, comment_text = line.partition(';')
            parts = pre_comment.split()
            if len(parts) >= 4 and parts[0].isdigit():
                i = int(parts[0]) - 1  # Convert to 0-based
                j = int(parts[1]) - 1
                funct = int(parts[2])
                length = float(parts[3])
                force_const = float(parts[4]) if len(parts) >= 5 else 10000
                comment = comment_text.strip() if comment_text else ""
                # Store as [i, j, funct, length, force_const, comment]
                topo.bonds.append([i, j, funct, length, force_const, comment])
        
        elif current_section == 'constraints':
            # Parse data and comment separately
            pre_comment, _, comment_text = line.partition(';')
            parts = pre_comment.split()
            if len(parts) >= 4 and parts[0].isdigit():
                i = int(parts[0]) - 1  # Convert to 0-based
                j = int(parts[1]) - 1
                funct = int(parts[2])
                length = float(parts[3])
                comment = comment_text.strip() if comment_text else ""
                # Store as [i, j, funct, length, comment]
                topo.constraints.append([i, j, funct, length, comment])

        elif current_section == 'exclusions':
            # Gromacs allows either pair-wise lines (i j) or multi-column (i j k ...)
            pre_comment, _, _comment_text = line.partition(';')
            parts = pre_comment.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue

            i = int(parts[0]) - 1
            # Deduplicate while preserving file order
            if not hasattr(topo, "_exclusions_seen"):
                topo._exclusions_seen = set()

            for pj in parts[1:]:
                if not pj.isdigit():
                    continue
                j = int(pj) - 1
                if i == j:
                    continue
                key = (i, j)
                if key in topo._exclusions_seen:
                    continue
                topo._exclusions_seen.add(key)
                topo.exclusions.append([i, j])
        
        elif current_section == 'angles':
            # Parse data and comment separately
            pre_comment, _, comment_text = line.partition(';')
            parts = pre_comment.split()
            if len(parts) >= 5 and parts[0].isdigit():
                i = int(parts[0]) - 1  # Convert to 0-based
                j = int(parts[1]) - 1
                k = int(parts[2]) - 1
                funct = int(parts[3])
                angle = float(parts[4])
                force_const = float(parts[5]) if len(parts) >= 6 else 0.0
                comment = comment_text.strip() if comment_text else ""
                # Store as [i, j, k, funct, angle, force_const, comment]
                topo.angles.append([i, j, k, funct, angle, force_const, comment])
        
        elif current_section == 'dihedrals':
            # Parse data and comment separately
            pre_comment, _, comment_text = line.partition(';')
            parts = pre_comment.split()
            if len(parts) >= 5 and parts[0].isdigit():
                i = int(parts[0]) - 1  # Convert to 0-based
                j = int(parts[1]) - 1
                k = int(parts[2]) - 1
                l = int(parts[3]) - 1
                funct = int(parts[4])
                comment = comment_text.strip() if comment_text else ""

                if funct == 11:
                    # funct=11: kphi a0 a1 a2 a3 a4
                    if len(parts) < 11:
                        raise ValueError(f"Invalid funct=11 dihedral line (expected 6 params): {line}")
                    kphi = float(parts[5])
                    a0 = float(parts[6])
                    a1 = float(parts[7])
                    a2 = float(parts[8])
                    a3 = float(parts[9])
                    a4 = float(parts[10])
                    topo.dihedrals.append([i, j, k, l, 11, kphi, a0, a1, a2, a3, a4, comment])
                else:
                    # Default: interpret as (angle, force_const, multiplicity) as used by funct=1/9.
                    # Note: other funct values (e.g. RB) are not explicitly supported here.
                    if len(parts) < 7:
                        raise ValueError(f"Invalid dihedral line (expected >=2 params): {line}")
                    angle = float(parts[5])
                    force_const = float(parts[6])
                    multiplicity = int(parts[7]) if len(parts) >= 8 else 1
                    topo.dihedrals.append([i, j, k, l, funct, angle, force_const, multiplicity, comment])
        
        elif current_section in {'virtual_sites2', 'virtual_sites3', 'virtual_sites4', 'virtual_sitesn'}:
            pre_comment, _, comment_text = line.partition(';')
            parts = pre_comment.split()
            if not parts or not parts[0].isdigit():
                continue

            if not topo.virtual_sites:
                topo.virtual_sites = Topology._init_virtual_sites_dict()
            else:
                for k in ("virtual_sites2", "virtual_sites3", "virtual_sites4", "virtual_sitesn"):
                    topo.virtual_sites.setdefault(k, [])

            comment = comment_text.strip() if comment_text else ""

            if current_section == 'virtual_sitesn':
                if len(parts) < 3:
                    continue
                site = int(parts[0]) - 1
                funct = int(parts[1])
                atoms = [int(p) - 1 for p in parts[2:]]
                entry = [site, funct, *atoms]
                if comment:
                    entry.append(comment)
                topo.virtual_sites['virtual_sitesn'].append(entry)
            else:
                required = {
                    'virtual_sites2': 2,
                    'virtual_sites3': 3,
                    'virtual_sites4': 4,
                }[current_section]

                # Expected order: site i j [k [l]] funct params...
                if len(parts) < 2 + required + 1:
                    continue
                site = int(parts[0]) - 1
                atoms = [int(p) - 1 for p in parts[1:1 + required]]
                funct = int(parts[1 + required])
                raw_params = parts[2 + required:]
                parsed_params = []
                for p in raw_params:
                    try:
                        parsed_params.append(int(p))
                        continue
                    except ValueError:
                        pass
                    try:
                        parsed_params.append(float(p))
                        continue
                    except ValueError:
                        parsed_params.append(p)

                entry = [site, *atoms, funct, *parsed_params]
                if comment:
                    entry.append(comment)
                topo.virtual_sites[current_section].append(entry)
        
    # Clean up any parser-only attributes
    if hasattr(topo, "_exclusions_seen"):
        delattr(topo, "_exclusions_seen")
    return topo


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


def letter_occurrences(string):
    """Count letter occurences"""
    frequencies = defaultdict(lambda: 0)
    for character in string:
        if character.isalnum():
            frequencies[character.upper()] += 1
    return frequencies


def get_charge(molecule):
    """Get net charge of molecule"""
    return Chem.rdmolops.GetFormalCharge(molecule)


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


def read_params(val, size):  # AutoM3 function 
    """Returns the closest force value to given parameter, 
    based on state-of-the-art parametrizations avaliablie in MAD (https://mad.ibcp.fr) """
    bonds = {'S-S': {0.36: 5000.0, 0.378: 5000.0, 0.321: 25000.0, 0.331: 5000.0, 0.3: 5000.0, 0.37: 5000.0, 0.281: 25000.0,
                     0.314: 25000.0, 0.32: 7500.0, 0.38: 5000.0, 0.33: 17000.0, 0.405: 5000.0, 0.395: 5000.0, 0.39: 5000.0,
                     0.385: 5000.0, 0.35: 5000.0, 0.375: 3500.0, 0.376: 7000.0, 0.34: 7000.0},
             'T-T': {0.32: 25000.0, 0.261: 25000.0, 0.376: 25000.0, 0.25: 25000.0, 0.401: 25000.0, 0.449: 100000.0,
                     0.251: 100000.0},
             'T-S': {0.364: 25000.0, 0.408: 25000.0, 0.272: 25000.0, 0.31: 7000.0, 0.3: 5000.0, 0.253: 5000.0, 0.387: 25000.0,
                     0.34: 5000.0, 0.29: 5000.0, 0.32: 5000.0, 0.33: 10000.0, 0.286: 100000.0, 0.371: 100000.0, 0.244: 100000.0,
                     0.355: 5000.0, 0.36: 5000.0},
             'R-T': {0.389: 5000.0},
             'R-R': {0.38: 50000.0, 0.475: 3800.0, 0.47: 3800.0, 0.468: 3800.0, 0.49: 3800.0, 0.46: 7000.0, 0.45: 7000.0,
                     0.455: 7000.0},
             'R-S': {0.385: 7000.0, 0.38: 7000.0, 0.405: 7000.0}
             }
    
    angles = {'T-T-S': {180.0: 250.0, 138.0: 250.0, 71.0: 250.0, 122.0: 50.0},
              'T-S-S': {155.0: 100.0, 148.0: 100.0},
              'T-S-T': {135.0: 30.0},
              'S-S-S': {150.6: 100.0, 130.0: 25.0, 150.0: 100.0, 135.0: 15.0},
              'T-T-R': {160.0: 180.0},
              'R-R-R': {180.0: 35.0, 100.0: 10.0},
              }
    
    dihedrals = {'S-S-S-T': {180.0: 100.0},
                 'S-S-T-T': {180.0: 100.0},
                 'S-T-T-S': {180.0: 75.0, 0.0: 50.0},
                 'S-T-T-T': {180.0: 200.0, 0.0: 100.0},
                 'T-T-T-T': {180.0: 200.0, 0.0: 100},#, 1.01: 1.01, 0.64: 0.605, 0.65: 0.6, -28.0: 200.0, 2.69: 14.12, 0.08: 2.31, 0.52: 0.373},
                 'T-S-S-T': {180.0: 100.0},
                 'R-T-T-T': {180.0: 50.0},
                 'T-T-S-S': {180.0: 50.0},
                 'S-T-S-S': {0.0: 50.0},
                 'T-T-T-S': {180.0: 20.0},
                 'T-R-R-T': {0.0: 1.8},
                 'T-T-S-T': {-45.0: 200.0},
                 'S-S-S-S': {180.0: 1.96, 0.0: 0.18}}
    
    if len(size) == 3: #bonds
        if size not in bonds.keys(): size = size[2]+'-'+size[0]
        for k,v in bonds.items():
            if k == size:
                closest_length = find_closest_key(v, val)
                force = v[closest_length]
                return force

    if len(size) == 5: #angles
        key_exists = False
        if size in angles.keys():
            for k,v in angles.items():
                if k == size:
                    closest_length = find_closest_key(v, val)
                    force = v[closest_length]
                    return force

    if len(size) == 7: #dihedrals
        key_exists = False
        if size in dihedrals.keys():
            for k,v in dihedrals.items():
                if k == size:
                    closest_length = find_closest_key(v, val)
                    force = v[closest_length]
                    return force


def substruct2smi(molecule, partitioning, cg_bead, atom_name_map=None):
    """Substructure to smiles conversion; also output Wildman-Crippen log_p;
    and charge of group."""
    frag = rdchem.EditableMol(molecule)
    # fragment smi: [H]N([H])c1nc(N([H])[H])n([H])n1
    num_atoms = molecule.GetConformer().GetNumAtoms()
    # First delete all hydrogens
    for i in range(num_atoms):
        if molecule.GetAtomWithIdx(i).GetSymbol() == "H":
            # find atom from coordinates
            submol = frag.GetMol()
            for j in range(submol.GetConformer().GetNumAtoms()):
                if (
                    molecule.GetConformer().GetAtomPosition(i)[0]
                    == submol.GetConformer().GetAtomPosition(j)[0]
                ):
                    frag.RemoveAtom(j)
    # Then heavy atoms that aren't part of the CG bead #(except those
    # involved in the same ring).
    for i in partitioning.keys():
        if partitioning[i] != cg_bead: # AutoM3 change
            # find atom from coordinates
            submol = frag.GetMol()
            for j in range(submol.GetConformer().GetNumAtoms()):
                if (
                    molecule.GetConformer().GetAtomPosition(i)[0]
                    == submol.GetConformer().GetAtomPosition(j)[0]
                ):
                    frag.RemoveAtom(j)
    # Wildman-Crippen log_p
    wc_log_p = rdMolDescriptors.CalcCrippenDescriptors(frag.GetMol())[0]
    # Charge -- look at atoms that are only part of the bead (no ring rule)
    chg = 0
    for i in partitioning.keys():
        if partitioning[i] == cg_bead:
            chg += molecule.GetAtomWithIdx(i).GetFormalCharge()

    smi = Chem.MolToSmiles(Chem.rdmolops.AddHs(frag.GetMol(), addCoords=True))
    ### AutoM3 ###
    atoms_in_smi = " ; atoms: "
    converted_smi = False
    real_smi = None
    for at, bd in partitioning.items():
        if bd == cg_bead:
            if atom_name_map:
                atom_label = atom_name_map[at]
            else:
                at_symbol = molecule.GetAtomWithIdx(at).GetSymbol()
                atom_label = at_symbol + str(at+1)
            atoms_in_smi += atom_label + ", "
    if "c" in smi or "n" in smi or "s" in smi:
        converted_smi = True
        real_smi = smi
        smi = cyclic_smi_conversion(smi)
    # fragment smi: Nc1ncnn1 ---------> FAILURE! Need to fix this Andrew! For now, just a hackish soln:
    # smi = smi.lower() if smi.islower() else smi.upper()
    return smi, wc_log_p, chg, atoms_in_smi, converted_smi, real_smi


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


def get_standard_mass(bead_type): # AutoM3
    """Gets standard mass of atoms in smile code"""
    if bead_type.startswith('T'): return 36
    else: 
        if bead_type.startswith('S'): return 54
        else: return 72



def smi2alogps(forcepred, smi, wc_log_p, bead, converted_smi, real_smi, logp_file=None, trial=False): 
    """
    Returns water/octanol partitioning free energy according to ALOGPS
    AutoM3 : Returns water/octanol partitioning free energy defined empiricaly from customized database
    """
    logger.debug("Entering smi2alogps()")

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


###################################################################################
### OLD STUFF
###################################################################################

def topout_noVS(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, bead_coords, ring_atoms, cg_beads, 
    write_exclusions=True): ### AutoM3 ###
    """AutoM3 : Print itp file without virtual sites, upon users wish"""
    text = ""

    molname=""
    for line in list(atoms_write.split("\n")):
        if line != "":
            x = line.split()
            molname=x[3]
    modified_header_write = header_write
    modified_bonds_write = bonds_write
    exclusions_net = ""
    if len(ring_atoms[0]) > 4 and len(ring_atoms[0]) < 10 and len(bead_coords) < 6:
        #changing nrexcl to 1 if 1 cycle and max 5 beads
        modified_lines_header = []
        for line in list(header_write.split("\n")):
            if ("  "+molname) not in line: modified_lines_header.append(line)
            else:
                lineH = line.split("         ")
                txt = lineH[0] + "          1"
                modified_lines_header.append(txt)
        modified_header_write="\n".join(modified_lines_header)

        #Adding force to constraints 
        modified_lines_bonds = []
        for line in list(bonds_write.split("\n")):
            if "1" in line and len(line.split("   ")) < 7: 
                modified_lines_bonds.append(line + "    1000000")
            else: modified_lines_bonds.append(line)
            if line=="[constraints]":
                if line in modified_lines_bonds : modified_lines_bonds.remove(line)
                txt = "#ifndef FLEXIBLE\n[constraints]\n#endif"
                modified_lines_bonds.append(txt)

        #adding exclusions for two most distant beads in ring
        if len(bead_coords) > 3:
            remote_dist = 0
            remote_beads = []
            bead_in_ring_coords = {}
            ring_atoms = ring_atoms[0]

            for nb,bead_nb in enumerate(cg_beads):
                bead_in_ring_coords[nb+1]=bead_coords[nb]

            for nb_bead1, coord1 in bead_in_ring_coords.items():
                for nb_bead2, coord2 in bead_in_ring_coords.items():
                    dist= math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 + (coord1[2]-coord2[2])**2)

                    if dist > remote_dist and nb_bead1!=nb_bead2:
                        remote_beads=[nb_bead1,nb_bead2]
                        remote_dist=dist
                        
            exclusions_net = ""
            exclusions_net = exclusions_net + "\n[exclusions]\n"
            exclusions_net = exclusions_net + "  " + str(remote_beads[0])+ " " + str(remote_beads[1])
            exclusions_net = exclusions_net + "\n"

            for line in modified_lines_bonds:
                if line!="" and len(line.split("   ")) > 6:
                    if str(remote_beads[0]) == line.split("   ")[1] and str(remote_beads[1]) == line.split("   ")[2] :
                        if line in modified_lines_bonds : modified_lines_bonds.remove(line)
                    else:
                        if str(remote_beads[1]) == line.split("   ")[1] and str(remote_beads[0]) == line.split("   ")[2] :
                            if line in modified_lines_bonds : modified_lines_bonds.remove(line)


        modified_bonds_write="\n".join(modified_lines_bonds)

    if len(cg_beads) > 4:
        #Clean angles already described by dihedrals
        modified_lines_angles = []
        for lineA in list(angles_write.split("\n")):
            if lineA not in modified_lines_angles: modified_lines_angles.append(lineA)
            for lineD in list(dihedrals_write.split("\n")):
                angle_line = lineA.split()
                dihed_line = lineD.split()
                if len(dihed_line) > 2 and not lineD.startswith(";") and len(angle_line) > 2 and not lineA.startswith(";"):
                    if angle_line[0] in dihed_line[:4] and angle_line[1] in dihed_line[:4] and angle_line[2] in dihed_line[:4] and lineA in modified_lines_angles:
                        pass
                        # modified_lines_angles.remove(lineA)
        modified_angles_write = "\n".join(modified_lines_angles)
    else: 
        modified_angles_write = angles_write

    #bartender info search
    bartender_input_info = {}
    bartender_input_info["BONDS"] = []
    for line in list(modified_bonds_write.split("\n")):
        if ";" not in line and len(line.split())>4:
            bartender_input_info["BONDS"].append(line.split()[:2])
    
    bartender_input_info["ANGLES"] = []
    for line in list(modified_angles_write.split("\n")):
        if ";" not in line and len(line.split())>5:
            bartender_input_info["ANGLES"].append(line.split()[:3])
    
    bartender_input_info["IMPROPERS"] = []
    for line in list(dihedrals_write.split("\n")): 
        if ";" not in line and len(line.split())>6:
            bartender_input_info["IMPROPERS"].append(line.split()[:4])
    
    position_restraints = write_position_restraints(range(1, len(cg_beads) + 1))

    text = (
        modified_header_write
        + "\n"
        + atoms_write
        + "\n"
        + modified_bonds_write
        + "\n"
        + modified_angles_write
        + "\n"
        + dihedrals_write
        + "\n"
        + exclusions_net
        + position_restraints
    )
    
    return text, bartender_input_info


def run_bartender(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, 
                    bead_coords, ring_atoms, cg_beads, molecule, molname, atoms_in_smi_dict,
                    write_exclusions=True):
    """
    Combined function to build topology output and bartender input.
    
    Args:
        header_write, atoms_write, bonds_write, angles_write, dihedrals_write: Formatted sections
        bead_coords: CG bead coordinates
        ring_atoms: Ring atom lists
        cg_beads: CG bead list
        molecule: RDKit molecule object
        molname: Molecule name
        atoms_in_smi_dict: Atoms in SMILES dictionary
        write_exclusions: Whether to write exclusions
        
    Returns:
        tuple: (topout_text, bartender_out or None)
    """
    # Build topology output
    _, bartender_input_info = topout_noVS(
        header_write, atoms_write, bonds_write, angles_write, dihedrals_write,
        bead_coords, ring_atoms, cg_beads, write_exclusions
    )
    
    # Build bartender output if requested
    bartender_out = bartender_input(molecule, molname, atoms_in_smi_dict, bartender_input_info)
    
    return _, bartender_out


def topout_vs(header_write, atoms_write, bonds_write, angles_write, dihedrals_write, virtual_sites, vs_write, rigid_dih, simple_model): ### AutoM3 ###

    """AutoM3 : Prints whole .itp file with all bonded and nonbonded parameters. """
    text = ""
    bartender_input_info = {}
    nb_beads = 0
    molname=""
    for line in list(atoms_write.split("\n")):
        if line != "":
            x = line.split()
            molname=x[3]
            nb_beads=int(x[0])

    #Atoms: add bead VS
    vs_bead_names=""
    #Atoms: change mass of VS to 0 and divide it between constructing beads
    modified_lines_atoms = list(atoms_write.split("\n"))
    vs_mass={}
    if virtual_sites and any(isinstance(k, str) for k in virtual_sites.keys()):
        legacy_vs = {}
        for entry in virtual_sites.get('virtual_sitesn', []):
            if isinstance(entry, dict) and 'site' in entry and 'atoms' in entry:
                legacy_vs[int(entry['site'])] = list(entry['atoms'])
        virtual_sites = legacy_vs

    for vs, cb in virtual_sites.items():
        for i, line in enumerate(modified_lines_atoms):
            if line:
                atom_line = line.split()
                if str(vs+1) == atom_line[0] and atom_line[7] != '0': 
                    vs_bead_names+=atom_line[1]
                    vs_mass[vs]=int(atom_line[7])
                    fields = line.split()
                    comments = line.split('   ;   ')[1]
                    modified_lines_atoms[i] = "   {:<5d}   {:5s}   1   {:5s}   {:7s}   {:<5d}   {:2d}     0   ;   {:24s}".format(
                        int(fields[0]), fields[1],  str(fields[3]), fields[4], int(fields[5]), int(fields[6]), comments)

    for vs, cb in virtual_sites.items():
        for vs_env in cb:
            for j, line2 in enumerate(modified_lines_atoms):
                if line2 :
                    atom_line2 = line2.split()
                    if str(vs_env+1) == atom_line2[0]:
                        new_mass = int(int(atom_line2[7]) + vs_mass[vs] / len(cb) ) #add 1/cb mass of VS
                        fields = line2.split()
                        comments = line2.split('   ;   ')[1]
                        modified_lines_atoms[j] = "   {:<5d}   {:5s}   1   {:5s}   {:7s}   {:<5d}   {:2d}   {:3d}   ;   {:24s}".format(
                            int(fields[0]), fields[1],  str(fields[3]), fields[4], int(fields[5]), int(fields[6]),
                            int(new_mass), comments
)
    modified_atoms_write = "\n".join(modified_lines_atoms)

    modified_lines_header=[]
    for line in list(header_write.split("\n")):
        if ("  "+molname) not in line: modified_lines_header.append(line)
        else:
            lineH = line.split("         ")
            txt = lineH[0] + "         1"
            modified_lines_header.append(txt)
    modified_header_write = "\n".join(modified_lines_header)

    # Adding force to constraints
    modified_lines_bonds = []
    bonds_list = []
    for line in list(bonds_write.split("\n")):
        if '1' in line:
            bonds_list.append(f"{line.split('   ')[1]},{line.split('   ')[2]}")
        if '1' in line and len(line.split("   ")) < 7:
            modified_lines_bonds.append(line + "    1000000")
        else:
            modified_lines_bonds.append(line)
        if line == "[constraints]":
            if line in modified_lines_bonds: modified_lines_bonds.remove(line)
            txt = "#ifndef FLEXIBLE\n[constraints]\n#endif"
            modified_lines_bonds.append(txt)
    modified_bonds_write = "\n".join(modified_lines_bonds)
    
    #Bonds / Constraints: delete lines describing interactions with VS
    bond_with_vs={}
    for line in list(modified_bonds_write.split("\n")):
        if line !="":
            bond_line = line.split()
            if len(bond_line)>2 and not line.startswith(";"):
                for vs, cb in virtual_sites.items():
                    if str(vs+1) in bond_line[:2]:
                        # memorizing atom bonded with VS = vs_bond
                        if str(vs+1) == bond_line[0] : vs_bond = bond_line[1] 
                        if str(vs+1) == bond_line[1] : vs_bond = bond_line[0]

                        #memorize VS bond count
                        if str(vs+1) not in bond_with_vs:
                            bond_with_vs[str(vs+1)]=[]
                        if vs_bond not in bond_with_vs.values():
                            bond_with_vs[str(vs+1)].append(vs_bond) #beads bounded to VS

                        #Check if bond between bead B and VS is the only bond connecting B to the rest of the molecule: if yes, don't remove it
                        nb_occ=0
                        if str(vs+1) == bond_line[0]:
                            for i in bonds_list:
                                if bond_line[1] in i: nb_occ+=1
                        else:
                            for i in bonds_list:
                                if bond_line[0] in i: nb_occ+=1
                        if line in modified_lines_bonds and nb_occ>1: modified_lines_bonds.remove(line)
    modified_bonds_write = "\n".join(modified_lines_bonds)

    #Angles: delete lines describing interactions with VS 
    modified_lines_angles = []
    for line in list(angles_write.split("\n")):
        if line !="":
            angle_line = line.split()
            if line not in modified_lines_angles: 
                modified_lines_angles.append(line)
            if len(angle_line)>2 and not line.startswith(";"):
                for vs, cb in virtual_sites.items():
                    #if str(vs+1) == angle_line[0] or str(vs+1) == angle_line[1] or str(vs+1) == angle_line[2] :
                    if str(vs+1) in angle_line[:3] :
                        if line in modified_lines_angles : modified_lines_angles.remove(line)

    #Clean angles already described by dihedrals 
    for lineA in modified_lines_angles:
        for lineD in list(dihedrals_write.split("\n")):
            angle_line = lineA.split()
            dihed_line = lineD.split()
            if len(dihed_line)>2 and not lineD.startswith(";") and len(angle_line)>2 and not lineA.startswith(";"):
                if angle_line[0] in dihed_line[:4] and angle_line[1] in dihed_line[:4] and angle_line[2] in dihed_line[:4]:
                    if lineA in modified_lines_angles : modified_lines_angles.remove(lineA)
    modified_angles_write = "\n".join(modified_lines_angles)

    if not simple_model:
        #Dihedrals: delete lines describing interactions with VS
        modified_lines_dihedrals = []
        dih_list = []
        for line in list(dihedrals_write.split("\n")):
            if line !="":
                dihed_line = line.split()
                if line not in modified_lines_dihedrals: modified_lines_dihedrals.append(line)
                if len(dihed_line)>2 and not line.startswith(";"):
                    dih_list.append(dihed_line[:4])
                    for vs, cb in virtual_sites.items():
                        if str(vs+1) in dihed_line[:4] :
                            if line in modified_lines_dihedrals : modified_lines_dihedrals.remove(line)
        for i in rigid_dih:
            modified_lines_dihedrals.append(i)
        
        modified_dihedrals_write = "\n".join(modified_lines_dihedrals)
    else:
        modified_dihedrals_write = dihedrals_write

    exclusions_net=""
    exclusions_net = exclusions_net + "\n[exclusions]\n"
    for i in range(1,nb_beads):
        row = " ".join(map(str, range(i, nb_beads + 1)))
        exclusions_net="   "+exclusions_net+row+"\n"
    

    # bartender info search
    bartender_input_info["VSITES"] = []
    for line in list(vs_write.split("\n")):
        if ";" not in line and len(line.split()) > 3:
            info_vs = f"{line.split()[0]} {','.join(line.split()[2:])} 1"
            bartender_input_info["VSITES"].append(info_vs)

    bartender_input_info["BONDS"] = []
    for line in list(modified_bonds_write.split("\n")):
        if ";" not in line and len(line.split()) > 4:
            bartender_input_info["BONDS"].append(line.split()[:2])

    bartender_input_info["ANGLES"] = []
    for line in list(modified_angles_write.split("\n")):
        if ";" not in line and len(line.split()) > 5:
            bartender_input_info["ANGLES"].append(line.split()[:3])

    bartender_input_info["IMPROPERS"] = []
    for line in list(dihedrals_write.split("\n")): # not modified_dihedrals_write
        if ";" not in line and len(line.split()) > 6:
            bartender_input_info["IMPROPERS"].append(line.split()[:4])

    position_restraints = write_position_restraints(range(1, nb_beads + 1))
    text = (
        modified_header_write
        + "\n"
        + modified_atoms_write
        + "\n"
        + modified_bonds_write
        + "\n"
        + modified_angles_write
        + "\n"
        + modified_dihedrals_write
        + "\n"
        + vs_write
        + exclusions_net
        + position_restraints
    )
    return text, vs_bead_names, bartender_input_info


def bartender_input(mol, molname, atoms_in_beads, bart_info_dict): ### AutoM3 ###
    """ Generates ready t use input data for Bartender """
    text=f"# INPUT data for bonded parameter definition by BARTENDER for molecule {molname}\n"

    text+="BEADS\n"
    heavy_at=[]
    hydr_at=[]
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol()!='H':
            heavy_at.append(i)
        else:
            hydr_at.append(i)
    heavy_hydro_pair =[]

    for ib in range(len(mol.GetBonds())):
        abond = mol.GetBondWithIdx(ib)
        if (abond.GetBeginAtomIdx() in heavy_at and abond.GetEndAtomIdx() in hydr_at) or (abond.GetBeginAtomIdx() in hydr_at and abond.GetEndAtomIdx() in heavy_at):
            heavy_hydro_pair.append([abond.GetBeginAtomIdx(),abond.GetEndAtomIdx()])

    for bead,atomlist in atoms_in_beads.items():
        atoms=re.findall(r'\d+',atomlist)
        for pair in heavy_hydro_pair:
            if str(pair[0]) in atoms:
                atoms.append(pair[1])
            if str(pair[1]) in atoms:
                atoms.append(pair[0])
        incr_at=[int(atom) + 1 for atom in atoms]
        at_str=",".join(map(str, incr_at))
        text+=str(bead)+" "+at_str+"\n"
    
    for tp, info in bart_info_dict.items():
        if info:
            text+=tp+'\n'
            if tp=='VSITES':
                for i in info:
                    text+=f"{i}\n"
            else:
                for i in info:
                    text+=f"{','.join(i)}\n"
    return text
