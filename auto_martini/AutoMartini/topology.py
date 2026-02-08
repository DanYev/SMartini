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
    atomnames: list = field(default_factory=list)
    beadtypes: list = field(default_factory=list)
    atoms_in_smi_dict: dict = field(default_factory=dict)
    
    # Bonds data: list of [i, j, dist]
    bonds: list = field(default_factory=list)
    
    # Constraints data: list of [i, j, dist]
    constraints: list = field(default_factory=list)
    
    # Angles data: list of [i, j, k, angle, force_const, angle_type]
    angles: list = field(default_factory=list)
    
    # Dihedrals data: list of [i, j, k, l, angle, force_const]
    dihedrals: list = field(default_factory=list)
    
    # Virtual sites (if any)
    virtual_sites: dict = field(default_factory=dict)
    
    # Rigid dihedrals (for virtual sites)
    rigid_dihedrals: list = field(default_factory=list)
    
    # Metadata
    nrexcl: int = 2
    
    # Build methods - update topology data
    def build_atoms(self, cgbeads, forcepred, molecule, hbonda, hbondd, partitioning, 
                    ringatoms, ringatoms_flat, logp_file, trial=False):
        """Build atoms data structure."""
        logger.debug("Entering Topology.build_atoms()")
        
        for bead in range(len(cgbeads)):
            try:
                smi_frag, wc_log_p, charge, atoms_in_smi, converted_smi, real_smi = substruct2smi(
                    molecule, partitioning, bead
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

            if errval == 0:
                try:
                    if charge_frag == 0:
                        alogps, logp_origin = smi2alogps(forcepred, smi_frag, wc_log_p, bead + 1, converted_smi, real_smi, logp_file, trial)
                    else:
                        alogps = 0.0
                        logp_origin = "; Charged fragment"
                except (NameError, TypeError, ValueError):
                    return

                hbond_a_flag = sum(1 for at in hbonda if partitioning[at] == bead)
                hbond_d_flag = sum(1 for at in hbondd if partitioning[at] == bead)
                in_ring = cgbeads[bead] in ringatoms_flat

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
    
    def build_bonds(self, cgbeads, cgbeads_ring, molecule, partitioning, cgbead_coords, ringatoms, cutoff=1e4):
        """Build bonds and constraints data."""
        logger.debug("Entering Topology.build_bonds()")
        cpt_ringatoms = 0
        
        if ringatoms != []:
            cpt_ringatoms = len(sum(ringatoms, []))

        if len(cgbeads) <= 1:
            return

        # Main bond/constraint detection logic
        for i in range(len(cgbeads)):
            for j in range(i + 1, len(cgbeads)):
                dist = np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                if dist > 0.54:
                    break
                if dist < 0.134:
                    raise NameError("Bond too short")

                added_to_constraints = False
                for ring in ringatoms:
                    if cgbeads[i] in ring and cgbeads[j] in ring:
                        self.constraints.append([i, j, dist])
                        added_to_constraints = True
                        break
                if added_to_constraints:
                    continue

                # Look for a bond between an atom of i and an atom of j
                found_connection = False
                atoms_in_bead_i = []
                for ii in partitioning.keys():
                    if partitioning[ii] == i:
                        atoms_in_bead_i.append(ii)
                
                atoms_in_bead_j = []
                for jj in partitioning.keys():
                    if partitioning[jj] == j:
                        atoms_in_bead_j.append(jj)
                        
                for ib in range(len(molecule.GetBonds())):
                    abond = molecule.GetBondWithIdx(ib)
                    if (
                        abond.GetBeginAtomIdx() in atoms_in_bead_i
                        and abond.GetEndAtomIdx() in atoms_in_bead_j
                    ) or (
                        abond.GetBeginAtomIdx() in atoms_in_bead_j
                        and abond.GetEndAtomIdx() in atoms_in_bead_i
                    ):
                        found_connection = True
                
                if found_connection:
                    self.bonds.append([i, j, dist])
                else:
                    if cpt_ringatoms < 7 and len(cgbeads) < 5 and [i, j, dist] not in self.constraints:
                        self.constraints.append([i, j, dist])

        # Ring beads check
        for ir in range(len(cgbeads_ring)):
            for jr in range(ir + 1, len(cgbeads_ring)):
                distr = np.linalg.norm(cgbead_coords[ir] - cgbead_coords[jr]) * 0.1
                if distr < 0.65:
                    for ring in ringatoms:
                        if ( cgbeads_ring[ir] in ring and cgbeads_ring[jr] in ring and distr <= 0.45 
                            ) and ([ir, jr, distr] not in self.constraints and [ir, jr, distr] not in  self.bonds ):
                            self.constraints.append([ir, jr, distr])

        # Go through list of constraints. If we find an extra
        # possible constraint between beads that have constraints, add it.
        beads_with_const = []
        for c in self.constraints:
            if c[0] not in beads_with_const:
                beads_with_const.append(c[0])
            if c[1] not in beads_with_const:
                beads_with_const.append(c[1])

        beads_with_const = sorted(beads_with_const)
        for i in range(len(beads_with_const)):
            for j in range(1 + i, len(beads_with_const)):
                const_exists = False
                for c in self.constraints:
                    if (c[0] == i and c[1] == j) or (c[0] == j and c[1] == i):
                        const_exists = True
                        break
                if not const_exists:
                    dist = np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                    if any(dist  != bl[2] for bl in self.bonds):
                        # Check that it's not in the bond list
                        in_bond_list = False
                        for b in self.bonds:
                            if (b[0] == i and b[1] == j) or (b[0] == j and b[0] == i):
                                in_bond_list = True
                                break
                        # Are atoms part of the same ring
                        in_ring = False
                        for ring in ringatoms:
                            if cgbeads[i] in ring and cgbeads[j] in ring and len(ring)<5:
                                in_ring = True
                                break
                        # If not in bondlist and in the same ring, add the constraint
                        if not in_bond_list and in_ring and [i, j, dist] not in self.constraints:
                            self.constraints.append([i, j, dist])
    
    def build_angles(self, cgbeads, molecule, partitioning, cgbead_coords, ringatoms, type_2_cutoff=160.0):
        """Build angles data structure."""
        logger.debug("Entering Topology.build_angles()")

        if len(cgbeads) <= 2:
            return

        bondlist = self.bonds
        constlist = self.constraints

        for i in range(len(cgbeads)):
            for j in range(len(cgbeads)): 
                for k in range(len(cgbeads)):     

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
                    for ring in ringatoms:
                        if cgbeads[i] in ring and cgbeads[j] in ring and cgbeads[k] in ring:
                            stop_iteration = True
                            break
                    if stop_iteration:
                        continue

                    # Check if all are bonded
                    ij_bonded = False
                    jk_bonded = False
                    ik_bonded = False
                    for b in bondlist + constlist:
                        connectivity = [b[0], b[1]]
                        if i in connectivity and j in connectivity:
                            ij_bonded = True
                        if j in connectivity and k in connectivity:
                            jk_bonded = True
                        if i in connectivity and k in connectivity:
                            ik_bonded = True
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
                    # Look for any double bond between atoms belonging to these CG beads.
                    atoms_in_fragment = []
                    for aa in partitioning.keys():
                        if partitioning[aa] == j:
                            atoms_in_fragment.append(aa)
                    force_const = 100.0
                    for ib in range(len(molecule.GetBonds())):
                        abond = molecule.GetBondWithIdx(ib)
                        if (
                            abond.GetBeginAtomIdx() in atoms_in_fragment
                            and abond.GetEndAtomIdx() in atoms_in_fragment
                        ):
                            bondtype = molecule.GetBondBetweenAtoms(
                                abond.GetBeginAtomIdx(), abond.GetEndAtomIdx()
                            ).GetBondType()
                            if bondtype == rdchem.BondType.DOUBLE:
                                force_const = 45.0

                    ### AutoM3 ###
                    if len(partitioning) > 15:
                        for a1 in range(len(self.angles)):
                            for a2 in range(len(self.angles)):
                                if i in self.angles[a1] and j in self.angles[a1] and j in self.angles[a2] and k in self.angles[a2]:
                                    break
                    
                    funct = 1
                    if angle > type_2_cutoff:
                        force_const = 250.0
                    self.angles.append([i, j, k, funct, angle, force_const])
    
    def build_dihedrals(self, cgbeads, ringatoms, cgbead_coords):
        """Build dihedrals data structure and return num_ar."""
        logger.debug("Entering Topology.build_dihedrals()")

        num_ar = 0
        constlist = self.constraints

        if len(cgbeads) <= 3:
            return num_ar

        # Dihedrals
        dihed_list = []
        # Three ring atoms and one non ring
        for i in range(len(cgbeads)):
            for j in range(len(cgbeads)):
                for k in range(len(cgbeads)):
                    for l in range(len(cgbeads)):
                        if i != j and i != k and i != l and j != k and j != l and k != l:

                            three_in_ring = False
                            for ring in ringatoms:
                                num_ar += len(ring)
                                if [
                                    [cgbeads[i] in ring],
                                    [cgbeads[j] in ring],
                                    [cgbeads[k] in ring],
                                    [cgbeads[l] in ring],
                                ].count([True]) >= 3:
                                    three_in_ring = True
                                    break
                            for b in constlist:
                                if i in [b[0], b[1]] and j in [b[0], b[1]]:
                                    pass
                                if j in [b[0], b[1]] and k in [b[0], b[1]]:
                                    pass
                                if k in [b[0], b[1]] and l in [b[0], b[1]]:
                                    pass
                            # Distance criterion--beads can't be far apart
                            disthres = 0.5
                            close_enough = False
                            if (
                                np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                                < disthres
                                and np.linalg.norm(cgbead_coords[j] - cgbead_coords[k]) * 0.1
                                < disthres
                                and np.linalg.norm(cgbead_coords[k] - cgbead_coords[l]) * 0.1
                                < disthres
                            ):
                                close_enough = True

                            already_dih = False
                            for dih in dihed_list:
                                if dih[0] == l and dih[1] == k and dih[2] == j and dih[3] == i:
                                    already_dih = True
                                    break

                            if three_in_ring and close_enough and not already_dih:
                                r1 = cgbead_coords[j] - cgbead_coords[i]
                                r2 = cgbead_coords[k] - cgbead_coords[j]
                                r3 = cgbead_coords[l] - cgbead_coords[k]
                                p1 = np.cross(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                                p2 = np.cross(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
                                r2 /= np.linalg.norm(r2)
                                cosphi = np.dot(p1, p2)
                                sinphi = np.dot(r2, np.cross(p1, p2))
                                angle = 180.0 / math.pi * np.arctan2(sinphi, cosphi)
                                r1_1 = cgbead_coords[i] - cgbead_coords[j]
                                r2_2 = cgbead_coords[j] - cgbead_coords[k]
                                angle_ijk = 180.0 / math.pi * math.acos(np.dot(r1_1,r2) / (np.linalg.norm(r1_1) * np.linalg.norm(r2)))
                                angle_jkl = 180.0 / math.pi * math.acos(np.dot(r2_2,r3) / (np.linalg.norm(r2_2) * np.linalg.norm(r3)))
                                forc_const = 10.0
                                if angle_ijk < 145.0 and angle_jkl < 145.0:
                                    dihed_list.append([i, j, k, l, angle, forc_const])

        if len(dihed_list) > 0:
            for dl in dihed_list:
                for di in dihed_list[1:]:
                    if dl != di:
                        # Check if beads are repeating
                        if  dl[0:2]==di[0:2] or dl[0:2]==di[2:4] or dl[2:4]==di[0:2] or dl[2:4]==di[2:4] or sorted(dl[:4])==sorted(di[:4]):
                            if di in dihed_list:
                                dihed_list.remove(di)
        
        self.dihedrals = dihed_list
        return num_ar
    
    def build_virtual_sites(self, ringatoms, cg_bead_coords, partitioning, molecule):
        """Build virtual sites data structure."""
        logger.debug("Entering Topology.build_virtual_sites()")

        # Get number of bonds for each atom
        atom_bond_counts = {atom.GetIdx(): 0 for atom in molecule.GetAtoms()}
        
        for bond in molecule.GetBonds():
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            
            if (molecule.GetAtomWithIdx(begin_atom_idx).GetSymbol() != "H" and 
                molecule.GetAtomWithIdx(end_atom_idx).GetSymbol() != "H") and \
               (partitioning[begin_atom_idx] != partitioning[end_atom_idx]):
                if begin_atom_idx not in atom_bond_counts:
                    atom_bond_counts[begin_atom_idx] = 1
                else:
                    atom_bond_counts[begin_atom_idx] += 1
                
                if end_atom_idx not in atom_bond_counts:
                    atom_bond_counts[end_atom_idx] = 1
                else:
                    atom_bond_counts[end_atom_idx] += 1
        
        bead_bond_counts = {}
        for a, b in partitioning.items():
            if b not in bead_bond_counts: 
                bead_bond_counts[b] = 0
            for at, cpt in atom_bond_counts.items():
                if at == a: 
                    bead_bond_counts[b] += cpt

        ring_atoms = []
        for ra in ringatoms:
            ring_atoms += ra

        # Find beads constructing rings
        bead_in_ring_coords = {}
        for atom, bead in partitioning.items():
            if atom in ring_atoms and bead not in bead_in_ring_coords:
                bead_in_ring_coords[bead] = cg_bead_coords[bead]
        
        # Count distances between each pair of beads
        distances = {}
        for bead, coord in bead_in_ring_coords.items():
            distances[bead] = {}
            for other_bead, other_coord in bead_in_ring_coords.items():
                if bead != other_bead:
                    distance = np.linalg.norm(coord - other_coord) 
                    distances[bead][other_bead] = distance
        
        def find_more_vs(num_vs, bead_bond_counts_sorted, cg_bead_coords, distances):
            virtual_sites = {}
            vs_list = []
            # Only select beads that are in rings (i.e., in distances dict)
            ring_beads = [bead for bead in bead_bond_counts_sorted.keys() if bead in distances]
            for i in range(min(num_vs, len(ring_beads))):
                vs_bead = int(ring_beads[i])
                vs_list.append(vs_bead)
            
            for vs in vs_list:
                constructing_beads_dist = dict(sorted(distances[vs].items(), key=lambda item: item[1]))
                constructing_beads = [bead for bead in constructing_beads_dist.keys()]
                for bead in constructing_beads:
                    if bead in vs_list:
                        constructing_beads.remove(bead)
                
                if vs not in virtual_sites.keys():
                    virtual_sites[vs] = constructing_beads[:4]
            return virtual_sites

        # Find number of fused cycles = number of needed virtual sites    
        bead_bond_counts_sorted = dict(sorted(bead_bond_counts.items(), key=lambda item: item[1], reverse=True))
        cpt_ringatoms = len(sum(ringatoms, []))

        for r_nb in range(len(ringatoms)):
            if cpt_ringatoms > 6 and cpt_ringatoms < 19: 
                self.virtual_sites = find_more_vs(1, bead_bond_counts_sorted, cg_bead_coords, distances)

            if cpt_ringatoms > 18:  # more than 4 fused cycles
                self.virtual_sites = find_more_vs(3, bead_bond_counts_sorted, cg_bead_coords, distances)

        # Calculate rigid dihedrals for 4-atom virtual sites
        for vs, cb in self.virtual_sites.items():
            if len(cb) == 4:
                # Find dihedral from constructing beads
                i = cb[0]
                j = cb[1]
                k = cb[2]
                l = cb[3]
                r1 = cg_bead_coords[j] - cg_bead_coords[i]
                r2 = cg_bead_coords[k] - cg_bead_coords[j]
                r3 = cg_bead_coords[l] - cg_bead_coords[k]
                p1 = np.cross(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                p2 = np.cross(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
                r2 /= np.linalg.norm(r2)
                cosphi = np.dot(p1, p2)
                sinphi = np.dot(r2, np.cross(p1, p2))
                angle = 180.0 / math.pi * np.arctan2(sinphi, cosphi)
                force = 100
                self.rigid_dihedrals.append({
                    'atoms': [cb[0], cb[1], cb[2], cb[3]],
                    'angle': round(angle, 2),
                    'force': force
                })
    
    # Format methods - return formatted strings
    def format_header(self):
        """Format Topology header section."""
        text = "; GENERATED WITH Auto_Martini M3FF for {}\n".format(self.molname)
        info = (
            "; Developed by: Kiran Kanekal, Tristan Bereau, and Andrew Abi-Mansour\n"
            + "; updated to Martini 3 force field by Magdalena Szczuka\n"
            + "; supervised by Matthieu Chavent, Pierre Poulain and Paulo C. T. Souza \n"
            + "; SMILES code : " + self.mol_smi + "\n\n"
            + "\n[moleculetype]\n"
            + "; molname       nrexcl\n"
            + "  {:5s}         {:d}\n\n".format(self.molname, self.nrexcl)
            + "[atoms]\n"
            + "; id      type   resnr residue atom    cgnr    charge  mass ;  smiles    ; atom_num"
        )
        return text + info
    
    def format_atoms(self, trial=False):
        """Format atoms list into ITP text."""
        if trial:
            return ""
        
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
    
    def format_bonds(self, ringatoms, trial=False, cutoff=1e4):
        """Format bonds and constraints into ITP text."""
        if trial:
            return ""
        
        text = ""
        cpt_ringatoms = 0
        if ringatoms != []:
            cpt_ringatoms = len(sum(ringatoms, []))
        
        # Create beadlist for read_params
        beadlist = []
        for bead in self.beadtypes:
            if not bead.startswith('T') and not bead.startswith('S'):
                beadlist.append('R')
            else:
                beadlist.append(bead[0])
        
        if len(self.bonds) > 0:
            text = "\n[bonds]\n" + ";  i   j     funct   length   force.c."
            for b in self.bonds:
                # Make sure atoms in bond are not part of the same ring
                fc = read_params(b[2], beadlist[b[0]] + "-" + beadlist[b[1]])
                if fc >= cutoff:
                    fc = cutoff
                text = text + "\n   {:<3d} {:<3d}   1       {:4.2f}       {:4.1f}".format(
                    b[0] + 1, b[1] + 1, b[2], fc,
                )
        else:
            text = "\n[bonds]\n"
        
        if len(self.constraints) > 0:
            text = text + "\n\n[constraints]\n" + ";  i   j     funct   length"
            for c in self.constraints:
                if c not in self.bonds:
                    if cpt_ringatoms > 18 and c[2] > 0.415:
                        continue
                    text = text + "\n   {:<3d} {:<3d}   1       {:4.2f}".format(
                        c[0] + 1, c[1] + 1, c[2]
                    )
        
        return text
    
    def format_angles(self):
        """Format angles into ITP text."""
        text = ""
        
        ### AutoM3 ###
        beadlist = []
        for bead in self.beadtypes:
            if not bead.startswith('T') and not bead.startswith('S'):
                beadlist.append('R')
            else:
                beadlist.append(bead[0])

        if len(self.angles) > 0:
            text = text + "\n[angles]\n"
            text = text + ";  i  j  k    funct  angle  force.c.\n"
            for a in self.angles:
                force = read_params(a[4], beadlist[a[0]] + "-" + beadlist[a[1]] + "-" + beadlist[a[2]])
                if force is None:
                    force = a[5]
                text = text + "  {:2} {:2} {:2}       {:2}    {:<5.1f}  {:5.1f}\n".format(
                    a[0] + 1, a[1] + 1, a[2] + 1, a[3], a[4], force
                )
        return text
    
    def format_dihedrals(self, num_ar, cgbeads, ringatoms, cgbead_coords):
        """Format dihedrals into ITP text."""
        text = ""
        
        ### AutoM3 ###
        bead_in_ring_coords = {}
        for nb, bead_nb in enumerate(cgbeads):
            for ring in ringatoms:
                if bead_nb in ring:
                    bead_in_ring_coords[nb] = cgbead_coords[nb]
        
        beadlist = []
        for bead in self.beadtypes:
            if not bead.startswith('T') and not bead.startswith('S'):
                beadlist.append('R')
            else:
                beadlist.append(bead[0])
        
        if len(self.dihedrals) > 0:
            text = text + "\n[dihedrals]\n"
            text = text + ";  i  j  k  l  funct  angle  force.c.\n"
            
            for d in self.dihedrals:
                ### AutoM3 ###
                force = read_params(d[4], beadlist[d[0]] + "-" + beadlist[d[1]] + "-" + beadlist[d[2]] + "-" + beadlist[d[3]])
                if num_ar > 0 and (d[0] or d[1] or d[2] or d[3] not in bead_in_ring_coords.keys()) and force is not None:
                    force = force / 2  # for dihedral between cycle-bead and non-cycled bead: decrease of force
                if force is None:
                    force = d[5]
                text = (
                    text
                    + "  {:2} {:2} {:2} {:2}    2    {:<5.1f}  {:5.1f}\n".format(
                        d[0] + 1, d[1] + 1, d[2] + 1, d[3] + 1, d[4], force
                    )
                )
        return text
    
    def format_virtual_sites(self):
        """Format virtual sites data for .itp output."""
        if not self.virtual_sites:
            return ""
        
        text = "\n[virtual_sitesn]\n"
        text += "; site funct  constructing atom indices"
        
        for vs, cb in self.virtual_sites.items():
            if len(cb) == 4:
                text += "\n   {:d}       1     {:d} {:d} {:d} {:d}".format(
                    vs+1, cb[0]+1, cb[1]+1, cb[2]+1, cb[3]+1
                )
            elif len(cb) == 3:
                text += "\n   {:d}       1     {:d} {:d} {:d}".format(
                    vs+1, cb[0]+1, cb[1]+1, cb[2]+1
                )
            elif len(cb) == 2:
                text += "\n   {:d}       1     {:d} {:d}".format(
                    vs+1, cb[0]+1, cb[1]+1
                )
        
        return text
    
    def to_itp(self, num_ar, cgbeads, ringatoms, cgbead_coords, trial=False):
        """Generate complete ITP file content."""
        text = self.format_header() + "\n"
        text += self.format_atoms(trial) + "\n"
        text += self.format_bonds(ringatoms, trial) + "\n"
        text += self.format_angles() + "\n"
        text += self.format_dihedrals(num_ar, cgbeads, ringatoms, cgbead_coords) + "\n"
        text += self.format_virtual_sites()
        return text


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
    if len(suppl) > 1:
        print("Error. Only one molecule may be provided.")
        exit(1)
    logger.debug("SDF supplier length: %d", len(suppl))
    molecule = ""
    for molecule in suppl:
        if molecule is None:
            print("Error. Can't read molecule.")
            exit(1)
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


def substruct2smi(molecule, partitioning, cg_bead):
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
            at_symbol = molecule.GetAtomWithIdx(at).GetSymbol()
            atoms_in_smi += at_symbol + str(at+1) + ", "
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


def build_virtual_sites_data(ringatoms, cg_bead_coords, partitioning, molecule):
    """
    Build virtual sites data structure (decoupled from formatting).
    Returns: (virtual_sites_dict, rigid_dihedrals_list)
    """
    logger.debug("Entering build_virtual_sites_data()")

    # Get number of bonds for each atom
    atom_bond_counts = {atom.GetIdx(): 0 for atom in molecule.GetAtoms()}
    
    for bond in molecule.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        
        if (molecule.GetAtomWithIdx(begin_atom_idx).GetSymbol() != "H" and 
            molecule.GetAtomWithIdx(end_atom_idx).GetSymbol() != "H") and \
           (partitioning[begin_atom_idx] != partitioning[end_atom_idx]):
            if begin_atom_idx not in atom_bond_counts:
                atom_bond_counts[begin_atom_idx] = 1
            else:
                atom_bond_counts[begin_atom_idx] += 1
            
            if end_atom_idx not in atom_bond_counts:
                atom_bond_counts[end_atom_idx] = 1
            else:
                atom_bond_counts[end_atom_idx] += 1
    
    bead_bond_counts = {}
    for a, b in partitioning.items():
        if b not in bead_bond_counts: 
            bead_bond_counts[b] = 0
        for at, cpt in atom_bond_counts.items():
            if at == a: 
                bead_bond_counts[b] += cpt

    ring_atoms = []
    virtual_sites = {}
    for ra in ringatoms:
        ring_atoms += ra

    # Find beads constructing rings
    bead_in_ring_coords = {}
    for atom, bead in partitioning.items():
        if atom in ring_atoms and bead not in bead_in_ring_coords:
            bead_in_ring_coords[bead] = cg_bead_coords[bead]
    
    # Count distances between each pair of beads
    distances = {}
    for bead, coord in bead_in_ring_coords.items():
        distances[bead] = {}
        for other_bead, other_coord in bead_in_ring_coords.items():
            if bead != other_bead:
                distance = np.linalg.norm(coord - other_coord) 
                distances[bead][other_bead] = distance
    
    def find_more_vs(num_vs, bead_bond_counts_sorted, cg_bead_coords, distances):
        virtual_sites = {}
        vs_list = []
        for i in range(num_vs):
            vs_bead = int(list(bead_bond_counts_sorted.keys())[i])
            vs_list.append(vs_bead)
        
        for vs in vs_list:
            constructing_beads_dist = dict(sorted(distances[vs].items(), key=lambda item: item[1]))
            constructing_beads = [bead for bead in constructing_beads_dist.keys()]
            for bead in constructing_beads:
                if bead in vs_list:
                    constructing_beads.remove(bead)
            
            if vs not in virtual_sites.keys():
                virtual_sites[vs] = constructing_beads[:4]
        return virtual_sites

    # Find number of fused cycles = number of needed virtual sites    
    bead_bond_counts_sorted = dict(sorted(bead_bond_counts.items(), key=lambda item: item[1], reverse=True))
    cpt_ringatoms = len(sum(ringatoms, []))

    for r_nb in range(len(ringatoms)):
        if cpt_ringatoms > 6 and cpt_ringatoms < 19: 
            virtual_sites = find_more_vs(1, bead_bond_counts_sorted, cg_bead_coords, distances)

        if cpt_ringatoms > 18:  # more than 4 fused cycles
            virtual_sites = find_more_vs(3, bead_bond_counts_sorted, cg_bead_coords, distances)

    # Calculate rigid dihedrals for 4-atom virtual sites
    rigid_dihedrals = []
    for vs, cb in virtual_sites.items():
        if len(cb) == 4:
            # Find dihedral from constructing beads
            i = cb[0]
            j = cb[1]
            k = cb[2]
            l = cb[3]
            r1 = cg_bead_coords[j] - cg_bead_coords[i]
            r2 = cg_bead_coords[k] - cg_bead_coords[j]
            r3 = cg_bead_coords[l] - cg_bead_coords[k]
            p1 = np.cross(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
            p2 = np.cross(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
            r2 /= np.linalg.norm(r2)
            cosphi = np.dot(p1, p2)
            sinphi = np.dot(r2, np.cross(p1, p2))
            angle = 180.0 / math.pi * np.arctan2(sinphi, cosphi)
            force = 100
            rigid_dihedrals.append({
                'atoms': [cb[0], cb[1], cb[2], cb[3]],
                'angle': round(angle, 2),
                'force': force
            })
        
    return virtual_sites, rigid_dihedrals


def format_virtual_sites(virtual_sites):
    """Format virtual sites data for .itp output."""
    if not virtual_sites:
        return ""
    
    text = "\n[virtual_sitesn]\n"
    text += "; site funct  constructing atom indices"
    
    for vs, cb in virtual_sites.items():
        if len(cb) == 4:
            text += "\n   {:d}       1     {:d} {:d} {:d} {:d}".format(
                vs+1, cb[0]+1, cb[1]+1, cb[2]+1, cb[3]+1
            )
        elif len(cb) == 3:
            text += "\n   {:d}       1     {:d} {:d} {:d}".format(
                vs+1, cb[0]+1, cb[1]+1, cb[2]+1
            )
        elif len(cb) == 2:
            text += "\n   {:d}       1     {:d} {:d}".format(
                vs+1, cb[0]+1, cb[1]+1
            )
    
    return text


def topout(header_write, atoms_write, bonds_write, angles_write):
    """Print simple itp file"""
    text = header_write + "\n" + atoms_write + "\n" + bonds_write + "\n" + angles_write

    # bartender info search
    bartender_input_info = {}
    bartender_input_info["BONDS"] = []
    for line in list(bonds_write.split("\n")):
        if ";" not in line and len(line.split()) > 4:
            bartender_input_info["BONDS"].append(line.split()[:2])

    bartender_input_info["ANGLES"] = []
    for line in list(angles_write.split("\n")):
        if ";" not in line and len(line.split()) > 5:
            bartender_input_info["ANGLES"].append(line.split()[:3])
    return text, bartender_input_info


def write_position_restraints(
    atom_indices,
    force_constant: str = "POSRES_FC",
    funct: int = 1,
    ifdef: str = "POSRES",
    include_end_if: bool = True,
):
    """Return a position restraints section for the provided atom indices.

    Parameters
    ----------
    atom_indices : iterable of int
        Atom indices (1-based) to restrain.
    force_constant : str
        Force constant label written for x/y/z (default: POSRES_FC).
    funct : int
        Gromacs function type (default: 1).
    ifdef : str
        Preprocessor symbol used for conditional inclusion.
    include_end_if : bool
        Whether to append a matching #endif line.
    """
    if not atom_indices:
        return ""

    lines = [
        "#ifndef POSRES_FC",
        "#define POSRES_FC 1000.0",
        "#endif",
        "[ position_restraints ]",
        f"#ifdef {ifdef}",
    ]
    for atom_index in atom_indices:
        atom_id = int(atom_index)
        lines.append(
            f"{atom_id:5d} {funct:d} {force_constant} {force_constant} {force_constant}"
        )
    if include_end_if:
        lines.append("#endif")
    return "\n".join(lines) + "\n"


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
        modified_lines_header=[]
        for line in list(header_write.split("\n")):
            if ("  "+molname) not in line: modified_lines_header.append(line)
            else:
                lineH=line.split("         ")
                txt=lineH[0]+"          1"
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
                                   bartender=False, write_exclusions=True):
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
        bartender: Whether to generate bartender output
        write_exclusions: Whether to write exclusions
        
    Returns:
        tuple: (topout_text, bartender_out or None)
    """
    # Build topology output
    topout_text, bartender_input_info = topout_noVS(
        header_write, atoms_write, bonds_write, angles_write, dihedrals_write,
        bead_coords, ring_atoms, cg_beads, write_exclusions
    )
    
    # Build bartender output if requested
    bartender_out = None
    if bartender:
        bartender_out = bartender_input(molecule, molname, atoms_in_smi_dict, bartender_input_info)
    
    return topout_text, bartender_out


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


# ==============================================================================
# NEW: Topology Building Functions (decoupled data from formatting)
# ==============================================================================

def build_topology(cgbeads, molname, mol_smi, forcepred, cgbeads_ring, molecule, hbonda, hbondd, 
                    partitioning, cgbead_coords, ringatoms, ringatoms_flat, logp_file, beadtypes=None, trial=False, simple_model=False):
    """
    Build complete Topology object with all CG data.
    
    This function centralizes topology construction and returns a Topology object
    containing all topology data in structured form (atoms, bonds, angles, dihedrals, etc.).
    
    Args:
        molname: Molecule name
        mol_smi: SMILES string
        forcepred: Force field prediction model
        cgbeads: List of CG beads
        cgbeads_ring: List of ring CG beads
        molecule: RDKit molecule object
        hbonda: H-bond acceptors
        hbondd: H-bond donors
        partitioning: Atom to bead mapping
        cgbead_coords: CG bead coordinates
        ringatoms: List of ring atoms
        ringatoms_flat: Flattened ring atoms
        logp_file: LogP file handle
        beadtypes: Pre-computed bead types (optional, will compute if None)
        trial: Whether this is a trial run
        simple_model: Whether to skip dihedrals
    
    Returns:
        Topology: object containing all topology data
    """
    topo = Topology(molname=molname, mol_smi=mol_smi)
    
    # Build atoms
    atomnames, computed_beadtypes, atoms_data, atoms_in_smi_dict = build_atoms_data(
        cgbeads, molname, forcepred,  molecule, hbonda, hbondd,
        partitioning, ringatoms, ringatoms_flat, logp_file, trial
    )
    topo.atomnames = atomnames
    topo.beadtypes = beadtypes if beadtypes is not None else computed_beadtypes
    topo.atoms = atoms_data
    topo.atoms_in_smi_dict = atoms_in_smi_dict
    
    # Build bonds and constraints
    bonds, constraints = build_bonds_data(
        cgbeads, cgbeads_ring, molecule, partitioning, cgbead_coords, ringatoms
    )
    topo.bonds = bonds
    topo.constraints = constraints
    
    # Build angles
    angles = build_angles_data(
        cgbeads, molecule, partitioning, cgbead_coords, bonds, constraints, ringatoms
    )
    topo.angles = angles
    
    # Build dihedrals (unless simple model)
    if not simple_model:
        dihedrals, num_ar = build_dihedrals_data(
            cgbeads, constraints, ringatoms, cgbead_coords
        )
        topo.dihedrals = dihedrals
    
    return topo


def build_atoms_data(cgbeads, molname, forcepred, molecule, hbonda, hbondd, partitioning, 
                      ringatoms, ringatoms_flat, logp_file, trial=False):
    """Build atoms data structure (decoupled from formatting)."""
    logger.debug("Entering build_atoms_data()")
    atomnames = []
    beadtypes = []
    atoms_data = []
    atoms_in_smi_dict = {}

    for bead in range(len(cgbeads)):
        try:
            smi_frag, wc_log_p, charge, atoms_in_smi, converted_smi, real_smi = substruct2smi(
                molecule, partitioning, bead
            )
        except Exception:
            raise
        atoms_in_smi_dict[bead + 1] = atoms_in_smi.replace(" ; atoms: ", "")

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

        if errval == 0:
            try:
                if charge_frag == 0:
                    alogps, logp_origin = smi2alogps(forcepred, smi_frag, wc_log_p, bead + 1, converted_smi, real_smi, logp_file, trial)
                else:
                    alogps = 0.0
                    logp_origin = "; Charged fragment"
            except (NameError, TypeError, ValueError):
                return atomnames, beadtypes, atoms_data, atoms_in_smi_dict

            hbond_a_flag = sum(1 for at in hbonda if partitioning[at] == bead)
            hbond_d_flag = sum(1 for at in hbondd if partitioning[at] == bead)
            in_ring = cgbeads[bead] in ringatoms_flat

            bead_type = determine_bead_type(alogps, charge, hbond_a_flag, hbond_d_flag, in_ring, smi_frag)
            atom_name = ""
            name_index = 0
            while atom_name in atomnames or name_index == 0:
                name_index += 1
                atom_name = "{:1s}{:02d}".format(bead_type[1], name_index)
            atomnames.append(atom_name)
            
            mass = get_standard_mass(bead_type)

            atom_dict = {
                'id': bead + 1,
                'type': bead_type,
                'resnr': 1,
                'residue': molname[:4] if len(molname) > 4 else molname,
                'atom': atom_name,
                'cgnr': bead + 1,
                'charge': charge,
                'mass': mass,
                'smiles': smi_frag,
                'atoms_in_smi': atoms_in_smi,
                'logp_origin': logp_origin
            }
            atoms_data.append(atom_dict)
            beadtypes.append(bead_type)

    return atomnames, beadtypes, atoms_data, atoms_in_smi_dict


def build_bonds_data(cgbeads, cgbeads_ring, molecule, partitioning, cgbead_coords, ringatoms, cutoff=1e4):
    """Build bonds and constraints data (decoupled from formatting)."""
    logger.debug("Entering build_bonds_data()")
    bondlist = []
    constlist = []
    cpt_ringatoms = 0
    
    if ringatoms != []:
        cpt_ringatoms = len(sum(ringatoms, []))

    if len(cgbeads) <= 1:
        return bondlist, constlist

    # Main bond/constraint detection logic (same as original print_bonds)
    for i in range(len(cgbeads)):
        for j in range(i + 1, len(cgbeads)):
            dist = np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
            if dist > 0.54:
                break
            if dist < 0.134:
                raise NameError("Bond too short")

            added_to_constraints = False
            for ring in ringatoms:
                if cgbeads[i] in ring and cgbeads[j] in ring:
                    constlist.append([i, j, dist])
                    added_to_constraints = True
                    break
            if added_to_constraints:
                continue

            # Look for a bond between an atom of i and an atom of j
            found_connection = False
            atoms_in_bead_i = []
            for ii in partitioning.keys():
                if partitioning[ii] == i:
                    atoms_in_bead_i.append(ii)
            
            atoms_in_bead_j = []
            for jj in partitioning.keys():
                if partitioning[jj] == j:
                    atoms_in_bead_j.append(jj)
                    
            for ib in range(len(molecule.GetBonds())):
                abond = molecule.GetBondWithIdx(ib)
                if (
                    abond.GetBeginAtomIdx() in atoms_in_bead_i
                    and abond.GetEndAtomIdx() in atoms_in_bead_j
                ) or (
                    abond.GetBeginAtomIdx() in atoms_in_bead_j
                    and abond.GetEndAtomIdx() in atoms_in_bead_i
                ):
                    found_connection = True
            
            if found_connection:
                bondlist.append([i, j, dist])
            else:
                if cpt_ringatoms < 7 and len(cgbeads) < 5 and [i, j, dist] not in constlist:
                    constlist.append([i, j, dist])

    # Ring beads check
    for ir in range(len(cgbeads_ring)):
        for jr in range(ir + 1, len(cgbeads_ring)):
            distr = np.linalg.norm(cgbead_coords[ir] - cgbead_coords[jr]) * 0.1
            if distr < 0.65:
                for ring in ringatoms:
                    if ( cgbeads_ring[ir] in ring and cgbeads_ring[jr] in ring and distr <= 0.45 
                        ) and ([ir, jr, distr] not in constlist and [ir, jr, distr] not in  bondlist ):
                        constlist.append([ir, jr, distr])

    # Go through list of constraints. If we find an extra
    # possible constraint between beads that have constraints, add it.
    beads_with_const = []
    for c in constlist:
        if c[0] not in beads_with_const:
            beads_with_const.append(c[0])
        if c[1] not in beads_with_const:
            beads_with_const.append(c[1])

    beads_with_const = sorted(beads_with_const)
    for i in range(len(beads_with_const)):
        for j in range(1 + i, len(beads_with_const)):
            const_exists = False
            for c in constlist:
                if (c[0] == i and c[1] == j) or (c[0] == j and c[1] == i):
                    const_exists = True
                    break
            if not const_exists:
                dist = np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                if any(dist  != bl[2] for bl in bondlist):
                    # Check that it's not in the bond list
                    in_bond_list = False
                    for b in bondlist:
                        if (b[0] == i and b[1] == j) or (b[0] == j and b[0] == i):
                            in_bond_list = True
                            break
                    # Are atoms part of the same ring
                    in_ring = False
                    for ring in ringatoms:
                        if cgbeads[i] in ring and cgbeads[j] in ring and len(ring)<5:
                            in_ring = True
                            break
                    # If not in bondlist and in the same ring, add the constraint
                    if not in_bond_list and in_ring and [i, j, dist] not in constlist:
                        constlist.append([i, j, dist])

    return bondlist, constlist


def format_topology_header(topo):
    """Format Topology header section."""
    text = "; GENERATED WITH Auto_Martini M3FF for {}\n".format(topo.molname)
    info = (
        "; Developed by: Kiran Kanekal, Tristan Bereau, and Andrew Abi-Mansour\n"
        + "; updated to Martini 3 force field by Magdalena Szczuka\n"
        + "; supervised by Matthieu Chavent, Pierre Poulain and Paulo C. T. Souza \n"
        + "; SMILES code : " + topo.mol_smi + "\n\n"
        + "\n[moleculetype]\n"
        + "; molname       nrexcl\n"
        + "  {:5s}         {:d}\n\n".format(topo.molname, topo.nrexcl)
        + "[atoms]\n"
        + "; id      type   resnr residue atom    cgnr    charge  mass ;  smiles    ; atom_num"
    )
    return text + info


def format_topology_atoms(atoms_data, trial=False):
    """Format atoms list into ITP text."""
    if trial:
        return ""
    
    text = ""
    for atom in atoms_data:
        text += (
            "   {:<5d}   {:5s}   {:d}   {:5s}   {:7s}   {:<5d}   {:2d}   {:3d}   ;   {:8s}{:8s}{:9s}\n".format(
                atom['id'], atom['type'], atom['resnr'], atom['residue'], atom['atom'],
                atom['cgnr'], atom['charge'], atom['mass'], atom['smiles'],
                atom['atoms_in_smi'], atom['logp_origin']
            )
        )
    return text


def format_topology_bonds(bonds, constraints, beadtypes, ringatoms, trial=False, cutoff=1e4):
    """Format bonds and constraints into ITP text."""
    if trial:
        return ""
    
    text = ""
    cpt_ringatoms = 0
    if ringatoms != []:
        cpt_ringatoms = len(sum(ringatoms, []))
    
    # Create beadlist for read_params
    beadlist = []
    for bead in beadtypes:
        if not bead.startswith('T') and not bead.startswith('S'):
            beadlist.append('R')
        else:
            beadlist.append(bead[0])
    
    if len(bonds) > 0:
        text = "\n[bonds]\n" + ";  i   j     funct   length   force.c."
        for b in bonds:
            # Make sure atoms in bond are not part of the same ring
            fc = read_params(b[2], beadlist[b[0]] + "-" + beadlist[b[1]])
            if fc >= cutoff:
                fc = cutoff
            text = text + "\n   {:<3d} {:<3d}   1       {:4.2f}       {:4.1f}".format(
                b[0] + 1, b[1] + 1, b[2], fc,
            )
    else:
        text = "\n[bonds]\n"
    
    if len(constraints) > 0:
        text = text + "\n\n[constraints]\n" + ";  i   j     funct   length"
        for c in constraints:
            if c not in bonds:
                if cpt_ringatoms > 18 and c[2] > 0.415:
                    continue
                text = text + "\n   {:<3d} {:<3d}   1       {:4.2f}".format(
                    c[0] + 1, c[1] + 1, c[2]
                )
    
    return text


def build_angles_data(cgbeads, molecule, partitioning, cgbead_coords, bondlist, constlist, ringatoms, type_2_cutoff=160.0):
    """Build angles data structure (decoupled from formatting)."""
    logger.debug("Entering build_angles_data()")
    
    angle_list = []

    # Angles
    if len(cgbeads) <= 2:
        return angle_list

    for i in range(len(cgbeads)):
        for j in range(len(cgbeads)): 
            for k in range(len(cgbeads)):     

                # Check if all indices are different
                if i == j or j == k or i == k:
                    continue

                # Check if angle already exists
                stop_iteration = False
                for a in angle_list:
                    it, jt, kt = a[0], a[1], a[2]
                    if i == kt and j == jt and k == it:
                        stop_iteration = True
                        break
                if stop_iteration:
                    continue

                # Check if all of them are in one ring
                for ring in ringatoms:
                    if cgbeads[i] in ring and cgbeads[j] in ring and cgbeads[k] in ring:
                        stop_iteration = True
                        break
                if stop_iteration:
                    continue

                # Check if all are bonded
                ij_bonded = False
                jk_bonded = False
                ik_bonded = False
                for b in bondlist + constlist:
                    connectivity = [b[0], b[1]]
                    if i in connectivity and j in connectivity:
                        ij_bonded = True
                    if j in connectivity and k in connectivity:
                        jk_bonded = True
                    if i in connectivity and k in connectivity:
                        ik_bonded = True
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
                # Look for any double bond between atoms belonging to these CG beads.
                atoms_in_fragment = []
                for aa in partitioning.keys():
                    if partitioning[aa] == j:
                        atoms_in_fragment.append(aa)
                force_const = 100.0
                for ib in range(len(molecule.GetBonds())):
                    abond = molecule.GetBondWithIdx(ib)
                    if (
                        abond.GetBeginAtomIdx() in atoms_in_fragment
                        and abond.GetEndAtomIdx() in atoms_in_fragment
                    ):
                        bondtype = molecule.GetBondBetweenAtoms(
                            abond.GetBeginAtomIdx(), abond.GetEndAtomIdx()
                        ).GetBondType()
                        if bondtype == rdchem.BondType.DOUBLE:
                            force_const = 45.0

                ### AutoM3 ###
                if len(partitioning) > 15:
                    for a1 in range(len(angle_list)):
                        for a2 in range(len(angle_list)):
                            if i in angle_list[a1] and j in angle_list[a1] and j in angle_list[a2] and k in angle_list[a2]:
                                break
                
                funct = 1
                if angle > type_2_cutoff:
                    force_const = 250.0
                angle_list.append([i, j, k, funct, angle, force_const])

    return angle_list


def build_dihedrals_data(cgbeads, constlist, ringatoms, cgbead_coords):
    """Build dihedrals data structure (decoupled from formatting)."""
    logger.debug("Entering build_dihedrals_data()")

    new_dihed_list = []
    num_ar = 0

    if len(cgbeads) > 3: 
        # Dihedrals
        dihed_list = []
        # Three ring atoms and one non ring
        for i in range(len(cgbeads)):
            for j in range(len(cgbeads)):
                for k in range(len(cgbeads)):
                    for l in range(len(cgbeads)):
                        if i != j and i != k and i != l and j != k and j != l and k != l:

                            three_in_ring = False
                            for ring in ringatoms:
                                num_ar += len(ring)
                                if [
                                    [cgbeads[i] in ring],
                                    [cgbeads[j] in ring],
                                    [cgbeads[k] in ring],
                                    [cgbeads[l] in ring],
                                ].count([True]) >= 3:
                                    three_in_ring = True
                                    break
                            for b in constlist:
                                if i in [b[0], b[1]] and j in [b[0], b[1]]:
                                    pass
                                if j in [b[0], b[1]] and k in [b[0], b[1]]:
                                    pass
                                if k in [b[0], b[1]] and l in [b[0], b[1]]:
                                    pass
                            # Distance criterion--beads can't be far apart
                            disthres = 0.5
                            close_enough = False
                            if (
                                np.linalg.norm(cgbead_coords[i] - cgbead_coords[j]) * 0.1
                                < disthres
                                and np.linalg.norm(cgbead_coords[j] - cgbead_coords[k]) * 0.1
                                < disthres
                                and np.linalg.norm(cgbead_coords[k] - cgbead_coords[l]) * 0.1
                                < disthres
                            ):
                                close_enough = True

                            already_dih = False
                            for dih in dihed_list:
                                if dih[0] == l and dih[1] == k and dih[2] == j and dih[3] == i:
                                    already_dih = True
                                    break

                            if three_in_ring and close_enough and not already_dih:
                                r1 = cgbead_coords[j] - cgbead_coords[i]
                                r2 = cgbead_coords[k] - cgbead_coords[j]
                                r3 = cgbead_coords[l] - cgbead_coords[k]
                                p1 = np.cross(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                                p2 = np.cross(r2, r3) / (np.linalg.norm(r2) * np.linalg.norm(r3))
                                r2 /= np.linalg.norm(r2)
                                cosphi = np.dot(p1, p2)
                                sinphi = np.dot(r2, np.cross(p1, p2))
                                angle = 180.0 / math.pi * np.arctan2(sinphi, cosphi)
                                r1_1 = cgbead_coords[i] - cgbead_coords[j]
                                r2_2 = cgbead_coords[j] - cgbead_coords[k]
                                angle_ijk = 180.0 / math.pi * math.acos(np.dot(r1_1,r2) / (np.linalg.norm(r1_1) * np.linalg.norm(r2)))
                                angle_jkl = 180.0 / math.pi * math.acos(np.dot(r2_2,r3) / (np.linalg.norm(r2_2) * np.linalg.norm(r3)))
                                forc_const = 10.0
                                if angle_ijk < 145.0 and angle_jkl < 145.0:
                                    dihed_list.append([i, j, k, l, angle, forc_const])

        new_dihed_list = dihed_list
        if len(dihed_list) > 0:
            for dl in dihed_list:
                for di in dihed_list[1:]:
                    if dl != di:
                        # Check if beads are repeating
                        if  dl[0:2]==di[0:2] or dl[0:2]==di[2:4] or dl[2:4]==di[0:2] or dl[2:4]==di[2:4] or sorted(dl[:4])==sorted(di[:4]):
                            new_dihed_list.remove(di)

    return new_dihed_list, num_ar


def format_topology_angles(angle_list, beadtypes):
    """Format angles into ITP text."""
    text = ""
    
    ### AutoM3 ###
    beadlist = []
    for bead in beadtypes:
        if not bead.startswith('T') and not bead.startswith('S'):
            beadlist.append('R')
        else:
            beadlist.append(bead[0])

    if len(angle_list) > 0:
        text = text + "\n[angles]\n"
        text = text + ";  i  j  k    funct  angle  force.c.\n"
        for a in angle_list:
            force = read_params(a[4], beadlist[a[0]] + "-" + beadlist[a[1]] + "-" + beadlist[a[2]])
            if force is None:
                force = a[5]
            text = text + "  {:2} {:2} {:2}       {:2}    {:<5.1f}  {:5.1f}\n".format(
                a[0] + 1, a[1] + 1, a[2] + 1, a[3], a[4], force
            )
    return text


def format_topology_dihedrals(dihed_list, num_ar, cgbeads, ringatoms, cgbead_coords, beadtypes):
    """Format dihedrals into ITP text."""
    text = ""
    
    ### AutoM3 ###
    bead_in_ring_coords = {}
    for nb, bead_nb in enumerate(cgbeads):
        for ring in ringatoms:
            if bead_nb in ring:
                bead_in_ring_coords[nb] = cgbead_coords[nb]
    
    beadlist = []
    for bead in beadtypes:
        if not bead.startswith('T') and not bead.startswith('S'):
            beadlist.append('R')
        else:
            beadlist.append(bead[0])
    
    if len(dihed_list) > 0:
        text = text + "\n[dihedrals]\n"
        text = text + ";  i  j  k  l  funct  angle  force.c.\n"
        
        for d in dihed_list:
            ### AutoM3 ###
            force = read_params(d[4], beadlist[d[0]] + "-" + beadlist[d[1]] + "-" + beadlist[d[2]] + "-" + beadlist[d[3]])
            if num_ar > 0 and (d[0] or d[1] or d[2] or d[3] not in bead_in_ring_coords.keys()) and force is not None:
                force = force / 2  # for dihedral between cycle-bead and non-cycled bead: decrease of force
            if force is None:
                force = d[5]
            text = (
                text
                + "  {:2} {:2} {:2} {:2}    2    {:<5.1f}  {:5.1f}\n".format(
                    d[0] + 1, d[1] + 1, d[2] + 1, d[3] + 1, d[4], force
                )
            )
    return text


def format_topology_to_itp(topo, trial=False):
    """
    Format complete Topology object to ITP file text.
    
    Args:
        topo: Topology object
        trial: Whether this is a trial run
        
    Returns:
        str: Complete ITP file content
    """
    text = ""
    
    # Header
    text += "; GENERATED WITH Auto_Martini M3FF for {}\n".format(topo.molname)
    text += (
        "; Developed by: Kiran Kanekal, Tristan Bereau, and Andrew Abi-Mansour\n"
        + "; updated to Martini 3 force field by Magdalena Szczuka\n"
        + "; supervised by Matthieu Chavent, Pierre Poulain and Paulo C. T. Souza \n"
        + "; SMILES code : " + topo.mol_smi + "\n\n"
        + "\n[moleculetype]\n"
        + "; molname       nrexcl\n"
        + "  {:5s}         {:d}\n\n".format(topo.molname, topo.nrexcl)
        + "[atoms]\n"
        + "; id      type   resnr residue atom    cgnr    charge  mass ;  smiles    ; atom_num\n"
    )
    
    # Atoms
    text += format_topology_atoms(topo.atoms, trial)
    
    # Bonds and constraints
    text += format_topology_bonds(topo.bonds, topo.constraints, topo.beadtypes, [], trial)
    
    # Angles
    text += format_topology_angles(topo.angles, topo.beadtypes)
    
    # Dihedrals
    if topo.dihedrals:
        # Need cgbeads and ringatoms for formatting dihedrals
        # For now, skip num_ar calculation - will need to refactor if needed
        text += "\n[dihedrals]\n;  i  j  k  l  funct  angle  force.c.\n"
        beadlist = []
        for bead in topo.beadtypes:
            if not bead.startswith('T') and not bead.startswith('S'):
                beadlist.append('R')
            else:
                beadlist.append(bead[0])
        
        for d in topo.dihedrals:
            force = read_params(d[4], beadlist[d[0]] + "-" + beadlist[d[1]] + "-" + beadlist[d[2]] + "-" + beadlist[d[3]])
            if force is None:
                force = d[5]
            text += "  {:2} {:2} {:2} {:2}    2    {:<5.1f}  {:5.1f}\n".format(
                d[0] + 1, d[1] + 1, d[2] + 1, d[3] + 1, d[4], force
            )
    
    return text
