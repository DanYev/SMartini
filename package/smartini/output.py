from pathlib import Path
import re
import logging
from sys import exit

logger = logging.getLogger(__name__)


def output_gro(sites, site_names, molname):
    """Output GRO file of CG structure"""
    logger.info("Writing GRO file")
    num_beads = len(sites)
    gro_out = ""
    if len(sites) != len(site_names):
        logger.warning("Error. Incompatible number of beads and bead names.")
        exit(1)
    gro_out += "{:s} generated from smartini\n".format(molname)
    gro_out += "{:5d}\n".format(num_beads)
    if len(molname)>4:molname=molname[:4]
    for i in range(num_beads):
        gro_out += "{:5d}{:<6s} {:3s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(
            1, # was i +1, but this is GRO file for one molecule, so all beads should be a part of the same molecule
            molname,
            site_names[i],
            i + 1,
            sites[i][0] / 10.0,
            sites[i][1] / 10.0,
            sites[i][2] / 10.0,
        )
    gro_out += "{:10.5f}{:10.5f}{:10.5f}\n".format(10.0, 10.0, 10.0)
    return gro_out


def output_pdb(sites, site_names, molname, bonds=None, constraints=None):
    """Output PDB file of CG structure with CONECT records
    
    Parameters
    ----------
    sites : array-like
        Coordinates of CG beads in Angstroms
    site_names : list
        Names of CG beads
    molname : str
        Molecule name
    bonds : list, optional
        List of bonds as [i, j, dist] where i, j are bead indices
    constraints : list, optional
        List of constraints as [i, j, dist] where i, j are bead indices
        
    Returns
    -------
    str
        PDB format string with ATOM and CONECT records
    """
    logger.debug("Writing PDB file")
    num_beads = len(sites)
    pdb_out = ""
    
    if len(sites) != len(site_names):
        logger.warning("Error. Incompatible number of beads and bead names.")
        exit(1)
    
    # Write header
    pdb_out += "REMARK   Generated from smartini\n"
    pdb_out += f"REMARK   Molecule: {molname}\n"
    
    # Truncate molname if needed for PDB format (3-letter residue name)
    resname = molname[:3] if len(molname) > 3 else molname
    
    # Write ATOM records
    # PDB format: ATOM serial name resName chainID resSeq X Y Z occupancy tempFactor element
    for i in range(num_beads):
        atom_name = site_names[i][:4] if len(site_names[i]) <= 4 else site_names[i][:4]
        pdb_out += "ATOM  {:5d} {:^4s} {:3s} A{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
            i + 1,                          # serial
            atom_name,                       # atom name (centered, max 4 chars)
            resname,                         # residue name  
            1,                               # residue sequence number
            sites[i][0],                    # X coordinate 
            sites[i][1],                    # Y coordinate
            sites[i][2],                    # Z coordinate
            1.00,                            # occupancy
            0.00,                            # temperature factor
            ""                               # element (left blank for CG beads)
        )
    
    pdb_out += "TER\n"
    
    # Write CONECT records from bonds and constraints
    # Build connectivity dictionary
    connections = {}
    
    if bonds:
        for bond in bonds:
            i, j = int(bond[0]), int(bond[1])
            if i not in connections:
                connections[i] = []
            if j not in connections:
                connections[j] = []
            if j not in connections[i]:
                connections[i].append(j)
            if i not in connections[j]:
                connections[j].append(i)
    
    if constraints:
        for constraint in constraints:
            i, j = int(constraint[0]), int(constraint[1])
            if i not in connections:
                connections[i] = []
            if j not in connections:
                connections[j] = []
            if j not in connections[i]:
                connections[i].append(j)
            if i not in connections[j]:
                connections[j].append(i)
    
    # Write CONECT records (PDB atom indices are 1-based)
    for atom_idx in sorted(connections.keys()):
        bonded = sorted(connections[atom_idx])
        # CONECT records can have multiple bonded atoms on one line
        # Format: CONECT serial serial serial serial...
        conect_line = "CONECT{:5d}".format(atom_idx + 1)
        for bonded_idx in bonded:
            conect_line += "{:5d}".format(bonded_idx + 1)
        pdb_out += conect_line + "\n"
    
    pdb_out += "END\n"
    
    return pdb_out


def output_map(topology, map_file: str, to_ff: str = "martini3001"):
    """Create a `.map` file from a Topology instance.

    Parameters
    ----------
    topology:
        A Topology object instance.
    map_file:
        Path to the output .map file.
    to_ff:
        String for the `[to]` block.
    """
    if not topology.atoms:
        raise ValueError("Topology has no atoms to map.")

    resname = topology.molname or "MOL"
    if len(resname) > 4:
        resname = resname[:4]

    bead_order = [atom['atom'] for atom in topology.atoms]
    bead_atomnames = {atom['atom']: atom['atomnames'] for atom in topology.atoms}

    out = ""
    out += "[ molecule ]\n"
    out += f"{resname}\n\n"
    out += "[ from ]\n"
    out += "amber charmm\n\n"
    out += "[ to ]\n"
    out += f"{to_ff}\n\n"
    out += "[ martini ]\n"
    out += "  " + " ".join(bead_order) + "\n\n"
    out += "[ mapping ]\n"
    out += "amber charmm\n\n"
    out += "[ atoms ]\n"
    num = 1
    for bead in bead_order:
        atomnames = bead_atomnames.get(bead, "")

        if isinstance(atomnames, str):
            atom_list = [name for name in re.split(r"[\s,]+", atomnames.strip()) if name]
        elif isinstance(atomnames, (list, tuple)):
            atom_list = [str(name).strip() for name in atomnames if str(name).strip()]
        else:
            atom_list = []

        for atom in atom_list:
            out += f"{num:>6d}{atom:>8s}{bead:>8s}\n"
            num += 1
    
    # Chiral information is not directly available in the same way.
    # This part is omitted as it was parsed from ITP comments.

    out += "\n"
    with open(Path(map_file), "w") as f:
        f.write(out)
