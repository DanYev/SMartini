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
from pathlib import Path
import logging
from sys import exit
from .common import *


logger = logging.getLogger(__name__)


def output_gro(sites, site_names, molname):
    """Output GRO file of CG structure"""
    logger.info("Writing GRO file")
    num_beads = len(sites)
    gro_out = ""
    if len(sites) != len(site_names):
        logger.warning("Error. Incompatible number of beads and bead names.")
        exit(1)
    gro_out += "{:s} generated from auto_martiniM3\n".format(molname)
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


def output_map(sites, site_names, molname):
    """Output MAP file of CG structure"""
    logger.info("Writing MAP file")
    num_beads = len(sites)
    gro_out = ""
    gro_out += "{:s} generated from auto_martiniM3\n".format(molname)
    gro_out += "{:5d}\n".format(num_beads)
    if len(molname)>4:molname=molname[:4]
    for i in range(num_beads):
        gro_out += "{:5d}{:<6s} {:3s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(
            1, #was i +1, but this is GRO file for one molecule, so all beads should be a part of the same molecule
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
    logger.info("Writing PDB file")
    num_beads = len(sites)
    pdb_out = ""
    
    if len(sites) != len(site_names):
        logger.warning("Error. Incompatible number of beads and bead names.")
        exit(1)
    
    # Write header
    pdb_out += "REMARK   Generated from auto_martiniM3\n"
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


def _parse_itp_atoms_mapping(itp_text: str):
    """Parse AutoMartini-generated `[atoms]` block and extract bead->atom mapping.

    Expected input format (as produced by `topology.format_topology_header()` + `topology.format_topology_atoms()`)
    includes per-atom comment fragments like:

        ; atoms: P0, O5, O14, ...

    Returns
    -------
    (molname, bead_atomnames, chiral_blocks)
      - molname: str or None
      - bead_atomnames: dict[str, list[str]] mapping bead name (e.g. "D01") -> list of atom labels (e.g. ["P0","O5"]).
      - chiral_blocks: list[list[str]] raw lines from any `[ chiral ]` sections (optional; may be empty)
    """

    molname = None
    bead_atomnames: dict[str, list[str]] = {}

    lines = itp_text.splitlines()
    # Molname from [moleculetype]
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "[moleculetype]":
            # find first non-empty, non-comment line after it
            for j in range(i + 1, min(i + 20, len(lines))):
                s = lines[j].strip()
                if not s or s.startswith(";"):
                    continue
                molname = s.split()[0]
                break
            break

    in_atoms = False
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            in_atoms = s.lower() == "[atoms]"
            continue
        if not in_atoms:
            continue
        if s.startswith(";"):
            continue

        # Split into pre-comment columns and optional comment
        pre, _, comment = ln.partition(";")
        cols = pre.split()
        if len(cols) < 5:
            continue
        bead_name = cols[4]

        # Find 'atoms:' segment in comment
        if "atoms:" not in comment:
            continue
        after = comment.split("atoms:", 1)[1]
        # Trim at next ';' if present (some lines contain "; ALOGPS..." etc)
        after = after.split(";", 1)[0]

        atom_labels = []
        for tok in after.replace(",", " ").split():
            tok = tok.strip()
            # Keep things like P0, O14, Cl12, etc.
            if not tok:
                continue
            # Defensive: avoid trailing punctuation
            tok = tok.strip(",")
            atom_labels.append(tok)
        if not atom_labels:
            continue
        bead_atomnames[bead_name] = atom_labels

    # Collect any `[ chiral ]` blocks as raw text, if present.
    chiral_blocks: list[list[str]] = []
    in_chiral = False
    current: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.lower() == "[ chiral ]":
            if current:
                chiral_blocks.append(current)
                current = []
            in_chiral = True
            continue
        if s.startswith("[") and s.endswith("]"):
            if in_chiral and current:
                chiral_blocks.append(current)
                current = []
            in_chiral = False
            continue
        if in_chiral:
            if s and not s.startswith(";"):
                current.append(ln.rstrip())
    if in_chiral and current:
        chiral_blocks.append(current)

    return molname, bead_atomnames, chiral_blocks


def make_map_from_itp(itp_file: str, map_file:str, resname: str | None = None, to_ff: str = "martini3001"):
    """Create a `.map` file (similar to `gln.amber.map`) from an AutoMartini `.itp`.

    This is intentionally minimal: we only require that the `.itp` contains an `[atoms]`
    section whose lines include an `atoms:` list in the comment (as produced by AutoMartini M3).

    Parameters
    ----------
    itp_text:
        Full `.itp` file content.
    resname:
        Optional residue name placed in `[ molecule ]`. If omitted, uses molname parsed from
        `[moleculetype]` (fallback: "MOL").
    from_ff / to_ff:
        Strings for the `[from]` and `[to]` blocks.
    """
    itp_text = Path(itp_file).read_text()

    molname, bead_atomnames, chiral_blocks = _parse_itp_atoms_mapping(itp_text)
    if not bead_atomnames:
        raise ValueError("Could not find any bead `atoms:` annotations in the `[atoms]` section.")

    if resname is None:
        resname = molname or "MOL"

    # The `.map` format expects a `[ martini ]` list of bead names.
    # Keep ITP ordering: atoms section is bead id order => bead name order should be stable.
    # We reconstruct ordering by scanning the ITP again.
    bead_order: list[str] = []
    for ln in itp_text.splitlines():
        pre, _, _comment = ln.partition(";")
        cols = pre.split()
        if len(cols) >= 5 and cols[0].isdigit():
            bead = cols[4]
            if bead in bead_atomnames and bead not in bead_order:
                bead_order.append(bead)
    if not bead_order:
        bead_order = list(bead_atomnames.keys())

    out = ""
    out += "[ molecule ]\n"
    out += f"{resname}\n\n"
    out += "[from]\n"
    out += "amber charmm\n\n"
    out += "[to]\n"
    out += f"{to_ff}\n\n"
    out += "[ martini ]\n"
    out += "  " + " ".join(bead_order) + "\n\n"
    out += "[ mapping ]\n"
    out += "amber charmm\n\n"
    out += "[ atoms ]\n"
    num = 1
    for bead in bead_order:
        for atom in bead_atomnames.get(bead, []):
            out += f"{num:>6d}  {atom:>6s}  {bead:>6s}\n"
            num += 1

    if chiral_blocks:
        # Preserve chiral info if present (can be used by martinize-type tools).
        for block in chiral_blocks:
            out += "\n[ chiral ]\n"
            for ln in block:
                out += f"{ln}\n"

    out += "\n"
    with open(Path(map_file), "w") as f:
        f.write(out)
