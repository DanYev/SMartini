from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def rebuild_partitioning_from_sdf(
    itp_partitioning: Dict[int, int],
    sdf_file: str | Path,
    *,
    require_all_heavy: bool = True,
) -> Dict[int, int]:
    """Rebuild an AA-style partitioning (including hydrogens) from an SDF.

    This is meant to fix atom-index mismatches between:
    - the partitioning embedded in the CG .itp header (typically heavy-atom based)
    - the atom order used by the AA topology/trajectory (derived from the same SDF)

    Strategy
    --------
    1) Load the molecule from the SDF using OpenFF (same stack used in AA pipeline).
    2) Ensure explicit hydrogens exist (OpenFF may add them).
    3) For heavy atoms: keep bead assignment from itp_partitioning[heavy_idx].
    4) For hydrogens: assign to the bead of their (unique) bonded heavy atom.
    """
    sdf_file = Path(sdf_file)
    itp_partitioning = {int(k): int(v) for k, v in dict(itp_partitioning).items()}

    try:
        from openff.toolkit import Molecule
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "OpenFF Toolkit is required to rebuild AA partitioning from SDF. "
            "Install it (and its RDKit backend) in your environment."
        ) from e

    offmol = Molecule.from_file(str(sdf_file))
    # Ensure explicit H atoms exist, matching the AA pipeline expectation.
    add_h = getattr(offmol, "add_hydrogens", None)
    if callable(add_h):
        try:
            add_h()
        except Exception:
            # If H addition fails, we continue (may already be explicit).
            pass

    atoms = list(offmol.atoms)
    bonded = {i: set() for i in range(len(atoms))}
    for b in offmol.bonds:
        i = int(b.atom1_index)
        j = int(b.atom2_index)
        bonded[i].add(j)
        bonded[j].add(i)

    heavy_indices = [i for i, a in enumerate(atoms) if int(a.atomic_number) != 1]
    missing_heavy = [i for i in heavy_indices if i not in itp_partitioning]
    if missing_heavy:
        msg = f"ITP partitioning missing {len(missing_heavy)} heavy atoms (e.g. {missing_heavy[:10]})."
        if require_all_heavy:
            raise ValueError(msg)
        logger.warning(msg)

    aa_partitioning: Dict[int, int] = {}
    # Heavy atoms: copy from ITP
    for i in heavy_indices:
        if i in itp_partitioning:
            aa_partitioning[i] = int(itp_partitioning[i])

    # Hydrogens: assign using bonded heavy atom
    for i, a in enumerate(atoms):
        if int(a.atomic_number) != 1:
            continue

        heavy_neighbors = [
            j for j in bonded.get(i, []) if int(atoms[j].atomic_number) != 1
        ]
        if not heavy_neighbors:
            continue
        if len(heavy_neighbors) > 1:
            # Unusual, but pick the first deterministically.
            heavy_neighbors = sorted(heavy_neighbors)

        heavy_idx = int(heavy_neighbors[0])
        bead = itp_partitioning.get(heavy_idx)
        if bead is None:
            continue
        aa_partitioning[i] = int(bead)

    return dict(sorted(aa_partitioning.items(), key=lambda kv: int(kv[0])))


def patch_topology_partitioning_from_sdf(topo, sdf_file: str | Path):
    """In-place patch: overwrite topo.partitioning with AA-consistent mapping."""
    topo.partitioning = rebuild_partitioning_from_sdf(topo.partitioning, sdf_file)
    return topo
