SMart
============

## What is SMart?

A toolkit for topology parameters generation for small molecules for Martini 3 force field. Bead type determination is the same as in Automartini M3 (https://github.com/Martini-Force-Field-Initiative/Automartini_M3/), while generation of the topology and bonded parameters is brand new and is based on fitting to the all-atom MD data.

## Developers
* Danis Yangaliev

## How it works (mapping + solver)

1. **Fragment-based mapping (`partitioning.py`)**
	- The molecule is converted to a heavy-atom graph (hydrogens are ignored at this stage).
	- The graph is split into overlapping fragments (ring-centered and linear/branch fragments).
	- For each fragment, possible bead anchors are enumerated and expanded into candidate local mappings.
	- Fragment mappings are stitched across overlap atoms, then filtered (no single-atom beads, bead-size limits, ring consistency) and ranked.
	- The result is a best-first list of full-molecule candidate mappings.

2. **Candidate evaluation (`solver.py`)**
	- `Cg_molecule` embeds/optimizes the AA structure (RDKit MMFF), builds molecular features, and requests mapping candidates from `partitioning.generate_mappings`.
	- Candidates are tried in ranked order; optional ring symmetrization and user bead constraints are applied.
	- For each candidate, bead types are assigned, hydrogen atoms are reattached to create an all-atom bead mapping, and bead coordinates are computed.
	- A Martini topology is then built (atoms and bonded terms). The first candidate that passes all checks is accepted and used for output.

In short: **partitioning generates chemically sensible mapping candidates efficiently, and the solver validates them end-to-end until it finds the first complete, consistent Martini 3 model.**