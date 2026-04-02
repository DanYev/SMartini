SMartini
============

## What is SMartini?

A pipeline for generating and iteratively refining Martini 3 small-molecule topologies from atomistic simulation data.

CG topology generation is now much faster: initial topologies are generated in seconds, even for molecules like chlorophyll (~64 heavy atoms).

SMartini runs an AA→CG fitting loop: initial mapping, atomistic sampling, Boltzmann inversion of bonded terms, CG simulation, and parameter updates from CG-vs-AA distribution mismatch.


## How it works (1-5 scripts)

1. **`1_gen_cg_topo.py` — Initial CG topology generation**
	- Reads the ligand (SDF/SMILES), builds an RDKit molecule, and runs `AutoMartini.solver.Cg_molecule`.
	- Writes initial outputs in the molecule directory: `*_initial.itp`, CG `*.pdb`, and `*.map`.

2. **`2_aa_md.py` — Atomistic reference simulation**
	- Generates ligand AA force-field parameters with OpenFF (SMIRNOFF), builds a solvated OpenMM system, then runs minimization/heating/equilibration/production MD.
	- Exports sampled AA trajectory files (`topology.pdb`, `samples.xtc`) for fitting.

3. **`3_boltz_inv.py` — First bonded-parameter fit from AA data**
	- Reads the initial ITP + AA trajectory, computes internal coordinates, and applies Boltzmann inversion.
	- Fits bonds, angles, and dihedrals, then writes an updated `molname.itp`.

4. **`4_cg_md.py` — CG simulation with current topology**
	- Builds/solvates a Martini CG system in GROMACS and runs EM + production MD.
	- Exports CG sampled trajectory (`topology.pdb`, `samples.xtc`) for comparison with AA.

5. **`5_cgmd_upd.py` — Iterative topology refinement**
	- Compares CG and AA internal-coordinate distributions.
	- Updates bonded parameters (bonds/constraints/angles/dihedrals) and overwrites `molname.itp` with refined values.

## Fitting cycle (`fitting_cycle.sh`)

`fitting_cycle.sh` automates one refinement loop:
- run CG MD (`4_cg_md.py`),
- update topology from CG-vs-AA mismatch (`5_cgmd_upd.py`),
- rerun CG MD with updated parameters,
- generate overlay plots (`5_cgmd_upd.py plot`).

This cycle is repeated until CG distributions reasonably match AA references.

## License and upstream attribution

This project uses and adapts parts of AutoMartini M3:
- https://github.com/Martini-Force-Field-Initiative/Automartini_M3/

The project is distributed under **GNU GPL v2.0 (or later)** terms, consistent with upstream usage of GPL-licensed code.

## Developers
* Danis Yangaliev