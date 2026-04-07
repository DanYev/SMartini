SMartini
========

## What is SMartini?

A pipeline for generating and iteratively refining Martini 3 small-molecule topologies from atomistic simulation data.

The determination of non-bonded terms (bead types) is based on AutoMartini M3. However, CG topology generation is now much faster: initial topologies are generated in seconds, even for larger molecules such as chlorophyll (~64 heavy atoms). Bonded terms are determined from atomistic reference MD simulations.

SMartini runs an AA→CG fitting pipeline: initial mapping, atomistic sampling, Boltzmann inversion of bonded terms from AA simulations, CG simulation, and parameter updates based on CG-vs-AA distribution mismatch.

## Installation
```bash
git clone https://github.com/DanYev/LigPar
cd LigPar
conda env create --file environment.yml
source activate smartini
```

## How it works
A directory `<CFG.sysdir>/<MOLNAME>` is expected, containing either:
- an `.sdf` file (if starting from a structure), or
- a `config.yml` file with `smiles: <SMILES>`.

The default configuration file is `config.py`, and parameters can optionally be overridden in `config.yml`. Examples are available in `examples`.

Run:

```bash
bash martinize_ligand.sh <MOLNAME>
```

This runs the following scripts:

- `1_gen_cg_topo.py` — Initial CG topology generation
- `2_aa_md.py` — Atomistic reference simulation
- `3_boltz_inv.py` — Initial bonded-parameter fit from AA data

Then, the bonded parameters are updated in iterative cycles (2 by default):

- `4_cg_md.py` — CG simulation with the current topology
- `5_cgmd_upd.py` — Bonded parameter update based on CG MD

This cycle is repeated until CG distributions reasonably match AA references.

## License and upstream attribution

This project uses and adapts parts of AutoMartini M3:
- https://github.com/Martini-Force-Field-Initiative/Automartini_M3/

The project is distributed under **GNU GPL v2.0 (or later)** terms, consistent with upstream usage of GPL-licensed code.

## Developers
* Danis Yangaliev