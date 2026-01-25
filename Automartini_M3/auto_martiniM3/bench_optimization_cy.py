"""Micro-benchmark for acceptable-trials filtering.

Runs the pure-Python filter vs the Cython NumPy fast path.

This is intentionally synthetic (no RDKit needed) so it's fast to run and
focused on the hot loops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchConfig:
    n_trials: int = 200_000
    n_beads: int = 8
    n_atoms: int = 120
    n_bonds: int = 140
    n_rings: int = 6
    seed: int = 0


def _make_inputs(cfg: BenchConfig):
    rng = np.random.default_rng(cfg.seed)

    # Trials: (n_trials, n_beads)
    seq_one_beads = rng.integers(0, cfg.n_atoms, size=(cfg.n_trials, cfg.n_beads), dtype=np.int32)

    # Bonds: (n_bonds, 2)
    bonds = rng.integers(0, cfg.n_atoms, size=(cfg.n_bonds, 2), dtype=np.int32)
    # Avoid self-bonds
    mask = bonds[:, 0] == bonds[:, 1]
    if mask.any():
        bonds[mask, 1] = (bonds[mask, 1] + 1) % cfg.n_atoms

    # ring_id_of_atom: (n_atoms,)
    ring_id_of_atom = np.full(cfg.n_atoms, -1, dtype=np.int32)
    # Mark some atoms as belonging to rings
    ring_atoms = rng.integers(0, cfg.n_atoms, size=(cfg.n_rings, cfg.n_atoms // 10), dtype=np.int32)
    for rid in range(cfg.n_rings):
        ring_id_of_atom[ring_atoms[rid]] = rid

    # Also produce Python list-of-lists ring_atoms for the pure python path
    ring_atoms_py = [list(map(int, ring_atoms[rid])) for rid in range(cfg.n_rings)]
    bonds_py = [list(map(int, b)) for b in bonds]

    return seq_one_beads, bonds, ring_id_of_atom, ring_atoms_py, bonds_py


def _time_call(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def main():
    from auto_martiniM3 import optimization

    cfg = BenchConfig()
    seq_one_beads, bonds, ring_id, ring_atoms_py, bonds_py = _make_inputs(cfg)

    # Pure python expects:
    # find_acceptable_trials(seq_one_beads, molecule, list_heavy_atoms, heavyatom_coords, ring_atoms, list_bonds, allatom_coords, force_map)
    # We can pass None / empty for unused args.
    py_args = (
        seq_one_beads,
        None,
        list(range(cfg.n_atoms)),
        None,
        ring_atoms_py,
        bonds_py,
        None,
        False,
    )

    print(f"Trials: {cfg.n_trials}  beads/trial: {cfg.n_beads}  atoms: {cfg.n_atoms}  bonds: {cfg.n_bonds}")

    out_py, dt_py = _time_call(optimization.find_acceptable_trials, *py_args)
    print(f"python: {len(out_py)} accepted in {dt_py:.3f}s")

    if getattr(optimization, "find_acceptable_trials_cy_np", None) is None:
        print("cython: not available (extension not built)")
        return

    out_cy, dt_cy = _time_call(optimization.find_acceptable_trials_cy_np, seq_one_beads, bonds, ring_id)
    print(f"cython: {len(out_cy)} accepted in {dt_cy:.3f}s")

    # Sanity: compare counts (not necessarily same exact trials if python path differs)
    print(f"speedup: {dt_py / dt_cy:.2f}x")


if __name__ == "__main__":
    main()
