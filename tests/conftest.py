"""Shared fixtures for the SMartini test suite.

The CG topology pipeline (``1_gen_cg_topo.py``) is configured through
environment variables, so each reference molecule is run once per session as
an isolated subprocess.  All generated files are kept under ``tests/output``
(never written back into ``tests/refs``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from itp_utils import (
    OUTPUT_DIR,
    PIPELINE_SCRIPT,
    REFS_DIR,
    REPO_ROOT,
    discover_molecules,
)

MOLECULES = discover_molecules()


@pytest.fixture(scope="session")
def base_config() -> Path:
    """Write a package-level config that points ``systems_dir`` into tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logp_file = REFS_DIR / "logP_smi_test.dat"
    path = OUTPUT_DIR / "_base_config.yaml"
    path.write_text(
        f"systems_dir: {OUTPUT_DIR.as_posix()}\n"
        f"logp_file: {logp_file.as_posix()}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="session")
def run_pipeline(base_config: Path):
    """Return a cached ``molname -> output directory`` pipeline runner."""
    cache: dict[str, Path] = {}

    def _run(molname: str) -> Path:
        if molname in cache:
            return cache[molname]

        ref_dir = REFS_DIR / molname
        out_dir = OUTPUT_DIR / molname

        # Start from a clean per-molecule directory so stale outputs from an
        # earlier session can never make a broken run look successful.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        # 1_gen_cg_topo.py expects <systems_dir>/<molname>/<molname>.sdf.
        sdf = ref_dir / f"{molname}.sdf"
        shutil.copy2(sdf, out_dir / sdf.name)

        env = os.environ.copy()
        env.update(
            {
                "SM_MOLNAME": molname,
                "SM_CONFIG_YML": str(ref_dir / "config.yml"),
                "SM_BASE_CONFIG": str(base_config),
            }
        )

        completed = subprocess.run(
            [sys.executable, str(PIPELINE_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        (out_dir / "pipeline.log").write_text(
            f"$ {sys.executable} {PIPELINE_SCRIPT}\n"
            f"SM_MOLNAME={molname}\n"
            f"SM_CONFIG_YML={ref_dir / 'config.yml'}\n"
            f"SM_BASE_CONFIG={base_config}\n\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"1_gen_cg_topo.py failed for {molname} "
                f"(exit {completed.returncode}).\n"
                f"--- stdout ---\n{completed.stdout}\n"
                f"--- stderr ---\n{completed.stderr}"
            )

        generated_itp = out_dir / f"{molname}_initial.itp"
        if not generated_itp.is_file():
            raise AssertionError(
                f"1_gen_cg_topo.py did not write {generated_itp}.\n"
                f"--- stdout ---\n{completed.stdout}\n"
                f"--- stderr ---\n{completed.stderr}"
            )

        cache[molname] = out_dir
        return out_dir

    return _run
