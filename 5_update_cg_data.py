import logging
from pathlib import Path

import AutoMartini as am

from lpmath import read_cg_trajectory, read_cog_trajectory, calculate_internal_coordinates
from plots import plot_internal_coordinates_overlay
from cg_refine import refine_topology_from_cg_vs_aa


logging.basicConfig(
	level=logging.INFO,
	format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
	force=True,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
	molname = "ANP"
	wdir = Path("systems") / molname

	itp_updated = wdir / "mapping" / f"{molname}_updated.itp"
	itp_default = wdir / "mapping" / f"{molname}.itp"
	in_itp = itp_updated if itp_updated.exists() else itp_default
	logger.info("Reading topology from %s", in_itp)
	topo = am.topology.read_itp(str(in_itp))
	unique_dihedrals = {(int(d[0]), int(d[1]), int(d[2]), int(d[3])) for d in topo.dihedrals}
	logger.info(
		"Loaded %s dihedral terms across %s unique torsions",
		len(topo.dihedrals),
		len(unique_dihedrals),
	)

	aa_dir = wdir / "aa_md"
	aa_pdb = aa_dir / "md.pdb"
	aa_xtc = aa_dir / "md.xtc"

	cg_dir = wdir / "cg_md"
	cg_pdb = cg_dir / "mdrun" / "topology.pdb"
	cg_xtc = cg_dir / "mdrun" / "samples.xtc"

	logger.info("Reading AA trajectory from %s", aa_dir)
	aa_traj = read_cog_trajectory(aa_pdb, aa_xtc, topo.partitioning)
	aa_internal = calculate_internal_coordinates(aa_traj, topo)

	logger.info("Reading CG trajectory from %s", cg_dir)
	cg_traj = read_cg_trajectory(cg_pdb, cg_xtc, start=0, stop=2000) # start and step can be adjusted to speed up processing for long trajectories
	cg_internal = calculate_internal_coordinates(cg_traj, topo)

	plot_internal_coordinates_overlay(
		aa_internal,
		cg_internal,
		topo,
		output_file=wdir / "png" / "cg_vs_aa.png",
	)

	# Refine the CG topology based on CG-vs-AA distribution mismatch.
	# Writes a new file and leaves the original ITP unchanged.
	# out_refined_itp = wdir / "mapping" / f"{molname}_cgrefined.itp"
	out_refined_itp = itp_updated
	# refine_topology_from_cg_vs_aa(topo, aa_internal, cg_internal, out_refined_itp)
