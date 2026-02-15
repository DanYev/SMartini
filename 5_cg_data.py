import logging
from pathlib import Path

import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import AutoMartini as am


logging.basicConfig(
	level=logging.INFO,
	format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
	force=True,
)
logger = logging.getLogger(__name__)


def read_cog_trajectory(in_pdb, in_xtc, partitioning):
	"""Read AA trajectory and calculate COG trajectory for CG beads.

	Parameters
	----------
	in_pdb : str or Path
		Path to atomistic PDB file
	in_xtc : str or Path
		Path to atomistic XTC trajectory
	partitioning : dict
		Mapping of atom indices to bead indices {atom_idx: bead_idx}

	Returns
	-------
	numpy.ndarray
		CG trajectory array with shape (n_frames, n_beads, 3) in nm
	"""
	logger.info("Reading AA trajectory: %s, %s", in_pdb, in_xtc)

	u = mda.Universe(str(in_pdb), str(in_xtc))
	n_frames = len(u.trajectory)

	n_beads = max(partitioning.values()) + 1
	bead_to_atoms = {i: [] for i in range(n_beads)}
	for atom_idx, bead_idx in partitioning.items():
		bead_to_atoms[bead_idx].append(atom_idx)

	cg_trajectory = np.zeros((n_frames, n_beads, 3))

	for frame_idx, _ in enumerate(u.trajectory):
		for bead_idx in range(n_beads):
			atom_indices = bead_to_atoms[bead_idx]
			if atom_indices:
				positions = u.atoms[atom_indices].positions
				cg_trajectory[frame_idx, bead_idx] = positions.mean(axis=0) / 10.0

	logger.info("COG trajectory computed: %s frames, %s beads", n_frames, n_beads)
	return cg_trajectory[:-1, :, :]


def calculate_internal_coordinates(cg_trajectory, topo):
	"""Calculate internal coordinates (bonds, angles, dihedrals)."""
	n_frames = cg_trajectory.shape[0]
	internal_coords = {}

	for bond in topo.bonds:
		i, j = int(bond[0]), int(bond[1])
		distances = np.zeros(n_frames)
		for frame_idx in range(n_frames):
			pos_i = cg_trajectory[frame_idx, i]
			pos_j = cg_trajectory[frame_idx, j]
			distances[frame_idx] = np.linalg.norm(pos_i - pos_j)
		internal_coords[(i, j, "bond")] = distances

	for constraint in topo.constraints:
		i, j = int(constraint[0]), int(constraint[1])
		distances = np.zeros(n_frames)
		for frame_idx in range(n_frames):
			pos_i = cg_trajectory[frame_idx, i]
			pos_j = cg_trajectory[frame_idx, j]
			distances[frame_idx] = np.linalg.norm(pos_i - pos_j)
		internal_coords[(i, j, "constraint")] = distances

	for angle in topo.angles:
		i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
		angles = np.zeros(n_frames)
		for frame_idx in range(n_frames):
			pos_i = cg_trajectory[frame_idx, i]
			pos_j = cg_trajectory[frame_idx, j]
			pos_k = cg_trajectory[frame_idx, k]
			v1 = pos_i - pos_j
			v2 = pos_k - pos_j
			cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
			cos_angle = np.clip(cos_angle, -1.0, 1.0)
			angles[frame_idx] = np.degrees(np.arccos(cos_angle))
		internal_coords[(i, j, k, "angle")] = angles
        
	# Calculate dihedrals
	for dihedral in topo.dihedrals:
		i, j, k, l = int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3])
		dihedrals = np.zeros(n_frames)
		for frame_idx in range(n_frames):
			pos_i = cg_trajectory[frame_idx, i]
			pos_j = cg_trajectory[frame_idx, j]
			pos_k = cg_trajectory[frame_idx, k]
			pos_l = cg_trajectory[frame_idx, l]
			b1 = pos_j - pos_i
			b2 = pos_k - pos_j
			b3 = pos_l - pos_k
			n1 = np.cross(b1, b2)
			n2 = np.cross(b2, b3)
			b2_norm = b2 / np.linalg.norm(b2)
			x = np.dot(n1, n2)
			y = np.dot(np.cross(n1, b2_norm), n2)
			dihedrals[frame_idx] = np.degrees(np.arctan2(y, x))
		internal_coords[(i, j, k, l, "dihedral")] = dihedrals


	# for dihedral in topo.dihedrals:
	# 	i, j, k, l = int(dihedral[0]), int(dihedral[1]), int(dihedral[2]), int(dihedral[3])
	# 	dihedrals = np.zeros(n_frames)
	# 	for frame_idx in range(n_frames):
	# 		pos_i = cg_trajectory[frame_idx, i]
	# 		pos_j = cg_trajectory[frame_idx, j]
	# 		pos_k = cg_trajectory[frame_idx, k]
	# 		pos_l = cg_trajectory[frame_idx, l]
	# 		b1 = pos_j - pos_i
	# 		b2 = pos_k - pos_j
	# 		b3 = pos_l - pos_k
	# 		n1 = np.cross(b1, b2)
	# 		n2 = np.cross(b2, b3)
	# 		b2_norm = b2 / np.linalg.norm(b2)
	# 		x = np.dot(n1, n2)
	# 		y = np.dot(np.cross(n1, b2_norm), n2)
	# 		dihedrals[frame_idx] = np.degrees(np.arctan2(y, x))
	# 	internal_coords[(i, j, k, l, "dihedral")] = dihedrals

	return internal_coords


def circular_mean_deg(angles):
	angles_rad = np.deg2rad(angles)
	sin_mean = np.mean(np.sin(angles_rad))
	cos_mean = np.mean(np.cos(angles_rad))
	mean_rad = np.arctan2(sin_mean, cos_mean)
	return np.rad2deg(mean_rad)


def wrap_to_180(angles):
	return (angles + 180) % 360 - 180


def read_cg_trajectory(in_pdb, in_xtc):
	"""Read CG trajectory and return positions in nm.

	Parameters
	----------
	in_pdb : str or Path
		Path to CG PDB file
	in_xtc : str or Path
		Path to CG XTC trajectory

	Returns
	-------
	numpy.ndarray
		CG trajectory array with shape (n_frames, n_beads, 3) in nm
	"""
	logger.info("Reading CG trajectory: %s, %s", in_pdb, in_xtc)
	u = mda.Universe(str(in_pdb), str(in_xtc))
	n_frames = len(u.trajectory)
	n_beads = len(u.atoms)
	cg_trajectory = np.zeros((n_frames, n_beads, 3))

	for frame_idx, _ in enumerate(u.trajectory):
		cg_trajectory[frame_idx] = u.atoms.positions / 10.0

	logger.info("Loaded CG trajectory: %s frames, %s beads", n_frames, n_beads)
	return cg_trajectory


def plot_internal_coordinates_overlay(aa_coords, cg_coords, topo, output_file=None):
	"""Plot AA and CG histograms for bonds, angles, and dihedrals."""
	bonds_aa = {k: v for k, v in aa_coords.items() if k[-1] in ["bond", "constraint"]}
	angles_aa = {k: v for k, v in aa_coords.items() if k[-1] == "angle"}
	dihedrals_aa = {k: v for k, v in aa_coords.items() if k[-1] == "dihedral"}

	bonds_cg = {k: v for k, v in cg_coords.items() if k[-1] in ["bond", "constraint"]}
	angles_cg = {k: v for k, v in cg_coords.items() if k[-1] == "angle"}
	dihedrals_cg = {k: v for k, v in cg_coords.items() if k[-1] == "dihedral"}

	if bonds_aa or bonds_cg:
		_plot_bonds_overlay(bonds_aa, bonds_cg, topo, output_file)
	if angles_aa or angles_cg:
		_plot_angles_overlay(angles_aa, angles_cg, topo, output_file)
	if dihedrals_aa or dihedrals_cg:
		_plot_dihedrals_overlay(dihedrals_aa, dihedrals_cg, topo, output_file)


def _resolve_keys(bonds_aa, bonds_cg, topo):
	keys = []
	for bond in topo.bonds:
		keys.append((int(bond[0]), int(bond[1]), "bond"))
	for constraint in topo.constraints:
		keys.append((int(constraint[0]), int(constraint[1]), "constraint"))
	if not keys:
		keys = list(set(bonds_aa.keys()) | set(bonds_cg.keys()))
	return keys


def _plot_bonds_overlay(bonds_aa, bonds_cg, topo, output_file):
	logger.info("Plotting %s bonds/constraints", len(set(bonds_aa) | set(bonds_cg)))

	bond_ref = {(int(b[0]), int(b[1])): b[3] for b in topo.bonds}
	constraint_ref = {(int(c[0]), int(c[1])): c[3] for c in topo.constraints}

	keys = _resolve_keys(bonds_aa, bonds_cg, topo)
	keys = [k for k in keys if k in bonds_aa or k in bonds_cg]
	n_plots = len(keys)
	if n_plots == 0:
		return

	n_cols = min(4, n_plots)
	n_rows = int(np.ceil(n_plots / n_cols))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
	if n_plots == 1:
		axes = [axes]
	else:
		axes = axes.flatten()

	for idx, key in enumerate(keys):
		ax = axes[idx]
		i, j, bond_type = key
		aa_vals = bonds_aa.get(key)
		cg_vals = bonds_cg.get(key)
		bins = _common_bins(aa_vals, cg_vals, bins=30)
		hist_range = _preferred_range(aa_vals, cg_vals)

		_plot_hist_pair(ax, aa_vals, cg_vals, bins=bins, hist_range=hist_range)

		if bond_type == "bond" and (i, j) in bond_ref:
			ref_length = bond_ref[(i, j)]
			ax.axvline(ref_length, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_length:.3f}")
		elif bond_type == "constraint" and (i, j) in constraint_ref:
			ref_length = constraint_ref[(i, j)]
			ax.axvline(ref_length, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_length:.3f}")

		ax.set_xlabel("Distance (nm)", fontsize=9)
		ax.set_title(f"{bond_type.capitalize()}: {i+1}-{j+1}", fontsize=10)
		ax.grid(alpha=0.3)
		ax.set_yticks([])
		ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
		if hist_range is not None:
			ax.set_xlim(hist_range)

		_add_stats_box(ax, aa_vals, cg_vals, value_type="bond")
		ax.legend(fontsize=8)

	for idx in range(n_plots, len(axes)):
		axes[idx].axis("off")

	plt.tight_layout()
	_save_or_show(output_file, "bonds")


def _plot_angles_overlay(angles_aa, angles_cg, topo, output_file):
	logger.info("Plotting %s angles", len(set(angles_aa) | set(angles_cg)))

	angle_ref = {(int(a[0]), int(a[1]), int(a[2])): a[4] for a in topo.angles}
	keys = [(int(a[0]), int(a[1]), int(a[2]), "angle") for a in topo.angles]
	if not keys:
		keys = list(set(angles_aa.keys()) | set(angles_cg.keys()))
	keys = [k for k in keys if k in angles_aa or k in angles_cg]

	n_plots = len(keys)
	if n_plots == 0:
		return

	n_cols = min(4, n_plots)
	n_rows = int(np.ceil(n_plots / n_cols))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
	if n_plots == 1:
		axes = [axes]
	else:
		axes = axes.flatten()

	for idx, key in enumerate(keys):
		ax = axes[idx]
		i, j, k, angle_type = key
		aa_vals = angles_aa.get(key)
		cg_vals = angles_cg.get(key)
		bins = _common_bins(aa_vals, cg_vals, bins=30)
		hist_range = _preferred_range(aa_vals, cg_vals)

		_plot_hist_pair(ax, aa_vals, cg_vals, bins=bins, hist_range=hist_range)

		if (i, j, k) in angle_ref:
			ref_angle = angle_ref[(i, j, k)]
			ax.axvline(ref_angle, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_angle:.1f} deg")

		ax.set_xlabel("Angle (degrees)", fontsize=9)
		ax.set_title(f"Angle: {i+1}-{j+1}-{k+1}", fontsize=10)
		ax.grid(alpha=0.3)
		ax.set_yticks([])
		ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
		if hist_range is not None:
			ax.set_xlim(hist_range)

		_add_stats_box(ax, aa_vals, cg_vals, value_type="angle")
		ax.legend(fontsize=8)

	for idx in range(n_plots, len(axes)):
		axes[idx].axis("off")

	plt.tight_layout()
	_save_or_show(output_file, "angles")


def _plot_dihedrals_overlay(dihedrals_aa, dihedrals_cg, topo, output_file):
	logger.info("Plotting %s dihedrals", len(set(dihedrals_aa) | set(dihedrals_cg)))

	dihedral_ref = {(int(d[0]), int(d[1]), int(d[2]), int(d[3])): d[5] for d in topo.dihedrals}
	keys = [(int(d[0]), int(d[1]), int(d[2]), int(d[3]), "dihedral") for d in topo.dihedrals]
	if not keys:
		keys = list(set(dihedrals_aa.keys()) | set(dihedrals_cg.keys()))
	keys = [k for k in keys if k in dihedrals_aa or k in dihedrals_cg]

	n_plots = len(keys)
	if n_plots == 0:
		return

	n_cols = min(4, n_plots)
	n_rows = int(np.ceil(n_plots / n_cols))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
	if n_plots == 1:
		axes = [axes]
	else:
		axes = axes.flatten()

	for idx, key in enumerate(keys):
		ax = axes[idx]
		i, j, k, l, _ = key
		aa_vals = dihedrals_aa.get(key)
		cg_vals = dihedrals_cg.get(key)

		circ_mean = _reference_circ_mean(aa_vals, cg_vals)
		aa_shifted = _shift_dihedrals(aa_vals, circ_mean)
		cg_shifted = _shift_dihedrals(cg_vals, circ_mean)

		_plot_hist_pair(ax, aa_shifted, cg_shifted, bins=30, hist_range=(-180, 180))

		if (i, j, k, l) in dihedral_ref:
			ref_dihedral = dihedral_ref[(i, j, k, l)]
			ref_shifted = wrap_to_180(ref_dihedral - circ_mean)
			ax.axvline(ref_shifted, color="red", linestyle="--", linewidth=1.5, label=f"ITP: {ref_dihedral:.1f} deg")

		ax.set_xlabel(f"Dihedral - {circ_mean:.1f} deg", fontsize=9)
		ax.set_title(f"Dihedral: {i+1}-{j+1}-{k+1}-{l+1}", fontsize=10)
		ax.grid(alpha=0.3)
		ax.set_yticks([])
		ax.set_xlim(-180, 180)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(60))

		_add_stats_box(ax, aa_vals, cg_vals, value_type="dihedral")
		ax.legend(fontsize=8)

	for idx in range(n_plots, len(axes)):
		axes[idx].axis("off")

	plt.tight_layout()
	_save_or_show(output_file, "dihedrals")


def _common_bins(aa_vals, cg_vals, bins=30):
	all_vals = []
	if aa_vals is not None:
		all_vals.append(aa_vals)
	if cg_vals is not None:
		all_vals.append(cg_vals)
	if not all_vals:
		return bins
	combined = np.concatenate(all_vals)
	vmin = float(np.min(combined))
	vmax = float(np.max(combined))
	if vmin == vmax:
		vmin -= 1e-3
		vmax += 1e-3
	return np.linspace(vmin, vmax, bins + 1)


def _plot_hist_pair(ax, aa_vals, cg_vals, bins=30, hist_range=None):
	if aa_vals is not None:
		ax.hist(
			aa_vals,
			bins=bins,
			range=hist_range,
			density=True,
			alpha=0.55,
			color="tab:blue",
			edgecolor="black",
			label="AA",
		)
	if cg_vals is not None:
		ax.hist(
			cg_vals,
			bins=bins,
			range=hist_range,
			density=True,
			alpha=0.55,
			color="tab:orange",
			edgecolor="black",
			label="CG",
		)


def _preferred_range(aa_vals, cg_vals):
	if aa_vals is not None and len(aa_vals) > 0:
		values = aa_vals
	elif cg_vals is not None and len(cg_vals) > 0:
		values = cg_vals
	else:
		return None

	vmin = float(np.min(values))
	vmax = float(np.max(values))
	if vmin == vmax:
		vmin -= 1e-3
		vmax += 1e-3
	return (vmin, vmax)


def _add_stats_box(ax, aa_vals, cg_vals, value_type):
	lines = []
	if aa_vals is not None:
		mu, sigma = _compute_stats(aa_vals, value_type)
		lines.append(f"AA mu={mu:.3f} sigma={sigma:.3f}")
	if cg_vals is not None:
		mu, sigma = _compute_stats(cg_vals, value_type)
		lines.append(f"CG mu={mu:.3f} sigma={sigma:.3f}")
	if not lines:
		return

	ax.text(
		0.98,
		0.98,
		"\n".join(lines),
		transform=ax.transAxes,
		fontsize=8,
		verticalalignment="top",
		horizontalalignment="right",
		bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
	)


def _compute_stats(values, value_type):
	if value_type == "dihedral":
		mean_val = circular_mean_deg(values)
		centered = wrap_to_180(values - mean_val)
		std_val = np.std(centered)
	else:
		mean_val = float(np.mean(values))
		std_val = float(np.std(values))
	return mean_val, std_val


def _reference_circ_mean(aa_vals, cg_vals):
	if aa_vals is not None and len(aa_vals) > 0:
		return circular_mean_deg(aa_vals)
	if cg_vals is not None and len(cg_vals) > 0:
		return circular_mean_deg(cg_vals)
	return 0.0


def _shift_dihedrals(values, center):
	if values is None:
		return None
	return wrap_to_180(values - center)


def _save_or_show(output_file, suffix):
	if output_file:
		base = Path(output_file).stem if isinstance(output_file, (str, Path)) else "internal_coords"
		out_path = Path(output_file).parent / f"{base}_{suffix}.png" if isinstance(output_file, (str, Path)) else f"{suffix}.png"
		logger.info("Saving %s plot to %s", suffix, out_path)
		plt.savefig(out_path, dpi=100, bbox_inches="tight")
		plt.close()
	else:
		plt.show()


if __name__ == "__main__":
	molname = "FTA"
	wdir = Path("systems") / molname

	itp_updated = wdir / "mapping" / f"{molname}_updated.itp"
	itp_default = wdir / "mapping" / f"{molname}.itp"
	in_itp = itp_updated if itp_updated.exists() else itp_default
	logger.info("Reading topology from %s", in_itp)
	topo = am.topology.read_itp(str(in_itp))

	aa_dir = wdir / "aa_md"
	aa_pdb = aa_dir / "md.pdb"
	aa_xtc = aa_dir / "md.xtc"

	cg_dir = wdir / "cg_md"
	cg_pdb = cg_dir / "topology.pdb"
	cg_xtc = cg_dir / "samples.xtc"

	logger.info("Reading AA trajectory from %s", aa_dir)
	aa_traj = read_cog_trajectory(aa_pdb, aa_xtc, topo.partitioning)
	aa_internal = calculate_internal_coordinates(aa_traj, topo)

	logger.info("Reading CG trajectory from %s", cg_dir)
	cg_traj = read_cg_trajectory(cg_pdb, cg_xtc)
	cg_internal = calculate_internal_coordinates(cg_traj, topo)

	plot_internal_coordinates_overlay(
		aa_internal,
		cg_internal,
		topo,
		output_file=wdir / "mapping" / "internal_coords_aa_vs_cg.png",
	)
