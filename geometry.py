import numpy as np
import os
import pickle
import sys
from pathlib import Path
import reforge.forge.forcefields as ffs
import reforge.forge.cgmap as cgmap
from reforge.forge.topology import Topology, BondList
from reforge.forge.geometry import get_cg_bonds, get_aa_bonds
from reforge.plotting import init_figure, make_hist, plot_figure
from reforge.mdsystem import gmxmd
from reforge.pdbtools import AtomList, pdb2system


def process_chain(_chain, _ff, _start_idx, _mol_name):
    """
    Process an individual RNA chain: map it to coarse-grained representation and
    generate a topology.

    Args:
        chain (iterable): An RNA chain from the parsed system.
        ff: Force field object.
        start_idx (int): Starting atom index for mapping.
        mol_name (str): Molecule name.

    Returns:
        tuple: (cg_atoms, chain_topology)
    """
    _cg_atoms = cgmap.map_chain(_chain, _ff, atid=_start_idx)
    sequence = [res.resname for res in _chain]
    chain_topology = Topology(forcefield=_ff, sequence=sequence, molname=_mol_name)
    chain_topology.process_atoms()
    chain_topology.process_bb_bonds()
    chain_topology.process_sc_bonds()
    return _cg_atoms, chain_topology


def merge_topologies(top_list):
    """
    Merge multiple Topology objects into one.

    Args:
        top_list (list): List of Topology objects.

    Returns:
        Topology: The merged Topology.
    """
    _merged_topology = top_list.pop(0)
    for new_top in top_list:
        _merged_topology += new_top
    return _merged_topology


def get_reference_topology(inpdb, mol_name='dsrna'):
    # Need to get the topology from the reference system
    print(f'Calculating the reference topology from {inpdb}...', file=sys.stderr)
    system = pdb2system(inpdb)
    cgmap.move_o3(system)  # Adjust O3 atoms as required
    structure = AtomList()
    topologies = []
    start_idx = 1
    for chain in system.chains():
        cg_atoms, chain_top = process_chain(chain, ff, start_idx, mol_name)
        structure.extend(cg_atoms)
        topologies.append(chain_top)
        start_idx += len(cg_atoms)
    top = merge_topologies(topologies)
    print('Done!', file=sys.stderr)
    return top


def prep_data_tmp(aabonds, cgbonds, resname):
    cg_dict = cgbonds.categorize()
    aa_dict = aabonds.categorize()
    if resname == 'all':
        keys = [comm.split()[1] for comm in aabonds.comms]
        keys = sorted(set(keys))
        aadatas, cgdatas = [], []
        for key in keys:
            filtered = BondList([bond for bond in aabonds if key in bond[2]])
            aadatas.append(filtered.measures)
            filtered = BondList([bond for bond in cgbonds if key in bond[2]])
            cgdatas.append(filtered.measures)
            axtitles = keys
    else:
        res_keys = [key for key in sorted(aa_dict.keys()) if key.startswith(resname)]
        aadatas = [aa_dict[key].measures for key in res_keys]
        cgdatas = [cg_dict[key].measures for key in res_keys]
        axtitles = [key.split()[1] for key in res_keys]
    return aadatas, cgdatas, axtitles


def prep_data(bonds, resname):
    adict = bonds.categorize()
    if resname == 'all':
        keys = [comm.split()[1] for comm in bonds.comms]
        keys = sorted(set(keys))
        datas = []
        for key in keys:
            filtered = BondList([bond for bond in bonds if key in bond[2]])
            datas.append(filtered.measures)
            axtitles = keys
    else:
        res_keys = [key for key in sorted(adict.keys()) if key.startswith(resname)]
        datas = [adict[key].measures for key in res_keys]
        axtitles = [key.split()[1] for key in res_keys]
    return datas, axtitles


def make_histograms(bonds_list, params_list, resid='all', resname='all', figname='all', grid=(3, 4), figpath=f'png/test.png'):
    print(f'Plotting {resname}...', file=sys.stderr)
    # prep data for plotting 
    datas_list = []
    axtitles_list = []
    for bonds in bonds_list:
        datas, axtitles = prep_data(bonds, resid)
        datas_list.append(datas)
        axtitles_list.append(axtitles)
    datas_tr = list(zip(*datas_list))
    axtitles = axtitles_list[0]
    # plotting 
    fig, axes = init_figure(grid=grid, axsize=(3, 3))
    for ax, axtitle, datas_tr_list in zip(axes, axtitles, datas_tr):
        datas_list = unwrap_list_of_dihedrals(datas_tr_list, figname)
        make_hist(ax, datas_list, params_list)
        ax.tick_params(left=False, labelleft=False)
        ax.set_title(axtitle, fontsize=12)
    # Hide the remaining unused subplots
    for j in range(len(datas_tr), len(axes.flat)):
        fig.delaxes(axes.flat[j])    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=1, bbox_to_anchor=(0.984, 0.062), frameon=True)
    plot_figure(fig, axes, figname=figname, figpath=figpath)
    print(f'Done!', file=sys.stderr)


def unwrap_dihedrals(dihs):
    dihs = np.array(dihs)
    neg_dihs = dihs[dihs < 0]
    pos_dihs = dihs[dihs > 0]
    neg_av = np.average(neg_dihs)
    pos_av = np.average(pos_dihs)
    if pos_av - neg_av > 180:
        dihs[dihs < 0] = dihs[dihs < 0] + 360
    return dihs


def unwrap_list_of_dihedrals(datas_list, figname):
    unwrapped_list = []
    if figname.split()[0] == 'Dihedral':
        for datas in datas_list:
            datas = unwrap_dihedrals(datas)
            unwrapped_list.append(datas)
    else:
        unwrapped_list = datas_list
    return unwrapped_list


def plot_all(dists_tuple, angles_tuple, dihs_tuple):
    bins = 200
    ch_params = {'label': 'CHARMM', 'bins': bins, 'density': True, 'histtype': 'step', 'color': 'blue', 'alpha': 1.0, 'linewidth': 1.5}
    am_params = {'label': 'AMBER', 'bins': bins, 'density': True, 'histtype': 'step', 'color': 'red', 'alpha': 1.0, 'linewidth': 1.5}
    cg_params = {'label': 'Martini', 'bins': bins, 'density': True, 'histtype': 'step', 'color': 'black', 'alpha': 1.0, 'linewidth': 1.5}
    params_list = [ch_params, am_params, cg_params]
    make_histograms(dihs_tuple, params_list, 
        figname=f'Dihedral distributions (degrees)', grid=(4, 4), figpath=os.path.join(figdir, 'dihs_all.png'))
    make_histograms(angles_tuple, params_list, 
        figname=f'Angle distributions (degrees)', grid=(3, 4), figpath=os.path.join(figdir, 'angles_all.png'))
    make_histograms(dists_tuple, params_list, 
        figname=f'Distance distributions (nm)', grid=(4, 4), figpath=os.path.join(figdir, 'bonds_all.png'))


def plot_by_residue(dists_tuple, angles_tuple, dihs_tuple): 
    bins = 200
    ch_params = {'label': 'CHARMM', 'bins': bins, 'density': True, 'histtype': 'step', 'color': 'blue', 'alpha': 1.0, 'linewidth': 1.5}
    am_params = {'label': 'AMBER', 'bins': bins, 'density': True, 'histtype': 'step', 'color': 'red', 'alpha': 1.0, 'linewidth': 1.5}
    cg_params = {'label': 'Martini', 'bins': bins, 'density': True, 'histtype': 'step', 'color': 'black', 'alpha': 1.0, 'linewidth': 1.5}
    params_list = [ch_params, am_params, cg_params]
    resnames = {'A': 'Adenine', 'C': 'Cytosine', 'G': 'Guanine', 'U': 'Uracil'}
    # resnames = {'A': 'Adenine', 'U': 'Uracil'}
    for resid, resname in resnames.items():
        make_histograms(dihs_tuple, params_list, resid, resname, 
            figname=f'Dihedrals {resname}', grid=(3, 4), figpath=os.path.join(figdir, f'dihs_{resid}.png'))
        make_histograms(angles_tuple, params_list, resid, resname, 
            figname=f'Angles {resname}', grid=(2, 4), figpath=os.path.join(figdir, f'angles_{resid}.png'))
        make_histograms(dists_tuple, params_list, resid, resname, 
            figname=f'Distances {resname}', grid=(3, 4), figpath=os.path.join(figdir, f'bonds_{resid}.png'))


def rename_residues_in_charmm_pdb(aapdb):
    res_map = {'ADE':'A', 'CYT':'C', 'GUA':'G', 'URA':'U'}
    system = pdb2system(aapdb)
    for atom in system.atoms:
        atom.resname = res_map[atom.resname]
    system.write_pdb(aapdb)


def rename_residues_in_amber_pdb(aapdb):
    res_map = {'A5':'A', 'C5':'C', 'G5':'G', 'U5':'U', 
        'A3':'A', 'C3':'C', 'G3':'G', 'U3':'U',
        'A':'A', 'C':'C', 'G':'G', 'U':'U'}
    system = pdb2system(aapdb)
    for atom in system.atoms:
        atom.resname = res_map[atom.resname]
    system.write_pdb(aapdb)


def save_datas(datas, fpath):
    with open(fpath, "wb") as file:
        pickle.dump(datas, file)


def read_datas(fpath):
    with open(fpath, "rb") as f:
        datas = pickle.load(f)
    return datas


if __name__ == "__main__":
    mol_name = 'dsrna'
    cg_sys = gmxmd.GmxSystem('systems', 'dsrna')
    ch_sys = gmxmd.GmxSystem('systems', 'dsrna_charmm')
    am_sys = gmxmd.GmxSystem('systems', 'dsrna_amber')
    figdir = os.path.join('png', mol_name)
    cg_pdb = cg_sys.root / 'mdruns' / 'mdrun_1' / 'conv.pdb' # mdrun_2 for the paper
    cg_refpdb = cg_sys.root / 'inpdb.pdb'
    ch_pdb = ch_sys.root / 'mdruns' / 'mdrun_2' / 'conv.pdb'
    ch_refpdb = ch_sys.root / 'ref.pdb'
    am_pdb = am_sys.root / 'mdruns' / 'mdrun_2' / 'conv.pdb'
    am_refpdb = am_sys.root / 'ref.pdb'
    # Neeed to rename the residues before using the PDB
    # rename_residues_in_charmm_pdb(ch_pdb)
    # rename_residues_in_amber_pdb(am_pdb)
    ff = ffs.Martini30RNA()
    ch_reftop = get_reference_topology(ch_refpdb, mol_name=mol_name)
    am_reftop = get_reference_topology(am_refpdb, mol_name=mol_name)
    cg_reftop = get_reference_topology(cg_refpdb, mol_name=mol_name)
    ch_dists, ch_angles, ch_dihs = get_aa_bonds(ch_pdb, ff, ch_reftop)
    am_dists, am_angles, am_dihs = get_aa_bonds(am_pdb, ff, am_reftop)
    cg_dists, cg_angles, cg_dihs = get_cg_bonds(cg_pdb, cg_reftop)
    # am_dists, am_angles, am_dihs = cg_dists, cg_angles, cg_dihs
    # ch_dists, ch_angles, ch_dihs = cg_dists, cg_angles, cg_dihs
    datas = (ch_dists, am_dists, cg_dists), (ch_angles, am_angles, cg_angles), (ch_dihs, am_dihs, cg_dihs)
    # save_datas(datas, 'data/datas.pkl')
    # datas = read_datas('data/datas.pkl')
    plot_all(*datas)
    plot_by_residue(*datas)


    

    






        
   
