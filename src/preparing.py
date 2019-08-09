# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name锛�     preparing
   Description :
   Author :       haxu
   date锛�          2019-06-18
-------------------------------------------------
   Change Activity:
                   2019-06-18:
-------------------------------------------------
"""
__author__ = 'haxu'

import os
import numpy as np
from sklearn.model_selection import GroupKFold
from rdkit.Chem import AllChem
from xyz2mol import xyz2mol, read_xyz_file
from pathlib import Path
import pandas as pd
import pickle
import multiprocessing as mp
from sklearn import preprocessing
from rdkit import Chem
from utils import mol_from_axyz
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from dscribe.descriptors import ACSF
from dscribe.core.system import System


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


class Coupling:
    def __init__(self, id, contribution, index, type, value, contribute_and_value):
        self.id = id
        self.contribution = contribution
        self.index = index
        self.type = type
        self.value = value
        self.contribute_and_value = contribute_and_value


class Graph:
    def __init__(self, coupling: Coupling,
                 molecule_name,
                 smiles,
                 axyz,
                 node,
                 edge,
                 edge_index,
                 mol_label,
                 node_label):
        self.coupling = coupling
        self.molecule_name = molecule_name
        self.smiles = smiles
        self.axyz = axyz
        self.node = node
        self.edge = edge
        self.edge_index = edge_index

        self.node_label = node_label
        self.mol_label = mol_label

    def __str__(self):
        return f'graph of {self.molecule_name} -- smiles:{self.smiles}'


COUPLING_TYPE = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
SYMBOL = ['H', 'C', 'N', 'O', 'F']
BOND_TYPE = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]

# xiong
atom2polar = {'H': 2.2, 'O': 3.44, 'C': 2.55, 'F': 3.98, 'N': 3.04}
atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}
fudge_factor = 0.05
atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}


def gaussian_rbf(x, min_x, max_x, center_num):
    center_point = np.linspace(min_x, max_x, center_num)
    x_vec = np.exp(np.square(center_point - x))
    return x_vec


dist_min = 0.95860666
dist_max = 12.040386

ACSF_GENERATOR = ACSF(
    species=['H', 'C', 'N', 'O', 'F'],
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)


def make_graph(name, gb_structure, gb_scalar_coupling, gb_mol):
    # ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type','scalar_coupling_constant']
    coupling_df = gb_scalar_coupling.get_group(name)

    # [molecule_name,atom_index,atom,x,y,z,EN,rad,n_bonds,bond_lengths_mean, XX	YX	ZX	XY	YY	ZY	XZ	YZ	ZZ	mulliken_charge]
    df = gb_structure.get_group(name)
    df = df.sort_values(['atom_index'], ascending=True)
    a = df.atom.values.tolist()
    xyz = df[['x', 'y', 'z']].values

    # new 0808
    en_rad_n_bonds_length = df[['EN', 'rad', 'n_bonds', 'bond_lengths_mean']].values
    node_label = df[['XX', 'YX', 'ZX', 'XY', 'YY', 'ZY', 'XZ', 'YZ', 'ZZ', 'mulliken_charge']].values.astype(np.float32)

    if name in gb_mol.groups.keys():
        mol = gb_mol.get_group(name)
        mol_label = mol[['X', 'Y', 'Z', 'potential_energy']].values.astype(np.float32)
    else:
        mol_label = np.array([[0, 0, 0, 0]], dtype=np.float32)

    # new0808

    mol = mol_from_axyz(a, xyz)

    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)

    num_atom = mol.GetNumAtoms()
    symbol = np.zeros((num_atom, len(SYMBOL)), np.uint8)
    acceptor = np.zeros((num_atom, 1), np.uint8)
    donor = np.zeros((num_atom, 1), np.uint8)
    aromatic = np.zeros((num_atom, 1), np.uint8)
    hybridization = np.zeros((num_atom, len(HYBRIDIZATION)), np.uint8)
    # num_h = np.zeros((num_atom, 1), np.float32)
    atomic = np.zeros((num_atom, 1), np.float32)

    valence = np.zeros((num_atom, 1), np.uint8)
    ring3 = np.zeros((num_atom, 1), np.uint8)
    ring4 = np.zeros((num_atom, 1), np.uint8)
    ring5 = np.zeros((num_atom, 1), np.uint8)
    ring6 = np.zeros((num_atom, 1), np.uint8)
    ring = np.zeros((num_atom, 1), np.uint8)
    charge = np.zeros((num_atom, 1), np.float32)
    num_h = np.zeros((num_atom, 5), np.uint8)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        symbol[i] = one_hot_encoding(atom.GetSymbol(), SYMBOL)
        aromatic[i] = atom.GetIsAromatic()
        hybridization[i] = one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION)

        num_h[i] = atom.GetTotalNumHs(includeNeighbors=True)
        atomic[i] = atom.GetAtomicNum()

        valence[i] = atom.GetExplicitValence()
        ring3[i] = int(atom.IsInRingSize(3))
        ring4[i] = int(atom.IsInRingSize(4))
        ring5[i] = int(atom.IsInRingSize(5))
        ring6[i] = int(atom.IsInRingSize(6))
        ring[i] = int(atom.IsInRing())

        AllChem.ComputeGasteigerCharges(mol)
        charge[i] = atom.GetProp('_GasteigerCharge')
        num_h[i] = one_hot_encoding(atom.GetTotalNumHs(includeNeighbors=True), range(5))

    for t in range(0, len(feature)):
        if feature[t].GetFamily() == 'Donor':
            for i in feature[t].GetAtomIds():
                donor[i] = 1
        elif feature[t].GetFamily() == 'Acceptor':
            for i in feature[t].GetAtomIds():
                acceptor[i] = 1

    num_edge = num_atom * num_atom - num_atom

    edge_index = np.zeros((num_edge, 2), np.uint32)
    bond_type = np.zeros((num_edge, len(BOND_TYPE)), np.uint32)
    distance = np.zeros((num_edge, 1), np.float32)
    angle = np.zeros((num_edge, 1), np.float32)

    norm_xyz = preprocessing.normalize(xyz, norm='l2')

    ij = 0
    for i in range(num_atom):
        for j in range(num_atom):
            if i == j: continue
            edge_index[ij] = [i, j]

            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_type[ij] = one_hot_encoding(bond.GetBondType(), BOND_TYPE)

            distance[ij] = np.linalg.norm(xyz[i] - xyz[j])
            angle[ij] = (norm_xyz[i] * norm_xyz[j]).sum()

            ij += 1

    atom = System(symbols=a, positions=xyz)
    acsf = ACSF_GENERATOR.create(atom)

    l = []
    for item in coupling_df[['atom_index_0', 'atom_index_1']].values.tolist():
        i = edge_index.tolist().index(item)
        l.append(i)

    l = np.array(l)

    coupling_edge_index = np.concatenate([coupling_df[['atom_index_0', 'atom_index_1']].values, l.reshape(len(l), 1)],
                                         axis=1)

    coupling = Coupling(coupling_df['id'].values,
                        coupling_df[['fc', 'sd', 'pso', 'dso']].values.astype(np.float32),
                        # coupling_df[['atom_index_0', 'atom_index_1']].values,
                        coupling_edge_index,
                        np.array([COUPLING_TYPE.index(t) for t in coupling_df.type.values], np.int32),
                        coupling_df['scalar_coupling_constant'].values,
                        contribute_and_value=np.concatenate([coupling_df[['scalar_coupling_constant']].values,
                                                             coupling_df[['fc', 'sd', 'pso', 'dso']].values],
                                                            axis=1).astype(np.float32)
                        )

    bins = np.arange(0.959, 12.05, 0.5)
    bins = [np.histogram(x, bins)[0].argmax() for x in distance]
    bins = np.array([one_hot_encoding(b, range(23)) for b in bins], dtype=np.uint)

    graph = Graph(
        coupling=coupling,
        molecule_name=name,
        smiles=Chem.MolToSmiles(mol),
        axyz=[a, xyz],
        node=[acsf, symbol, acceptor, donor, aromatic, hybridization, num_h, atomic, en_rad_n_bonds_length, valence,
              ring3, ring4, ring5, ring6, ring, charge],

        edge=[bond_type, distance, angle, bins],
        edge_index=edge_index,

        node_label=node_label,
        mol_label=mol_label,
    )

    return graph


# def do_one(p):
#     i, molecule_name, gb_structure, gb_scalar_coupling, graph_file = p
#     g = make_graph(molecule_name, gb_structure, gb_scalar_coupling)
#
#     with open(graph_file, 'wb') as f:
#         pickle.dump(g, f)
#     return


if __name__ == '__main__':
    df_structure = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
    df_train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
    df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

    # 0808 new
    # 原子
    magnetic_shielding_tensors = pd.read_csv('../input/champs-scalar-coupling/magnetic_shielding_tensors.csv')
    mulliken_charges = pd.read_csv('../input/champs-scalar-coupling/mulliken_charges.csv')
    df_structure = pd.merge(df_structure, magnetic_shielding_tensors, how='left', on=['molecule_name', 'atom_index'])
    df_structure = pd.merge(df_structure, mulliken_charges, how='left', on=['molecule_name', 'atom_index']).fillna(0)

    # 分子
    dipole_moments = pd.read_csv('../input/champs-scalar-coupling/dipole_moments.csv')
    potential_energy = pd.read_csv('../input/champs-scalar-coupling/potential_energy.csv')
    mol = pd.merge(dipole_moments, potential_energy, how='left', on=['molecule_name'])
    # 0808 new

    df_test['scalar_coupling_constant'] = 0
    df_scalar_coupling = pd.concat([df_train, df_test])
    df_scalar_coupling_contribution = pd.read_csv('../input/champs-scalar-coupling/scalar_coupling_contributions.csv')
    df_scalar_coupling = pd.merge(df_scalar_coupling, df_scalar_coupling_contribution,
                                  how='left',
                                  on=['molecule_name', 'atom_index_0', 'atom_index_1', 'atom_index_0', 'type'])

    gb_scalar_coupling = df_scalar_coupling.groupby('molecule_name')
    gb_structure = df_structure.groupby('molecule_name')
    gb_mol = mol.groupby('molecule_name')

    molecule_names = df_scalar_coupling.molecule_name.unique()

    g = make_graph('dsgdb9nsd_000001', gb_structure, gb_scalar_coupling, gb_mol)

    print(g.node)
    print(g.mol_label)
    print(g.node_label)

    param = []


    def do_one(p):
        molecule_name, graph_file = p
        g = make_graph(molecule_name, gb_structure, gb_scalar_coupling, gb_mol)
        with open(graph_file, 'wb') as f:
            pickle.dump(g, f)


    for i, molecule_name in enumerate(molecule_names):
        graph_file = f'../input/graph/{molecule_name}.pickle'
        p = (molecule_name, graph_file)
        param.append(p)

    print('load done.')

    pool = mp.Pool(processes=55)
    _ = pool.map(do_one, param)

    pool.close()
    pool.join()
