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


def split_fold():
    df = pd.read_csv('../input/champs-scalar-coupling/train.csv')

    X = df['scalar_coupling_constant']
    y = df['scalar_coupling_constant']
    groups = df['molecule_name']

    kfold = GroupKFold(n_splits=5)
    res = pd.DataFrame({})
    for fold, (_, val_idx) in enumerate(kfold.split(X, y, groups)):
        tmp = df.loc[val_idx]
        tmp['fold'] = fold
        res = pd.concat([res, tmp])
    res.to_csv('../input/champs-scalar-coupling/train_split_group.csv', index=False)


path = Path('../input/champs-scalar-coupling/structures')


def fun(df):
    name = df['molecule_name']
    atomicNumList, charge, xyz_coordinates = read_xyz_file(path / f'{name}.xyz')
    mol = xyz2mol(atomicNumList, xyz_coordinates, charge, True, True)
    atom = mol.GetAtomWithIdx(df['atom_index'])
    AllChem.ComputeGasteigerCharges(mol)
    df['charge'] = atom.GetProp('_GasteigerCharge')
    nb = [a.GetSymbol() for a in atom.GetNeighbors()]
    df['nb_h'] = sum([_ == 'H' for _ in nb])
    df['nb_c'] = sum([_ == 'C' for _ in nb])
    df['nb_n'] = sum([_ == 'N' for _ in nb])
    df['nb_o'] = sum([_ == 'O' for _ in nb])
    df['nb_f'] = sum([_ == 'F' for _ in nb])
    df['hybridization'] = int(atom.GetHybridization())

    df['inring'] = int(atom.IsInRing())
    df['inring3'] = int(atom.IsInRingSize(3))
    df['inring4'] = int(atom.IsInRingSize(4))
    df['inring5'] = int(atom.IsInRingSize(5))
    df['inring6'] = int(atom.IsInRingSize(6))
    df['inring7'] = int(atom.IsInRingSize(7))
    df['inring8'] = int(atom.IsInRingSize(8))

    return df


def multi_task(df):
    res = df(lambda x: fun(x), axis=1)
    return res


def apply_mul_core(df):
    import multiprocessing as mlp
    num_cpu = 50
    pool = mlp.Pool(num_cpu)
    batch_num = 1 + len(df) // num_cpu
    results = []
    for i in range(num_cpu):
        task = df[i * batch_num: (i + 1) * batch_num]
        result = pool.apply_async(multi_task, (task,))
        results.append(result)
    pool.close()
    pool.join()
    res = pd.DataFrame({})
    for result in results:
        feat = result.get()
        res = pd.concat([res, feat])
    return res


def struct_feature():
    df_structures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')

    df_structures_new = apply_mul_core(df_structures)
    df_structures_new.to_csv('../input/champs-scalar-coupling/structures_621.csv', index=False)


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


class Coupling:
    def __init__(self, id, contribution, index, type, value):
        self.id = id
        self.contribution = contribution
        self.index = index
        self.type = type
        self.value = value


class Graph:
    def __init__(self, molecule_name, smiles, axyz, node, edge, edge_index, coupling: Coupling):
        self.molecule_name = molecule_name
        self.smiles = smiles
        self.axyz = axyz
        self.node = node
        self.edge = edge
        self.edge_index = edge_index
        self.coupling = coupling

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


def make_graph(name, gb_structure, gb_scalar_coupling):
    # ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type','scalar_coupling_constant']
    df = gb_scalar_coupling.get_group(name)

    coupling = Coupling(df['id'].values,
                        df[['fc', 'sd', 'pso', 'dso']].values,
                        df[['atom_index_0', 'atom_index_1']].values,
                        np.array([COUPLING_TYPE.index(t) for t in df.type.values], np.int32),
                        df['scalar_coupling_constant'].values,
                        )

    # [molecule_name,atom_index,atom,x,y,z]
    df = gb_structure.get_group(name)
    df = df.sort_values(['atom_index'], ascending=True)
    a = df.atom.values.tolist()
    xyz = df[['x', 'y', 'z']].values

    mol = mol_from_axyz(a, xyz)

    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)

    num_atom = mol.GetNumAtoms()
    symbol = np.zeros((num_atom, len(SYMBOL)), np.uint8)  # category
    acceptor = np.zeros((num_atom, 1), np.uint8)
    donor = np.zeros((num_atom, 1), np.uint8)
    aromatic = np.zeros((num_atom, 1), np.uint8)
    hybridization = np.zeros((num_atom, len(HYBRIDIZATION)), np.uint8)
    num_h = np.zeros((num_atom, 1), np.float32)  # real
    atomic = np.zeros((num_atom, 1), np.float32)
    valence = np.zeros((num_atom, 1), np.float32)
    peak = np.zeros((num_atom, 1), np.float32)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        symbol[i] = one_hot_encoding(atom.GetSymbol(), SYMBOL)
        aromatic[i] = atom.GetIsAromatic()
        hybridization[i] = one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION)

        num_h[i] = atom.GetTotalNumHs(includeNeighbors=True)
        atomic[i] = atom.GetAtomicNum()

        # new feautures
        valence[i] = atom.GetExplicitValence()
        nei = [n.GetTotalNumHs() for n in atom.GetNeighbors()]
        peak[i] = sum([n + 1 if n > 0 and num_h[i] > 0 else 0 for n in nei])

    for t in range(0, len(feature)):
        if feature[t].GetFamily() == 'Donor':
            for i in feature[t].GetAtomIds():
                donor[i] = 1
        elif feature[t].GetFamily() == 'Acceptor':
            for i in feature[t].GetAtomIds():
                acceptor[i] = 1

    num_edge = num_atom * num_atom - num_atom
    # bug
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

    graph = Graph(
        name,
        Chem.MolToSmiles(mol),
        [a, xyz],
        [symbol, acceptor, donor, aromatic, hybridization, num_h, atomic, valence, peak],
        [bond_type, distance, angle, ],
        edge_index,
        coupling,
    )

    return graph


def do_one(p):
    i, molecule_name, gb_structure, gb_scalar_coupling, graph_file = p
    g = make_graph(molecule_name, gb_structure, gb_scalar_coupling)

    with open(graph_file, 'wb') as f:
        pickle.dump(g, f)
    return


if __name__ == '__main__':
    df_structure = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
    df_train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
    df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv')
    df_test['scalar_coupling_constant'] = 0
    df_scalar_coupling = pd.concat([df_train, df_test])
    df_scalar_coupling_contribution = pd.read_csv('../input/champs-scalar-coupling/scalar_coupling_contributions.csv')
    df_scalar_coupling = pd.merge(df_scalar_coupling, df_scalar_coupling_contribution,
                                  how='left',
                                  on=['molecule_name', 'atom_index_0', 'atom_index_1', 'atom_index_0', 'type'])

    gb_scalar_coupling = df_scalar_coupling.groupby('molecule_name')
    gb_structure = df_structure.groupby('molecule_name')

    molecule_names = df_scalar_coupling.molecule_name.unique()

    g = make_graph('dsgdb9nsd_000214', gb_structure, gb_scalar_coupling)

    # print(g.node)
    # print(g.edge)
    # print(g.smiles)

    # param = []
    #
    # for i, molecule_name in enumerate(molecule_names):
    #     graph_file = f'../input/graph/{molecule_name}.pickle'
    #     p = (i, molecule_name, gb_structure, gb_scalar_coupling, graph_file)
    #     param.append(p)
    #
    # print('load done.')
    #
    # pool = mp.Pool(processes=55)
    # _ = pool.map(do_one, param)
    #
    # pool.close()
    # pool.join()
