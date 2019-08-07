# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     preparing
   Description :
   Author :       haxu
   date：          2019-06-18
-------------------------------------------------
   Change Activity:
                   2019-06-18:
-------------------------------------------------
"""
__author__ = 'haxu'

import os
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
from sklearn import preprocessing
from rdkit import Chem
from utils import mol_from_axyz
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures, AllChem


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
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]


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
    symbol = np.zeros((num_atom, len(SYMBOL)), np.uint8)
    hybridization = np.zeros((num_atom, len(HYBRIDIZATION)), np.uint8)
    atomic = np.zeros((num_atom, 1), np.uint8)
    valence = np.zeros((num_atom, 6), np.uint8)
    aromatic = np.zeros((num_atom, 1), np.uint8)
    ring3 = np.zeros((num_atom, 1), np.uint8)
    ring4 = np.zeros((num_atom, 1), np.uint8)
    ring5 = np.zeros((num_atom, 1), np.uint8)
    ring6 = np.zeros((num_atom, 1), np.uint8)
    ring = np.zeros((num_atom, 1), np.uint8)
    charge = np.zeros((num_atom, 1), np.float32)
    num_h = np.zeros((num_atom, 8), np.uint8)

    acceptor = np.zeros((num_atom, 1), np.uint8)
    donor = np.zeros((num_atom, 1), np.uint8)
    chirality = np.zeros((num_atom, 3), np.uint8)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        symbol[i] = one_hot_encoding(atom.GetSymbol(), SYMBOL)
        hybridization[i] = one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION)
        atomic[i] = atom.GetAtomicNum()
        aromatic[i] = atom.GetIsAromatic()
        valence[i] = one_hot_encoding(atom.GetExplicitValence(), range(6))
        aromatic[i] = atom.GetIsAromatic()

        ring3[i] = int(atom.IsInRingSize(3))
        ring4[i] = int(atom.IsInRingSize(4))
        ring5[i] = int(atom.IsInRingSize(5))
        ring6[i] = int(atom.IsInRingSize(6))
        ring[i] = int(atom.IsInRing())
        AllChem.ComputeGasteigerCharges(mol)
        charge[i] = atom.GetProp('_GasteigerCharge')
        num_h[i] = one_hot_encoding(atom.GetTotalNumHs(includeNeighbors=True), range(8))

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

    graph = Graph(
        name,
        Chem.MolToSmiles(mol),
        [a, xyz],
        [symbol, hybridization, atomic, valence,
         aromatic, ring3, ring4, ring5, ring6, ring, charge, num_h, acceptor, donor],
        [bond_type, distance, angle],
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

    g = make_graph('dsgdb9nsd_052330', gb_structure, gb_scalar_coupling)

    print(g.node)
    print(g.edge)
    print(g.smiles)

    # param = []
    #
    # for i, molecule_name in enumerate(molecule_names):
    #     graph_file = f'../input/graph0805/{molecule_name}.pickle'
    #     p = (i, molecule_name, gb_structure, gb_scalar_coupling, graph_file)
    #     param.append(p)
    #
    # print('load done.')
    #
    # pool = mp.Pool(processes=3)
    # _ = pool.map(do_one, param)
    #
    # pool.close()
    # pool.join()
