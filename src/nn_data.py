# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     nn_data
   Description :
   Author :       haxu
   date：          2019-06-26
-------------------------------------------------
   Change Activity:
                   2019-06-26:
-------------------------------------------------
"""
__author__ = 'haxu'

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import numpy as np
from dscribe.descriptors import ACSF
from dscribe.core.system import System
from nn_utils import Graph, Coupling

ACSF_GENERATOR = ACSF(
    species=['H', 'C', 'N', 'O', 'F'],
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)


def null_collate(batch):
    batch_size = len(batch)

    node = []
    edge = []
    edge_index = []
    node_index = []

    coupling_value = []
    coupling_atom_index = []
    coupling_type_index = []
    coupling_batch_index = []
    infor = []

    offset = 0
    for b in range(batch_size):
        graph = batch[b]

        num_node = len(graph.node)
        node.append(graph.node)
        edge.append(graph.edge)

        graph.edge_index = graph.edge_index.astype(np.int64)

        edge_index.append(graph.edge_index + offset)
        node_index.append(np.array([b] * num_node))

        num_coupling = len(graph.coupling.value)
        coupling_value.append(graph.coupling.value)

        coupling_atom_index.append(graph.coupling.index + offset)
        coupling_type_index.append(graph.coupling.type)
        coupling_batch_index.append(np.array([b] * num_coupling))

        infor.append((graph.molecule_name, graph.smiles, graph.coupling.id))
        offset += num_node

    node = torch.from_numpy(np.concatenate(node)).float()
    edge = torch.from_numpy(np.concatenate(edge)).float()
    edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int64)).long()
    node_index = torch.from_numpy(np.concatenate(node_index)).long()

    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1, 1),
        np.concatenate(coupling_batch_index).reshape(-1, 1),
    ], -1)
    coupling_index = torch.from_numpy(coupling_index).long()
    return node, edge, edge_index, node_index, coupling_value, coupling_index, infor


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


class PMPDataset(Dataset):
    def __init__(self, names, is_seven=False):
        self.path = Path('../input/graph')
        if is_seven:
            self.path = Path('/opt/ml/disk/PMP/input/graph')
        self.names = names
        self.bins = np.arange(0.959, 12.05, 0.5)  # 23

    def __getitem__(self, x):
        # molecule_name, smiles, axyz(atom, xyz), node, edge, edge_index

        name = self.names[x]
        with open(self.path / f'{name}.pickle', 'rb') as f:
            g = pickle.load(f)

        assert isinstance(g, Graph)
        assert (g.molecule_name == name)

        atom = System(symbols=g.axyz[0], positions=g.axyz[1])

        acsf = ACSF_GENERATOR.create(atom)
        g.node += [acsf, g.axyz[1]]

        # bins = [np.histogram(x, self.bins)[0].argmax() for x in g.edge[1]]
        # bins = np.array([one_hot_encoding(b, range(23)) for b in bins])
        # g.edge += [bins]

        g.node = np.concatenate(g.node, -1)
        g.edge = np.concatenate(g.edge, -1)

        return g

    def __len__(self):
        return len(self.names)


COUPLING_TYPE = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']


class PMPDatasetType(Dataset):
    def __init__(self, names, type):
        self.path = Path('../input/graph0805')
        assert type in COUPLING_TYPE
        self.type = COUPLING_TYPE.index(type)
        self.names = names

    def __getitem__(self, x):
        # molecule_name, smiles, axyz(atom, xyz), node, edge, edge_index

        name = self.names[x]
        with open(self.path / f'{name}.pickle', 'rb') as f:
            g = pickle.load(f)

        assert isinstance(g, Graph)
        assert (g.molecule_name == name)

        atom = System(symbols=g.axyz[0], positions=g.axyz[1])

        acsf = ACSF_GENERATOR.create(atom)
        g.node += [acsf, g.axyz[1]]

        index = np.where(g.coupling.type == self.type)[0]
        g.coupling.id = g.coupling.id[index]
        g.coupling.type = g.coupling.type[index]
        g.coupling.type = [0] * len(g.coupling.type)

        g.coupling.index = g.coupling.index[index]
        g.coupling.value = g.coupling.value[index]

        g.node = np.concatenate(g.node, -1)
        g.edge = np.concatenate(g.edge, -1)

        return g

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    names = ['dsgdb9nsd_000001', 'dsgdb9nsd_000002']

    train_loader = DataLoader(PMPDataset(names), batch_size=48, collate_fn=null_collate)
    # train_loader = DataLoader(PMPDatasetType(names, type='1JHN'), batch_size=48, collate_fn=null_collate)
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(train_loader):
        print(node.size())
        print(edge_index.size())

        print(coupling_index)

        print(coupling_value)

        break

    # print(max_value)
    # print(min_value)
