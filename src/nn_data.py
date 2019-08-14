# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name锛�     nn_data
   Description :
   Author :       haxu
   date锛�          2019-06-26
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
from nn_utils import Graph, Coupling
import random

from dscribe.descriptors import ACSF
from dscribe.core.system import System

ACSF_GENERATOR = ACSF(
    species=['H', 'C', 'N', 'O', 'F'],
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)


def null_collate(batch, is_train=False):
    batch_size = len(batch)

    node = []
    edge = []
    edge_index = []
    node_index = []

    # 边属于哪一个分子
    edge_belong_index = []

    coupling_value = []
    coupling_edge_index = []
    coupling_atom_index = []
    coupling_type_index = []
    coupling_batch_index = []
    infor = []

    offset = 0
    edge_offset = 0
    for b in range(batch_size):
        graph = batch[b]

        num_node = len(graph.node)
        node.append(graph.node)

        edge.append(graph.edge)

        graph.edge_index = graph.edge_index.astype(np.int64)

        edge_index.append(graph.edge_index + offset)
        node_index.append([b] * num_node)

        # # edge index
        # edge_belong_index.append([b] * (num_node * (num_node - 1)))

        num_coupling = len(graph.coupling.value)

        coupling_value.append(graph.coupling.value)

        cp_edge = graph.coupling.index[:, :2] + offset

        if is_train:
            if random.random() < 0.5:
                coupling_atom_index.append(cp_edge)
            else:
                coupling_atom_index.append(cp_edge[:, ::-1])
        else:
            coupling_atom_index.append(graph.coupling.index[:, :2] + offset)

        coupling_type_index.append(graph.coupling.type)
        coupling_batch_index.append([b] * num_coupling)

        infor.append((graph.molecule_name, graph.smiles, graph.coupling.id))
        offset += num_node
        edge_offset += (num_node - 1) * num_node

    edge_index_tmp = np.concatenate(edge_index).astype(np.int64).tolist()
    coupling_atom_index_tmp = np.concatenate(coupling_atom_index).tolist()

    l = []
    for item in coupling_atom_index_tmp:
        i = edge_index_tmp.index(item)
        l.append(i)

    node = torch.from_numpy(np.concatenate(node)).float()
    edge = torch.from_numpy(np.concatenate(edge)).float()
    edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int64)).long()
    node_index = torch.from_numpy(np.concatenate(node_index)).long()
    # edge_belong_index = torch.from_numpy(np.concatenate(edge_belong_index)).long()

    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1, 1),
        np.concatenate(coupling_batch_index).reshape(-1, 1),
        np.array(l).reshape(-1, 1),
    ], -1)

    coupling_index = torch.from_numpy(coupling_index).long()
    return node, edge, edge_index, node_index, coupling_value, coupling_index, infor


COUPLING_TYPE = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


train_collect = lambda x: null_collate(x, is_train=True)
valid_collect = lambda x: null_collate(x, is_train=False)


class PMPDataset(Dataset):
    def __init__(self, names, type='1JHC', is_seven=False):
        self.path = Path('../input/graph_old')
        if is_seven:
            self.path = Path('/opt/ml/disk/PMP/input/graph_old')
        self.names = names
        self.type = COUPLING_TYPE.index(type)
        # self.bins = np.arange(0.959, 12.05, 0.5)  # 23

    def __getitem__(self, x):
        # molecule_name, smiles, axyz(atom, xyz), node, edge, edge_index

        name = self.names[x]
        with open(self.path / f'{name}.pickle', 'rb') as f:
            g = pickle.load(f)

        assert isinstance(g, Graph)
        assert (g.molecule_name == name)

        atom = System(symbols=g.axyz[0], positions=g.axyz[1])

        # g.node = [g.node[0]]

        acsf = ACSF_GENERATOR.create(atom)
        g.node += [acsf]
        g.node += [g.axyz[1]]

        g.node = np.concatenate(g.node, -1)
        g.edge = np.concatenate(g.edge, -1)

        return g

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    names = ['dsgdb9nsd_000001', 'dsgdb9nsd_000002', 'dsgdb9nsd_000030', 'dsgdb9nsd_000038']

    train_loader = DataLoader(PMPDataset(names), batch_size=1, collate_fn=train_collect)
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(
            train_loader):
        print(node.size())
        print(edge.size())
        print(edge_index)
        print(node_index)
        print(coupling_index)
        print(coupling_index)

        break

    # print(max_value)
    # print(min_value)
