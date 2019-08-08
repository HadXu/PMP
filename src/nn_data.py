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


def null_collate(batch):
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

        # edge index
        edge_belong_index.append([b] * (num_node * (num_node - 1)))

        num_coupling = len(graph.coupling.value)
        coupling_value.append(graph.coupling.value)

        coupling_atom_index.append(graph.coupling.index[:, :2] + offset)
        coupling_edge_index.append(graph.coupling.index[:, 2] + edge_offset)

        coupling_type_index.append(graph.coupling.type)
        coupling_batch_index.append([b] * num_coupling)

        infor.append((graph.molecule_name, graph.smiles, graph.coupling.id))
        offset += num_node
        edge_offset += (num_node - 1) * num_node

    node = torch.from_numpy(np.concatenate(node)).float()
    edge = torch.from_numpy(np.concatenate(edge)).float()
    edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int64)).long()
    node_index = torch.from_numpy(np.concatenate(node_index)).long()
    edge_belong_index = torch.from_numpy(np.concatenate(edge_belong_index)).long()

    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1, 1),
        np.concatenate(coupling_batch_index).reshape(-1, 1),
        np.concatenate(coupling_edge_index).reshape(-1, 1),
    ], -1)

    coupling_index = torch.from_numpy(coupling_index).long()
    return node, edge, edge_index, node_index, edge_belong_index, coupling_value, coupling_index, infor


COUPLING_TYPE = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


class PMPDataset(Dataset):
    def __init__(self, names, type='1JHC', is_seven=False):
        self.path = Path('../input/graph')
        if is_seven:
            self.path = Path('/opt/ml/disk/PMP/input/graph')
        self.names = names
        self.type = COUPLING_TYPE.index(type)
        self.bins = np.arange(0.959, 12.05, 0.5)  # 23

    def __getitem__(self, x):
        # molecule_name, smiles, axyz(atom, xyz), node, edge, edge_index

        name = self.names[x]
        with open(self.path / f'{name}.pickle', 'rb') as f:
            g = pickle.load(f)

        assert isinstance(g, Graph)
        assert (g.molecule_name == name)

        g.node += [g.axyz[1]]

        g.node = np.concatenate(g.node, -1)
        g.edge = np.concatenate(g.edge, -1)

        return g

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    names = ['dsgdb9nsd_000001', 'dsgdb9nsd_000002', 'dsgdb9nsd_000030', 'dsgdb9nsd_000038']

    train_loader = DataLoader(PMPDataset(names), batch_size=2, collate_fn=null_collate)
    for b, (node, edge, edge_index, node_index, edge_belong_index, coupling_value, coupling_index, infor) in enumerate(
            train_loader):
        print(node.size())
        print(edge.size())
        print(edge_index)
        print(node_index)
        print(edge_belong_index)
        break

    # print(max_value)
    # print(min_value)
