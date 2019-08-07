import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from nn_data import PMPDataset, null_collate
from torch_geometric.utils import scatter_
from torch_scatter import *
import math
import random
from nn_utils import Graph, Coupling


class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-05, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=256):
        super(GraphConv, self).__init__()

        self.encoder = nn.Sequential(
            # LinearBn(edge_dim, hidden_dim),
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # LinearBn(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # LinearBn(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            # LinearBn(hidden_dim // 2, node_dim * node_dim),
            nn.Linear(hidden_dim // 2, node_dim * node_dim),
        )

        # self.gru = nn.GRU(node_dim, node_dim, batch_first=False, bidirectional=False)
        self.gru = nn.GRU(node_dim, node_dim // 2, batch_first=False, bidirectional=True)
        self.bias = nn.Parameter(torch.zeros(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim), 1.0 / math.sqrt(node_dim))

    def forward(self, node, edge_index, edge, hidden):
        num_node, node_dim = node.shape  # 4,128
        num_edge, edge_dim = edge.shape  # 12,6
        edge_index = edge_index.t().contiguous()

        x_i = torch.index_select(node, 0, edge_index[0])

        edge = self.encoder(edge).view(-1, node_dim, node_dim)

        message = x_i.view(-1, 1, node_dim) @ edge  # 12,1,128

        message = message.view(-1, node_dim)  # 12,128

        message = scatter_('mean', message, edge_index[1], dim_size=num_node)  # 4,128

        message = F.relu(message + self.bias)  # 4, 128

        update = message  # 9, 128

        update, hidden = self.gru(update.view(1, -1, node_dim), hidden)

        update = update.view(-1, node_dim)

        return update, hidden


class Set2Set(torch.nn.Module):
    def softmax(self, x, index, num=None):
        x = x - scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_layer = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1  # bs

        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),  # hidden
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))  # cell

        q_star = x.new_zeros(batch_size, self.out_channel)  # bs,256
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)  # bs,128

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True)  # num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)  # num_node x 1

            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size)  #

            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Net(torch.nn.Module):
    def __init__(self, node_dim=96, edge_dim=6, num_target=8):
        super(Net, self).__init__()
        self.num_propagate = 4
        self.num_s2s = 4
        self.hidden_dim = 128
        self.edge_dim = edge_dim

        self.preprocess = nn.Sequential(
            LinearBn(node_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            # LinearBn(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.propagate = GraphConv(self.hidden_dim, self.edge_dim)

        self.set2set = Set2Set(self.hidden_dim, processing_step=self.num_s2s)

        self.predict = nn.Sequential(
            # LinearBn(5 * self.hidden_dim, 1024),
            nn.Linear(5 * self.hidden_dim, 1024),
            nn.ReLU(inplace=True),
            # LinearBn(1024, 512),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_target),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):
        num_node, node_dim = node.shape

        node = self.preprocess(node)  # 9,128

        x = node  # 9, 128

        hidden = torch.zeros(2, x.size(0), 64)

        nodes = []
        for i in range(self.num_propagate):
            nodes.append(node + x)
            node, hidden = self.propagate(node, edge_index, edge, hidden)  # (9,128)

        node = torch.stack(nodes)  #

        node = torch.mean(node, dim=0)

        pool = self.set2set(node, node_index)  # 2, 256

        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = torch.split(
            coupling_index, 1, dim=1)

        pool = torch.index_select(pool, dim=0, index=coupling_batch_index.view(-1))  # 16,256
        node0 = torch.index_select(node, dim=0, index=coupling_atom0_index.view(-1))  # 16,128
        node1 = torch.index_select(node, dim=0, index=coupling_atom1_index.view(-1))  # 16,128

        att = node0 + node1 - node0 * node1

        predict = self.predict(torch.cat([pool, node0, node1, att], -1))

        predict = torch.gather(predict, 1, coupling_type_index).view(-1)  # 16

        return predict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TuningNet(torch.nn.Module):
    def __init__(self, url=None):
        super(TuningNet, self).__init__()
        self.base = Net(node_dim=96, edge_dim=6, num_target=8)

        if url:
            self.base.load_state_dict(
                torch.load(url, map_location=lambda storage, loc: storage))

        # for param in self.base.parameters():
        #     param.requires_grad = False

        self.base.predict = nn.Sequential(
            LinearBn(5 * self.base.hidden_dim, 1024),
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):
        num_node, node_dim = node.shape

        node = self.base.preprocess(node)

        x = node

        hidden = node.unsqueeze(0)
        nodes = []
        for i in range(self.base.num_propagate):
            nodes.append(node + x)
            node, hidden = self.base.propagate(node, edge_index, edge, hidden)  # (9,128)

        node = torch.stack(nodes)
        node = torch.mean(node, dim=0)

        pool = self.base.set2set(node, node_index)  # 2, 256

        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = torch.split(
            coupling_index, 1, dim=1)

        pool = torch.index_select(pool, dim=0, index=coupling_batch_index.view(-1))  # 16,256
        node0 = torch.index_select(node, dim=0, index=coupling_atom0_index.view(-1))  # 16,128
        node1 = torch.index_select(node, dim=0, index=coupling_atom1_index.view(-1))  # 16,128

        att = node0 + node1 - node0 * node1

        predict = self.base.predict(torch.cat([pool, node0, node1, att], -1))

        predict = torch.gather(predict, 1, coupling_type_index).view(-1)  # 16

        return predict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    names = ['dsgdb9nsd_000002', 'dsgdb9nsd_000001', 'dsgdb9nsd_000030', 'dsgdb9nsd_000038']
    train_loader = DataLoader(PMPDataset(names), batch_size=2, collate_fn=null_collate)
    net = Net(node_dim=96, edge_dim=6, num_target=8)

    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(train_loader):
        _ = net(node, edge, edge_index, node_index, coupling_index)

        break

    print('model success!')
