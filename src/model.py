import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from nn_data import PMPDataset, null_collate, Graph, Coupling
from torch_geometric.utils import scatter_
from torch_scatter import *
import math


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
    def __init__(self, node_dim, num_step):
        super(GraphConv, self).__init__()

        self.gru = nn.GRU(node_dim, node_dim, batch_first=False, bidirectional=False)
        self.edge_embedding = LinearBn(128, node_dim * node_dim)
        self.bias = nn.Parameter(torch.zeros(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim), 1.0 / math.sqrt(node_dim))
        self.num_step = num_step
        self.node_dim = node_dim

    def forward(self, node, edge_index, edge):
        x = node
        hidden = node.unsqueeze(0)
        edge = self.edge_embedding(edge).view(-1, self.node_dim, self.node_dim)
        num_node, node_dim = node.shape  # 4,128
        nodes = []
        for i in range(self.num_step):
            nodes.append(node + x)

            x_i = torch.index_select(node, 0, edge_index[0])

            message = x_i.view(-1, 1, node_dim) @ edge  # 12,1,128

            message = message.view(-1, node_dim)  # 12,128

            message = scatter_('mean', message, edge_index[1], dim_size=num_node)  # 4,128

            message = F.relu(message + self.bias)  # 4, 128

            node = message  # 9, 128

            node, hidden = self.gru(node.view(1, -1, node_dim), hidden)  # (1,9,128)  (1,9,128)

            node = node.view(-1, node_dim)

        node = torch.stack(nodes)
        node = torch.mean(node, dim=0)
        return node


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

    def __init__(self):
        super(Net, self).__init__()
        self.hidden_dim = 128

        self.node_embedding = nn.Sequential(
            LinearBn(96, self.hidden_dim),
            nn.ReLU(inplace=True),
            LinearBn(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.edge_embedding = nn.Sequential(
            LinearBn(29, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True)
        )

        self.encoder = GraphConv(self.hidden_dim, 4)

        self.decoder = Set2Set(self.hidden_dim, processing_step=4)

        self.predict = nn.Sequential(
            LinearBn(6 * self.hidden_dim, 1024),
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 8),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):
        edge_index = edge_index.t().contiguous()

        # coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = \
        #     torch.split(coupling_index, 1, dim=1)

        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index, edge_coupling_index = \
            torch.split(coupling_index, 1, dim=1)

        node = self.node_embedding(node)
        edge = self.edge_embedding(edge)

        node = self.encoder(node, edge_index, edge)

        pool = self.decoder(node, node_index)  # 2, 256

        pool = torch.index_select(pool, dim=0, index=coupling_batch_index.view(-1))  # 16,256
        node0 = torch.index_select(node, dim=0, index=coupling_atom0_index.view(-1))  # 16,128
        node1 = torch.index_select(node, dim=0, index=coupling_atom1_index.view(-1))  # 16,128
        edge = torch.index_select(edge, dim=0, index=edge_coupling_index.view(-1))

        att = node0 + node1 - node0 * node1

        # predict
        predict = self.predict(torch.cat([pool, node0, node1, att, edge], -1))
        predict = torch.gather(predict, 1, coupling_type_index).view(-1)  # 16

        return predict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    names = ['dsgdb9nsd_000002', 'dsgdb9nsd_000001', 'dsgdb9nsd_000030', 'dsgdb9nsd_000038']
    train_loader = DataLoader(PMPDataset(names), batch_size=2, collate_fn=null_collate)
    net = Net()

    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(train_loader):
        _ = net(node, edge, edge_index, node_index, coupling_index)

        break

    print('model success!')
