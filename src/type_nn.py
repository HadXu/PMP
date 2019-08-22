from nn_data import PMPDatasetType, null_collate
from model import Net, LinearBn
from torch import nn
import torch
import numpy as np
from itertools import chain
import random
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from Nadam import Nadam
from nn_utils import do_train, do_valid, do_valid_Type, time_to_str, Graph, Coupling
from timeit import default_timer as timer
from tqdm import tqdm

parser = ArgumentParser(description='train PMP')

parser.add_argument('--fold', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--lr', type=float, default=3e-4)

args = parser.parse_args()

fold = args.fold
lr = args.lr
name = args.name

device = torch.device(args.gpu)
cuda_aviable = torch.cuda.is_available()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if cuda_aviable:
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True


class TypeNet(nn.Module):
    def __init__(self):
        super(TypeNet, self).__init__()

        self.base = Net()
        self.base.load_state_dict(torch.load('../checkpoint/fold0_model_0819.pth'))

        for p in self.base.parameters():
            p.requires_grad = False

        self.predict = nn.Sequential(
            LinearBn(6 * 128, 1024),
            nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):
        edge_index = edge_index.t().contiguous()

        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index, edge_coupling_index = \
            torch.split(coupling_index, 1, dim=1)

        node = self.base.node_embedding(node)
        edge = self.base.edge_embedding(edge)

        node = self.base.encoder1(node, edge_index, edge)

        node = self.base.encoder2(node, edge_index)

        node = self.base.encoder3(node, edge_index)

        pool = self.base.decoder(node, node_index)  # set2set

        pool = torch.index_select(pool, dim=0, index=coupling_batch_index.view(-1))  # 16,256
        node0 = torch.index_select(node, dim=0, index=coupling_atom0_index.view(-1))  # 16,128
        node1 = torch.index_select(node, dim=0, index=coupling_atom1_index.view(-1))  # 16,128
        edge = torch.index_select(edge, dim=0, index=edge_coupling_index.view(-1))

        att = node0 + node1 - node0 * node1

        predict = self.predict(torch.cat([pool, node0, node1, att, edge], -1))

        predict = torch.gather(predict, 1, coupling_type_index).view(-1)  # 16

        return predict


def main():
    names = np.load('../input/champs-scalar-coupling/names.npy')
    for k in range(5):
        if k != 0:
            continue

        val_names = names[k]
        tr_names = list(chain(*(names[:k].tolist() + names[k + 1:].tolist())))

        train_loader = DataLoader(PMPDatasetType(tr_names, type='1JHC'), batch_size=48, collate_fn=null_collate,
                                  num_workers=8,
                                  pin_memory=False,
                                  shuffle=True)
        val_loader = DataLoader(PMPDatasetType(val_names, type='1JHC'), batch_size=128, collate_fn=null_collate,
                                num_workers=8,
                                pin_memory=False,
                                )

        net = TypeNet().to(device)

        optimizer = Nadam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

        start = timer()
        for e in range(200):
            if e > 5:
                for p in net.base.parameters():
                    p.requires_grad = True
                optimizer = Nadam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)

            train_loss = do_train(net, train_loader, optimizer, device)
            valid_loss, log_mae = do_valid_Type(net, val_loader, device, '1JHC')
            timing = time_to_str((timer() - start), 'min')
            print(e, train_loss, valid_loss, log_mae, timing)


if __name__ == '__main__':
    main()
