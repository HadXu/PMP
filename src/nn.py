import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from model import Net
from nn_utils import do_train, do_valid, time_to_str, Graph, Coupling, Logger, adjust_learning_rate
import random
from sklearn.model_selection import KFold
from timeit import default_timer as timer
from nn_data import PMPDataset, null_collate
from argparse import ArgumentParser

parser = ArgumentParser(description='train PMP')

parser.add_argument('--fold', type=int, required=True)
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--bs', type=int, default=48)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()

fold = args.fold
lr = args.lr
bs = args.bs
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

log = Logger()
log.open(f'{name}_fold{fold}.txt')
log.write(str(args) + '\n')


def train_fold(fold):
    df_train = pd.read_csv('../input/champs-scalar-coupling/train.csv', usecols=['molecule_name'])
    names = df_train['molecule_name'].unique()

    kfold = KFold(n_splits=5, random_state=42)
    for k, (tr_idx, val_idx) in enumerate(kfold.split(names)):
        if k != fold:
            continue
        log.write(f'~~~~~~~~~~~~ fold {fold} ~~~~~~~~~~~~\n')
        best_score = 999
        best_epoch = 0

        log.write(f'raw train:{len(tr_idx)} -- raw val:{len(val_idx)}\n')

        tr_names = names[tr_idx]
        val_names = names[val_idx]

        log.write(f'train:{len(tr_names)} --- val:{len(val_names)}\n')

        train_loader = DataLoader(PMPDataset(tr_names), batch_size=bs, collate_fn=null_collate, num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
        val_loader = DataLoader(PMPDataset(val_names), batch_size=bs, collate_fn=null_collate, num_workers=4,
                                pin_memory=True,
                                )

        net = Net(node_dim=96, edge_dim=6).to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        if 'fine' in name:
            # name: att-fine
            net.load_state_dict(
                torch.load(f'../checkpoint/fold{fold}_model_{name.split("-")[0]}.pth',
                           map_location=lambda storage, loc: storage))

            log.write('load pre-trained done.....\n')

        f_normal = '{:^5} | {:^3.4f} | {:^7.4f} | {:^7.4f}, {:^7.4f}, {:^7.4f}, {:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f} | {:^7.4f} | {:^7} \n'
        f_boost = '{:^5}* | {:^3.4f} | {:^7.4f} | {:^7.4f}, {:^7.4f}, {:^7.4f}, {:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f},{:^7.4f} | {:^7.4f} | {:^7} \n'

        log.write(
            'epoch | train loss |  valid loss |  1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH  |  log_mae  |  time \n')

        start = timer()
        for e in range(200):

            train_loss = do_train(net, train_loader, optimizer, device)
            valid_loss, log_mae, log_mae_mean = do_valid(net, val_loader, device)
            timing = time_to_str((timer() - start), 'min')
            if log_mae_mean < best_score:
                best_score = log_mae_mean
                best_epoch = e
                torch.save(net.state_dict(), f'../checkpoint/fold{k}_model_{name}.pth')
                log.write(f_boost.format(e, train_loss, valid_loss, *log_mae, log_mae_mean, timing))
            else:
                log.write(f_normal.format(e, train_loss, valid_loss, *log_mae, log_mae_mean, timing))


if __name__ == '__main__':
    train_fold(fold)
