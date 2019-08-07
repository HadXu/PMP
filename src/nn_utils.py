# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     nn_utils
   Description :
   Author :       haxu
   date：          2019-06-26
-------------------------------------------------
   Change Activity:
                   2019-06-28:
-------------------------------------------------
"""
__author__ = 'haxu'

import sys
import os
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        if os.path.exists(file):
            os.remove(file)
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass


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


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


COUPLING_TYPE = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']


def compute_kaggle_metric(predict, coupling_value, coupling_type):
    mae = [0] * 8
    log_mae = [0] * 8
    diff = np.fabs(predict - coupling_value)
    num_type = 8

    for t in range(num_type):
        index = np.where(coupling_type == t)[0]
        if len(index) > 0:
            m = diff[index].mean()
            log_m = np.log(m + 1e-2)
            mae[t] = m
            log_mae[t] = log_m
        else:
            pass

    return mae, log_mae


def criterion(predict, coupling_value):
    predict = predict.view(-1)
    coupling_value = coupling_value.view(-1)
    assert (predict.shape == coupling_value.shape)

    loss = torch.abs(predict - coupling_value)
    loss = loss.mean()
    loss = torch.log(loss)

    # loss = torch.nn.SmoothL1Loss()(predict, coupling_value)

    return loss


norm = {
    '1JHC': [94.976153, 18.277236880290022],
    '2JHH': [-10.286605, 3.979607163730381],
    '1JHN': [47.479884, 10.922171556272295],
    '2JHN': [3.124754, 3.6734741723096236],
    '2JHC': [-0.270624, 4.523610750196486],
    '3JHH': [4.771023, 3.704984434128579],
    '3JHC': [3.688470, 3.0709074866562176],
    '3JHN': [0.990730, 1.315393353533751],
}


def do_valid(net, valid_loader, device):
    net.eval()
    valid_num = 0
    valid_predict = []
    valid_coupling_type = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in tqdm(enumerate(valid_loader)):
        node = node.to(device)
        edge = edge.to(device)
        edge_index = edge_index.to(device)
        node_index = node_index.to(device)
        coupling_value = coupling_value.to(device)
        coupling_index = coupling_index.to(device)

        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = criterion(predict, coupling_value)

        batch_size = len(infor)
        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_index[:, 2].data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += batch_size * loss.item()
        valid_num += batch_size

    assert (valid_num == len(valid_loader.dataset))

    valid_loss = valid_loss / valid_num

    # compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)
    coupling_type = np.concatenate(valid_coupling_type).astype(np.int32)

    mae, log_mae = compute_kaggle_metric(predict, coupling_value, coupling_type)

    log_mae_mean = sum(log_mae) / 8
    return valid_loss, log_mae, log_mae_mean


def do_train(net, train_loader, optimizer, device):
    net.train()
    optimizer.zero_grad()
    losses = []

    # debug

    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in tqdm(enumerate(train_loader)):
        node = node.to(device)
        edge = edge.to(device)
        edge_index = edge_index.to(device)
        node_index = node_index.to(device)
        coupling_value = coupling_value.to(device)
        coupling_index = coupling_index.to(device)

        predict = net(node, edge, edge_index, node_index, coupling_index)

        loss = criterion(predict, coupling_value)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return np.mean(losses)


def do_train_type(net, train_loader, optimizer, device, type):
    net.train()
    optimizer.zero_grad()
    losses = []

    # debug

    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in tqdm(enumerate(train_loader)):
        node = node.to(device)
        edge = edge.to(device)
        edge_index = edge_index.to(device)
        node_index = node_index.to(device)
        coupling_value = coupling_value.to(device)
        coupling_index = coupling_index.to(device)
        coupling_value = (coupling_value - norm[type][0]) / norm[type][1]
        predict = net(node, edge, edge_index, node_index, coupling_index)
        loss = criterion(predict, coupling_value)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return np.mean(losses)


def do_valid_type(net, valid_loader, device, type):
    net.eval()
    valid_num = 0
    valid_predict = []
    valid_coupling_type = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in tqdm(enumerate(valid_loader)):
        node = node.to(device)
        edge = edge.to(device)
        edge_index = edge_index.to(device)
        node_index = node_index.to(device)
        coupling_value = coupling_value.to(device)
        coupling_index = coupling_index.to(device)

        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index)
            predict = predict * norm[type][1] + norm[type][0]
            loss = criterion(predict, coupling_value)

        batch_size = len(infor)
        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_index[:, 2].data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += batch_size * loss.item()
        valid_num += batch_size

    assert (valid_num == len(valid_loader.dataset))

    valid_loss = valid_loss / valid_num

    # compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)

    diff = np.fabs(predict - coupling_value)
    m = diff.mean()
    log_m = np.log(m + 1e-2)

    return valid_loss, log_m


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    assert (len(lr) == 1)
    lr = lr[0]

    return lr


class StepScheduler():
    def __init__(self, pairs):
        super(StepScheduler, self).__init__()

        N = len(pairs)
        rates = []
        steps = []
        for n in range(N):
            steps.append(pairs[n][0])
            rates.append(pairs[n][1])

        self.rates = rates
        self.steps = steps

    def __call__(self, epoch):
        N = len(self.steps)
        lr = -1
        for n in range(N):
            if epoch >= self.steps[n]:
                lr = self.rates[n]
        return lr

    def __str__(self):
        string = 'Step Learning Rates\n' \
                 + 'rates=' + str(['%7.4f' % i for i in self.rates]) + '\n' \
                 + 'steps=' + str(['%7.0f' % i for i in self.steps]) + ''
        return string
