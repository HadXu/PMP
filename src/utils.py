# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       haxu
   date：          2019-06-20
-------------------------------------------------
   Change Activity:
                   2019-06-20:
-------------------------------------------------
"""
__author__ = 'haxu'

import os
import sys
import numpy as np
import pandas as pd
from xyz2mol import get_atomicNumList, xyz2mol


def mol_from_axyz(symbol, xyz):
    charged_fragments = True
    quick = True
    charge = 0
    atom_no = get_atomicNumList(symbol)
    mol = xyz2mol(atom_no, xyz, charge, charged_fragments, quick)
    return mol


def read_list_from_file(list_file, comment='#'):
    with open(list_file) as f:
        lines = f.readlines()

    strings = []
    for line in lines:
        if comment is not None:
            s = line.split(comment, 1)[0].strip()
        else:
            s = line.strip()

        if s != '':
            strings.append(s)

    return strings


def read_champs_xyz(xyz_file):
    line = read_list_from_file(xyz_file, comment=None)
    num_atom = int(line[0])
    xyz = []
    symbol = []
    for n in range(num_atom):
        l = line[1 + n]
        l = l.replace('\t', ' ').replace('  ', ' ')
        l = l.split(' ')
        symbol.append(l[0])
        xyz.append([float(l[1]), float(l[2]), float(l[3]), ])

    return symbol, xyz


def map_atom_info(df):
    df_structures = pd.read_csv('../input/champs-scalar-coupling/structures_621.csv')
    for atom_idx in range(2):
        df = pd.merge(df, df_structures, how='left',
                      left_on=['molecule_name', f'atom_index_{atom_idx}'],
                      right_on=['molecule_name', 'atom_index'])

        df = df.drop('atom_index', axis=1)
        df = df.rename(columns={'atom': f'atom_{atom_idx}',
                                'x': f'x_{atom_idx}',
                                'y': f'y_{atom_idx}',
                                'z': f'z_{atom_idx}'})

    atom_count = df_structures.groupby(['molecule_name', 'atom']).size().unstack(fill_value=0)
    df = pd.merge(df, atom_count, how='left', left_on='molecule_name', right_on='molecule_name')
    del df_structures
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(
                        np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(
                        np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df









