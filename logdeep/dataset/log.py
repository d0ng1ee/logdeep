#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
import pandas as pd
from collections import Counter


def prepare_log(data_dir,datatype,window_size):
    num_sessions = 0
    Sequentials = []
    Quantitatives = []
    labels = []
    if datatype == 'train':
        data_dir += 'hdfs/hdfs_train'
    with open(data_dir, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line =  tuple(map(lambda n: n - 1,map(int, line.strip().split())))

            for i in range(len(line) - window_size):
                Sequential_pattern = line[i:i + window_size]
                Quantitative_pattern = [0]*28
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                Sequentials.append(Sequential_pattern)
                Quantitatives.append(Quantitative_pattern)
                labels.append(line[i + window_size])
    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir, len(Sequentials)))

    return Sequentials, Quantitatives, labels
                

class log_dataset(Dataset):
    def __init__(self, log, seq=True,quan=False):
        if seq:
            self.Sequentials = log[0]
        if quan:
            self.Quantitatives = log[1]
        self.labels = log[-1]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.Sequentials[idx],dtype=torch.float),torch.tensor(self.Quantitatives[idx],dtype=torch.float),self.labels[idx]


if __name__ == '__main__':
    data_dir = '../../data/hdfs/hdfs_train'
    window_size=10
    train_logs = prepare_log(data_dir=data_dir, datatype='train', window_size=window_size)
    train_dataset = log_dataset(log =train_logs,seq=True,quan=True)
    print(train_dataset[0])
    print(train_dataset[100])
