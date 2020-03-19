#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
import pandas as pd
from collections import Counter


def prepare_log(data_dir,datatype,window_size):
    num_sessions = 0
    Sequential = []
    Quantitative = []
    labels = []
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
                Sequential.append(Sequential_pattern)
                Quantitative_pattern.append(Quantitative_pattern)
                labels.append(line[i + window_size])
    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir, len(Sequential)))

    return Sequential, Quantitative, labels
                

class log_dataset(Dataset):
    def __init__(self, Sequential, Quantitative, labels):
        self.Sequential = Sequential
        self.Quantitative = Quantitative
        self.labels = labels
    def __len__(self):
        return len(self.Sequential)
    def __getitem__(self, idx):
        return torch.tensor(self.Sequential[idx],dtype=torch.float),torch.tensor(self.Quantitative[idx],dtype=torch.float),self.labels[idx]

