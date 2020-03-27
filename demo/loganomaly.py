#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import deeplog, loganomaly, robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *


# Config Parameters

options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cpu"

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# Features
options['sequentials'] = True
options['quantitatives'] = True
options['semantics'] = False
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 28

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 370
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "loganomaly"
options['save_dir'] = "../result/loganomaly/"

# Predict
options['model_path'] = "../result/loganomaly/loganomaly_epoch299.pth"
options['num_candidates'] = 9

seed_everything(seed=1234)


def train():
    Model = loganomaly(input_size=options['input_size'],
                       hidden_size=options['hidden_size'],
                       num_layers=options['num_layers'],
                       num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = loganomaly(input_size=options['input_size'],
                       hidden_size=options['hidden_size'],
                       num_layers=options['num_layers'],
                       num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
