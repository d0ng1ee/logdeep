#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from logdeep.tools.train import Trainer
from logdeep.tools.predict import Predicter
from logdeep.tools.utils import *
from logdeep.models.lstm import loganomaly, deeplog, robustlog


# Config Parameters

options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cpu"

# Smaple
options['sample'] = "fix_window"
options['window_size'] = 10 # if fix_window

# Features
options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False
options['feature_num'] = sum([options['sequentials'], options['quantitatives'], options['semantics']])

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
options['max_epoch'] = 300
options['lr_step'] = (300,350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = "../result/deeplog/"

# Predict
options['model_path'] = "../result/deeplog/deeplog_last.pth"
options['num_candidates'] = 9

seed_everything(seed=1234)

def train():
    Model = deeplog(input_size=1, hidden_size=64, num_layers=2, num_keys=28)
    trainer = Trainer(Model, options)
    trainer.start_train()

def predict():
    Model = deeplog(input_size=1, hidden_size=64, num_layers=2, num_keys=28)
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()

if __name__ == "__main__":
    train()