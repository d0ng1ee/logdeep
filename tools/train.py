#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import torch
import pandas as pd 
import os
import gc
from torch.utils.data import DataLoader
import torch.nn as nn
from logdeep.dataset.log import prepare_log, log_dataset
from logdeep.models.lstm import Model_attention
import time
from tqdm import tqdm
from utils import *



class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        os.makedirs(self.save_dir,exist_ok = True)

        train_logs = prepare_log(self.data_dir, datatype = 'train', window_size=self.window_size)
        # valid_logs = prepare_log(self.data_dir, datatype = 'valid',window_size=self.window_size)

        train_dataset = log_dataset(log = train_logs, seq=True,quan=True)
        # valid_dataset = log_dataset(log = valid_logs, seq=True,quan=True)

        del train_logs
        # del valid_logs
        gc.collect()

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        # self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self.num_train_log = len(train_dataset)
        # self.num_valid_log = len(valid_dataset)

        # print('Find %d train logs, %d validation logs' % (self.num_train_log, self.num_valid_log))
        print('Train batch size %d ,Validation batch size %d' % (options['batch_size'], options['batch_size']))

        self.device = options['device']
        self.model = model.to(self.device)

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=options['lr'], momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=options['lr'], betas=(0.9, 0.999),
                                              )
        else:
            raise NotImplementedError

        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']

        self.accumulation_step = options['accumulation_step']

        self.start_epoch = 0
        self.best_loss=1e10
        self.best_score = -1
        self.max_epoch = options['max_epoch']
        save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: [] for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: [] for key in ["epoch", "lr", "time", "f1_score"]}
        }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv", index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" % (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, (seq, quan, label) in enumerate(tbar):
            seq = seq.clone().detach().view(-1, self.window_size, 1).to(self.device)
            quan = quan.clone().detach().view(-1, 28, 1).to(self.device)
            output = self.model(seq, quan, device = self.device)
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))

        self.log['train']['loss'].append(total_losses / num_batch)

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        scores = 0
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, batch in enumerate(tbar):
            with torch.no_grad():
                x, y = [elem.to(self.device,non_blocking=True) for elem in batch]
                pred = self.model(x, y,device = self.device)
                score = macro_recall(pred, y)
                scores += score
        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_train_img))
        macro_recall_score = scores / num_batch
        print("Validation macro_recall_score:", macro_recall_score)
        self.log['valid']['size'].append(self.train_size)
        self.log['valid']['score'].append(macro_recall_score)
        
        if macro_recall_score > self.best_macro_recall_score:
            print("Find a better model with score, {} -> {}".format(self.best_macro_recall_score, macro_recall_score))
            self.best_macro_recall_score = macro_recall_score
            self.save_checkpoint(epoch, save_optimizer=False, suffix="best_macro_recall_score")
        else:
            print("This model {}, best model {}".format(macro_recall_score, self.best_macro_recall_score))

    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch ==0:
               self.optimizer.param_groups[0]['lr'] /=32
            if epoch in [1,2,3,4,5]:
               self.optimizer.param_groups[0]['lr'] *=2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.train(epoch)
            # self.valid(epoch)
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            if epoch>=200 and epoch % 3 == 2:
                self.save_checkpoint(epoch, save_optimizer=True, suffix="epoch" + str(epoch))
            self.save_log()

# todo:把配置参数单独放到一个文件中保存修改

def main():
    options = dict()
    options['data_dir'] = '../data/'
    options['window_size'] = 10
    options['debug'] = False

    options['device'] = "cpu"

    options['batch_size'] = 2048
    options['accumulation_step'] = 1

    options['optimizer'] = 'adam'
    options['lr'] = 0.001
    options['max_epoch'] = 300
    options['lr_step'] = (200,250,275)
    options['lr_decay_ratio'] = 0.1

    options['resume_path'] = None
    options['model_name'] = "model_test"
    options['save_dir'] = "../result/model_test/"

    # --- Model ---

    Model = Model_attention(input_size=1, hidden_size=64, num_layers=2, num_keys=28)
    trainer = Trainer(Model, options)
    trainer.start_train()


if __name__ == "__main__":
    main()