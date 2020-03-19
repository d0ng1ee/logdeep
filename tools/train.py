#!/usr/bin/env python
# -*- coding: utf-8 -*-

sys.path.append('../')
import torch
import pandas as pd 
import os
import gc
from torch.utils.data import DataLoader
from logdeep.dataset.log import prepare_log, log_dataset
from logdeep.models import Model_attention



class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        os.makedirs(self.save_dir,exist_ok = True)

        train_logs = prepare_log(self.data_dir, datatype = 'train', windows_size=self.windows_size)
        valid_logs = prepare_log(self.data_dir, datatype = 'valid',windows_size=self.windows_size)

        train_dataset = log_dataset(train_logs)
        valid_dataset = log_dataset(valid_logs)

        del train_logs
        del valid_logs
        gc.collect()

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.batch_size = options['batch_size']

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        # print('Find %d train logs, %d validation logs' % (self.num_train_log, self.num_valid_log))
        print('Train batch size %d ,Validation batch size %d' % (options['batch_size'], options['batch_size']))

        self.device = options['device']
        self.model = model.to(self.device)

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(param_groups, lr=options['lr'], momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(param_groups, lr=options['lr'], betas=(0.9, 0.999),
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
            "train": {loss_name: [] for loss_name in loss_stats + ["epoch", "lr", "time","size"]},
            "valid": {key: [] for key in ["epoch", "time","size","score"]}
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
        self.best_macro_recall_score = checkpoint['best_macro_recall_score']
        if self.parallel:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
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
        total_losses = {loss: 0 for loss in loss_stats}
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        scores = 0
        for i, batch in enumerate(tbar):

            for loss_name in total_losses:
                total_losses[loss_name] += float(loss)    
            loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses['loss'] / (i + 1)))
        for k in total_losses:
            self.log['train'][k].append(total_losses[k] / num_batch)
        self.log['train']['size'].append(self.train_size)
        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_train_img))
        for k in total_losses:
            print("{} : {}".format(k, total_losses[k] / num_batch))
        print()

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        # self.model.eval()
        total_losses = {loss: 0 for loss in loss_stats}
        scores = 0
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, batch in enumerate(tbar):
            with torch.no_grad():
                x, y = [elem.to(self.device,non_blocking=True) for elem in batch]
                pred = self.model(x, y)
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
            self.valid(epoch)
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            if epoch>=30 and epoch % 3 == 2:
                self.save_checkpoint(epoch, save_optimizer=True, suffix="epoch" + str(epoch))
            self.save_log()

# todo:把配置参数单独放到一个文件中保存修改
def main():
    options = dict()
    options['datadir'] = '../data/'
    options['debug'] = False

    options['device'] = "cpu"

    # options['parallel']=True
    options['batch_size'] = 2048
    options['accumulation_step'] = 1

    options['optimizer'] = 'sgd'
    options['lr'] = 0.05
    options['max_epoch'] = 150
    options['lr_step'] = (75,120,135,145)
    options['lr_decay_ratio'] = 0.2

    options['resume_path'] = None
    options['model_name'] = "model_test"
    options['save_dir'] = "./result/model_test/"

    # --- Model ---

    Model = Model_attention()
    trainer = Trainer(Model, options)
    trainer.start_train()


if __name__ == "__main__":
    main()