# author:donglee
# email:donglee-afar@outlook.com
import torch
import pandas as pd 
import os
import gc

class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        os.makedirs(self.save_dir,exist_ok = True)

        logs = read_io

        train_dataset
        valid_dataset

        del 
        gc.collect()

        self.train_loader
        self.valid_loader
        self.batch_size = options['batch_size']

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' % (self.num_train_log, self.num_valid_log))
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