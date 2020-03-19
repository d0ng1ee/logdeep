import torch
import os
import numpy as np
import random
import sklearn
from torch import nn
import torch.nn.functional as F

def save_parameters(options,filename):
    with open(filename,"w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key,options[key]))