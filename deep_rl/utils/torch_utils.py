#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .config import *
import torch
import os
import random

def select_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        #onfig.DEVICE = torch.device('cuda:%d' % (gpu_id))
        Config.DEVICE = torch.device('cuda')
    else:
        Config.DEVICE = torch.device('cpu')


# COME BACK AND CHECK IF 32 IS NEEDED OR IF WE CAN USE FLOAT16
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    #x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x


def range_tensor(end):
    return torch.arange(end).to(Config.DEVICE)


def to_np(t):
    return t.cpu().detach().numpy()


def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


