#######################################################################
# Copyright (C) 2017 Shangtong Zhang (zhangshangtong.cpp@gmail.com)   #
# Copyright (C) 2025 Saptarshi Nath, Christos Peridis,                #
# Eseoghene Benjamin, Andrea Soltoggio                                #
#                                                                     #
# Licensed under the Apache License, Version 2.0 (the "License");     #
# you may not use this file except in compliance with the License.    #
# You may obtain a copy of the License at                             #
#     http://www.apache.org/licenses/LICENSE-2.0                      #
#                                                                     #
# Unless required by applicable law or agreed to in writing, software #
# distributed under the License is distributed on an "AS IS" BASIS,   #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or     #
# implied. See the License for the specific language governing        #
# permissions and limitations under the License.                      #
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


