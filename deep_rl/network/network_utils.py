#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from ..utils import *
import torch.cuda.amp as amp

class BaseNet:
    def __init__(self):
        pass


# fp16 computation
#def layer_init(layer, w_scale=1.0):
#    nn.init.orthogonal_(layer.weight.data)
#    layer.weight.data = layer.weight.data.to(torch.float16).mul_(w_scale)#layer.weight.data.mul_(w_scale)
#    nn.init.constant_(layer.bias.data, 0)
#    layer.bias.data = layer.bias.data.to(torch.float16)
#    return layer

def layer_init(layer, w_scale=1.0, dtype=torch.float16):
    with torch.no_grad():
        torch.nn.init.orthogonal_(layer.weight)
        layer.weight.mul_(w_scale).to(dtype)
        torch.nn.init.constant_(layer.bias, 0)
        layer.bias.to(dtype)
    return layer
    


# fp32 computation
#def layer_init(layer, w_scale=1.0):
#    nn.init.orthogonal_(layer.weight.data)
#    layer.weight.data.mul_(w_scale)
#    nn.init.constant_(layer.bias.data, 0)
#    return layer