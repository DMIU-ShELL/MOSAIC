#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


#   _________   _____  _________      _____                         __   
#  /   _____/  /  _  \ \_   ___ \    /  _  \    ____   ____   _____/  |_ 
#  \_____  \  /  /_\  \/    \  \/   /  /_\  \  / ___\_/ __ \ /    \   __\
#  /        \/    |    \     \____ /    |    \/ /_/  >  ___/|   |  \  |  
# /_______  /\____|__  /\______  / \____|__  /\___  / \___  >___|  /__|  
#         \/         \/        \/          \//_____/      \/     \/      

from copy import deepcopy
import multiprocessing as mp
import torch
import torch.nn as nn
import numpy as mp
from queue import Empty
from ..network import network_heads as nethead
from ..shell_modules.mmn.ssmask_utils import set_model_task, consolidate_mask, cache_masks, set_num_tasks_learned, get_mask, set_mask, erase_masks
from .BaseAgent import BaseAgent, BaseContinualLearnerAgent
from ..network.network_bodies import FCBody_SS, DummyBody_CL
from ..utils.torch_utils import random_seed, select_device, tensor
from ..utils.misc import Batcher
from ..component.replay import Replay
import numpy as np
import torch.nn.functional as F
import gym
from torch.distributions import Normal
import traceback
import torch.optim as optim