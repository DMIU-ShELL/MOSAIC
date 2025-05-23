#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class CTgraphConvBody(nn.Module):
    def __init__(self, in_channels=1):
        super(CTgraphConvBody, self).__init__()
        self.feature_dim = 16
        self.conv1 = layer_init(nn.Conv2d(in_channels, 4, kernel_size=5, stride=1))
        self.conv2 = layer_init(nn.Conv2d(4, 8, kernel_size=3, stride=1))
        self.conv3 = layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(4 * 4 * 16, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class MNISTConvBody(nn.Module):
    def __init__(self, in_channels=1, noisy_linear=False):
        super(MNISTConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        #self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        if noisy_linear:
            self.fc4 = NoisyLinear(6 * 6 * 64, self.feature_dim)
        else:
            self.fc4 = layer_init(nn.Linear(6 * 6 * 64, self.feature_dim))
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            self.fc4.reset_noise()

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        #y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x
    
class FCBody_Baseline(nn.Module): # fcbody for supermask superposition continual learning algorithm
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu):
        super(FCBody_Baseline, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))

        return x, ret_act

class FCBody_CL(nn.Module): # fcbody for continual learning setup
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu):
        super(FCBody_CL, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for layer in self.layers:
                x = self.gate(layer(x))
        return x, ret_act

from ..shell_modules.mmn.ssmask_utils import MultitaskMaskLinear, MultitaskMaskConv2d, ComposeMultitaskMaskLinear, CompBLC_MultitaskMaskLinear, ComposeMultitaskMaskConv2d
from ..shell_modules.mmn.ssmask_utils import NEW_MASK_RANDOM
from ..shell_modules.mmn.ssmask_utils import NEW_MASK_LINEAR_COMB
class FCBody_SS(nn.Module): # fcbody for supermask superposition continual learning algorithm
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu, discrete_mask=True, num_tasks=3, new_task_mask=NEW_MASK_RANDOM):
        super(FCBody_SS, self).__init__()
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList([MultitaskMaskLinear(dim_in, dim_out, discrete=discrete_mask, \
            num_tasks=num_tasks, new_mask_type=new_task_mask) \
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))

        return x, ret_act

class FCBody_SS_Comp(nn.Module): # fcbody for supermask superposition continual learning algorithm
    def __init__(self, state_dim, task_label_dim=None, hidden_units=(64, 64), gate=F.relu, discrete_mask=True, num_tasks=3, new_task_mask=NEW_MASK_RANDOM, seed=1, use_naive_blc=False):
        super(FCBody_SS_Comp, self).__init__()
        print("\n\n\n\nSTATE_DIM", state_dim)
        if task_label_dim is None:
            dims = (state_dim, ) + hidden_units
        else:
            dims = (state_dim + task_label_dim, ) + hidden_units

        self.layers = nn.ModuleList([CompBLC_MultitaskMaskLinear(dim_in, dim_out, discrete=discrete_mask, \
            num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed, use_naive_blc=use_naive_blc) \
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))

        return x, ret_act

class ConvBody_SS(nn.Module): # conv body for supermask lifelong learning algorithm
    def __init__(self, state_dim, lstm_hidden_size=200, hidden_units = (64, 64), kernels=[(3,3), (3,3)], strides=[1,1], paddings=[1,1], feature_dim=512, task_label_dim=None, gate=F.relu, discrete_mask=True, num_tasks=3, new_task_mask=NEW_MASK_RANDOM, seed=1):
        super(ConvBody_SS, self).__init__()

        print(state_dim)
        in_channels = state_dim[2] # assumes state_state with dim: num_channels x height x width
        self.conv1 = ComposeMultitaskMaskConv2d(in_channels, 32, kernel_size=8, stride=4, padding=1, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed)
        self.conv2 = ComposeMultitaskMaskConv2d(32, 64, kernel_size=4, stride=2, padding=1, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed)
        self.conv3 = ComposeMultitaskMaskConv2d(64, 64, kernel_size=3, stride=1, padding=1, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed)

        
        '''if task_label_dim is None: dims = (state_dim[0], ) + hidden_units
        else: dims = (state_dim[0] + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList(
            [
                ComposeMultitaskMaskConv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed) \
                for dim_in, dim_out, kernel, stride, padding in zip(dims[:-1], dims[1:], kernels, strides, paddings)
            ]
        )

        flattened_in = 128 * max(state_dim) * min(state_dim)
        self.layers.append(CompBLC_MultitaskMaskLinear(flattened_in, feature_dim, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed))

        print(f'Network: {self.layers}')'''

        #self.direction_emb = nn.Embedding(4, 4)
        #self.mission_emb = nn.Embedding(100, 16)
        #self.lstm = nn.LSTM(input_size=32 * 7 * 7 + 4 + 16, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        #self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer for output
        self.flatten = nn.Flatten()
        flattened_in = 32 * max(state_dim) * min(state_dim)
        self.fc = CompBLC_MultitaskMaskLinear(64 * 7 * 7, feature_dim, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed)

        self.gate = gate
        self.feature_dim = feature_dim
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'

        ret_act = []

        # conv1
        y = self.gate(self.conv1(x))
        #print(y.shape)
        if return_layer_output:
            ret_act.append(('{0}.conv.1'.format(prefix), y.detach().cpu().reshape(-1,)))
        
        # maxp1
        #y = self.maxp1(y)
        
        # conv2
        y = self.gate(self.conv2(y))
        #print(y.shape)
        if return_layer_output:
            ret_act.append(('{0}.conv.2'.format(prefix), y.detach().cpu().reshape(-1,)))
        
        # conv3
        y = self.gate(self.conv3(y))
        #print(y.shape)
        if return_layer_output:
            ret_act.append(('{0}.conv.3'.format(prefix), y.detach().cpu().reshape(-1,)))

        # flatten
        y = self.flatten(y)
        #y = y.view(y.shape[0], -1)
        #print(y.shape)
        if self.task_label_dim is not None:
            y = torch.cat([y, task_label], dim=1)
        
        # fc1
        y = self.gate(self.fc(y))
        if return_layer_output:
            ret_act.append(('{0}.fc.1'.format(prefix), y.detach().cpu().reshape(-1,)))
        return y, ret_act

    '''def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))

        return x, ret_act'''
    
class ConvBody_SS_Modified(nn.Module): # conv body for supermask lifelong learning algorithm
    def __init__(self, state_dim, kernels=[(3,3), (3,3)], strides=[1,1], paddings=[1,1], feature_dim=512, task_label_dim=None, gate=F.relu, discrete_mask=True, num_tasks=3, new_task_mask=NEW_MASK_RANDOM, seed=1, use_naive_blc=False):
        super(ConvBody_SS_Modified, self).__init__()

        print('State dim: ',state_dim)
        in_channels = 1#state_dim[2] # assumes state_state with dim: num_channels x height x width
        self.conv1 = ComposeMultitaskMaskConv2d(in_channels, 16, kernel_size=8, stride=4, padding=0, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed, use_naive_blc=use_naive_blc)
        self.conv2 = ComposeMultitaskMaskConv2d(16, 32, kernel_size=4, stride=2, padding=0, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed, use_naive_blc=use_naive_blc)
        
        '''if task_label_dim is None: dims = (state_dim[0], ) + hidden_units
        else: dims = (state_dim[0] + task_label_dim, ) + hidden_units
        self.layers = nn.ModuleList(
            [
                ComposeMultitaskMaskConv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed) \
                for dim_in, dim_out, kernel, stride, padding in zip(dims[:-1], dims[1:], kernels, strides, paddings)
            ]
        )

        flattened_in = 128 * max(state_dim) * min(state_dim)
        self.layers.append(CompBLC_MultitaskMaskLinear(flattened_in, feature_dim, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed))

        print(f'Network: {self.layers}')'''

        #self.direction_emb = nn.Embedding(4, 4)
        #self.mission_emb = nn.Embedding(100, 16)
        #self.lstm = nn.LSTM(input_size=32 * 7 * 7 + 4 + 16, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        #self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer for output
        self.flatten = nn.Flatten()
        flattened_in = 32 * max(state_dim) * min(state_dim)
        #self.fc = CompBLC_MultitaskMaskLinear(64 * 7 * 7, feature_dim, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed)
        self.fc = CompBLC_MultitaskMaskLinear(32 * 16 * 16, feature_dim, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed, use_naive_blc=use_naive_blc)

        self.gate = gate
        self.feature_dim = feature_dim
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'

        ret_act = []

        # conv1
        y = self.gate(self.conv1(x))
        if return_layer_output:
            ret_act.append(('{0}.conv.1'.format(prefix), y.detach().cpu().reshape(-1,)))
        
        # maxp1
        #y = self.maxp1(y)
        
        # conv2
        y = self.gate(self.conv2(y))
        #print(y.shape)
        if return_layer_output:
            ret_act.append(('{0}.conv.2'.format(prefix), y.detach().cpu().reshape(-1,)))

        # flatten
        y = self.flatten(y)
        #y = y.view(y.shape[0], -1)
        #print(y.shape)
        if self.task_label_dim is not None:
            y = torch.cat([y, task_label], dim=1)
        
        # fc1
        y = self.gate(self.fc(y))
        if return_layer_output:
            ret_act.append(('{0}.fc.1'.format(prefix), y.detach().cpu().reshape(-1,)))
        return y, ret_act

    '''def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        #if task_label is not None: x = torch.cat([x, task_label], dim=1)
       
        ret_act = []
        if return_layer_output:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))
                ret_act.append(('{0}.layers.{1}'.format(prefix, i), x))
        else:
            for i, layer in enumerate(self.layers):
                x = self.gate(layer(x))

        return x, ret_act'''
    
class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class DummyBody_CL(nn.Module):
    def __init__(self, state_dim, task_label_dim=None):
        super(DummyBody_CL, self).__init__()
        self.feature_dim = state_dim + (0 if task_label_dim is None else task_label_dim)
        self.task_label_dim = task_label_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix=''):
        if self.task_label_dim is not None:
            assert task_label is not None, '`task_label` should be set'
            x = torch.cat([x, task_label], dim=1)
        
        return x, []

class DummyBody_CL_Mask(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody_CL_Mask, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x, task_label=None, return_layer_output=False, prefix='', mask=None):
        return x, []
