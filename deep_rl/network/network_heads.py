#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import BaseNet
from .network_bodies import *
from ..utils.config import Config
import torch.nn as nn
from ..utils.torch_utils import tensor

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class VanillaNet_CL(nn.Module, BaseNet):
    def __init__(self, output_dim, task_label_dim, body):
        super(VanillaNet_CL, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, x, task_label=None, to_numpy=False):
        x = tensor(x)
        task_label = tensor(task_label)
        phi = self.body(x, task_label)
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class DuelingNet_CL(nn.Module, BaseNet):
    def __init__(self, action_dim, task_label_dim, body):
        super(DuelingNet_CL, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, x, task_label=None, to_numpy=False):
        x = tensor(x)
        task_label = tensor(task_label)
        phi = self.body(x, task_label)
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.cpu().detach().numpy()
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        #self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        #self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        self.fc_action = nn.Linear(actor_body.feature_dim, action_dim)
        self.fc_critic = nn.Linear(critic_body.feature_dim, 1)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class ActorCriticNetSS(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, new_task_mask, discrete_mask=True):
        super(ActorCriticNetSS, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body

        self.fc_action = MultitaskMaskLinear(actor_body.feature_dim, action_dim, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.fc_critic = MultitaskMaskLinear(critic_body.feature_dim, 1, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)

        ap = [p for p in self.actor_body.parameters() if p.requires_grad is True]
        ap += [p for p in self.fc_action.parameters() if p.requires_grad is True]
        self.actor_params = ap

        cp = [p for p in self.critic_body.parameters() if p.requires_grad is True]
        cp += [p for p in self.fc_critic.parameters() if p.requires_grad is True]
        self.critic_params = cp

        self.phi_params = [p for p in self.phi_body.parameters() if p.requires_grad is True]

class ActorCriticNetSSComp(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, new_task_mask, discrete_mask=True, seed=1, use_naive_blc=False):
        super(ActorCriticNetSSComp, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body

        self.fc_action = CompBLC_MultitaskMaskLinear(actor_body.feature_dim, action_dim, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed, use_naive_blc=use_naive_blc)
        self.fc_critic = CompBLC_MultitaskMaskLinear(critic_body.feature_dim, 1, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask, seed=seed, use_naive_blc=use_naive_blc)

        ap = [p for p in self.actor_body.parameters() if p.requires_grad is True]
        ap += [p for p in self.fc_action.parameters() if p.requires_grad is True]
        self.actor_params = ap

        cp = [p for p in self.critic_body.parameters() if p.requires_grad is True]
        cp += [p for p in self.fc_critic.parameters() if p.requires_grad is True]
        self.critic_params = cp

        self.phi_params = [p for p in self.phi_body.parameters() if p.requires_grad is True]

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        action = self.actor(phi)
        if to_numpy:
            return action.cpu().detach().numpy()
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, to_numpy=False):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v

# actor-critic net for continual learning where tasks are labelled using
# supermask superposition algorithm
class GaussianActorCriticNet_SS(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931 #-20.
    LOG_STD_MAX = 0.4055 #1.3
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random'):
        super(GaussianActorCriticNet_SS, self).__init__()
        # continuous values mask is used for Gaussian (continuous control policies)
        discrete_mask = False
        self.network = ActorCriticNetSS(state_dim, action_dim, phi_body, actor_body, critic_body, \
            num_tasks, new_task_mask, discrete_mask=discrete_mask)
        self.task_label_dim = task_label_dim

        self.network.fc_log_std = MultitaskMaskLinear(self.network.actor_body.feature_dim, \
            action_dim, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.network.actor_params += [p for p in self.network.fc_log_std.parameters() if p.requires_grad is True]
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        #mean = F.tanh(self.network.fc_action(phi_a))
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        log_std = self.network.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, GaussianActorCriticNet_SS.LOG_STD_MIN, \
            GaussianActorCriticNet_SS.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), \
                ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output
    
class GaussianActorCriticNet_SS_Comp(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931 #-20.
    LOG_STD_MAX = 0.4055 #1.3
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random',
                 seed=1):
        super(GaussianActorCriticNet_SS_Comp, self).__init__()
        # continuous values mask is used for Gaussian (continuous control policies)
        discrete_mask = False
        self.network = ActorCriticNetSSComp(state_dim, action_dim, phi_body, actor_body, critic_body, \
            num_tasks, new_task_mask, discrete_mask=discrete_mask, seed=seed)
        self.task_label_dim = task_label_dim

        self.network.fc_log_std = CompBLC_MultitaskMaskLinear(self.network.actor_body.feature_dim, \
            action_dim, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.network.actor_params += [p for p in self.network.fc_log_std.parameters() if p.requires_grad is True]
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        #mean = F.tanh(self.network.fc_action(phi_a))
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        log_std = self.network.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, GaussianActorCriticNet_SS_Comp.LOG_STD_MIN, \
            GaussianActorCriticNet_SS_Comp.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), \
                ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output
    

class GaussianActorCriticNet_SS_Comp_FixedStd(nn.Module):
    LOG_STD_MIN = -0.6931  # Lower bound for log std
    LOG_STD_MAX = 0.4055   # Upper bound for log std
    FIXED_LOG_STD = -0.5    # Set your fixed log standard deviation value

    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random',
                 seed=1):
        super(GaussianActorCriticNet_SS_Comp_FixedStd, self).__init__()
        
        discrete_mask = False
        
        # Initialize the Actor-Critic Network
        self.network = ActorCriticNetSSComp(state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, new_task_mask, discrete_mask=discrete_mask, seed=seed)
        self.task_label_dim = task_label_dim

        log_std_values = torch.zeros(action_dim, dtype=torch.float32)
        log_std_values[-1] = -0.5  # Gripper has lower variance
        self.fixed_log_std = nn.Parameter(log_std_values, requires_grad=False)
        self.to(Config.DEVICE)  # Move the model to the appropriate device

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()

        v = self.network.fc_critic(phi_v)

        # Use fixed standard deviation from Composuites
        std = torch.exp(self.fixed_log_std)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()
        
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std),
                            ('policy_action', action), ('value_fn', v)]
        
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        
        return mean, action, log_prob, entropy, v, layers_output


class GaussianActorCriticNet_FixedStd(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931  # -20.
    LOG_STD_MAX = 0.4055  # 1.3
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet_FixedStd, self).__init__()
        
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim
        
        # Remove masking and replace with fixed log stdlog_std_values = torch.zeros(action_dim, dtype=torch.float32)
        log_std_values = torch.zeros(action_dim, dtype=torch.float32)
        log_std_values[-1] = -0.5  # Gripper has lower variance
        self.fixed_log_std = nn.Parameter(log_std_values, requires_grad=False)
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        layers_output = []
        phi, out = self.network.phi_body(obs, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        
        #mean = torch.nn.functional.tanh(self.network.fc_action(phi_a))
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        
        # Use fixed standard deviation from CompoSuite
        std = torch.exp(self.fixed_log_std)
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), 
                              ('policy_action', action), ('value_fn', v)]
        
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        
        return mean, action, log_prob, entropy, v, layers_output



# actor-critic net for continual learning where tasks are labelled
class GaussianActorCriticNet_CL(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931 #-20.
    LOG_STD_MAX = 0.4055 #1.3
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet_CL, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim

        self.network.fc_log_std = layer_init(nn.Linear(self.network.actor_body.feature_dim, \
            action_dim), 1e-3)
        self.network.actor_params += [p for p in self.network.fc_log_std.parameters() if p.requires_grad is True]
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        #mean = F.tanh(self.network.fc_action(phi_a))
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        log_std = self.network.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, GaussianActorCriticNet_CL.LOG_STD_MIN, \
            GaussianActorCriticNet_CL.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), \
                ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def predict(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v

# actor-critic net for continual learning where tasks are labelled using
# supermask superposition algorithm
class CategoricalActorCriticNet_SS(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random'
                 ):
        super(CategoricalActorCriticNet_SS, self).__init__()
        self.network = ActorCriticNetSS(state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, new_task_mask)
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output

class CategoricalActorCriticNet_SS_Comp(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random',
                 seed=1,
                 use_naive_blc=False):
        super(CategoricalActorCriticNet_SS_Comp, self).__init__()
        self.network = ActorCriticNetSSComp(state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, new_task_mask, seed=seed, use_naive_blc=use_naive_blc)
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output


# actor-critic net for continual learning where tasks are labelled
class CategoricalActorCriticNet_CL(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet_CL, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output

class TD3Net(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 actor_body_fn,
                 critic_body_fn,
                 actor_opt_fn,
                 critic_opt_fn,
                 ):
        super(TD3Net, self).__init__()
        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()

        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        obs = tensor(obs)
        return torch.tanh(self.fc_action(self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2
    
class SACNet(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 actor_body_fn,
                 critic_body_fn,
                 value_body_fn,
                 actor_opt_fn,
                 critic_opt_fn,
                 value_opt_fn,
                 ):
        super(SACNet, self).__init__()
        # I think this code might be redundant and could be replaced with what is in
        # the Gaussian actor critic net class. But regardless I think for now this should
        # do the job.

        # Network bodies
        self.actor_body = actor_body_fn()
        # SAC uses two critic networks
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()
        self.value_body = value_body_fn()

        # Network final output layers (maps body to the action space dimension)
        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)
        self.fc_value = layer_init(nn.Linear(self.value_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) + \
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())
        self.value_params = list(self.value_body.parameters()) + list(self.fc_value.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.value_opt = value_opt_fn(self.value_params)

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.to(Config.DEVICE)

    def forward(self, obs):
        # Takes an observation and outputs the action from policy network (actor network)
        obs = tensor(obs)

        # tanh is used to sqush the output values between -1 and 1 (useful for continous action spaces)
        return torch.tanh(self.fc_action(self.actor_body(obs)))
    
    def sample(self, obs, reparameterize=True):
        obs = tensor(obs)
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        # NOTE: Reparameterize trick is used to carry out backpropagation through a differntiable
        # noise source (i.e, the gaussian (normal) distribution we are using in this code)
        # and then transforming it using a determinsitic function. This incorporates the actor networks
        # output mean and std to produce the sampled action.
        #
        # In SAC we use it to enable efficient and differentiable sampling of actions druing both exploration
        # and policy optimization. It allows the algorithm to explore and learn from stochastic actions while
        # maintaining differentiability for gradient-based updates, facilitatiing effective policy learning.
        # It is core to the SAC implementation
        if reparameterize == True:
            action = dist.rsample()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def q(self, obs, a):
        # Computes the Q-values for the given state-action pairs using hte critic networks
        obs = tensor(obs) # state
        a = tensor(a) # action
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2

    def value(self, obs):
        # computes the target value for the value network during the update step.
        # estimates the state-value function using the next state observation (obs) and next action sampled from the target policy
        obs = tensor(obs)
        return self.fc_value(self.value_body(obs))