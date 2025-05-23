#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


# ____________________________       _____                         __   
# \______   \______   \_____  \     /  _  \    ____   ____   _____/  |_ 
#  |     ___/|     ___//   |   \   /  /_\  \  / ___\_/ __ \ /    \   __\
#  |    |    |    |   /    |    \ /    |    \/ /_/  >  ___/|   |  \  |  
#  |____|    |____|   \_______  / \____|__  /\___  / \___  >___|  /__|  
#                             \/          \//_____/      \/     \/      

from .BaseAgent import BaseAgent, BaseContinualLearnerAgent
from ..utils.torch_utils import random_seed, tensor
from ..utils.misc import Batcher
from ..component.replay import Replay
from ..shell_modules.mmn.ssmask_utils import set_model_task, consolidate_mask, cache_masks, set_num_tasks_learned, get_mask, set_mask, GetSubnetDiscrete, GetSubnetContinuous, GetSubnetContinuousV2, mask_init, signed_constant, consolidate_comm_mask, update_comm_masks, erase_masks, get_current_betas

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from scipy.stats import wasserstein_distance
import traceback
from sklearn.cluster import Birch
import tensorboardX as tf
from collections import deque
import pandas as pd
import time
import gym

# Base PPO agent implementations
class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def iteration(self):
        config = self.config
        rollout = []
        states = self.states
        for _ in range(config.rollout_length):
            actions, log_probs, _, values = self.network.predict(states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - terminals])
            states = next_states

        self.states = states
        pending_value = self.network.predict(states)[-1]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals * next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        advantages = (advantages - advantages.mean()) / advantages.std()

        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, log_probs, entropy_loss, values = self.network.predict(sampled_states, sampled_actions)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps


class PPOContinualLearnerAgent(BaseContinualLearnerAgent):
    def __init__(self, config):
        BaseContinualLearnerAgent.__init__(self, config)
        self.config = config
        self.task = None if config.task_fn is None else config.task_fn()
        if config.eval_task_fn is None:
            self.evaluation_env = None
        else:
            self.evaluation_env = config.eval_task_fn(config.log_dir)
            self.task = self.evaluation_env if self.task is None else self.task

        tasks = []
        if config.task_ids is not None:
            tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)
            print(config.task_ids)
            [print(dumb) for dumb in tasks_]
            tasks = [tasks_[task_id] for task_id in config.task_ids]
            del tasks_
        elif config.task_paths is not None:
            tasks_ = self.task.get_all_tasks(config.cl_requires_task_label)

            task_paths_set = set(tuple(path) for path in config.task_paths)

            for task_dict in tasks_:
                if tuple(task_dict['task']) in task_paths_set:
                    tasks.append(task_dict)
        
        print(tasks)
        #self.config.task_ids = []
        for task in tasks:
            # Convert one-hot encoding to integer
            task_label = np.array(task['task_label'])
            task_label_integer = np.argmax(task_label, axis=0)
            
            print(f"Task Name: {task['name']}, Task Label (One-Hot): {task['task_label']}, Task Label (Integer): {task_label_integer}, Path: {task['task']}")
            #self.config.task_ids.append(task_label_integer)

        
        
        self.config.cl_tasks_info = tasks
        label_dim = None if not config.use_task_label else len(tasks[0]['task_label'])
        #label_dim = 0 if tasks[0]['task_label'] is None else len(tasks[0]['task_label']) #CHANGE THAT FOR THE
        self.task_label_dim = label_dim



        # set seed before creating network to ensure network parameters are
        # same across all shell agents
        #torch.manual_seed(config.seed)
        
        random_seed(config.backbone_seed)   # Chris

        
        #Ihave changed the bellow commande by substitue the label dim with embedding dim
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim, label_dim)#self.task_emb_size)
        _params = list(self.network.parameters())
        self.opt = config.optimizer_fn(_params, config.lr)

        #self.scheduler = lr_scheduler.StepLR(self.opt, step_size=config.lr_decay_step, gamma=config.lr_decay_gamma)
        #lambda_lr = lambda: config.lr / ((self.total_steps + 1) / 512) ** 0.5
        #self.scheduler = lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda_lr)

        self.total_steps = 0
        #for name, para in self.network.named_parameters():
        #    print('{}: {}'.format(name, para.shape))

        # Network initialised with backbone network so we set the seed with the experiment seed
        random_seed(config.seed)    # Chris

        self.episode_rewards = np.zeros(config.num_workers)
        self.last_episode_rewards = np.zeros(config.num_workers)
        # running reward: used to compute average across all episodes
        # that may occur in an iteration
        self.running_episodes_rewards = [[] for _ in range(config.num_workers)]
        self.iteration_rewards = np.zeros(config.num_workers)

        self.states = self.task.reset()
        print(f'FIRST PRINT: {self.states.shape}')
        self.states = config.state_normalizer(self.states)
        print(f'SECOND PRINT: {self.states.shape}')
        
        self.layers_output = None
        self.data_buffer = Replay(memory_size=int(1.5*1e2), batch_size=130) #batch_size=130

        self.curr_train_task_label = None
        self.curr_eval_task_label = None

        # other performance metric (specifically for metaworld environment)
        if self.task.name == config.ENV_METAWORLD or self.task.name == config.ENV_CONTINUALWORLD or self.task.name == config.ENV_COMPOSUITE:
            self._rollout_fn = self._rollout_metaworld
            self.episode_success_rate = np.zeros(config.num_workers)
            self.last_episode_success_rate = np.zeros(config.num_workers)
            # used to compute average across all episodes that may occur in an iteration
            self.running_episodes_success_rate = [[] for _ in range(config.num_workers)]
            self.iteration_success_rate = np.zeros(config.num_workers)
            print('')
        else:
            self._rollout_fn = self._rollout_normal
            self.episode_success_rate = None
            self.last_episode_success_rate = None
            self.running_episodes_success_rate = None
            self.iteration_success_rate = None

    def iteration(self):
        '''This function performs the training iteration.
        It is where the learner of the agent (the neural network) is being optimized'''

        start_time = time.time()
        
        config = self.config
        rollout = []
        states = self.states

        # Rollout function.
        states, rollout = self._rollout_fn(states)
        #self.config.logger.info(f'----------------------- Rollout function complete in {time.time() - start_time} seconds -----------------------\n')

        self.states = states
        pending_value = self.network.predict(states)[-2]#self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount * terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]
        
        #self.config.logger.info(f'----------------------- Rollout processing complete in {time.time() - start_time} seconds -----------------------\n')

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        eps = 1e-6
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        # Setup logging
        grad_norm_log = []
        policy_loss_log = []
        value_loss_log = []
        log_probs_log = []
        entropy_log = []
        ratio_log = []

        if config.use_full_batch:
            print('Using full batch')
            for _ in range(config.optimization_epochs):
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(states, actions)

                ratio = (log_probs - log_probs_old).exp()
                obj = ratio * advantages
                obj_clipped = ratio.clamp(1.0 - config.ppo_ratio_clip, 1.0 + config.ppo_ratio_clip) * advantages
                
                # Compute policy loss
                policy_loss = -torch.min(obj, obj_clipped).mean(0) - config.entropy_weight * entropy_loss.mean()
                
                # Compute value loss
                value_loss = 0.5 * (returns - values).pow(2).mean()

                # Compute KL divergence for early stopping
                approx_kl = (log_probs_old - log_probs).mean().item()
                if approx_kl > 1.5 * config.target_kl:
                    break  # Stop training if KL exceeds threshold

                # Logging
                log_probs_log.append(log_probs.detach().cpu().numpy().mean())
                entropy_log.append(entropy_loss.detach().cpu().numpy().mean())
                ratio_log.append(ratio.detach().cpu().numpy().mean())
                policy_loss_log.append(policy_loss.detach().cpu().numpy())
                value_loss_log.append(value_loss.detach().cpu().numpy())

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norm_log.append(norm_.detach().cpu().numpy())
                self.opt.step()

        # PPO with mini batching
        else:
            print('Using mini batching')
            batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
            for _ in range(config.optimization_epochs):
                batcher.shuffle()
                while not batcher.end():
                    batch_indices = batcher.next_batch()[0]
                    batch_indices = tensor(batch_indices).long()
                    sampled_states = states[batch_indices]
                    sampled_actions = actions[batch_indices]
                    sampled_log_probs_old = log_probs_old[batch_indices]
                    sampled_returns = returns[batch_indices]
                    sampled_advantages = advantages[batch_indices]

                    _, _, log_probs, entropy_loss, values, outs = self.network.predict(
                        sampled_states,
                        sampled_actions,
                        return_layer_output=True
                    )

                    ratio = (log_probs - sampled_log_probs_old).exp()
                    obj = ratio * sampled_advantages
                    obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                            1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                    
                    # Compute losses
                    policy_loss = -torch.min(obj, obj_clipped).mean(0) - config.entropy_weight * entropy_loss.mean()
                    value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                    # Logging
                    log_probs_log.append(log_probs.detach().cpu().numpy().mean())
                    entropy_log.append(entropy_loss.detach().cpu().numpy().mean())
                    ratio_log.append(ratio.detach().cpu().numpy().mean())
                    policy_loss_log.append(policy_loss.detach().cpu().numpy())
                    value_loss_log.append(value_loss.detach().cpu().numpy())

                    self.opt.zero_grad()
                    (policy_loss + value_loss).backward()
                    norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    grad_norm_log.append(norm_.detach().cpu().numpy())
                    self.opt.step()

        #self.config.logger.info(f'----------------------- Optimization epochs complete in {time.time() - start_time} seconds -----------------------\n')

        # Update total steps
        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return {'grad_norm': grad_norm_log, 'policy_loss': policy_loss_log, \
            'value_loss': value_loss_log, 'log_prob': log_probs_log, 'entropy': entropy_log, \
            'ppo_ratio': ratio_log}

    def _rollout_normal(self, states):
        '''It runs the agent on the environment and collects SAR data to staore in the Replay
        Buffer'''
        start_time = time.time()
        # Clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]
        
        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states)
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)

            # Save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions.cpu(), rewards, terminals, next_states])

            rollout.append(
                [
                    states,
                    values.detach(), 
                    actions.detach(), 
                    log_probs.detach(),
                    rewards, 1 - terminals
                ]
            )
            states = next_states

        #self.config.logger.info(f'----------------------- Rollout function data collection complete in {time.time() - start_time} seconds -----------------------\n')

        # Compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])

        return states, rollout
    
    # rollout for metaworld and continualworld environments. it is similar to normal
    # rollout with the inclusion of the capture of success rate metric.
    def _rollout_metaworld(self, states):
        # clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]
        self.running_episodes_success_rate = [[] for _ in range(self.config.num_workers)]

        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states)
            next_states, rewards, terminals, infos = self.task.step(actions.cpu().detach().numpy())
            success_rates = [info['success'] for info in infos]
            self.episode_rewards += rewards
            self.episode_success_rate += success_rates
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
                    self.episode_success_rate[i] = (self.episode_success_rate[i] > 0).astype(np.uint8)
                    self.running_episodes_success_rate[i].append(self.episode_success_rate[i])
                    self.last_episode_success_rate[i] = self.episode_success_rate[i]
                    self.episode_success_rate[i] = 0
            next_states = config.state_normalizer(next_states)

            # save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions.cpu(), rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])
            self.iteration_success_rate[i] = self._avg_episodic_perf(self.running_episodes_success_rate[i])

        return states, rollout


    # BACKUP (Iterations and rollout functions using the task label)
    """
    def iteration(self):
        '''This function performs the training iteration.
        It is where the learner of the agent (the neural network) is being optimized'''
        config = self.config
        rollout = []
        states = self.states
        if self.curr_train_task_label is not None:
            task_label = self.curr_train_task_label
        else:
            task_label = self.task.get_task()['task_label']
            assert False, 'manually set (temporary) breakpoint. code should not get here.'

        task_label = tensor(task_label)
        batch_dim = config.num_workers
        if batch_dim == 1:
            batch_task_label = task_label.reshape(1, -1)
        else:
            batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, dim=0)

        states, rollout = self._rollout_fn(states, batch_task_label)

        self.states = states
        pending_value = self.network.predict(states, task_label=batch_task_label)[-2]
        rollout.append([states, pending_value, None, None, None, None])
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, terminals = rollout[i]
            terminals = tensor(terminals).unsqueeze(1)
            rewards = tensor(rewards).unsqueeze(1)
            actions = tensor(actions)
            states = tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + config.discount * terminals * returns
            if not config.use_gae:
                advantages = returns - value.detach()
            else:
                td_error = rewards + config.discount*terminals*next_value.detach() - value.detach()
                advantages = advantages * config.gae_tau * config.discount * terminals + td_error
            processed_rollout[i] = [states, actions, log_probs, returns, advantages]

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), \
            zip(*processed_rollout))
        eps = 1e-6
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        grad_norm_log = []
        policy_loss_log = []
        value_loss_log = []
        log_probs_log = []
        entropy_log = []
        ratio_log = []
        batcher = Batcher(states.size(0) // config.num_mini_batches, [np.arange(states.size(0))])
        for _ in range(config.optimization_epochs):
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                batch_dim = sampled_states.shape[0]
                batch_task_label = torch.repeat_interleave(task_label.reshape(1, -1), batch_dim, \
                    dim=0)
                _, _, log_probs, entropy_loss, values, outs = self.network.predict(sampled_states, \
                    sampled_actions, task_label=batch_task_label, return_layer_output=True)
                ratio = (log_probs - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean(0) \
                    - config.entropy_weight * entropy_loss.mean()

                value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

                log_probs_log.append(log_probs.detach().cpu().numpy().mean())
                entropy_log.append(entropy_loss.detach().cpu().numpy().mean())
                ratio_log.append(ratio.detach().cpu().numpy().mean())
                policy_loss_log.append(policy_loss.detach().cpu().numpy())
                value_loss_log.append(value_loss.detach().cpu().numpy())

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                norm_ = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                grad_norm_log.append(norm_.detach().cpu().numpy())
                self.opt.step()

        steps = config.rollout_length * config.num_workers
        self.total_steps += steps
        self.layers_output = outs
        return {'grad_norm': grad_norm_log, 'policy_loss': policy_loss_log, \
            'value_loss': value_loss_log, 'log_prob': log_probs_log, 'entropy': entropy_log, \
            'ppo_ratio': ratio_log}

    def _rollout_normal(self, states, batch_task_label):
        '''It runs the agent on the environment and collects SAR data to staore in the Replay
        Buffer'''
        # clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]

        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label) #NOTE HERE WE BREAK in iteration 315!!!!!!!!!!!!!!!!!!!!!!!
            
            next_states, rewards, terminals, _ = self.task.step(actions.cpu().detach().numpy())
            self.episode_rewards += rewards
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
            next_states = config.state_normalizer(next_states)
            #print(actions.cpu())
            # save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions.cpu(), rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])

        return states, rollout

    # rollout for metaworld and continualworld environments. it is similar to normal
    # rollout with the inclusion of the capture of success rate metric.
    def _rollout_metaworld(self, states, batch_task_label):
        # clear running performance buffers
        self.running_episodes_rewards = [[] for _ in range(self.config.num_workers)]
        self.running_episodes_success_rate = [[] for _ in range(self.config.num_workers)]

        config = self.config
        rollout = []
        for _ in range(config.rollout_length):
            _, actions, log_probs, _, values, _ = self.network.predict(states, \
                task_label=batch_task_label)
            next_states, rewards, terminals, infos = self.task.step(actions.cpu().detach().numpy())
            success_rates = [info['success'] for info in infos]
            self.episode_rewards += rewards
            self.episode_success_rate += success_rates
            rewards = config.reward_normalizer(rewards)
            for i, terminal in enumerate(terminals):
                if terminals[i]:
                    self.running_episodes_rewards[i].append(self.episode_rewards[i])
                    self.last_episode_rewards[i] = self.episode_rewards[i]
                    self.episode_rewards[i] = 0
                    self.episode_success_rate[i] = (self.episode_success_rate[i] > 0).astype(np.uint8)
                    self.running_episodes_success_rate[i].append(self.episode_success_rate[i])
                    self.last_episode_success_rate[i] = self.episode_success_rate[i]
                    self.episode_success_rate[i] = 0
            next_states = config.state_normalizer(next_states)

            # save data to buffer for the detect module
            self.data_buffer.feed_batch([states, actions.cpu(), rewards, terminals, next_states])

            rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), \
                rewards, 1 - terminals])
            states = next_states

        # compute average performance across episodes in the rollout
        for i in range(config.num_workers):
            self.iteration_rewards[i] = self._avg_episodic_perf(self.running_episodes_rewards[i])
            self.iteration_success_rate[i] = self._avg_episodic_perf(self.running_episodes_success_rate[i])

        return states, rollout
    
    """
   
    def _avg_episodic_perf(self, running_perf):
        if len(running_perf) == 0: return 0.
        else: return np.mean(running_perf)


class PPOBaselineAgent(PPOContinualLearnerAgent):
    '''PPO continual learning agent baseline (experiences catastrophic forgetting). Uses simplified task_*_start and task_*_end methods to override masking.'''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)

    def task_train_start(self, task_label):
        self.curr_train_task_label = task_label
        return

    def task_train_end(self):
        self.curr_train_task_label = None
        return

    def task_eval_start(self, task_label):
        self.curr_eval_task_label = task_label
        return

    def task_eval_end(self):
        self.curr_eval_task_label = None
        return


class PPOLLAgent(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using supermask superposition algorithm
    task oracle available: agent informed about task boundaries (i.e., when
    one task ends and the other begins)

    supermask lifelong learning algorithm: https://arxiv.org/abs/2006.14769
    '''
    def __init__(self, config):
        PPOContinualLearnerAgent.__init__(self, config)
        # TODO: Modify the self.seen_tasks to use the mp.Manager(). modify to store the reward for the task as well.
        self.seen_tasks = {} # contains task labels / embeddings that agent has experienced so far.
        self.new_task = False
        self.curr_train_task_label = None
        self.curr_eval_task_label = None

    def get_seen_tasks(self):
        '''A getter method for deriving which tasks the DMIU agent has encountered.'''
        return self.seen_tasks
    
    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, seen_task_label in self.seen_tasks.items():
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:#Chnage this with the embedding distance threshold Perhaps.
                found_task_idx = task_idx
                break
        return found_task_idx

    def erase_memory(self, current_task_label):
        self.seen_tasks = {0: current_task_label}
        erase_masks(self.network, self.config.DEVICE)

    def task_train_start(self, task_label):
        '''Method for starting the new <<Detected>> task. It takes as an argument the tasklabel given by the 
        the "trainer_shell" from the "shell_tasks" list of dictionaries.'''

        task_idx = self._label_to_idx(task_label)
        if task_idx is None:
            # new task. add it to the agent's seen_tasks dictionary
            task_idx = len(self.seen_tasks) # generate an internal task index for new task
            self.seen_tasks[task_idx] = task_label
            self.new_task = True
            set_model_task(self.network, task_idx, new_task=True)
        else:
            set_model_task(self.network, task_idx)

        self.curr_train_task_label = task_label
        return

    def task_train_end(self):
        # NOTE, comment/uncomment alongside a block of code in `_forward_mask_lnear_comb` method in
        # MultitaskMaskLinear and MultiMaskLinearSparse classes
        consolidate_mask(self.network)

        self.curr_train_task_label = None
        cache_masks(self.network)
        if self.new_task:
            set_num_tasks_learned(self.network, len(self.seen_tasks))
        self.new_task = False # reset flag
        return

    def task_eval_start(self, task_label):
        self.network.eval()
        task_idx = self._label_to_idx(task_label)
        print(f'Evaluating for task ID {task_idx} {np.argmax(task_label, axis=0)}')
        if task_idx is None:
            # agent has not been trained on current task
            # being evaluated. therefore use a random mask
            # TODO: random task hardcoded to the first learnt
            # task/mask. update this later to use a random
            # previous task, or implementing a way for
            # agent to use an ensemble of different mask
            # internally for the task not yet seen.
            task_idx = 0
        set_model_task(self.network, task_idx)
        self.curr_eval_task_label = task_label
        print(f'Current task label: {self.curr_eval_task_label}')
        return

    def task_eval_end(self):
        self.curr_eval_task_label = None
        self.network.train()
        # resume training the model on train task label if training
        # was on before running evaluations.
        if self.curr_train_task_label is not None:
            task_idx = self._label_to_idx(self.curr_train_task_label)
            set_model_task(self.network, task_idx)
        return


class PPOShellAgent(PPOLLAgent):
    '''Lifelong learning (ppo continual learning with supermask) agent in ShELL
    settings. Agent runs with asynchronous multiprocessing/multithreading and communicates with other agents.

    Approach used for L2D2-C (Sharing Lifelong Reinforcement Learning via Modulating Masks).'''
    def __init__(self, config):
        PPOLLAgent.__init__(self, config)
        _mask = get_mask(self.network, task=0)
        self.mask_info = {}
        for key, value in _mask.items():
            self.mask_info[key] = tuple(value.shape)
        model_mask_dim = 0
        for k, v in self.mask_info.items():
            model_mask_dim += np.prod(v)
        self.model_mask_dim = model_mask_dim

    def _select_mask(self, masks, ensemble=False):
        found_mask = None
        if ensemble:
            raise NotImplementedError
        else:
            for mask in masks:
                if mask is not None:
                    found_mask = mask
                    break
        return found_mask
            
    def label_to_mask(self, task_label):
        task_idx = self._label_to_idx(task_label)
        # get task mask
        if task_idx is None:
            mask = None
        else:
            # function from deep_rl/shell_modules/mmn/ssmask_utils.py
            mask = get_mask(self.network, task_idx)
            mask = self.mask_to_vec(mask)
        return mask

    def distil_task_knowledge(self, masks):
        # TODO fix algorithm. Current solution invloves selecting the first mask 
        # that is not Noneand using it as the knowledge from which the agent can 
        # continue training on current task.
        task_label = self.curr_train_task_label
        #task_label = self.task.get_task()['task_label'] # use this to pass label from outside fn
        task_idx = self._label_to_idx(task_label)

        masks = [self.vec_to_mask(mask.to(self.config.DEVICE)) for mask in masks]
        mask = self._select_mask(masks)

        if mask is not None:
            # function from deep_rl/shell_modules/mmn/ssmask_utils.py
            set_mask(self.network, mask, task_idx)
            return True
        else:
            return False
    
    def distil_task_knowledge_single(self, mask, task_label):
        # New distil task knowledge algorithm
        # this function receives only one mask
        # which is the best mask.
        #print(mask)

        #task_label = self.curr_train_task_label
        task_idx = self._label_to_idx(task_label)
        print(f'DISTILLING KNOWLEDGE TO TASK IDX: {task_idx}')
        # Process the single mask as opposed to multiple
        mask = self.vec_to_mask(mask.to(self.config.DEVICE))

        if mask is not None:
            set_mask(self.network, mask, task_idx)
            return True
        else:
            return False

    def distil_task_knowledge_single_eval(self, mask, task_label):
        # New distil task knowledge algorithm
        # this function receives only one mask
        # which is the best mask.
        #print(mask)

        #task_label = self.curr_eval_task_label
        #print(task_label)
        task_idx = self._label_to_idx(task_label)
        #print(task_idx)

        # Process the single mask as opposed to multiple
        mask = self.vec_to_mask(mask.to(self.config.DEVICE))

        if mask is not None:
            set_mask(self.network, mask, task_idx)
            return True
        else:
            return False

    def mask_to_vec(self, dict_mask):
        with torch.no_grad():
            vec_mask = torch.zeros(self.model_mask_dim,)
            start = 0
            stop = None
            for k, v in dict_mask.items():
                stop = start + np.prod(v.shape)
                vec_mask[start : stop] = v.reshape(-1,)
                start = stop
        return vec_mask

    def vec_to_mask(self, vec_mask):
        dict_mask = {}
        start = 0
        stop = None
        for key, value in self.mask_info.items():
            stop = start + np.prod(value)
            dict_mask[key] = vec_mask[start : stop].reshape(*value)
            start = stop
        return dict_mask


# Detect L2D2-C agent implementation. Uses modified methods to support use of the detect module. Approach used for detect-l2d2-c (Title TBD).
class PPODetectShell(PPOShellAgent):
    '''Detect L2D2-C agent implementation. Uses modified methods to support use of the detect module.'''
    def __init__(self, config):
        PPOShellAgent.__init__(self, config)

        # Saptarshi: Updating the seen_tasks dictionary to be use the SyncManager() internal server solution.
        # this will allow us to use the seen_tasks dictionary across our entire parallelised system.
        self.seen_tasks = config.seen_tasks
        self.current_task_key = 0
        self.continuous = config.continuous

        # BIRCH online clustering
        #self.threshold = 1
        #self.branching_factor = 50
        #self.birch = Birch(threshold=self.threshold, branching_factor=self.branching_factor, n_clusters=None)
        #self.current_cluster_label = 0

        self.embedding_history = []

        #self.distance_history = deque([])   # List of computed embedding distances for the last 10 timesteps

        ###############################################################################
        # Detect Module Attributes

        # Variable for storing the action space size of the task for using it to 
        # convert the actions to one-hot vectors.
        self.task_action_space_size = self.task.action_dim

        # Variable for storing the detect reference number.
        self.detect_reference_num = config.detect_reference_num

        # Variable for storing the number of samples hyperperameter of the detect module.
        self.detect_num_samples = config.detect_num_samples

        # Varible for checking for dishiding wether the agent has encountered a new task or not.
        self.emb_dist_threshold = config.emb_dist_threshold

        # Variable for storing the frequency of the detect module activation.
        self.detect_module_activation_frequency = config.detect_module_activation_frequency

        # Create a list for saving the calculated embeddings
        self.encountered_task_embs = []

        # Varible for storing the current embedding label as an attribute of the agent itself.
        self.current_task_emb =  None #self.task.get_task()['task_emb']

        #Varibles for tracking the current task embedding, influenced by the two varibles above, and working at a same fashion.
        self.curr_train_task_emb = None
        self.curr_eval_task_emb = None

        self.new_task_emb = None


        ###############################################################################
        # Precalculate the embedding size based on the reference and the network observation size.
        tmp_state_obs  = self.task.reset()
        tmp_state_obs = config.state_normalizer(tmp_state_obs)
        observation_size = tmp_state_obs.ravel().shape[0]       # ravel() to accomodate for 2D obs as well when using a convolutional model...

        # Initialise detect module within the agent
        #self.detect = config.detect_fn(self.detect_reference_num, observation_size, self.detect_num_samples)
        self.detect = config.detect_fn(self.detect_reference_num, observation_size, self.task.action_dim, self.detect_num_samples)

        # Variable for saving the size of the task embedding that the detect module has produced.
        # Initially we store the precalculated embedding size
        self.detect.set_reference(observation_size, self.detect_reference_num, self.task.action_dim)
        self.task_emb_size  = self.detect.precalculate_embedding_size(self.detect_reference_num, observation_size, self.task.action_dim)
        self.config.logger.info(f'task embedding size: {self.task_emb_size}')
        self.current_task_emb = torch.zeros(self.get_task_emb_size())
        self.new_task_emb =  torch.zeros(self.get_task_emb_size())      # FIXME: We don't use this! Consider removal.


        # Beta parameters for shared masks from network
        #self.shared_betas = nn.Parameter(torch.zeros(32).type(torch.float32))
        self.shared_betas = nn.Parameter(torch.zeros(32, 32).type(torch.float32)) #Ideally can be a 1D vector. 2D for analysis.
        self.discrete = True
        self._subnet_class = GetSubnetDiscrete if self.discrete else GetSubnetContinuous

        
        self.betas_log_path = self.config.logger.log_dir + '/betas.csv'
        self.iteration_entropy = 1.0


        # Debugging outputs.
        print(f'Observation size: {observation_size}, Task embedding size: {self.task_emb_size}')
        print(f'Task action space size: {self.task_action_space_size}')


    ###############################################################################
    # Modified agent methods to accomodate changes for detect module integration
    def _embedding_to_idx(self, task_embedding):
        '''Private method that checks wether the agent has already seen the task with the
        embedding given before or not. It checks this by searching the dictionary of seen
        tasks and compering the given embedding with the stored emebddings of each stored
        task. If an an embedding with high similarity (based on the distance threshold) is
        found then it return the index of the task for this embedding.
        
        Important NOTE: The index for each of the seen tasks that the agent has encountered
        is different from the original index that the tasks already have in their info 
        dictonary which is being set upon their creation (from the task class). When a task
        is registered in the seen_tasks dictonary of the agent the agent automatically creates
        an unique index for this specific task, which is based on which turn the agent first
        encounters this specific task.'''

        found_task_idx = None
        for task_idx, task_dict in self.seen_tasks.items():
            seen_task_embedding = task_dict['task_emb']
            print(f'Seen task embedding: {seen_task_embedding}, embedding: {task_embedding}')

            if task_embedding is None:
                found_task_idx = self.current_task_key

            elif seen_task_embedding is None:
                found_task_idx = self.current_task_key
                self.update_seen_tasks(
                    embedding=task_embedding,
                    reward=0,
                    label=self.task.get_task()['task_label']
                )

            else:
                cosine_similarity = F.cosine_similarity(task_embedding, seen_task_embedding, dim=0)
                self.config.logger.info(f'Cosine similarity: {cosine_similarity}')
                if cosine_similarity > 0.4:#np.linalg.norm(task_embedding - seen_task_embedding) < self.emb_dist_threshold:
                    found_task_idx = task_idx
                    break
        return found_task_idx
        
        '''found_task_idx = None
        for task_idx, task_dict in self.seen_tasks.items():
            seen_task_embedding = task_dict['task_emb']
            print(f'Seen task embedding: {seen_task_embedding}, embedding: {task_embedding}')
            cosine_similarity = F.cosine_similarity(task_embedding, seen_task_embedding, dim=0)
            self.config.logger.info(cosine_similarity)
            if cosine_similarity > 0.4:#np.linalg.norm(task_embedding - seen_task_embedding) < self.emb_dist_threshold:
                found_task_idx = task_idx
                break
        return found_task_idx'''
        
    def task_train_start_emb(self, task_embedding, current_reward):
        '''Method for starting the training procedure upon a new Detected task form the Detect Module.
        It is based on the "task_train_start()" method.'''

        # Get the task idx (key) for if the task record exists (based on embedding similarity)
        self.current_task_key = self._embedding_to_idx(task_embedding)

        print(f'Current task key: {self.current_task_key}')
 
        if self.current_task_key is None:
            # If no similar embedding record was found then its a new task
            self.current_task_key = len(self.seen_tasks)                                        # Generate an internal task index for new task

            # Create the dictionary key-value pair for the new task
            self.update_seen_tasks(
                embedding=task_embedding, 
                reward=0,
                label=self.task.get_task()['task_label']
            )

            self.new_task = True                                                                # Set the new_task flag to True
            set_model_task(self.network, self.current_task_key, new_task=True, current_reward=current_reward)                  # Set the new task mask inside the model
            self.set_current_task_embedding(task_embedding)        # Set the self.current_task_emb for the newly detected task

        else:
            # Set model to use the existing task mask.
            set_model_task(self.network, self.current_task_key, current_reward=current_reward)

        return
    
    def task_train_end_emb(self):
        '''Method for stopping the training upon a specific task. It is being used when
        the Detect Module perceives a task change, in order to stop training for the old
        task and start training for the new one. Inspired from the original "task_train_end()"
        to work with embeddings instead of labels.'''
        # NOTE, comment/uncomment alongside a block of code in `_forward_mask_lnear_comb` method in
        # MultitaskMaskLinear and MultiMaskLinearSparse classes
        consolidate_mask(self.network)

        self.current_task_emb = None
        cache_masks(self.network)
        
        if self.new_task:
            set_num_tasks_learned(self.network, len(self.seen_tasks))

        self.new_task = False # reset flag
        return

    def embedding_to_mask(self, task_embedding):
        '''A method for finding the correspoding mask for the task tha it is currently encountered
        based on the agents task/mask registry (seen tasks dictionary)'''
        task_idx = self._embedding_to_idx(task_embedding)
        #get the stored correspoding mask for that specific task
        if task_idx is None:
            mask = None
        else:
            #function from ssmask_utils.py
            mask = get_mask(self.network, task_idx)
            mask = self.mask_to_vec(mask)
        return mask
    
    def idx_to_mask(self, task_idx):
        '''A method for finding the correspoding mask for the task tha it is currently encountered
        based on the agents task/mask registry (seen tasks dictionary)'''
        #get the stored correspoding mask for that specific task
        if task_idx is None:
            mask = None
        else:
            #function from ssmask_utils.py
            mask = get_mask(self.network, task_idx)
            print(f'Mask after get_mask')
            mask = self.mask_to_vec(mask)
            print(f'Mask after mask_to_vec: {mask}')
        return mask


    ###############################################################################
    # Mask linear combination methods.
    def linear_combination(self, masks):
        """
        linearly combines incoming masks from the network
        """
        try:
            # Get current training mask (this starts off as the random trainig mask for the task)
            _subnet = self.idx_to_mask(self.current_task_key)

            # Get incoming masks
            _subnets = [masks[idx].detach() for idx in range(len(masks))]
            _subnets.append(_subnet)

            #assert len(_subnets) > 0, 'an error occured'
            #_betas = self.shared_betas[len(masks), 0:len(masks)]    # 2D beta parameter vector
            #_betas = self.shared_betas[0:len(masks)]   # 1D beta parameter vector

            # Initialise beta parameters. By default zeros tensor will mean equal probability across
            # all masks in the linear combination step.
            _betas = torch.zeros(len(_subnets))

            ########################################################################
            # mask BLC modification. 1/2 on current mask and 1/(2k) on the incoming masks
            _betas[-1] = 0.5
            k = len(_subnets) - 1   # num masks excluding the current training mask
            _betas[:-1] = 1 / (2 * k)
            _betas = torch.log(_betas)
            ########################################################################
            print(f'Betas before softmax: {_betas}')
            # Convert betas to probability weights
            _betas = torch.softmax(_betas, dim=-1)  
            print(f'Betas after softmax {_betas}')

            #assert len(_betas) == len(_subnets), 'an error ocurred'
            
            # Apply beta coefficients to masks
            _subnets = [_b * _s for _b, _s in  zip(_betas, _subnets)]
            
            # Element wise sum/average of the incoming masks and current training mask (weighted sum/average)
            _subnet_linear_comb = torch.stack(_subnets, dim=0).sum(dim=0)
            #_subnet_linear_comb = torch.stack(_subnets, dim=0).mean(dim=0)  # using mean offers some more stability than sum when we have lots of masks or if one or more masks have large values.
            return _subnet_linear_comb.data #self._subnet_class.apply(_subnet_linear_comb)
        
        except Exception as e:
            traceback.print_exc()
    
    def consolidate_incoming(self, masks):
        """
        consolidates all incoming knowledge from the collective
        """
        # TODO: This needs to be changed to first consolidate the current shared masks
        #       and then perform the linear combination of the new shared masks
        #       with trainable beta parameters.

        # i.e.,
        
        '''with torch.no_grad():
            task_mask = self.consolidate_current_shared_masks()
            self.distil_task_knowledge_embedding(task_mask)
            
            self.betas = nn.Parameter(torch.zeros(masks, masks).type(torch.float32))
            
            self.betas = self.betas[masks, 0:self.task+1]

            ...
            '''
        
        
        with torch.no_grad():
        # Initalise random mask
            new_mask = self.linear_combination(masks)
            print(f'In consolidate_incoming: {new_mask}')
            self.distil_task_knowledge_embedding(new_mask)

            #set_mask(new_mask, self.current_task_key)


    ###############################################################################
    # Methods for detect module    
    def extract_sar(self):
        """Extracts (state, action, reward) tuples from the replay buffer
        and preprocesses them for further use.

        - Handles both discrete and continuous action spaces.
        - Ensures consistent shapes for concatenation.
        """

        buffer_data = self.data_buffer.sample()
        processed_data = []

        for state, action, reward, *_ in buffer_data:
            # Convert state to a flattened array
            state_array = state.ravel()

            # Convert action to a NumPy array
            action_array = np.array(action)

            # Ensure correct shape for action
            if isinstance(self.task.action_space, gym.spaces.Discrete):
                action_array = action_array.reshape(1,)  # Convert scalar action to (1,)
            else:
                action_array = action_array.reshape(self.task.action_space.shape)  # Keep continuous actions in proper shape

            # Convert reward to a NumPy array with shape (1,)
            reward_array = np.array(reward).reshape(1,)

            # Concatenate state, action, and reward into a single array
            sar_entry = np.concatenate([state_array, action_array, reward_array])
            processed_data.append(sar_entry)

        # Convert the list of processed data to a NumPy array
        return np.array(processed_data)

    def compute_task_embedding(self, sar_data, action_space_size):
        '''Function for computing the task embedding based on the current
        batch of SAR data derived from the replay buffer.'''
        #print("TASK NUM OF ACTIONS:", self.task.action_space.n)
        task_embedding = self.detect.lwe(sar_data, action_space_size)
        self.new_task_emb = task_embedding
        #self.current_task_emb = task_embedding
        #self.task.set_current_task_info('task_emb', task_embedding)
        #self.task.get_task()['task_emb'] = task_embedding
        self.task_emb_size = len(task_embedding)
        return task_embedding

    def update_seen_tasks(self, embedding, reward, label):
        self.seen_tasks.update({self.current_task_key : {
            'task_emb': embedding,
            'reward': reward,
            'ground_truth': label
        }})
    
    def assign_task_emb(self, new_emb, emb_distance):
        '''Assigns the most up to date embedding to the current task based
        on the embedding distance threshold.

        If agent is working on the same task (< threshold) then we update embedding
        with the moving average of the new and old embeddings.

        If the agent has encountered a task shift (> threshold) then we setup a new mask
        and record in the internal registry for the new task. If it is a previously encountered
        task then we just start updating the existing record.'''

        # LEGACY CODE:
        '''# Variable used to check for key that stores the task embedding of the agent's current task
        key_to_check = 'task_emb'

        # Flag for task change detected by the detect module (True if task change happend, False otherwise).
        task_chng_bool = None

        if not key_to_check in self.task.get_task().keys():                                                 # If 'task_emb' key is not in the task_fn() registry then...
            self.task.set_current_task_info(key_to_check, torch.zeros(self.get_task_emb_size()))            # ...create the key-value pair with zeros embedding tensor
            print("TASK INFO REGISTRY UPDATED WITH EMBEDDING KEY-VALUE PAIR!!!!!!!!!!!!!!!!")

        if emb_distance < self.emb_dist_threshold:                                                          # If the computed embedding distance is below the preset fixed threshold then...
             self.task.get_task()['task_emb'] = (self.task.get_task()['task_emb'] + new_emb) / 2            # ...calculate the moving average of the new embedding and the previous average and update the task_fn() registry

             self.set_current_task_embedding(self.task.get_task()['task_emb'])                              # Update self.current_task_emb with the new averaged embedding. NOTE: We don't currently use the self.current_task_emb for anything. It's likely that this was created as a placeholder replacement of the self.current_task_label
             
             current_task_idx = self._embedding_to_idx(self.current_task_emb)                               # Gets the generated task ID if the 'new' current_task_emb is similar to any of the previously seen tasks in self.seen_tasks. Essentially we want to check if we have done this task before or not using embeddings instead of task labels. If not we get NoneType here.
             
             self.seen_tasks[current_task_idx] = self.get_current_task_embedding()                          # Update the self.seen_tasks dictionary with the self.current_task_emb (the averaged 'new' embedding)
             str_task_chng_msg = "TASK CHNAGE NNNNOOOOTTTTT DETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

             task_chng_bool = False
             
        else:
            self.task_train_end_emb()                                                                       # End training on the previous task. This means consolidating the task mask, setting current_task_emb to NoneType and updating the number of tasks learned by the network.
            self.task_train_start_emb(new_emb)                                                              # Start training on the new task. This means checking if we have encountered the task before then set the existing model mask, otherwise, set up a new task and set the model to use a new task mask
            self.task.get_task()['task_emb'] = new_emb                                                      # Since we are switching to a different task then we update the task_fn() registry pointer with the new embedding. NOTE: The pointer is updated by the self.task.reset_task() method which sets the new task and resets the state/observations.
            self.set_current_task_embedding(new_emb)                                                        # Set the self.current_task_emb to this new task embedding. NOTE: Again this is not used for anything.
            self.set_new_task_emb(new_emb)                                                                  # Set the self.new_task_emb to this new embedding. NOTE: This is also not used for anything ???
            str_task_chng_msg = "TASK CHNAGE DETECTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            
            task_chng_bool = True

        return str_task_chng_msg, task_chng_bool'''
    
        # NEW CODE:
        # Debugging CLI outputs. We track the registry for embedding updates across all tasks.

        task_change_bool = None
            

        # Saptarshi: Modified algorithm to use the SyncManager() internal server dictionary (self.seen_tasks) instead of the registry inside the task_fn().
        # IF SAME TASK
        #if not(self.total_steps == 512 * 200):
        if emb_distance < self.emb_dist_threshold:
            old_emb = self.seen_tasks[self.current_task_key]['task_emb']
            self.current_task_emb = (old_emb + new_emb) / 2   # Compute moving average.

            # Update dictionary at current_task_key. Ideally we would have used a pointer to get this done but unfortunately we can't do so for mp.Manager() dictionaries because it is a proxy.
            if self.continuous == True:
                self.update_seen_tasks(
                    embedding = self.current_task_emb,
                    reward = np.mean(self.iteration_success_rate),
                    label = self.task.get_task()['task_label']
                )
            else:
                self.update_seen_tasks(
                    embedding = self.current_task_emb,
                    reward = np.mean(self.iteration_rewards),
                    label = self.task.get_task()['task_label']
                )

            task_change_bool = False

        # IF NEW TASK
        else:

            self.task_train_end_emb()                       # End training on previous task mask.
            if self.continuous == True:
                self.task_train_start_emb(new_emb, np.mean(self.iteration_success_rate))      # np.mean(self.iteration_rewards) was giving           # Start training on new mask for the newly detected task
            else:
                self.task_train_start_emb(new_emb, np.mean(self.iteration_rewards))
            self.current_task_emb = new_emb                 # Set the current task embedding to the first one produced for the new task
            task_change_bool = True

        return task_change_bool

    def log_betas(self, iteration):
        betas_logs = get_current_betas(self.network)
        data = [
            {
                'iteration' : iteration,
                'betas' : betas_logs
            }
        ]
        df = pd.DataFrame(data)
        df.to_csv(self.betas_log_path, mode='a', header=not pd.io.common.file_exists(self.betas_log_path), index=False)


    ###############################################################################
    # Methods for distilling incoming masks to model
    def distil_task_knowledge_embedding(self, mask):
        '''Method performs the distilation of task knowledge based on embeddings. The alorithmic approach
        is identical to the "distil_task_knowledge_single.'''

        # New distil task knowledge algorithm
        # this function receives only one mask
        # which is the best mask.
        #print(f'Mask in distil: {mask}')
        #print(f'Current task key: {self.current_task_key}')

        #task_label = self.curr_train_task_label
        #task_idx = self._embedding_to_idx(task_embedding)

        # Process the single mask as opposed to multiple
        mask = self.vec_to_mask(mask.to(self.config.DEVICE))

        if mask is not None:
            set_mask(self.network, mask, self.current_task_key)
            return True
        else:
            return False
        
    # TODO: Saptarshi: I think we will take a task label approach to the evaluation agent
    def distil_task_knowledge_single_eval_embedding(self, mask, label):
        
        '''This metehod performs the distilation of mask based on task embedding 
        for the evaluator agent.' The aglorithmic apporach is identical to the
        "distil_task_knowledge_single" method.'''
        # New distil task knowledge algorithm
        # this function receives only one mask
        # which is the best mask.
        #print(mask)

        #task_label = self.curr_eval_task_label
        #print(task_label)
        task_idx = self._label_to_idx(label)
        #print(task_idx)

        # Process the single mask as opposed to multiple
        mask = self.vec_to_mask(mask.to(self.config.DEVICE))

        if mask is not None:
            set_mask(self.network, mask, task_idx)
            return True
        else:
            return False

    def update_community_masks(self, masks, current_reward):
        update_comm_masks(self.network, masks, current_reward)


    ###############################################################################
    # Getters and Setters
    def get_current_task_label(self):
        '''Get current task label'''
        return self.task.get_task()['task_label']
    def set_current_task_label(self, new_task_label):
        '''This is unlikely to be used.'''
        self.task.get_task()['task_label'] = new_task_label

    def get_task_action_space_size(self):
        '''A getter method for retreiving the task action space.'''
        return self.task_action_space_size
    def set_task_action_space_size(self, new_action_space):
        '''A setter method for manually assining the task action space
        size for letter use it for converting the actions to one-hot
        representation.'''
        self.task_action_space_size = new_action_space
    
    def get_task_emb_size(self):
        '''''A getter method for retreiving the task embedding size.'''
        return self.task_emb_size
    def set_task_emb_size(self, new_embedding_size):
        '''A setter for dynamically setting the mebedding size.'''
        self.task_emb_size = new_embedding_size
    
    def get_emb_dist_threshold(self):
        '''A getter method for reteiving the distnce threshold for the embeddings.'''
        return self.emb_dist_threshold
    def set_emb_dist_threshold(self, new_distance_threshold):
        '''A setter for dynamically setting the distance threshold of the embeddings.'''
        self.emb_dist_threshold = new_distance_threshold
    
    def get_detect_module_activation_frequency(self):
        '''A getteer for the detect module activation frequency'''
        return self.detect_module_activation_frequency
    def set_detect_module_activation_frequency(self, new_activation_frequency):
        '''A setter method for manually setting the detect module activation frequnecy.'''
        self.detect_module_activation_frequncy = new_activation_frequency
    
    def get_current_task_embedding(self):
        '''A getter method for retreiving the current task embedding.'''
        return self.current_task_emb
    def set_current_task_embedding(self, new_current_task_embedding):
        '''A setter method for updating the current task embedding with the new
        more acurate estimate of the embedding produced by the detect module.'''
        self.current_task_emb = new_current_task_embedding
    
    def get_new_task_embedding(self):
        ''''''
        return self.new_task_emb
    def set_new_task_emb(self, new_emb):
        ''''''
        self.new_task_emb = new_emb

# I don't think we need this anymore. Was a placeholder barebones implementation of what the DetectShell could have looked like prior to detect module implementation.
class LLAgent_NoOracle(PPOContinualLearnerAgent):
    '''
    PPO continual learning agent using supermask superposition algorithm
    with *no task oracle*: agent is not informed about task boundaries 
    (i.e., when one task ends and the other begins) and has to detect task
    change by itself.

    supermask lifelong learning algorithm: https://arxiv.org/abs/2006.14769
    '''
    def __init__(self, config):
       PPOContinualLearnerAgent.__init__(self, config)
       self.seen_tasks = {} # contains task labels that agent has experienced so far.
       self.new_task = False
       self.curr_train_task_label = None

    def _name_to_idx(self, name):
        found_task_idx = None
        for task_idx, value in self.seen_tasks.items():
            seen_task_label, task_name = value
            if name == task_name:
                found_task_idx = task_idx
                break
        return found_task_idx

    def _label_to_idx(self, task_label):
        eps = 1e-5
        found_task_idx = None
        for task_idx, value in self.seen_tasks.items():
            seen_task_label, task_name = value
            if np.linalg.norm((task_label - seen_task_label), ord=2) < eps:
                found_task_idx = task_idx
                break
        return found_task_idx
        
    # TODO block of code to remove
    #def _select_mask(self, agents, masks, ensemble=False):
    #    found_mask = None
    #    if ensemble:
    #        raise NotImplementedError
    #    else:
    #        for agent, mask in zip(agents, masks):
    #            if mask is not None:
    #                found_mask = mask
    #                break
    #    return found_mask

    def update_task_label(self, task_label):
        # TODO: consider other ways to update the label as detect module
        # alters it. Maybe moving average?
        task_idx = self._label_to_idx(self.curr_train_task_label)
        self.seen_tasks[task_idx][0] = task_label
        self.curr_train_task_label = task_label

    def set_first_task(self, task_label, task_name):
        # start first task
        task_idx = 0 # first task idx is 0
        self.seen_tasks[task_idx] = [task_label, task_name]
        self.new_task = True
        set_model_task(self.network, task_idx)
        self.curr_train_task_label = task_label
        return

    def task_change_detected(self, task_label, task_name):
        # end current task (if any)
        if self.curr_train_task_label is not None:
            cache_masks(self.network)
            if self.new_task:
                set_num_tasks_learned(self.network, len(self.seen_tasks))
            self.new_task = False # reset flag
            self.curr_train_task_label = None

        # start next task
        # use task label or task name to check if task already exist in model
        task_idx = self._label_to_idx(task_label)
        if task_idx is None:
            # new task. add it to the agent's seen_tasks dictionary
            task_idx = len(self.seen_tasks) # generate an internal task index for new task
            self.seen_tasks[task_idx] = [task_label, task_name]
            self.new_task = True
        set_model_task(self.network, task_idx)
        self.curr_train_task_label = task_label
        return

    def task_eval_start(self, task_name):
        self.network.eval()
        task_idx = self._name_to_idx(task_name)
        if task_idx is None:
            # agent has not been trained on current task
            # being evaluated. therefore use a random mask
            # TODO: random task hardcoded to the first learnt
            # task/mask. update this later to use a random
            # previous task, or implementing a way for
            # agent to use an ensemble of different mask
            # internally for the task not yet seen.
            task_idx = 0
        set_model_task(self.network, task_idx)
        return

    def task_eval_end(self):
        self.network.train()
        # resume training the model on train task label if training
        # was on before running evaluations.
        if self.curr_train_task_label is not None:
            task_idx = self._label_to_idx(self.curr_train_task_label)
            set_model_task(self.network, task_idx)
        return
