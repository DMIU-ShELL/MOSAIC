#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .normalizer import *
import argparse
import torch

class Config:
    DEVICE = torch.device('cpu')
    ENV_MINIGRID = 'Minigrid'
    ENV_METACTGRAPH = 'MetaCTgraph'
    ENV_METAWORLD = 'MetaWorld'
    ENV_CONTINUALWORLD = 'ContinualWorld'
    ENV_PROCGEN = 'Procgen'
    ENV_COMPOSUITE = 'Composuite'
    ENV_MINIHACK = 'Minihack'
    ENV_METAWORLD = 'Metaworld'
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.policy_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.discount = 0.99
        self.target_network_update_freq = 0
        self.max_episode_length = 0
        self.exploration_steps = 0
        self.logger = None
        self.history_length = 1
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.update_interval = 1
        self.gradient_clip = 0.5
        self.entropy_weight = 0.01
        self.use_gae = False
        self.gae_tau = 1.0
        self.noise_decay_interval = 0
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.hybrid_reward = False
        self.episode_limit = 0
        self.min_memory_size = 200
        self.master_fn = None
        self.master_optimizer_fn = None
        self.num_heads = 10
        self.min_epsilon = 0
        self.save_interval = 0
        self.max_steps = 0
        self.render_episode_freq = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.episode_log_interval = 1
        self.iteration_log_interval = 30
        self.categorical_v_min = -10
        self.categorical_v_max = 10
        self.categorical_n_atoms = 51
        self.num_quantiles = 10
        self.gaussian_noise_scale = 0.3
        self.optimization_epochs = 4
        self.num_mini_batches = 32
        self.test_interval = 0
        self.test_repetitions = 10
        self.evaluation_env = None
        self.termination_regularizer = 0
        self.evaluation_episodes_interval = 0
        self.evaluation_episodes = 0
        self.sgd_update_frequency = 4
        self.seed = 1

        # extra config
        self.lr = 0.00025
        self.agent_name = None
        self.env_name = None
        self.env_config_path = None
        self.eval_task_fn = None
        #self.reg_loss_coeff = 1e-3

        # extra config for continual learning (cl) experiments
        self.cl_num_learn_blocks = 1
        self.cl_requires_task_label = True
        self.cl_num_tasks = 1
        self.task_ids = None
        #self.cl_alpha = 0.25
        #self.cl_n_slices = 50
        #self.cl_loss_coeff = 1e6
        #self.cl_preservation = 'mas' # note, this should be 'mas' or 'scp'
        #self.cl_tasks_info = None
        #self.cl_pm_min = -np.inf
        #self.cl_pm_max = np.inf
        #self.cl_learn_task_label = True
        self.eval_interval = None
        self.use_task_label = True
        self.continuous = False
        self.use_full_batch = False
        self.target_kl = None

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])

    def log_hyperparameters(self, log_file_path):
        with open(log_file_path, 'a') as f:
            f.write("Experiment Hyperparameters:\n")
            for key, value in vars(self).items():
                f.write(f"{key}: {value}\n")

    def log_hyperparameters_tensorboard(self, writer):
        for key, value in vars(self).items():
            writer.add_text(key, str(value))