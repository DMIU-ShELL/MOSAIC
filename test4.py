import gym
import gym_CTgraph
import numpy as np
from itertools import product
import os
import json

path = 'env_configs/ctgraph/seed1/meta_ctgraph_ctwtf.json'

with open(path, 'r') as f:
    env_meta_config = json.load(f)

task_order = env_meta_config['task_order']
base_path = os.path.dirname(path)

envs = []
for config_path in env_meta_config['config_paths']:
    env = gym.make('CTgraph-v0', config_path='{0}/{1}'.format(base_path, config_path))
    envs.append(env)


# observation/action space configuration
___observation_space = envs[0].observation_space
___action_space = envs[0].action_space
___action_dim = envs[0].action_space.n
___state_dim = envs[0].observation_space.shape


_all_tasks = []
for idx, env in enumerate(envs):
    depth = env.DEPTH
    branch = env.BRANCH
    img_seed = env.conf_data['image_dataset']['seed']
    tasks = list(product(list(range(branch)), repeat=depth))
    names = ['ctgraph_d{0}_b{1}_imgseed_{2}_task_{3}'.format(depth, branch, img_seed, j) \
        for j in range(len(tasks))] 
    #NOTE: tasks Includes the 'task-label' whichh is  a disaprated valvue
    tasks = [{'name': name, 'task': np.array(task), 'task_label': None, 'env_idx': idx} \
        for name, task in zip(names, tasks)]
    _all_tasks.append(tasks)

all_tasks = []
for env_tasks in _all_tasks: all_tasks += env_tasks
del _all_tasks


___envs = envs
___tasks = all_tasks
___env = None

# task label config
___task_label_dim = env_meta_config['label_dim']
___one_hot_labels = env_meta_config['one_hot']

# generate label for each task
if ___one_hot_labels:
    for idx in range(len(___tasks)):
        label = np.zeros((___task_label_dim,)).astype(np.float32)
        label[idx] = 1.
        ___tasks[idx]['task_label'] = label


if 'filter_tasks' in env_meta_config.keys():
    filtered_tasks = []
    for idx_ in env_meta_config['filter_tasks']:
        filtered_tasks.append(___tasks[idx_])
    ___tasks_ = ___tasks
    ___tasks = filtered_tasks


print(len(___tasks))
taskinfo = ___tasks[27]
___env = ___envs[taskinfo['env_idx']]
___env.unwrapped.set_high_reward_path(taskinfo['task'])
___current_task = taskinfo

print(___current_task)

env = ___env


state = env.reset()
print(state)
