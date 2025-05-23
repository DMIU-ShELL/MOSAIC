# MOSAIC

**MOSAIC (Modular Sharing and Composition in Collective Learning)** is a decentralized, asynchronous, and collaborative reinforcement learning system that enables autonomous agents to identify, share, and reuse modular knowledge. It uses task similarity and reward-based heuristics to allow agents to independently select, learn and act in RL environments. MOSAIC improves learning speed, task generalization, and scalability without centralized coordination.

This work is inspired and supported by the work conducted in [ShELL (Shared Experience Lifelong Learning)](https://rdcu.be/dB9zt).

## Overview

MOSAIC belongs to a novel paradigm of distributed AI systems where each agent is an independent learner capable of collaboration via sharing of knowledge. These agents communicate peer-to-peer, exchange task embeddings, and combine learned knowledge through modular neural masks guided by Wasserstein task similarity.

### Key Features

- Modular policy composition via neural masks.
- Task similarity estimation using Wasserstein embeddings.
- Asynchronous knowledge exchange guided by performance and similarity.
- Full support for decentralized, scalable training across tasks and environments.

## Agent Architecture

Each agent in MOSAIC:
- Utilizes [PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347) for reinforcement learning.
- Implements [Modulating Masks](https://arxiv.org/abs/2212.11110) to represent and isolate task-specific knowledge.
- Dynamically selects and blends external knowledge from peer agents via a two-phase heuristic protocol (similarity + performance).

Baseline agents are PPO-only and prone to catastrophic forgetting.

## Supported Environments

- [MiniGrid](https://github.com/Farama-Foundation/gym-minigrid)
- [CT-graph](https://github.com/soltoggio/CT-graph)
- [Procgen](https://github.com/openai/procgen)

## Requirements

- Requirements from [DeepRL](https://github.com/ShangtongZhang/DeepRL)
- Additional:
  - `gym-minigrid`
  - `ctgraph`
  - `minihack`
- Environment setup YAMLs are located in `./ymls/`

## Usage

### Run a Single Agent
To run a single C3L agent on Minigrid.

```
python run_minigrid.py <curriculum index> <port> -p <experiment name>
```
- The curriculum index tells the agent which curriculum of tasks to select for learning in the experiment.
- The listening port defines which port the server will listen on for incoming communication.
- The -p argument is optional and will default to the environment name.


CT-graph and MiniHack experiments can be run using run_mctgraph.py and run_procgen.py

### Run a distributed experiment
To run a multi-agent experiment with multiple agents, each with their own environment.
```
python launcher.py --env minigrid --exp <experiment folder name>
```
- The --env argument defines which setup to use. Each experiment has its own setup.
- The --exp argument defines the name of the folder in which the experiment data will be contained.
- The launcher.py file is setup to use CUDA_VISIBLE_DEVICES to define the GPU used by the agent. Our experiments have been run on Nvidia A100s using MiG configurations.

### Setting up communication
Ensure that the references.csv file contains the IPs and ports for your agents. 
```
<ip>, <port>
```

By default reference.csv file contains:

```
127.0.0.1, 29500
127.0.0.1, 29501
127.0.0.1, 29502
127.0.0.1, 29503
127.0.0.1, 29504
127.0.0.1, 29505
```

### Running on multiple devices
To run multiple agents on seperate devices, please update the addresses.csv file. This can contain one or more ip ports of other agents. For example:
```
xxx.xxx.x.x, 29500
xxx.xxx.x.x, 29501
```
To then run two agents on two different devices simply run the following commands:
```
Device 1:
python run_minigrid.py 0 29500

Device 2:
python run_minigrid.py 1 29501
```

Additional parameters are also available in the system
```
--num_agents: Modify the default value of the initial world size (default starts at 1)
--shell_config_path: Modify the default path to the shell configuration JSON.
--exp_id: A unique ID/name for an experiment. Can be useful to seperate logging
--eval: Launches in evaluation mode
--localhost: Launches in localhost mode. Can be useful for debugging
--shuffle: Randomly shuffles the curriculum from the shell.json configuration. Can be useful for testing.
--comm_interval: An integer value to indicate the number of communications to perform per task.
--device: An integer value to indicate the device selection. By default it will select the GPU if available. Otherwise a value of 0 will indicate CPU.
--reference: The file path to the .csv file containing the address table of bootstrapping agents. These are the addresses the agent will use to connect to an existing network, or form a new one.
```

### Configuring environments/curriculum
Curriculums and environments can be modified from the shell.json files in shell_configs/. This file contains the curriculum for each agent. Per-environment specifications can be found in env_configs/.

## Maintainers
The repository is currently developed and maintained by researchers from Loughborough University, Vanderbilt University, UC Riverside, and UT Dallas

## Bug Reporting
If you encounter any bugs using the code or have any questions, please raise an issue in the repository on GitHub.

## Acknowledgement
This material is based upon work supported by the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) under Contract No. HR00112190132.

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the United States Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA).