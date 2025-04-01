#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################



#   __________ .__                              __   .__     .__                                           
#   \______   \|  |    ____    ______  ______ _/  |_ |  |__  |__|  ______   _____    ____    ______  ______
#    |    |  _/|  |  _/ __ \  /  ___/ /  ___/ \   __\|  |  \ |  | /  ___/  /     \ _/ __ \  /  ___/ /  ___/
#    |    |   \|  |__\  ___/  \___ \  \___ \   |  |  |   Y  \|  | \___ \  |  Y Y  \\  ___/  \___ \  \___ \ 
#    |______  /|____/ \___  >/____  >/____  >  |__|  |___|  /|__|/____  > |__|_|  / \___  >/____  >/____  >
#           \/            \/      \/      \/              \/          \/        \/      \/      \/      \/ 
#
#                                                     :')


import numpy as np
import time
import torch
from .torch_utils import *
from tensorboardX import SummaryWriter
from ..shell_modules import *

import multiprocessing.dummy as mpd
from colorama import Fore
import psutil
import pandas as pd

try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

from memory_profiler import profile
import csv


def _shell_itr_log(logger, agent, agent_idx, itr_counter, task_counter, dict_logs, mask_interval):
    logger.info(Fore.BLUE + 'agent %d, task %d / iteration %d, total steps %d, ' \
    'mean/max/min reward %f/%f/%f' % (agent_idx, task_counter, \
        itr_counter,
        agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards)
    ))
    logger.scalar_summary('agent_{0}/last_episode_avg_reward'.format(agent_idx), \
        np.mean(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_std_reward'.format(agent_idx), \
        np.std(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_max_reward'.format(agent_idx), \
        np.max(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_min_reward'.format(agent_idx), \
        np.min(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/iteration_avg_reward'.format(agent_idx), \
        np.mean(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_std_reward'.format(agent_idx), \
        np.std(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_max_reward'.format(agent_idx), \
        np.max(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_min_reward'.format(agent_idx), \
        np.min(agent.iteration_rewards))

    prefix = 'agent_{0}_'.format(agent_idx)
    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('{0}debug/{1}_avg_norm'.format(prefix, tag), np.mean(value_norm))
            logger.scalar_summary('{0}debug/{1}_avg'.format(prefix, tag), value.mean())
            logger.scalar_summary('{0}debug/{1}_std'.format(prefix, tag), value.std())
            logger.scalar_summary('{0}debug/{1}_max'.format(prefix, tag), value.max())
            logger.scalar_summary('{0}debug/{1}_min'.format(prefix, tag), value.min())

    # Check if the file exists to control writing the header
    file_exists = os.path.isfile(f'{logger.log_dir}/logits_data.csv')
    if hasattr(agent, 'layers_output'):
        # Open the CSV file in append mode to keep adding data
        with open(f'{logger.log_dir}/logits_data.csv', mode='a') as file:
            writer = csv.writer(file)
            
            # Write the header only if the file is being created for the first time
            if not file_exists:
                n_actions = agent.task_action_space_size  # Assuming layers_output is not empty
                header = ['Step'] + [f'Logit_Action{i+1}' for i in range(n_actions)]
                writer.writerow(header)
            
            for tag, value in agent.layers_output:
                if tag == 'policy_logits':
                    # Write each timestep's logits in a single row
                    for step, logits in enumerate(value):
                        row = [step] + [logit.item() for logit in logits]
                        writer.writerow(row)

    #print(dict_logs)
    for key, value in dict_logs.items():
        #print(key, value)
        logger.scalar_summary('{0}debug_extended/{1}_avg'.format(prefix, key), np.mean(value))
        logger.scalar_summary('{0}debug_extended/{1}_std'.format(prefix, key), np.std(value))
        logger.scalar_summary('{0}debug_extended/{1}_max'.format(prefix, key), np.max(value))
        logger.scalar_summary('{0}debug_extended/{1}_min'.format(prefix, key), np.min(value))

    logger.scalar_summary('{0}communication_interval/'.format(prefix), mask_interval)

    return

# metaworld/continualworld
def _shell_itr_log_mw(logger, agent, agent_idx, itr_counter, task_counter, dict_logs, mask_interval):
    logger.info(Fore.BLUE + 'agent %d, task %d / iteration %d, total steps %d, ' \
    'mean/max/min reward %f/%f/%f, mean/max/min success rate %f/%f/%f' % (agent_idx, \
        task_counter,
        itr_counter,
        agent.total_steps,
        np.mean(agent.iteration_rewards),
        np.max(agent.iteration_rewards),
        np.min(agent.iteration_rewards),
        np.mean(agent.iteration_success_rate),
        np.max(agent.iteration_success_rate),
        np.min(agent.iteration_success_rate)
    ))
    logger.scalar_summary('agent_{0}/last_episode_avg_reward'.format(agent_idx), \
        np.mean(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_std_reward'.format(agent_idx), \
        np.std(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_max_reward'.format(agent_idx), \
        np.max(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/last_episode_min_reward'.format(agent_idx), \
        np.min(agent.last_episode_rewards))
    logger.scalar_summary('agent_{0}/iteration_avg_reward'.format(agent_idx), \
        np.mean(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_std_reward'.format(agent_idx), \
        np.std(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_max_reward'.format(agent_idx), \
        np.max(agent.iteration_rewards))
    logger.scalar_summary('agent_{0}/iteration_min_reward'.format(agent_idx), \
        np.min(agent.iteration_rewards))

    logger.scalar_summary('agent_{0}/last_episode_avg_success_rate'.format(agent_idx), \
        np.mean(agent.last_episode_success_rate))
    logger.scalar_summary('agent_{0}/last_episode_std_success_rate'.format(agent_idx), \
        np.std(agent.last_episode_success_rate))
    logger.scalar_summary('agent_{0}/last_episode_max_success_rate'.format(agent_idx), \
        np.max(agent.last_episode_success_rate))
    logger.scalar_summary('agent_{0}/last_episode_min_success_rate'.format(agent_idx), \
        np.min(agent.last_episode_success_rate))
    logger.scalar_summary('agent_{0}/iteration_avg_success_rate'.format(agent_idx), \
        np.mean(agent.iteration_success_rate))
    logger.scalar_summary('agent_{0}/iteration_std_success_rate'.format(agent_idx), \
        np.std(agent.iteration_success_rate))
    logger.scalar_summary('agent_{0}/iteration_max_success_rate'.format(agent_idx), \
        np.max(agent.iteration_success_rate))
    logger.scalar_summary('agent_{0}/iteration_min_success_rate'.format(agent_idx), \
        np.min(agent.iteration_success_rate))

    prefix = 'agent_{0}_'.format(agent_idx)
    if hasattr(agent, 'layers_output'):
        for tag, value in agent.layers_output:
            value = value.detach().cpu().numpy()
            value_norm = np.linalg.norm(value, axis=-1)
            logger.scalar_summary('{0}debug/{1}_avg_norm'.format(prefix, tag), np.mean(value_norm))
            logger.scalar_summary('{0}debug/{1}_avg'.format(prefix, tag), value.mean())
            logger.scalar_summary('{0}debug/{1}_std'.format(prefix, tag), value.std())
            logger.scalar_summary('{0}debug/{1}_max'.format(prefix, tag), value.max())
            logger.scalar_summary('{0}debug/{1}_min'.format(prefix, tag), value.min())

    for key, value in dict_logs.items():
        logger.scalar_summary('{0}debug_extended/{1}_avg'.format(prefix, key), np.mean(value))
        logger.scalar_summary('{0}debug_extended/{1}_std'.format(prefix, key), np.std(value))
        logger.scalar_summary('{0}debug_extended/{1}_max'.format(prefix, key), np.max(value))
        logger.scalar_summary('{0}debug_extended/{1}_min'.format(prefix, key), np.min(value))

    logger.scalar_summary('{0}communication_interval/'.format(prefix), mask_interval)

    return

# Concurrent implementations
'''
shell training: concurrent processing for event-based communication. a multitude of improvements have been made compared
to the previous shell_dist_train.
'''
#@profile
def trainer_learner(agent, comm, agent_id, manager, mask_interval, mode):
    ###############################################################################
    ### Setup logger
    logger = agent.config.logger
    #print(Fore.WHITE, end='') 
    #logger.info('***** start l2d2-c training')


    ###############################################################################
    ### Setup trainer loop pre-requisites
    shell_done = False
    shell_iterations = 0
    shell_tasks = agent.config.cl_tasks_info # tasks for agent
    #shell_task_ids = agent.config.task_ids
    shell_task_counter = 0
    #shell_eval_tracker = False
    shell_eval_data = []
    num_eval_tasks = len(agent.evaluation_env.get_all_tasks())
    shell_eval_data.append(np.zeros((num_eval_tasks, ), dtype=np.float32))
    #shell_metric_icr = [] # icr => instant cumulative reward metric. NOTE may be redundant now
    #eval_data_fh = open(logger.log_dir + '/eval_metrics_agent_{0}.csv'.format(agent_id), 'a', 
    #    buffering=1) # buffering=1 means flush data to file after every line written
    #shell_eval_end_time = None

    idling = True   # Flag to handle idling behaviour when curriculum ends.
    dict_to_query = None      # Variable to store the current task dictionary recrod to query for.


    ###############################################################################
    ### Select iteration logging function based on environment. Required for Meta World and Continual World
    if agent.task.name == agent.config.ENV_METAWORLD or agent.task.name == agent.config.ENV_CONTINUALWORLD or agent.task.name == agent.config.ENV_COMPOSUITE:
        itr_log_fn = _shell_itr_log_mw

    else:
        itr_log_fn = _shell_itr_log


    ###############################################################################
    ### Set the first task each agent is meant to train on
    states_ = agent.task.reset_task(shell_tasks[0])
    agent.states = agent.config.state_normalizer(states_)
    #logger.info(f'***** ENVIRONMENT SWITCHING TASKS')
    #logger.info(f'***** agent {agent_id} / setting first task (task 0)')
    #logger.info(f"***** task: {shell_tasks[0]['task']}")
    #logger.info(f"***** task_label: {shell_tasks[0]['task_label']}")

    # Set first task mask and record manually otherwise we run into issues with the implementation in the model.
    #agent.task_train_start_emb(task_embedding=None)

    # NOTE: ADDED detect.add_embedding() to accomodate the WEIGHTED AVG COSINE SIM
    agent.current_task_emb = torch.zeros(agent.get_task_emb_size())
    if agent.config.continuous == True:
        agent.task_train_start_emb(task_embedding=agent.current_task_emb, current_reward=agent.iteration_success_rate)   # TODO: There is an issue with this which is that the first task will be set as zero and then the detect module with do some learning, find that the task does not match the zero embedding and start another task change. This leaves the first entry to a task change as useless. Also issues if we try to moving average this
    else:
        agent.task_train_start_emb(task_embedding=agent.current_task_emb, current_reward=agent.iteration_rewards)
    #agent.detect.add_embedding(agent.current_task_emb, np.mean(agent.iteration_rewards))
    del states_


    ###############################################################################
    ### Start the comm module with the initial states and the first task label.
    # Returns shared queues to enable interaction between comm and agent.
    queue_label, queue_mask, queue_label_send, queue_mask_recv = comm.parallel(manager)


    ###############################################################################
    ### Logging setup (continued)
    tb_writer_emb = SummaryWriter(logger.log_dir + '/Detect_Component_Generated_Embeddings')
    _embeddings, _labels, exchanges, task_times, detect_module_activations = [], [], [], [], []
    task_times.append([0, shell_iterations, np.argmax(shell_tasks[0]['task_label'], axis=0), time.time()])
    detect_activations_log_path = logger.log_dir + '/detect_activations.csv'
    masks_log_path = logger.log_dir + '/exchanges.csv'
    emb_dist_log = logger.log_dir + '/distances.csv'
    m_dist_log1 = logger.log_dir + '/maha_cov_ident.csv'
    m_dist_log2 = logger.log_dir + '/maha_cov_mean.csv'
    cossim_log = logger.log_dir + '/cos_sim.csv'
    density_log = logger.log_dir + '/density.csv'
    emd_log = logger.log_dir + '/emd.csv'
    wdist_log = logger.log_dir + '/wdist_log.csv'


    ###############################################################################
    '''### Comm module event handlers. These run in parallel to enable the interactions between the comm and agent.
    def mask_handler():
        """
        Handles incoming masks from other agents. Linearly combines masks and adds resulting mask to network.
        """
        while True:
            masks_list  = queue_mask.get()
            
            #logger.info(Fore.WHITE + f'\n######### MASK RECEIVED FROM COMM #########')

            _masks = []
            _avg_embeddings = []
            _avg_rewards = []
            _mask_labels = []
            #print(f'masks list length: {len(masks_list)}')

            try:
                if len(masks_list) > 0:
                    for mask_response_dict in masks_list:

                        mask = mask_response_dict['mask']
                        embedding = mask_response_dict['embedding']
                        reward = mask_response_dict['reward']
                        label = mask_response_dict['label']
                        ip = mask_response_dict['ip']
                        port = mask_response_dict['port']

                        #print(type(label), len(label), label)

                        #_masks.append(mask)
                        _masks.append(agent.vec_to_mask(mask.to(agent.config.DEVICE))) # Use this one if using unified LC
                        _avg_embeddings.append(embedding)
                        _avg_rewards.append(reward)
                        _mask_labels.append(label)

                        # Log successful mask transfer
                        data = [
                            {
                                'iteration': shell_iterations,
                                'ip': ip,
                                'port': port,
                                'task_id': np.argmax(label,axis=0),
                                'reward': reward,
                                'embedding': embedding,
                                'mask_dim': len(mask),
                                'mask_tensor': mask
                            }
                        ]
                    
                        df = pd.DataFrame(data)
                        df.to_csv(masks_log_path, mode='a', header=not pd.io.common.file_exists(masks_log_path), index=False)


                        #exchanges.append([shell_iterations, ip, port, np.argmax(label, axis=0), reward, embedding, len(mask), mask])
                        #np.savetxt(logger.log_dir + '/exchanges_{0}.csv'.format(agent_id), exchanges, delimiter=',', fmt='%s')
                    
                    #logger.info(Fore.WHITE + f'Updating seen tasks dictionary with new data')
                    # Update the knowledge base with the expected reward
                    #agent.update_seen_tasks(_avg_embeddings[0], _avg_rewards[0], _mask_labels[0])#knowledge_base.update({tuple(label.tolist()): reward})
                    
                    # Traceback (most recent call last):
                    # File "/home/lunet/cosn2/detect-l2d2c/deeprl-shell/deep_rl/utils/trainer_shell.py", line 671, in mask_handler
                    #     agent.update_seen_tasks(_avg_embeddings[0], _avg_rewards[0], _mask_labels[0])
                    # IndexError: list index out of range

                    logger.info(Fore.WHITE + f'COMPOSING RECEIVED MASKS')
                    # Update the network with the linearly combined mask
                    #agent.distil_task_knowledge_embedding(_masks[0])       # This will only take the first mask in the list
                    #agent.consolidate_incoming(_masks)                      # This will take all the masks in the list and linearly combine with the random/current mask
                    if agent.config.continuous == True:
                        agent.update_community_masks(_masks, np.mean(agent.iteration_success_rate))
                    else:
                        agent.update_community_masks(_masks, np.mean(agent.iteration_rewards))
                    _masks = []

                    #logger.info(Fore.WHITE + 'COMPOSED MASK ADDED TO NETWORK!')
            except Exception as e:
                traceback.print_exc()

    def conv_handler():
        """
        Handles interval label to mask conversions for outgoing mask responses.
        """
        while True:
            try:
                to_convert = queue_label_send.get()

                logger.info(Fore.WHITE + 'GOT ID TO CONVERT TO MASK')
                logger.info(f"MASK/TASK ID: {to_convert['sender_task_id']}")

                sender_task_id = to_convert['sender_task_id']
                mask = agent.idx_to_mask(sender_task_id)

                #print(sender_task_id)
                #print(agent.seen_tasks[sender_task_id])

                reward = agent.seen_tasks[sender_task_id]['reward']
                emb = agent.seen_tasks[sender_task_id]['task_emb']
                label = agent.seen_tasks[sender_task_id]['ground_truth']
                print(Fore.LIGHTRED_EX + f'Found valid mask: {mask} with reward: {reward} and emb: {emb}')

                to_convert['response_mask'] = mask
                to_convert['response_reward'] = reward
                to_convert['response_embedding'] = emb
                to_convert['response_label'] = label
                queue_mask_recv.put((to_convert))
            except Exception as e:
                traceback.print_exc()
    

    ###############################################################################
    ### Start threads for the mask and conversion handlers.
    t_mask = mpd.Pool(processes=1)
    t_conv = mpd.Pool(processes=1)
    t_mask.apply_async(mask_handler)
    t_conv.apply_async(conv_handler)'''
    

    ###############################################################################
    ### MAIN OPERATIONAL LOOP
    while True:
        start_time = time.time()
        ###############################################################################
        ### Idling behaviour. Idles until terminated when curriculum is completed.
        # While idling the agent acts as a server that can be queried for knowledge by
        # other agents.
        if shell_done:
            if idling:
                print('Agent is idling...') # Once idling the agent acts as a server that can be queried for knowledge until the agent encounters a new task (support not implemented yet)
                
                # Log all the embeddings and labels to tensorboard projector
                # NOTE: RE-ENABLE FOR MAIN EXPERIMENTS!!!!!!!!!!!!!
                #emb_t = torch.stack(tuple(_embeddings))
                #tb_writer_emb.add_embedding(emb_t, metadata=_labels, global_step=shell_iterations)
                #agent.env.close()
                
                idling = False
                # Alternatively we can shutdown the agent here or do something for the experiment termination.
                
            #if omniscient_mode:
            #    if shell_iterations % mask_interval == 0:
            #        queue_label.put(None)

            #    shell_iterations += 1

            time.sleep(2) # Sleep value ensures the idling works as intended (fixes a problem encountered with the multiprocessing)
            continue
        #print()


        
        ###############################################################################
        ### Registry logging output.
        logger.info(Fore.RED + 'GLOBAL REGISTRY (seen_tasks dict)')
        for key, val in agent.seen_tasks.items(): logger.info(f"{key} --> embedding: {val['task_emb']}, reward: {val['reward']}, ground truth task id: {np.argmax(val['ground_truth'], axis=0)}, label length: {len(val['ground_truth'])}")
        for key, val in agent.seen_tasks.items(): logger.info(f"{key} --> reward: {val['reward']}, ground truth task id: {np.argmax(val['ground_truth'], axis=0)}, label length: {len(val['ground_truth'])}")
        #logger.info(f'{Fore.BLUE}----------------------- Text logging complete in {time.time() - start_time} seconds -----------------------\n')


        ###############################################################################
        ### Query for knowledge using communication process. Send label/embedding to the communication module to query for relevant knowledge from other peers.
        if dict_to_query is not None:
            #print('Entropy:', np.mean(dict_logs['entropy']))
            #print('Reward:', np.mean(agent.iteration_rewards))
            #if agent.config.continuous == True:
            #    mask_interval = adaptive_communication_interval(np.mean(dict_logs['entropy']), agent.iteration_success_rate, agent.task.action_space.n)
            #else:
            #    mask_interval = adaptive_communication_interval(np.mean(dict_logs['entropy']), agent.iteration_rewards, agent.task.action_space.n)

            if shell_iterations % mask_interval == 0:
                # Approach 2: At this point consolidate masks and then we can reset beta parameters. Then we can get new masks from network and combine.
                dict_to_query['shell_iteration'] = shell_iterations
                queue_label.put(dict_to_query)

        # Report performance to evaluation agent if present. Otherwise skip.
        if agent.config.evaluator_present.value == True:
            logger.info('Reporting performance to evaluation agent')
            dict_to_report = agent.seen_tasks[agent.current_task_key]
            dict_to_report['shell_iteration'] = shell_iterations
            dict_to_report['mask'] = agent.idx_to_mask(agent.current_task_key)
            dict_to_report['eval'] = True
            dict_to_report['parameters'] = None
            queue_label.put(dict_to_report)

        #logger.info(f'{Fore.BLUE}----------------------- Communication querying complete in {time.time() - start_time} seconds -----------------------\n')


        
        ###############################################################################
        ### Agent training iteration: collect on policy data and optimise the agent
        '''
        Handles the data collection and optimisation within the agent.
        TODO: Look into multihreading/multiprocessing the data collection and optimisation of the agent
                to achieve the continous data collection we want for a real world scenario.
            
            Possibly the optimisation could be made a seperate process parallel to the data collection
                process, similar to the communication-agent(trainer) architecture. Data collection and the code
                below would run together in the main loop.

        NOTE: Parallelising the backward and forward passes may not be possible using CUDA due to limitations
                on CUDA parallelisation. It could be done if we train the system on CPU using
                mulithreading or multiprocessing however it is challenging as there needs to be some sort of
                synchronisation.
        '''
        dict_logs = agent.iteration()
        shell_iterations += 1
        agent.iteration_entropy = dict_logs['entropy']

        # Log the beta parameters for the curren task
        agent.log_betas(shell_iterations)
        #logger.info(f'{Fore.BLUE}----------------------- Iteration function complete in {time.time() - start_time} seconds -----------------------\n')

        
        ###############################################################################
        ### Run detect module. Generates embedding for SAR. Perform check to see if there has been a task change or not.
        _dist_threshold = agent.emb_dist_threshold
        if shell_iterations != 0 and shell_iterations % agent.detect_module_activation_frequency == 0 and agent.data_buffer.size() >= (agent.detect.get_num_samples()):
            # Run the detect module on SAR and return some logging output.
            task_change_flag, new_emb, ground_truth_task_label, dist_arr, emb_bool, agent_seen_tasks = run_detect_module(agent)

            emb_dist = dist_arr[0]
            m_dist1 = dist_arr[1]
            m_dist2 = dist_arr[2]
            cos_sim = dist_arr[3]
            density = dist_arr[4]
            emd = dist_arr[5]
            w_dist = dist_arr[6]
            #reduced_emb_dist = dist_arr[7]

            # Log euclidean distance with moving average on current embedding
            '''data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(emb_dist)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(emb_dist_log, mode='a', header=not pd.io.common.file_exists(emb_dist_log), index=False)

            # Log mahalanobis distance with moving average with identity covariance matrix
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(m_dist1)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(m_dist_log1, mode='a', header=not pd.io.common.file_exists(m_dist_log1), index=False)

            # Log mahalanobis distance with moving average with mean covariance matrix
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(m_dist2)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(m_dist_log2, mode='a', header=not pd.io.common.file_exists(m_dist_log2), index=False)

            # Log cosine similarity with moving average
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(cos_sim)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(cossim_log, mode='a', header=not pd.io.common.file_exists(cossim_log), index=False)

            # Kernel density with moving average
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(density)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(density_log, mode='a', header=not pd.io.common.file_exists(density_log), index=False)

            # Wasserstein distance / Earth Mover's Distance
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(emd)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(emd_log, mode='a', header=not pd.io.common.file_exists(emd_log), index=False)

            # Wasserstein distance reference
            data = [
                {
                    'iteration': shell_iterations,
                    'distance' : float(w_dist)    # convert from tensor to float
                }
            ]
            df = pd.DataFrame(data)
            df.to_csv(wdist_log, mode='a', header=not pd.io.common.file_exists(wdist_log), index=False)'''
            
            if task_change_flag:
                #logger.info(Fore.YELLOW + f'TASK CHANGE DETECTED! NEW MASK CREATED. CURRENT TASK INDEX: {agent.current_task_key}')
            
                #log_string = f'Time: {time.time()}, Iteration: {shell_iterations}, Num samples for detection: {agent.detect.get_num_samples()}, Task change flag: {task_change_flag}, New embedding: {new_emb}, Ground truth label: {ground_truth_task_label}, Current embedding: {agent.current_task_emb}, Threshold: {_dist_threshold}, Distance: {emb_dist}, Embedding similarity: {emb_bool}, Agent seen tasks: {agent_seen_tasks}'
                #detect_module_activations.append([log_string])
                #np.savetxt(logger.log_dir + '/detect_activations_{0}.csv'.format(agent_id), detect_module_activations, delimiter=',', fmt='%s')

                data = [
                    {
                        'Iteration': shell_iterations,
                        'Time': time.time(),
                        'Num samples for detection': agent.detect.get_num_samples(),
                        'New embedding': new_emb,
                        'Ground truth label': ground_truth_task_label,
                        'Current embedding': agent.current_task_emb,
                        'Threshold': _dist_threshold,
                        'Distance': emb_dist,
                        'Similar': emb_bool,
                        'Agent seen_tasks()': agent_seen_tasks 
                    }
                ]
                df = pd.DataFrame(data)
                df.to_csv(detect_activations_log_path, mode='a', header=not pd.io.common.file_exists(detect_activations_log_path), index=False)
                del data
            
            # Update the dictionary containing the current task embedding to query for.
            dict_to_query = agent.seen_tasks[agent.current_task_key]
            dict_to_query['parameters'] = 0.5 #€ Cosine similarity threshold

            # Logging embeddings and labels
            if new_emb is not None:
                _label_one_hot = torch.tensor(np.array([ground_truth_task_label]))
                
                # Convert one-hot label to integer
                _label = torch.argmax(_label_one_hot).item()

                #_embeddings.append(new_emb)
                #_labels.append(_label)

                #logger.info(Fore.WHITE + f'Embedding: {new_emb}')
                #logger.info(f'Task ID: {_label}')
                #logger.info(f'Distance: {emb_dist}')
                #logger.info(f'Threshold: {agent.emb_dist_threshold}')
                #emb_t = torch.stack(tuple(_embeddings))
                #l_t = torch.stack(tuple(_labels))
                #tb_writer_emb.add_embedding(emb_t, metadata=_labels, global_step=shell_iterations)
            
            del task_change_flag, new_emb, ground_truth_task_label, dist_arr, emb_bool, agent_seen_tasks
        
        #logger.info(f'{Fore.BLUE}----------------------- Run detect method complete in {time.time() - start_time} seconds -----------------------\n')
        
        ###############################################################################
        ### Logs metrics to tensorboard log file and updates the embedding, reward pair in this cycle for a particular task.
        if shell_iterations % agent.config.iteration_log_interval == 0:
            itr_log_fn(logger, agent, agent_id, shell_iterations, shell_task_counter, dict_logs, mask_interval)
                
            # Save agent model
            agent.save(agent.config.log_dir + '/%s-%s-model-%s.bin' % (agent.config.agent_name, agent.config.tag, agent.task.name))
        
        #logger.info(f'{Fore.BLUE}----------------------- Iteration logging complete in {time.time() - start_time} seconds -----------------------\n')


        
        ###############################################################################
        ### Environment task change at the end of the max steps for each task. Agent is not aware of this change and must detect it using the detect module.
        '''
        # end of current task training. move onto next task or end training if last task.
        # i.e., Task Change occurs here. For detect module, if the task embedding signifies a task
        # change then that should occur here.

        If we want to use a Fetch All mode for ShELL then we need to add a commmunication component
        at task change which broadcasts the mask to all other agents currently on the network.

        Otherwise the current implementation is a On Demand mode where each agent requests knowledge
        only when required.
        '''
        if not agent.config.max_steps: raise ValueError('`max_steps` should be set for each agent')
        task_steps_limit = agent.config.max_steps[shell_task_counter] * (shell_task_counter + 1)

        # If agent completes the maximum number of steps for a task then switch to the next task in the curriculum.
        if agent.total_steps >= task_steps_limit:
            task_counter_ = shell_task_counter
            logger.info('\n' + Fore.WHITE + f'*****agent {agent_id} / end of training on task {task_counter_}')
            
            #MOVED to Assing EMB in PPO agent.task_train_end()

            # Increment task counter
            task_counter_ += 1
            shell_task_counter = task_counter_

            # If curriculum is not completed, switch to the next task in the curriculum
            if task_counter_ < len(shell_tasks):
                # new task
                logger.info(Fore.WHITE + f'***** ENVIRONMENT SWITCHING TASKS')
                logger.info(Fore.WHITE + f'***** agent {agent_id} / set next task {task_counter_}')
                logger.info(Fore.WHITE + f"***** task: {shell_tasks[task_counter_]['task']}")
                logger.info(Fore.WHITE + f"***** task_label: {shell_tasks[task_counter_]['task_label']}")
                
                # Set the new task from the environment. Agent remains unaware of this change and will continue until
                # detect module detects distrubtion shift.
                states_ = agent.task.reset_task(shell_tasks[task_counter_]) # reset_task sets the new task and returns the reset intial states.
                agent.states = agent.config.state_normalizer(states_)
                
                #MOVED to Assing EMB in PPO agent agent.task_train_start(shell_tasks[task_counter_]['task_label'])

                del states_

                task_times.append([task_counter_, shell_iterations, np.argmax(shell_tasks[task_counter_]['task_label'], axis=0), time.time()])
                np.savetxt(logger.log_dir + '/task_changes_{0}.csv'.format(agent_id), task_times, delimiter=',', fmt='%s')

            else:
                shell_done = True # training done for all task for agent. This leads to the idling behaviour in next iteration.
                logger.info(f'*****agent {agent_id} / end of all training')

            del task_counter_
        


        logger.info(f'{Fore.BLUE}----------------------- Iteration complete in {time.time() - start_time} seconds -----------------------\n')




#########################################################################################################################################
########################################  U T I L I T Y      F U N C T I O N S  #########################################################
#########################################################################################################################################

def run_detect_module(agent):
    '''Uitility function for running all the necassery methods and function for the detect module
    so the approprate embeddings are generated for each batch of SAR data'''
    
    #Initilize the retun varibles with None values in the case of the detect module not being appropriate to run.
    task_change_detected, emb_dist, emb_bool, ground_truth_task_label = None, None, None, torch.tensor(0)
    emb_dist, m_dist1, m_dist2, cos_sim, density, emd, wasserstein_distance = 0, 0, 0, 0, 0, 0, 0
    
    # Extract SAR data from agent's replay buffer
    sar_data = agent.extract_sar()

    # Compute embedding
    new_embedding = agent.compute_task_embedding(sar_data, agent.get_task_action_space_size())
    
    # Get current embedding and task label
    current_embedding = agent.current_task_emb
    ground_truth_task_label = agent.get_current_task_label()

    # Compute embedding distance
    emb_dist = agent.detect.emb_distance(current_embedding, new_embedding)

    # Check if new embedding and current embedding match
    emb_bool = current_embedding == new_embedding

    # Check if task has changed
    task_change_detected = agent.assign_task_emb(new_embedding, emb_dist)
    agent_seen_tasks = agent.get_seen_tasks()

    # Log all distances
    distances = [emb_dist, m_dist1, m_dist2, cos_sim, density, emd, wasserstein_distance]

    # Return all data
    return task_change_detected, new_embedding, ground_truth_task_label, distances, emb_bool, agent_seen_tasks

def adaptive_communication_interval(policy_entropy, recent_returns, lambda_base=20, alpha=4, beta=10):
    """
    Dynamically adjusts the communication interval based on policy entropy and return change.

    - High entropy (H) => frequent communication (low interval)
    - Large return improvement (Δr) => allows more communication
    - Low entropy & stable returns => longer intervals (less communication)
    
    Args:
        policy_entropy (float): Current policy entropy (0 to 1).
        recent_returns (list): List of recent episodic returns.
        lambda_base (int): Base interval factor (higher = less frequent communication).
        alpha (float): Scaling factor for entropy decay.
        beta (float): Scaling factor for return influence.

    Returns:
        int: Adjusted communication interval.
    """

    # Compute return difference (Δr), default to large value if not enough history
    delta_r = (recent_returns[-1] - recent_returns[-2]) if len(recent_returns) > 1 else 1.0

    # Ensure delta_r is non-negative for log scaling
    delta_r = max(delta_r, 1e-6)

    # Compute adaptive communication interval
    comm_interval = lambda_base * np.exp(-alpha * policy_entropy) + beta * np.log(1 + delta_r)

    return int(np.clip(comm_interval, 5, 200))