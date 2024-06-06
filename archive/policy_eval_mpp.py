import numpy as np
import tqdm
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
import gym
import math
import os
import torch.optim as optim
import diff_discrete_event as des
import matplotlib.pyplot as plt
import routing as rt
import json
import pathos.multiprocessing as mp
import policies as pol
import copy
import torch.distributions.one_hot_categorical as one_hot_sample
import argparse
import yaml

def run_policy(policy, seed, T, network, mu, h, queue_event_options, inter_arrival_dists, service_dists, init_queues):
    
    dq = des.DiffDiscreteEventSystem(network, mu, h, 
                                        inter_arrival_dists, service_dists, 
                                        init_queues = init_queues, 
                                        queue_event_options= queue_event_options,
                                        straight_through_min = False,
                                        batch = 1, 
                                        temp = 0.5, seed = seed,
                                        device = 'cpu')
    with torch.no_grad():
        for _ in trange(T):
            action = policy(dq.queues.detach(), dq.time_elapsed.detach())
            queue, cost, terminated, info = dq.step(action)
    
    return {'avg_cost': dq.cost / dq.time_elapsed,
            'seed': seed}

if __name__ == '__main__':


    # Settings
    num_runs = 100
    num_cores = 100
    policy_name = 'backpressure'
    

    # Parse and load env
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', type=str)

    args = parser.parse_args()

    # load config
    with open(f'./configs/env/{args.e}', 'r') as f:
        env_config = yaml.safe_load(f)
    
    name = env_config['name']
    print(f'env: {name}')

    ## Environment Parameters
    # load network
    if env_config['network'] is None:
        network = np.load(f'./env_data/{name}/{name}_network.npy')
    else:
        network = env_config['network']

    # load mu
    if env_config['mu'] is None:
        mu = np.load(f'./env_data/{name}/{name}_mu.npy')
    else:
        mu = env_config['mu']

    network = torch.tensor(network).float()
    mu = torch.tensor(mu).float()

    orig_s, orig_q = network.size()

    # repeat if server pools
    num_pool = env_config['num_pool']
    network = network.repeat_interleave(num_pool, dim = 0)
    mu = mu.repeat_interleave(num_pool, dim = 0)

    queue_event_options = env_config['queue_event_options']
    if queue_event_options is not None:
        if queue_event_options == 'custom':
            queue_event_options = torch.tensor(np.load(f'./env_data/{name}/{name}_delta.npy'))
        else:
            queue_event_options = torch.tensor(queue_event_options)

    h = torch.tensor(env_config['h'])

    init_queues = torch.tensor([env_config['init_queues']]).float()

    train_T = env_config['train_T']
    test_T = env_config['test_T']

    # arrival and service rates
    lam_type = env_config['lam_type']
    lam_params = env_config['lam_params']

    if lam_params['val'] is None:
        lam_r = np.load(f'./env_data/{name}/{name}_lam.npy')
    else:
        lam_r = lam_params['val']

    def lam(t):
        if lam_type == 'constant':
            lam = lam_r
        elif lam_type == 'step':
            is_surge = 1*(t.data.cpu().numpy() <= lam_params['t_step'])
            lam = is_surge * np.array(lam_params['val1']) + (1 - is_surge) * np.array(lam_params['val2'])
        else:
            return 'Nonvalid arrival rate'
        
        return lam
    
    def inter_arrival_dists(state, batch, t):
        exps = state.exponential(1, (batch, orig_q))
        lam_rate = lam(t)
        return exps / lam_rate
    
    def service_dists(state, batch, s, q, t):
        return state.exponential(1, (batch, s, q))

    ### Run policy
    run_policy_mp = lambda x: run_policy(**x)
    
    ### define policy
    policy = lambda queues, free_servers: pol.max_weight(queues, free_servers = torch.ones((1,network.size()[0])), 
                                                         network = network, mu = mu, h = h, queue_event_options = queue_event_options)

    eval_seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(num_runs)]
    rollout_seeds = [int.from_bytes(os.urandom(4), 'big') for _ in range(num_runs)]

    policy_jobs = []
    for i in range(num_runs):
        policy_jobs.append({'seed': eval_seeds[i], 
                           'T': test_T,
                           'num_pool': num_pool,
                           'network': network, 
                           'mu': mu, 
                           'h': h, 
                           'queue_event_options': queue_event_options,
                           'inter_arrival_dists': inter_arrival_dists, 
                           'service_dists': service_dists,
                           'init_queues': init_queues})

    # map with pool
    with mp.ProcessingPool(num_cores) as pool:
        results = pool.amap(run_policy_mp, policy_jobs)

    policy_loss = results.get()
    
    with open(f'{policy_name}_loss.json', 'w') as f:
        json.dump(policy_loss, f)        
