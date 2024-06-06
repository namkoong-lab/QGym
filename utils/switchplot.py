import sys 
sys.path.append('../')

import os
import numpy as np
import torch

import utils.routing as rt

from matplotlib import pyplot as plt


def plot_policy_switching_curve(policy_plot, 
                                net,
                                model_config,
                                env_config,
                                fig_dir = None, 
                                device = "cpu", 
                                base_level = 5, 
                                q = 2, 
                                max_queue = 50, 
                                inds = (0,1),
                                val_inds = (0,0)):
    
    X = np.arange(0, max_queue, 1)
    Y = np.arange(0, max_queue, 1)
    Z = np.zeros((max_queue,max_queue))

    for i in range(max_queue):
        for j in range(max_queue):
            obs = torch.tensor([base_level]*q)
            obs[inds[0]] = X[i]
            obs[inds[1]] = Y[j]

            obs = obs.float().unsqueeze(0).to(device)
            Z[i][j] = policy_plot(obs, net, model_config, env_config)[0][val_inds[0]][val_inds[0]]

    plt.imshow(Z, interpolation='nearest', origin='lower')
    if fig_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(fig_dir)
        plt.close()


def policy_plot(obs, net, model_config, env_config):
    pr = net(obs)

    if model_config['policy']['test_policy'] == 'sinkhorn':
        v, s_bar, q_bar = rt.pad(2*pr, obs, network = env_config['network'].unsqueeze(0), device = model_config['env']['device'])
        pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                model_config['policy']['sinkhorn']['num_iter'],
                                model_config['policy']['sinkhorn']['temp'],
                                model_config['policy']['sinkhorn']['eps'],
                                model_config['policy']['sinkhorn']['back_temp'],
                                model_config['env']['device'])
    elif model_config['policy']['test_policy'] == 'linear_assigment':
        v, s_bar, q_bar = rt.pad(pr + lex, test_dq_batch[-1].queues.detach(), network = env_config['network'], device = 'cpu')
        pr = rt.linear_assignment_batch(v, s_bar, q_bar)
    elif model_config['policy']['test_policy'] == 'softmax':
        pr = pr
    else:
        pass

    return pr

def create_plot_dir(model_config, env_config, experiment_name):
    model_name = model_config['name']
    fig_dir = f'../logs/plot/{experiment_name}/{model_name}/{model_name}'
    if not os.path.exists(f'../logs/plot/{experiment_name}'):
        os.makedirs(f'../logs/plot/{experiment_name}')
    if not os.path.exists(f'../logs/plot/{experiment_name}/{model_name}'):
        os.makedirs(f'../logs/plot/{experiment_name}/{model_name}')
    return fig_dir

def create_loss_dir(model_config, env_config, experiment_name):
    model_name = model_config['name']
    loss_dir = f'../logs/loss/{experiment_name}/{model_name}'
    if not os.path.exists(f'../logs/loss/{experiment_name}'):
        os.makedirs(f'../logs/loss/{experiment_name}')
    if not os.path.exists(f'../logs/loss/{experiment_name}/{model_name}'):
        os.makedirs(f'../logs/loss/{experiment_name}/{model_name}')
    return loss_dir

