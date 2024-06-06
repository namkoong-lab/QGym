import numpy as np
import tqdm
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
import gym
import argparse
import math
import os
import torch.optim as optim
import diff_discrete_event as des
import matplotlib.pyplot as plt
import routing as rt
import policies as pol
import json
import torch.distributions.one_hot_categorical as one_hot_sample

import yaml

def plot_policy_switching_curve(net, 
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
            Z[i][j] = net(obs)[0][val_inds[0]][val_inds[0]]

    plt.imshow(Z, interpolation='nearest', origin='lower')
    if fig_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(fig_dir)
        plt.close()

class AdvNet(nn.Module):
    def __init__(self, q, layers, hidden_dim):
        super().__init__()
        self.q = q
        self.layers = layers
        self.hidden_dim = hidden_dim

        self.input_fc = nn.Linear(self.q, hidden_dim)
        self.layers_fc = nn.ModuleList()
        for _ in range(layers):
            self.layers_fc.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_fc = nn.Linear(hidden_dim, self.q)
        #self.output_fc = nn.Linear(hidden_dim, self.q * self.s)
        
    def forward(self, x):
        
        # Input layer
        x = F.relu(self.input_fc(x))

        # Hidden layer
        for l in range(self.layers):
            x = F.relu(self.layers_fc[l](x))

        # Output layer
        x = self.output_fc(x)
        return F.softmax(x, dim = 1)
    
class PriorityNet(nn.Module):
    def __init__(self, s, q, layers, hidden_dim, f_time = False, x_stats = None, t_stats = None):
        super().__init__()
        self.s = s
        self.q = q
        self.x_stats = x_stats
        self.t_stats = t_stats
        self.layers = layers
        self.hidden_dim = hidden_dim
        
        self.f_time = f_time
        
        if self.f_time:
            self.input_fc = nn.Linear(self.q + 1, hidden_dim)
        else:
            self.input_fc = nn.Linear(self.q, hidden_dim)
            
        self.layers_fc = nn.ModuleList()
        for _ in range(layers):
            self.layers_fc.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_fc = nn.Linear(hidden_dim, self.s * self.q)
        #self.output_fc = nn.Linear(hidden_dim, self.q * self.s)
        
    def forward(self, x, t = 0):
        
        # Input layer
        batch = x.size()[0]
        
        if self.x_stats is not None:
            x = (x - self.x_stats[0]) / self.x_stats[1]

        if self.t_stats is not None:    
            t = (t - self.t_stats[0]) / self.t_stats[1]
        
        if self.f_time:
            x = torch.cat((x, t), 1)
            
        x = F.relu(self.input_fc(x))

        # Hidden layer
        for l in range(self.layers):
            x = F.relu(self.layers_fc[l](x))

        # Output layer
        x = self.output_fc(x)
        return 2*F.softmax(torch.reshape(x, (batch, self.s , self.q)), dim = 2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', type=str)
    parser.add_argument('-m', type=str)

    args = parser.parse_args()

    # load config
    with open(f'./configs/env/{args.e}', 'r') as f:
        env_config = yaml.safe_load(f)

    with open(f'./configs/model/{args.m}', 'r') as f:
        model_config = yaml.safe_load(f)
    
    name = env_config['name']
    print(f'env: {name}')

    # Set seed
    if model_config['env']['model_seed'] is not None:
        torch.manual_seed(model_config['env']['model_seed'])

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
    
    init_test_queues = torch.tensor([env_config['init_queues']]).float()
    init_train_queues = torch.tensor([env_config['init_queues']]).float()

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
    
    ## Model parameters
    model_name = model_config['name']
    print(f'model: {model_name}')

    checkpoint = model_config['checkpoint']
    checkpoint_dir = f"./models/{name}/{model_name}/{model_name}"

    num_epochs = model_config['opt']['num_epochs']
    train_batch = model_config['opt']['train_batch']
    test_batch = model_config['opt']['test_batch']
    lr = model_config['opt']['lr']
    betas = model_config['opt']['betas']
    threshold = model_config['opt']['threshold']
    
    device = model_config['env']['device']
    test_seed = model_config['env']['test_seed']
    train_seed = model_config['env']['train_seed']
    env_temp = model_config['env']['env_temp']
    test_freq = model_config['env']['test_freq']
    straight_through_min = model_config['env']['straight_through_min']

    layers = model_config['param']['layers']
    width = model_config['param']['width']
    f_time = model_config['param']['f_time']

    test_policy = model_config['policy']['test_policy']
    train_policy = model_config['policy']['train_policy']
    randomize = model_config['policy']['randomize']

    sink_num_iter = model_config['policy']['sinkhorn']['num_iter']
    sink_temp = model_config['policy']['sinkhorn']['temp']
    sink_eps = model_config['policy']['sinkhorn']['eps']
    sink_back_temp = model_config['policy']['sinkhorn']['back_temp']

    test_loss = []

    ## Train Loop
    
    for epoch in range(num_epochs):

        # Load checkpoint
        print(f'Testing: {checkpoint_dir}_{epoch}.pt')
        net = torch.load(f'{checkpoint_dir}_{epoch}.pt')
        
        if epoch % test_freq == 0:
            ## _________________________ Test _________________________
            dq = des.DiffDiscreteEventSystem(network, mu, h, 
                                        inter_arrival_dists, service_dists, 
                                        init_queues = init_test_queues, 
                                        queue_event_options= queue_event_options,
                                        straight_through_min = straight_through_min,
                                        batch = test_batch, 
                                        temp = env_temp, seed = test_seed,
                                        device = device)
            
            # Lexicographic ordering if server pools
            if num_pool > 1:
                lex = torch.tensor([0.1*i for i in range(num_pool)]).unsqueeze(1).repeat(1, orig_s)
                lex = lex.repeat(orig_s, 1)
                lex = lex.unsqueeze(0).repeat(dq.batch, 1, 1).to(device)
            else:
                lex = torch.zeros(dq.batch, dq.s, dq.q).to(device)
            
            with torch.no_grad():
                for _ in trange(test_T):
                    pr = net(dq.queues, dq.time_elapsed)
                    pr = pr.repeat_interleave(num_pool, dim = 1)

                    # test policy
                    if test_policy == 'sinkhorn':
                        v, s_bar, q_bar = rt.pad(2*pr + lex, dq.free_servers.data, dq.queues.data, network = dq.network, device = device)
                        pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                                sink_num_iter,
                                                sink_temp,
                                                sink_eps,
                                                sink_back_temp,
                                                device)[:,:dq.s,:dq.q]
                    elif test_policy == 'linear_assigment':
                        v, s_bar, q_bar = rt.pad(pr + lex, dq.free_servers.data, dq.queues.data, network = dq.network, device = 'cpu')
                        pr = rt.linear_assignment_batch(v, s_bar, q_bar)
                    elif test_policy == 'softmax':
                        pr = pr
                    else:
                        pass

                    # randomize policy or not
                    if randomize:
                        action = one_hot_sample.OneHotCategorical(probs = pr).sample()
                    else:
                        action = torch.round(pr)

                    queue, cost, terminated, info = dq.step(action)

            # Test cost metrics
            test_cost = torch.mean(dq.cost.unsqueeze(1) / dq.time_elapsed)
            test_loss.append({'epoch': epoch,
                            'test_loss': float(test_cost),
                            'test_loss_std': float(torch.std(dq.cost.unsqueeze(1) / dq.time_elapsed)) })
            
            print(f"queue lengths: \t{torch.mean(dq.time_weight_queue_len / dq.time_elapsed, dim = 0)}")
            print(f"final cost: \t{torch.mean(torch.matmul(dq.queues, dq.h))}")
            print(f"test cost: \t{test_cost}")

            if not model_config['env']['test_restart']:
                # for each epoch start where you left off
                init_test_queues = dq.queues.data

            if device == 'cpu':
                def policy_plot(obs):
                    pr = net(obs)

                    if test_policy == 'sinkhorn':
                        v, s_bar, q_bar = rt.pad(2*pr, torch.ones((1,dq.s)), obs, network = network.unsqueeze(0), device = device)
                        pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                                sink_num_iter,
                                                sink_temp,
                                                sink_eps,
                                                sink_back_temp,
                                                device)[:,:dq.s,:dq.q]
                    elif test_policy == 'linear_assigment':
                        v, s_bar, q_bar = rt.pad(pr + lex, dq.free_servers.data, dq.queues.data, network = network, device = 'cpu')
                        pr = rt.linear_assignment_batch(v, s_bar, q_bar)
                    elif test_policy == 'softmax':
                        pr = pr
                    else:
                        pass

                    return pr

                # Plot
                fig_dir = f'./plot/{name}/{model_name}/{model_name}'
                if model_config['plot']['plot_policy_curve']:
                    plot_policy_switching_curve(policy_plot, 
                                                fig_dir = f'{fig_dir}_{epoch}.png', 
                                                device = "cpu", 
                                                base_level = 5, 
                                                q = dq.q, 
                                                max_queue = 50, 
                                                inds = model_config['plot']['inds'],
                                                val_inds = model_config['plot']['val_inds'])

        # Test only
        
        with open(f'./loss/{name}_test.json', 'w') as f:
            json.dump(test_loss, f)

