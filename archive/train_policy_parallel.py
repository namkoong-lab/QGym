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
# import diff_discrete_event as des
import NEW_des_v2 as des 
import matplotlib.pyplot as plt
import routing as rt
import policies as pol
import json
import torch.distributions.one_hot_categorical as one_hot_sample
from multiprocessing import Pool

import yaml


import pdb


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

        x = F.relu(self.layers_fc[0](x))

        # Hidden layer
        for l in range(self.layers):
            x = F.relu(self.layers_fc[l](x))

        # Output layer
        x = self.output_fc(x)
        return F.softmax(torch.reshape(x, (batch, self.s , self.q)), dim = 2)

def testing(test_seed):
    dq = des.DiffDiscreteEventSystem(network, mu, h, 
                                        inter_arrival_dists, service_dists, 
                                        queue_event_options= queue_event_options,
                                        batch = test_batch, 
                                        temp = env_temp, seed = test_seed,
                                        device = torch.device(device))
    
    obs, state = dq.reset(seed = test_seed)
    
    total_cost = torch.tensor([[0.]]*test_batch).to(device)
    time_weight_queue_len = torch.tensor([[0.]]*test_batch).to(device)
    
    with torch.no_grad():
        for _ in trange(test_T):
            
            queues, time = obs


            pr = net(queues, time)
            pr = pr.repeat_interleave(num_pool, dim = 1)

            # test policy
            if test_policy == 'sinkhorn':
                v, s_bar, q_bar = rt.pad(2*pr + lex, queues.detach(), network = dq.network, device = device)
                pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                        sink_num_iter,
                                        sink_temp,
                                        sink_eps,
                                        sink_back_temp,
                                        device)[:,:dq.s,:dq.q]
            elif test_policy == 'linear_assigment':
                v, s_bar, q_bar = rt.pad(pr + lex, queues.detach(), network = dq.network, device = 'cpu')
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

            obs, state, cost, event_time = dq.step(state, action)
            total_cost = total_cost + cost
            time_weight_queue_len = time_weight_queue_len + queues * event_time

    # Test cost metrics
    test_cost = torch.mean(total_cost / state.time)

    return test_cost

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

    model_name = model_config['name']
    print(f'model: {model_name}')

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
    
    def service_dists(state, batch, t):
        return state.exponential(1, (batch, orig_q))
    
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
    # threshold = model_config['opt']['threshold']
    
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

    if checkpoint is None:
        net = PriorityNet(orig_s, orig_q, layers, width, f_time = f_time).to(device)
    else:
        net = torch.load(f'{checkpoint_dir}_{checkpoint}.pt')
    optimizer = optim.Adam(net.parameters(), lr = lr, betas = betas)

    test_loss = []

    if not os.path.exists(f'./loss'):
        os.makedirs(f'./loss')
    if not os.path.exists(f"./models/{name}"):
        os.makedirs(f"./models/{name}")
    if not os.path.exists(f"./models/{name}/{model_name}"):
        os.makedirs(f"./models/{name}/{model_name}")
    ## Train Loop
        
    for epoch in range(num_epochs):
        
        print(f"epoch: {epoch}")
        if True:#epoch % 50 == 0:
            ## _________________________ Test _________________________
            dq = des.DiffDiscreteEventSystem(network, mu, h, 
                                        inter_arrival_dists, service_dists, 
                                        queue_event_options= queue_event_options,
                                        batch = test_batch, 
                                        temp = env_temp, seed = test_seed,
                                        device = torch.device(device))
            
            # Lexicographic ordering if server pools
            if num_pool > 1:
                lex = torch.tensor([0.1*i for i in range(num_pool)]).unsqueeze(1).repeat(1, orig_s)
                lex = lex.repeat(orig_s, 1)
                lex = lex.unsqueeze(0).repeat(dq.batch, 1, 1).to(device)
            else:
                lex = torch.zeros(dq.batch, dq.s, dq.q).to(device)
            
            obs, state = dq.reset(seed = test_seed)
            
            total_cost = torch.tensor([[0.]]*test_batch).to(device)
            time_weight_queue_len = torch.tensor([[0.]]*test_batch).to(device)
            
            if epoch % 10000 == 0:
                cpu = 1
                with Pool(cpu) as p:
                    test_cost = p.map(testing, range(0, cpu))

                # Test cost metrics
                mean_test_cost = torch.mean(torch.tensor(test_cost))
                std_test_cost = torch.std(torch.tensor(test_cost))
                

                test_loss.append({'epoch': epoch,
                                'test_loss': float(mean_test_cost),
                                'test_loss_std': float(std_test_cost) })
                
                # print(f"queue lengths: \t{torch.mean(time_weight_queue_len / state.time, dim = 0)}")
                # print(f"final cost: \t{torch.mean(torch.matmul(queues, dq.h))}")
                print(f"test cost: \t{mean_test_cost}")
                print(f"test cost std: \t{std_test_cost}")

                # with open(f'./loss/{name}_{model_name}.json', 'w') as f:
                #     json.dump(test_loss, f)

            if not model_config['env']['test_restart']:
                # for each epoch start where you left off
                init_test_queues = queues.detach()

            if device == 'cpu':
                def policy_plot(obs):
                    pr = net(obs)

                    if test_policy == 'sinkhorn':
                        v, s_bar, q_bar = rt.pad(2*pr, obs, network = network.unsqueeze(0), device = device)
                        pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                                sink_num_iter,
                                                sink_temp,
                                                sink_eps,
                                                sink_back_temp,
                                                device)[:,:dq.s,:dq.q]
                    elif test_policy == 'linear_assigment':
                        v, s_bar, q_bar = rt.pad(pr + lex, dq.queues.detach(), network = network, device = 'cpu')
                        pr = rt.linear_assignment_batch(v, s_bar, q_bar)
                    elif test_policy == 'softmax':
                        pr = pr
                    else:
                        pass

                    return pr

                # Plot
                fig_dir = f'./plot/{name}/{model_name}/{model_name}'
                if not os.path.exists(f'./plot/{name}'):
                    os.makedirs(f'./plot/{name}')
                if not os.path.exists(f'./plot/{name}/{model_name}'):
                    os.makedirs(f'./plot/{name}/{model_name}')
                if model_config['plot']['plot_policy_curve']:
                    plot_policy_switching_curve(policy_plot, 
                                                fig_dir = f'{fig_dir}_{epoch}.png', 
                                                device = "cpu", 
                                                base_level = 5, 
                                                q = dq.q, 
                                                max_queue = 50, 
                                                inds = model_config['plot']['inds'],
                                                val_inds = model_config['plot']['val_inds'])

        ## _________________________ Train _________________________
        
        if train_seed is None:
            train_seed = int.from_bytes(os.urandom(4), 'big')

        # When training, 'straight_through_min = True'
        net = net.to(device)

        dq = des.DiffDiscreteEventSystem(network, mu, h, 
                                    inter_arrival_dists, service_dists, 
                                    queue_event_options = queue_event_options,
                                    batch = test_batch, 
                                    temp = env_temp, seed = train_seed,
                                    device = torch.device(device))

        # zero out the optimizer
        optimizer.zero_grad()

        # save grads
        back_outs = []
        def action_hook(grad):
            #grad = torch.clamp(grad, -threshold,threshold)
            back_outs.append(grad.tolist())
            #return grad
        
        nn_back_ins = []
        def priority_hook(grad):
            nn_back_ins.append(grad.tolist())

        # Train loop
        obs, state = dq.reset(seed = train_seed)
        total_cost = torch.tensor([[0.]]*train_batch).to(device)
        time_weight_queue_len = torch.tensor([[0.]]*train_batch).to(device)
        
        for _ in trange(train_T):
            queues, time = obs
            
            pr = net(queues, time)
            # print(f"step {_}")
            # pdb.set_trace()
            pr.register_hook(priority_hook)
            
            # server pools
            pr = pr.repeat_interleave(num_pool, dim = 1)
            
            if train_policy == 'sinkhorn':
                v, s_bar, q_bar = rt.pad(2*pr + lex, queues.detach(), network = dq.network, device = device)
                pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                        sink_num_iter,
                                        sink_temp,
                                        sink_eps,
                                        sink_back_temp,
                                        device)[:,:dq.s,:dq.q]
            elif train_policy == 'softmax':
                pr = pr
            else:
                pass

            action = pr
            action.register_hook(action_hook)
            
            obs, state, cost, event_time = dq.step(state, action)
            
            total_cost = total_cost + cost
            
            time_weight_queue_len = time_weight_queue_len + queues * event_time
        
        # Backward
        loss = torch.mean(total_cost / train_T)
        loss.backward()

        print(f"train cost:\t{torch.mean(total_cost / state.time)}")
        print(f"queue lengths: \t{torch.mean(time_weight_queue_len / state.time, dim = 0)}")
        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = model_config['opt']['grad_clip_norm'])
        optimizer.step()

        if not model_config['env']['train_restart']:
            init_train_queues = queues.detach()

        if model_config['env']['print_grads']:
            print("Action Grads")
            print(torch.mean(torch.sum(torch.tensor(back_outs),0),0))

            print("Priority Grads")
            print(torch.mean(torch.sum(torch.tensor(nn_back_ins),0),0))

        # Save checkpoint
        torch.save(net, checkpoint_dir + f'_{epoch}.pt')
        
        
        

