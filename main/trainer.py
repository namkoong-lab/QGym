import sys
sys.path.append('/user/hc3295/queue-learning')

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
import matplotlib.pyplot as plt
import utils.routing as rt
import json
from main.env import DiffDiscreteEventSystem

import yaml
import pdb
from datetime import datetime


from utils.switchplot import *

class Trainer:
    def __init__(self, model_config, env_config, policy, optimizer, draw_service, draw_inter_arrivals, experiment_name):
        self.model_config = model_config
        self.env_config = env_config
        self.policy = policy
        self.optimizer = optimizer
        self.draw_service = draw_service
        self.draw_inter_arrivals = draw_inter_arrivals

        self.test_loss = []


        self.fig_dir = create_plot_dir(self.model_config, self.env_config, experiment_name = experiment_name)
        self.loss_dir = create_loss_dir(self.model_config, self.env_config, experiment_name = experiment_name)

        self.experiment_name = experiment_name



    def train_epoch(self):
        dq = DiffDiscreteEventSystem(self.env_config['network'], self.env_config['mu'], self.env_config['h'], 
                                    queue_event_options = self.env_config['queue_event_options'],
                                    batch = 1, 
                                    temp = self.model_config['env']['env_temp'], seed = self.model_config['env']['train_seed'],
                                    device = torch.device(self.model_config['env']['device']), draw_service = self.draw_service, draw_inter_arrivals = self.draw_inter_arrivals)

        # zero out the optimizer
        self.optimizer.zero_grad()

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
        obs, state = dq.reset(seed = self.model_config['env']['train_seed'])
        total_cost = torch.tensor([[0.]]*self.model_config['opt']['train_batch']).to(self.model_config['env']['device'])
        time_weight_queue_len = torch.tensor([[0.]]*self.model_config['opt']['train_batch']).to(self.model_config['env']['device'])
        
        for _ in trange(self.env_config['train_T']):
            queues, time = dq.obs.queues, dq.obs.time
            
            pr = self.policy.train_forward(queues, time, dq.network, dq.h.unsqueeze(0).repeat(1,dq.s,1), dq.mu)
            
            pr.register_hook(priority_hook)
            
            # server pools
            pr = pr.repeat_interleave(1, dim = 1)
            
            if self.model_config['policy']['train_policy'] == 'sinkhorn':
                lex = torch.zeros(dq.batch, dq.s, dq.q).to(self.model_config['env']['device'])
                v, s_bar, q_bar = rt.pad_pool(2*pr + lex, queues.detach(), network = dq.network, device = self.model_config['env']['device'], server_pool_size = self.env_config['server_pool_size'])

                pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                        self.model_config['policy']['sinkhorn']['num_iter'],
                                        self.model_config['policy']['sinkhorn']['temp'],
                                        self.model_config['policy']['sinkhorn']['eps'],
                                        self.model_config['policy']['sinkhorn']['back_temp'],
                                        self.model_config['env']['device'])[:,:dq.s,:dq.q]
                
                
            elif self.model_config['policy']['train_policy'] == 'softmax':
                
                pr = F.softmax(pr) * self.env_config['network'].unsqueeze(0)
                pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1,self.env_config['network'].shape[-2],1)).clip(min = 1e-4)
                pr /= torch.sum(pr, dim = -1).unsqueeze(-1)
            else:
                pass
            
            # print(pr)
            action = pr
            # print(action)
            # print(queues)
            action.register_hook(action_hook)
            
            queues, reward, done, truncated, info = dq.step(action)
            # print(f"info: {info['cost']}", total_cost)
            total_cost = total_cost + info['cost']
            
            
            time_weight_queue_len = time_weight_queue_len + info['queues'] * info['event_time']
        
        # Backward
        
        loss = torch.mean(total_cost / self.env_config['train_T'])
        loss.backward()


        state = dq.obs
        # print(f"total_cost:\t{total_cost}, time:{state.time}")
        print(f"train cost:\t{torch.mean(total_cost / state.time)}")
        print(f"queue lengths: \t{torch.mean(time_weight_queue_len / state.time, dim = 0)}")
        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(), max_norm = self.model_config['opt']['grad_clip_norm'])
        self.optimizer.step()

        if not self.model_config['env']['train_restart']:
            init_train_queues = queues.detach()

        if self.model_config['env']['print_grads']:
            print("Action Grads")
            print(torch.mean(torch.sum(torch.tensor(back_outs),0),0))

            print("Priority Grads")
            print(torch.mean(torch.sum(torch.tensor(nn_back_ins),0),0))

    def test_epoch(self, epoch):

        test_dq_batch, lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch = self.create_batch_dq(self.model_config['opt']['test_batch'], self.model_config['env']['test_seed'])
        
        with torch.no_grad():
            pbar = trange(self.env_config['test_T'], desc=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')} - {self.experiment_name}")
            for step in pbar:
                
                current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                description = f"{current_time} - {self.experiment_name}"
                # Update the description in each iteration
                pbar.set_description(description)
            

                batch_queue = torch.cat([obs[0] for obs in obs_batch], dim = 0)

                batch_time = torch.cat([obs[1] for obs in obs_batch], dim = 0)
                repeated_queue = batch_queue.unsqueeze(1).repeat(1,self.env_config['network'].shape[-2],1)
                repeated_network = test_dq_batch[-1].network.repeat(self.model_config['opt']['test_batch'], 1, 1)
                repeated_mu = self.env_config['mu'].repeat(self.model_config['opt']['test_batch'], 1, 1)
                repeated_h = torch.tensor(self.env_config['h']).repeat(self.model_config['opt']['test_batch'],self.env_config['network'].shape[-2],1)

                pr = self.policy.test_forward(step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h)

                pr = pr.repeat_interleave(1, dim = 1)
                # test policy
                if self.model_config['policy']['test_policy'] == 'sinkhorn':
                    lex = torch.zeros(test_dq_batch[-1].batch, test_dq_batch[-1].s, test_dq_batch[-1].q).to(self.model_config['env']['device'])
                    v, s_bar, q_bar = rt.pad_pool(2*pr + lex, batch_queue.detach(), network = test_dq_batch[-1].network.repeat(self.model_config['opt']['test_batch'], 1, 1), device = self.model_config['env']['device'], server_pool_size = self.env_config['server_pool_size'])

                    pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                            self.model_config['policy']['sinkhorn']['num_iter'],
                                            self.model_config['policy']['sinkhorn']['temp'],
                                            self.model_config['policy']['sinkhorn']['eps'],
                                            self.model_config['policy']['sinkhorn']['back_temp'],
                                            self.model_config['env']['device'])[:,:test_dq_batch[-1].s,:test_dq_batch[-1].q]
                elif self.model_config['policy']['test_policy'] == 'linear_assigment':
                    lex = torch.zeros(test_dq_batch[-1].batch, test_dq_batch[-1].s, test_dq_batch[-1].q).to(self.model_config['env']['device'])
                    v, s_bar, q_bar = rt.pad_pool(2*pr + lex, batch_queue.detach(), network = test_dq_batch[-1].network.repeat(self.model_config['opt']['test_batch'], 1, 1), device = self.model_config['env']['device'], server_pool_size = self.env_config['server_pool_size'])
                    pr = rt.linear_assignment_batch(v, s_bar, q_bar)
                    
                    
                elif self.model_config['policy']['test_policy'] == 'softmax':
                    pr = F.softmax(pr) * repeated_network
                    pr = torch.minimum(pr, repeated_queue).clip(min = 1e-4)
                    pr /= torch.sum(pr, dim = -1).unsqueeze(-1)
                else:
                    pass
                # print(pr[0])
                action = torch.round(pr)

                # print(action[0])
                # print(batch_queue[0], total_cost_batch[0]/ state_batch[0].time)
                # print(batch_queue)
                
                for test_dq_idx in range(len(test_dq_batch)):
                    # print(action)
                    _, _, _, _, info = test_dq_batch[test_dq_idx].step(action[test_dq_idx])
                    obs_batch[test_dq_idx], state_batch[test_dq_idx], cost, event_time  = info['obs'], info['state'], info['cost'], info['event_time']
                    total_cost_batch[test_dq_idx] = total_cost_batch[test_dq_idx] + cost
                    time_weight_queue_len_batch[test_dq_idx] = time_weight_queue_len_batch[test_dq_idx] + info['queues'] * info['event_time']
                # print(obs_batch[0])
        # Test cost metrics
        # pdb.set_trace()
        test_cost_batch = [total_cost_batch[test_dq_idx] / state_batch[test_dq_idx].time for test_dq_idx in range(len(test_dq_batch))]
        test_cost = torch.mean(torch.concat(test_cost_batch))
        test_std = torch.std(torch.concat(test_cost_batch))
        test_queue_len = torch.mean(torch.concat([time_weight_queue_len_batch[test_dq_idx] / state_batch[test_dq_idx].time for test_dq_idx in range(len(test_dq_batch))]), dim = 0)
        test_queue_len = [float(_item) for _item in test_queue_len.to('cpu').detach().numpy().tolist()]
        self.test_loss.append({'epoch': epoch,
                        'test_loss': float(test_cost),
                        'test_loss_std': float(test_std),
                        'mean_queue_length': test_queue_len})
        
        print(f"queue lengths: \t{test_queue_len}")
        print(f"test cost: \t{test_cost}")

    def create_batch_dq(self, bs, seed):
        dq_batch = []
        lex_batch = []
        obs_batch = []
        state_batch = []
        total_cost_batch = []
        time_weight_queue_len_batch = []
        for dq_idx in range(bs):

            dq = DiffDiscreteEventSystem(self.env_config['network'], self.env_config['mu'], self.env_config['h'], 
                                        queue_event_options= self.env_config['queue_event_options'],
                                        batch = 1, 
                                        temp = self.model_config['env']['env_temp'], seed = seed + dq_idx,
                                        device = torch.device(self.model_config['env']['device']), draw_service = self.draw_service, draw_inter_arrivals = self.draw_inter_arrivals)


            lex = torch.zeros(dq.batch, dq.s, dq.q).to(self.model_config['env']['device'])
        
            obs, state = dq.reset(seed = seed)
            
            total_cost = torch.tensor([[0.]]).to(self.model_config['env']['device'])
            time_weight_queue_len = torch.tensor([[0.]]).to(self.model_config['env']['device'])

            lex_batch.append(lex)
            obs_batch.append(obs)
            state_batch.append(state)
            total_cost_batch.append(total_cost)
            time_weight_queue_len_batch.append(time_weight_queue_len)

            dq_batch.append(dq) 

        return dq_batch, lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch


