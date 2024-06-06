import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import torch.optim as optim
# from gym import spaces
import gym
import sys
#sys.path.append('../Gymnasium')
#from gymnasium import spaces
#import gymnasium as gym
import itertools

class STargmin(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x):
        return F.one_hot(torch.argmin(x, dim = 1), num_classes = x.size()[1]) - self.softmax(-x/self.temp).detach() + self.softmax(-x/self.temp)
            
class DiffDiscreteEventSystem(gym.Env):
    def __init__(self, network, mu, h, inter_arrival_dists, service_dists, 
                 init_queues, init_time = 0, batch = 1, queue_event_options = None,
                 straight_through_min = False,
                 queue_lim = None, temp = 1, seed = 3003,
                 device = "cpu", f_hook = False, f_verbose = False):
        
        self.device = device
        self.network = network.repeat(batch,1,1).to(self.device)
        self.mu = mu.repeat(batch,1,1).to(self.device)
        self.inter_arrival_dists = inter_arrival_dists
        self.service_dists = service_dists
        self.q = self.network.size()[2]
        self.s = self.network.size()[1]
        self.h = h.float().to(device)
        self.temp = temp
        self.st_argmin = STargmin(temp = self.temp)
        self.f_hook = f_hook
        self.f_verbose = f_verbose
        self.straight_through_min = straight_through_min
        
        #if self.f_preemptive:
        self.prev_action = torch.zeros(self.mu.size()).to(self.device)
        
        if queue_event_options is None:
            self.queue_event_options = torch.cat((F.one_hot(torch.arange(0,self.q)), -F.one_hot(torch.arange(0,self.q)))).float().to(self.device)
        else:
            self.queue_event_options = queue_event_options.float().to(self.device)

        self.temp = temp
        self.eps = 1e-8
        self.inv_eps = 1/self.eps
        self.batch = batch
        self.state = np.random.default_rng(seed)
        
        #self.service_times = self.draw_service() + torch.where(self.queues == 0.0, torch.inf, 0.0) * torch.ones(self.q)
        #self.queues = init_queues.float().unsqueeze(0).repeat(batch, 1).to(self.device)

        if init_queues.size()[0] == 1:
            self.queues = init_queues.float().repeat(self.batch, 1).to(self.device)
        else:
            self.queues = init_queues.float().to(self.device)

        #self.service_times = self.draw_service() + (self.inv_eps * (self.queues == 0.0)).to(self.device)

        
        self.free_servers = torch.ones((self.batch, self.s)).to(self.device)
        self.avail_jobs = self.queues
        

        self.cost = torch.tensor([0]).repeat(self.batch).to(self.device)

        if isinstance(init_time, torch.Tensor):
            self.time_elapsed = init_time.float().to(self.device)
        else:
            self.time_elapsed = torch.tensor([0.]).repeat(self.batch).unsqueeze(1).to(self.device)

        self.time_weight_queue_len = torch.zeros(self.queues.size()).to(self.device)
        self.queue_len_dist = {}
        self.marg_queue_len_dist = [{} for _ in range(self.q)]
        self.terminated = False

        self.service_times = self.inv_eps * torch.ones(self.draw_service().size()).to(self.device)
        self.arrival_times = self.draw_inter_arrivals()

        self.work = torch.zeros(self.service_times.size()).to(self.device)
        
        
    # def draw_service(self):
    #     service_mat = torch.tensor([[[service(self.state, self.time_elapsed[b]) 
    #                                   for service in self.service_dists] 
    #                                  for s in range(self.s)] 
    #                                 for b in range(self.batch)]).float().to(self.device)
    #     return service_mat
    
    # def draw_inter_arrivals(self):
    #     return torch.tensor([[inter_arrival(self.state, self.time_elapsed[b]) for inter_arrival in self.inter_arrival_dists]for b in range(self.batch)]).float().to(self.device)

    def draw_service(self):
        service_mat = torch.tensor(self.service_dists(self.state, self.batch, self.s, self.q, self.time_elapsed)).float().to(self.device)
        return service_mat

    def draw_inter_arrivals(self):
        interarrivals = torch.tensor(self.inter_arrival_dists(self.state, self.batch, self.time_elapsed)).float().to(self.device)
        return interarrivals
        
    def reset(self, init_queues, time, seed = None):
        #self.episode += 1
        self.cost = torch.tensor([0]).repeat(self.batch).to(self.device)
        self.time_elapsed = torch.tensor([time]).repeat(self.batch).unsqueeze(1).to(self.device)

        if init_queues.size()[0] == 1:
            self.queues = init_queues.float().repeat(self.batch, 1).to(self.device)
        else:
            self.queues = init_queues.float().to(self.device)
        
        if seed is not None:
            self.state = np.random.RandomState(seed)

        self.service_times = self.inv_eps * torch.ones(self.draw_service().size()).to(self.device)
        self.arrival_times = self.draw_inter_arrivals()

        self.work = torch.zeros(self.service_times.size()).to(self.device)

        #if self.f_preemptive:
        self.prev_action = torch.zeros(self.mu.size()).to(self.device)


        

    def step(self, action):
        #print(self.queues)
        #work = torch.sum(action * self.mu * self.network, 1) * (self.queues > 0).to(self.device)
        # B x s x q
        new_service_times = self.draw_service()
        
        # Compliance with network
        action = action * self.network

        # action is zero if queues are zero
        #if self.f_preemptive:
        action = torch.minimum(action, self.queues.unsqueeze(1).repeat(1,self.s,1))
        # else:
        #     self.avail_jobs = self.queues.unsqueeze(1).repeat(1,self.s,1)
        #     avail_jobs_action = torch.ones(self.service_times.size()).to(self.device) * self.avail_jobs.detach()
        #     action = torch.minimum(action, avail_jobs_action)

        #     free_servers_action = self.free_servers.detach().reshape(self.batch, self.s, 1).to(self.device) * torch.ones(self.service_times.size()).to(self.device)
        #     action = torch.minimum(action, free_servers_action)
            
        #     if self.f_verbose:
        #         print(f"zero_servers:\t{free_servers_action}")            
        #         print(f"zero_queues:\t{avail_jobs_action}")
        #         print()
        
        if self.f_verbose:
            print(f"masked action:\t{action}")
            print()
        
        # non-preemptive, we keep track of who has been worked on
        # new service times are drawn for newly assigned servers
        
        self.work = action
        
        # NO_GRAD: mask new service times except for newly assigned servers
        newly_served = 1*(self.service_times > 10000.).to(self.device)
        new_service_times = new_service_times / torch.clamp(newly_served.detach().float(), min = self.eps)
        
        if self.f_verbose:
            print(f"newly_served:\t\t{newly_served}")

        # NO_GRAD: mask service times for servers who were reassigned away
        # newly_diverted = 1.*(self.prev_action > 0.) * (action == 0.)
        # zero_newly_diverted = torch.ones(action.size()).float().to(self.device) - newly_diverted
        #self.service_times = self.service_times / torch.clamp(zero_to_reset_newly_left.detach().float(), min = self.eps)
        #self.service_times = self.service_times - zero_to_reset_newly_left * self.service_times.detach() + zero_to_reset_newly_left * self.service_times.detach()
        #self.service_times = torch.minimum(self.service_times/torch.clamp(zero_newly_diverted.detach(), min = self.eps), self.service_times / torch.clamp(newly_diverted.detach(), min = self.eps))

        if self.f_verbose:
            #print(f"newly_left:\t\t{zero_to_reset_newly_left}")
            print(f"work:\t\t{self.work}")
            print()
            
        
        if self.f_hook:
            self.work.register_hook(lambda grad: print(f"work_grad: {grad}"))
        
        # by taking min with new_service_times, we update service times for jobs
        self.service_times = torch.minimum(self.service_times, new_service_times)
        
        # effective service times are service_time divided by mu
        eff_service_times = torch.minimum(self.service_times / torch.clamp(self.work * self.mu, min = self.eps), self.inv_eps * torch.ones(self.service_times.size()).to(self.device))
        
        if self.f_verbose:
            print(f"new serv:\t\t{new_service_times}")
            print(f"service:\t\t{self.service_times}")
            print(f"eff service:\t\t{eff_service_times}")
            print()
        
        # for each queue, get next service time (B x q)
        min_eff_service_times = torch.min(eff_service_times, dim = 1).values
        
        if self.f_hook:
            eff_service_times.register_hook(lambda grad: print(f"eff_service_times: {grad}"))
            min_eff_service_times.register_hook(lambda grad: print(f"min_eff_service_grad: {grad}"))
        
        # arrival times and service times are both q vectors
        event_times = torch.cat((self.arrival_times, min_eff_service_times), dim = 1).float()
        
        # outcome is one_hot argmin of the event times
        outcome = F.softmax(-event_times/self.temp, dim = 1)
        
        # if a server served, which server? NO BACKWARD
        # B x s x q
        which_server = torch.transpose(F.one_hot(torch.argmin(eff_service_times, dim = 1), num_classes = self.s).float().to(self.device), 1, 2)
        
        # Determine event
        # B x q
        delta_q = torch.matmul(outcome, self.queue_event_options)

        if not self.straight_through_min:
            event_time = torch.min(event_times, dim = 1).values[:,None]
        else:
            event_time = torch.sum(event_times * outcome, dim = 1)[:,None]
        
        if self.f_verbose:
            print(f"outcome:\t\t{outcome}")
            print(f"which_server:\t{which_server}")
            print(f"delta_q:\t\t{delta_q}")
        #vent_time = torch.sum(event_times * outcome, dim = 1)
        
        if self.f_hook:
            event_times.register_hook(lambda grad: print(f"event_times: {grad}"))
            outcome.register_hook(lambda grad: print(f"outcome_grad: {grad}"))
            event_time.register_hook(lambda grad: print(f"outcome_grad: {grad}"))
        
        # update joint state dist: state is concatenated string
        if self.batch == 1:
            state_record = self.queues.data[0].numpy().astype("int")
            joint_state_key = tuple(state_record)
            if joint_state_key in self.queue_len_dist.keys():
                self.queue_len_dist[joint_state_key] += float(event_time.data.numpy())
            else:
                self.queue_len_dist[joint_state_key] = float(event_time.data.numpy())
            
            # update marginal state dist:
            for qu, qu_len in enumerate(state_record):
                if qu_len in self.marg_queue_len_dist[qu].keys():
                    self.marg_queue_len_dist[qu][int(qu_len)] += float(event_time.data.numpy())
                else:
                    self.marg_queue_len_dist[qu][int(qu_len)] = float(event_time.data.numpy())
        
        # update time elapsed, cost, queues
        self.time_elapsed = self.time_elapsed + event_time
        current_cost = torch.matmul(event_time * self.queues, self.h)
        self.cost = self.cost + current_cost

        self.time_weight_queue_len = self.time_weight_queue_len + event_time * self.queues
        
        self.queues = F.relu(self.queues + delta_q)
        
        if self.f_verbose:
            print(f"event_time:\t\t{event_time}")
            print(f"eff_elapsed_time:{self.work * self.mu * event_time.unsqueeze(1)}")
            print()
        
        self.service_times = self.service_times - self.work * self.mu * event_time.unsqueeze(1)
        self.arrival_times = self.arrival_times - event_time

        # If arrival occurred, insert new interarrivals
        new_arrival_times = self.draw_inter_arrivals()
        self.arrival_times = self.arrival_times + (new_arrival_times) * outcome[:,:self.q].detach()
        #self.arrival_times = self.arrival_times + (new_arrival_times) * outcome[:,:self.q]
        
        # If a service occurred, which server and what job were they serving
        # B x s: 1 if server was served
        with torch.no_grad():
            #freed_server = torch.matmul(outcome[:,self.q:].detach(), which_server).squeeze(1)
            freed_server = torch.sum(which_server * outcome[:,self.q:].unsqueeze(1).detach(), 2).to(self.device)
            
            # B x s x q: -1 if service happened for (s,q)
            # only the negative part
            removed_jobs = -F.relu(-delta_q).detach()
            freed_server_queue = torch.einsum('bi,bj->bij', (freed_server, removed_jobs))
            # If a queue drops to zero, release all work
            zero_out_just_emptied_queues = (removed_jobs* 1*(self.queues == 0.)).unsqueeze(1).to(self.device)
            service_happened_pos_queue = freed_server_queue*(1*(self.queues > 0.).unsqueeze(1)).to(self.device)
            
            # release
            freed_server_queue = zero_out_just_emptied_queues + service_happened_pos_queue
            
            if self.f_verbose:
                print(f"freed_server:\t{freed_server}")
                print(f"free_s_q:\t{freed_server_queue}")
            
            self.prev_action = action.detach() + freed_server_queue * action.detach()
            
            # Reset service times for freed servers
            freed_servers_inv_eps = (self.inv_eps) * -freed_server_queue
            self.service_times = torch.maximum(self.service_times, freed_servers_inv_eps)
        
        return self.get_observation(), current_cost, self.terminated, {}
    
    def get_observation(self):
        return self.queues
        
    def print_state(self):
        print(f"Total Cost:\t{self.cost}")
        print(f"Time Elapsed:\t{self.time_elapsed}")
        print(f"Queue Len:\t{self.queues}")
        print(f"Free Serv:\t{self.free_servers}")
        print(f"Avail Jobs:\t{self.avail_jobs}")
        print(f"Service times:\t{self.service_times}")
        print(f"Arrival times:\t{self.arrival_times}")
        #if self.f_preemptive:
        print(f"Prev Action:\t{self.prev_action}")
        # else:
        #     print(f"Work:\t{self.work}")