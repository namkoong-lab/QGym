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
from typing import NamedTuple
import pdb



class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor

class EnvState(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor
    service_times: torch.Tensor
    arrival_times: torch.Tensor


class STargmin(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.softmax = nn.Softmax(dim = 0)
    
    def forward(self, x):
        return F.one_hot(torch.argmin(x), num_classes = x.size()[1]) - self.softmax(-x/self.temp).detach() + self.softmax(-x/self.temp)
    
def allocator(action, mu, queue_service_times):
    
    adj_const = action.clone().detach()
    adj_const[adj_const < 1] = 1


    mu_with_grad = mu * action#/ adj_const
    a = mu * action
    # print(a)
    num_q = a.size()[-1]
    
    # identify non-zero actions
    nonzero_inds = a[0].detach().numpy().nonzero()
    nonzero_inds = np.transpose(nonzero_inds).tolist()
    
    # for each queue, identify servers with positive action
    queue_nonzero_inds = {i:[] for i in range(num_q)}
    for ind in nonzero_inds:
        if ind[1] in queue_nonzero_inds.keys():
            # for _server in range(int(torch.round(action[0][ind[0]][ind[1]]).item())):
            queue_nonzero_inds[ind[1]].append(ind)
    
    
    # build allocated action
    allocated_a = []
    num_allocated = []
    for q in range(num_q):
        q_allocated_a = []

        # sort the indices by work
        queue_nonzero_inds[q].sort(key = lambda x: float(a.detach()[0][x[0]][x[1]]), reverse = True)

        num_allocated_jobs = min(len(queue_service_times[q]), len(queue_nonzero_inds[q]))
        num_allocated.append(num_allocated_jobs)

        for j in range(num_allocated_jobs):
            # allocate job j in queue q to the jth non-empty server
            q_server = queue_nonzero_inds[q][j]
            q_allocated_a.append(a[0][q_server[0]][q_server[1]])
        
        # if num_allocated_jobs < len(queue_service_times[q]):
        #     num_unallocated_jobs = len(queue_service_times[q]) - num_allocated_jobs
        #     q_allocated_a = q_allocated_a + [torch.tensor(0) for _ in range(num_unallocated_jobs)]

        allocated_a.append(q_allocated_a)
    
    # print(queue_nonzero_inds)
    return allocated_a, queue_nonzero_inds, num_allocated
            
class DiffDiscreteEventSystem(gym.Env):
    def __init__(self, network, mu, h, inter_arrival_dists, service_dists, init_time = 0, batch = 1, queue_event_options = None,
                 straight_through_min = False,
                 queue_lim = None, temp = 1, seed = 3003,
                 device = "cpu", f_hook = False, f_verbose = False, reset = False):
        

        self.device = device
        self.state = np.random.default_rng(seed)
        self.network = network.repeat(batch,1,1).to(self.device)
        self.mu = mu.repeat(batch,1,1).to(self.device)
        self.inter_arrival_dists = inter_arrival_dists
        self.service_dists = service_dists
        self.q = self.network.size()[-1]
        self.s = self.network.size()[-2]
        self.h = h.float().to(device)
        self.temp = temp
        self.st_argmin = STargmin(temp = self.temp)
        self.f_hook = f_hook
        self.f_verbose = f_verbose
        self.straight_through_min = straight_through_min
        self.batch = batch

        self.eps = 1e-8
        self.inv_eps = 1/self.eps
        
        if queue_event_options is None:
            self.queue_event_options = torch.cat((F.one_hot(torch.arange(0,self.q)), -F.one_hot(torch.arange(0,self.q)))).float().to(self.device)
        else:
            self.queue_event_options = queue_event_options.float().to(self.device)
        
        # self.queues = init_queues.float().to(self.device)
        self.free_servers = torch.ones((self.batch, self.s)).to(self.device)
        self.cost = torch.tensor([0]).to(self.device)

        if isinstance(init_time, torch.Tensor):
            self.time_elapsed = init_time.float().to(self.device)
        else:
            self.time_elapsed = torch.tensor([0.]).to(self.device)

        self.time_weight_queue_len = torch.zeros(self.network.size()[-1]).to(self.device)
        self.queue_len_dist = {}
        self.marg_queue_len_dist = [{} for _ in range(self.q)]
        self.terminated = False

        # service times is a list of lists containing tensors
        # self.service_times = [[self.draw_service() for _ in range(int(self.queues[q]))] for q in range(self.q)]
        # self.arrival_times = self.draw_inter_arrivals()
        
    def draw_service(self, time):
        service = torch.tensor(self.service_dists(self.state, self.batch, time)).to(self.device)
        return service

    def draw_inter_arrivals(self, time):
        interarrivals = torch.tensor(self.inter_arrival_dists(self.state, self.batch, time)).to(self.device)
        return interarrivals
        
    def reset(self, init_queues=None, time=None, seed = None):
        #self.episode += 1
        cost = torch.tensor([0]).to(self.device)
        if time is None:
            time = torch.tensor([[0.]]).repeat(self.batch, 1).to(self.device)
        else:
            time = time.repeat(self.batch).unsqueeze(1).to(self.device)

        if init_queues is None:
            queues = torch.tensor([[0.]*self.q]).repeat(self.batch, 1).to(self.device)
        elif init_queues.size()[0] == 1:
            queues = init_queues.float().repeat(self.batch, 1).to(self.device)
        else:
            queues = init_queues.float().to(self.device)
        
        if seed is not None:
            self.state = np.random.RandomState(seed)

        service_times = [[self.draw_service(time) for _ in range(int(queues_sample[q]))] for q in range(self.q) for queues_sample in queues]
        arrival_times = self.draw_inter_arrivals(time)
        
        


        return Obs(queues, time), EnvState(queues, time, service_times, arrival_times)

    def step(self, state, action):
        
        
        #work = torch.sum(action * self.mu * self.network, 1) * (self.queues > 0).to(self.device)
        
        # Compliance with network
        
        queues, time, service_times, arrival_times = state        
        
        action = action * self.network
        

        # action is zero if queues are zero
        #if self.f_preemptive:
        action = torch.minimum(action, queues)

        # work is action times mu

        # allocate work to jobs
        #allocated_work = work
        allocated_work, queue_nonzero_inds, num_allocated = allocator(action, self.mu, service_times)
        # print(allocated_work)


        if self.f_verbose:
            print(f"action:\t{action}")
            print(f"allocated work:\t{allocated_work}")
            print(f"queue_nonzero_inds:\t{queue_nonzero_inds}")

        # print(queues)
        # print(action)
        
        # effective service times are service_time divided by mu
        #eff_service_times = torch.stack([torch.min(torch.stack(self.service_times[q]) / torch.stack(allocated_work[q])) for q in range(self.q)])
        
        
        eff_service_times = [torch.tensor([self.inv_eps])]*self.q
        for q in range(self.q):
            if num_allocated[q] > 0:
                eff_service_times[q] = torch.stack(service_times[q][:num_allocated[q]])[:,0,q] / torch.clip(torch.stack(allocated_work[q]), min = self.eps)
                #eff_service_times[q] = torch.stack(self.service_times[q])[:num_allocated[q]] / allocated_work[q]
        
        min_eff_service_times = torch.stack([torch.min(eff_service_times[q]) for q in range(self.q)])
        min_eff_service_times = min_eff_service_times.unsqueeze(0)
        
        
        # arrival times and service times are both q vectors
        event_times = torch.cat((arrival_times, min_eff_service_times), dim=1).float()

        if self.f_verbose:
            print(f"service:\t\t{service_times}")
            print(f"eff service:\t\t{eff_service_times}")
            print(f"eff service:\t\t{min_eff_service_times}")
            print(f"event times:\t\t{event_times}")
            print()

        # if a job was served, which job in which queue
        if True:
        # with torch.no_grad():
            which_job = [0]*self.q
            for q in range(self.q):
                if num_allocated[q] > 0:
                    which_job[q] = int(torch.argmin(torch.stack(service_times[q][:num_allocated[q]]).detach()[:,0,q] / torch.stack(allocated_work[q])).detach())
                    #which_job[q] = int(torch.argmin(torch.stack(self.service_times[q])[:num_allocated[q]].detach().squeeze(1) / allocated_work[q]).detach())

            which_queue = int(torch.argmin(min_eff_service_times).detach())
            
            if self.f_verbose:
                print(f"which_queue:\t\t{which_queue}")
                print(f"which_job:\t\t{which_job}")
                #print(f"which_arrival:\t\t{which_arrival}")
                print()
        
        
        # outcome is one_hot argmin of the event times
        outcome = self.st_argmin(event_times)
        
        
        # update state based on event time
        delta_q = torch.matmul(outcome, self.queue_event_options)
        
        # compute min event
        if not self.straight_through_min:
            event_time = torch.min(event_times)
        else:
            event_time = torch.sum(event_times * outcome)

        # if self.f_verbose:
        #     print(f"outcome:\t\t{outcome}")
        #     print(f"delta_q:\t\t{delta_q}")
        
        # if self.f_hook:
        #     if outcome.requires_grad:
        #         event_times.register_hook(lambda grad: print(f"event_times: {grad}"))
        #         outcome.register_hook(lambda grad: print(f"outcome_grad: {grad}"))
        #         event_time.register_hook(lambda grad: print(f"event time grad: {grad}"))
        #         delta_q.register_hook(lambda grad: print(f"delta grad: {grad}"))
        
        # # update joint state dist: state is concatenated string
        # with torch.no_grad():
        #     state_record = self.queues.data.numpy().astype("int")
        #     joint_state_key = tuple(state_record)
        #     if joint_state_key in self.queue_len_dist.keys():
        #         self.queue_len_dist[joint_state_key] += float(event_time.data.numpy())
        #     else:
        #         self.queue_len_dist[joint_state_key] = float(event_time.data.numpy())
            
        #     # update marginal state dist:
        #     for qu, qu_len in enumerate(state_record):
        #         if qu_len in self.marg_queue_len_dist[qu].keys():
        #             self.marg_queue_len_dist[qu][int(qu_len)] += float(event_time.data.numpy())
        #         else:
        #             self.marg_queue_len_dist[qu][int(qu_len)] = float(event_time.data.numpy())

        # time weighted queue length
        self.time_weight_queue_len = self.time_weight_queue_len + event_time * queues
        
        # update time elapsed, cost, queues
        time = time + event_time
        cost = torch.matmul(event_time * queues, self.h)
        
        
        queues = F.relu(queues + delta_q)
        # pdb.set_trace()

        
        
        if self.f_verbose:
            print(f"event_time:\t\t{event_time}")
            #print(f"eff_elapsed_time:{allocated_work * event_time}")
            print()
        
        # update service times for all jobs with positive work
        # for q in range(self.q):
        #     for j in range(num_allocated[q]):
        #         self.service_times[q][j] = F.relu(self.service_times[q][j] - allocated_work[q][j] * event_time)



        for q in range(self.q):
            if num_allocated[q] > 0:
                service_times[q][:num_allocated[q]] = list(torch.unbind(torch.stack(service_times[q][:num_allocated[q]]) - event_time * torch.stack(allocated_work[q]).unsqueeze(1).unsqueeze(-1).repeat((1,1,self.network.shape[-1]))))
                # service_times[q][:num_allocated[q]] = list(torch.unbind(torch.stack(service_times[q][:num_allocated[q]]) - event_time.detach() * torch.stack(allocated_work[q]).detach().unsqueeze(1).unsqueeze(-1).repeat((1,1,self.network.shape[-1]))))
        # update arrival times
        
        arrival_times = arrival_times - event_time

        if self.f_verbose:
            print(f"new service times:\t\t{service_times}")
            print(f"new arrival times:\t\t{arrival_times}")
            print()

        # Reset timers and add service
        # with torch.no_grad():
        if True:

            delta = delta_q.data.int()
            delta_arrived = torch.where(delta == 1, 1, 0)
            delta_left = torch.where(delta == -1, 1, 0)

            # if a new job arrives
            if torch.sum(delta_arrived) > 0:
                # arrival occurs
                new_arrival_times = self.draw_inter_arrivals(time)
                new_service_time = self.draw_service(time)                
                
                # new arrival counter
                if torch.sum(delta_arrived) == 1:
                    arrival_times = arrival_times + torch.nan_to_num((new_arrival_times) * delta_arrived, nan = self.inv_eps)

                which_arrival = torch.argmax(delta_arrived)
                
                # service time of the new arrival
                service_times[which_arrival].append(new_service_time)

                if self.f_verbose:
                    print('Arrival!')
                    print(f"new service times:\t\t{service_times}")
                    print(f"new arrival times:\t\t{arrival_times}")
                    print()
            
            if torch.sum(delta_left) > 0:
                # remove a served job
                service_times[which_queue][which_job[which_queue]] = service_times[which_queue][which_job[which_queue]].detach()
                popped_job = service_times[which_queue].pop(which_job[which_queue])
                del popped_job

                if self.f_verbose:
                    print('Service!')
                    #print(f"popped job:\t\t{popped_job}")
                    print(f"service times:\t\t{service_times}")
                    print()

        
        next_state = EnvState(queues, time, service_times, arrival_times)
        obs = Obs(queues, time)
        return obs, next_state, cost, event_time
    
    def get_observation(self):
        return self.queues
        
    def print_state(self):
        print(f"Total Cost:\t{self.cost}")
        print(f"Time Elapsed:\t{self.time_elapsed}")
        print(f"Queue Len:\t{self.queues}")
        print(f"Service times:\t{self.service_times}")
        print(f"Arrival times:\t{self.arrival_times}")
        # else:
        #     print(f"Work:\t{self.work}")
