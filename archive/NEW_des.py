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
        self.softmax = nn.Softmax(dim = 0)
    
    def forward(self, x):
        return F.one_hot(torch.argmin(x), num_classes = x.size()[0]) - self.softmax(-x/self.temp).detach() + self.softmax(-x/self.temp)
    
def allocator(a, queue_service_times):
    
    num_q = a.size()[1]
    
    # identify non-zero actions
    nonzero_inds = a.detach().numpy().nonzero()
    nonzero_inds = np.transpose(nonzero_inds).tolist()
    
    # for each queue, identify servers with positive action
    queue_nonzero_inds = {i:[] for i in range(num_q)}
    for ind in nonzero_inds:
        if ind[1] in queue_nonzero_inds.keys():
            queue_nonzero_inds[ind[1]].append(ind)
            
    # build allocated action
    allocated_a = []
    num_allocated = []
    for q in range(num_q):
        q_allocated_a = []

        # sort the indices by work
        queue_nonzero_inds[q].sort(key = lambda x: float(a.detach()[x[0]][x[1]]), reverse = True)

        num_allocated_jobs = min(len(queue_service_times[q]), len(queue_nonzero_inds[q]))
        num_allocated.append(num_allocated_jobs)

        for j in range(num_allocated_jobs):
            # allocate job j in queue q to the jth non-empty server
            q_server = queue_nonzero_inds[q][j]
            q_allocated_a.append(a[q_server[0]][q_server[1]])
        
        # if num_allocated_jobs < len(queue_service_times[q]):
        #     num_unallocated_jobs = len(queue_service_times[q]) - num_allocated_jobs
        #     q_allocated_a = q_allocated_a + [torch.tensor(0) for _ in range(num_unallocated_jobs)]

        allocated_a.append(q_allocated_a)
        
    return allocated_a, queue_nonzero_inds, num_allocated
            
class DiffDiscreteEventSystem(gym.Env):
    def __init__(self, network, mu, h, inter_arrival_dists, service_dists, 
                 init_queues, init_time = 0, batch = 1, queue_event_options = None,
                 straight_through_min = False,
                 queue_lim = None, temp = 1, seed = 3003,
                 device = "cpu", f_hook = False, f_verbose = False, reset = False):
        

        self.device = device
        self.state = np.random.default_rng(seed)
        self.network = network.to(self.device)
        self.mu = mu.to(self.device)
        self.inter_arrival_dists = inter_arrival_dists
        self.service_dists = service_dists
        self.q = self.network.size()[1]
        self.s = self.network.size()[0]
        self.h = h.float().to(device)
        self.temp = temp
        self.st_argmin = STargmin(temp = self.temp)
        self.f_hook = f_hook
        self.f_verbose = f_verbose
        self.straight_through_min = straight_through_min
        
        if queue_event_options is None:
            self.queue_event_options = torch.cat((F.one_hot(torch.arange(0,self.q)), -F.one_hot(torch.arange(0,self.q)))).float().to(self.device)
        else:
            self.queue_event_options = queue_event_options.float().to(self.device)
        
        self.queues = init_queues.float().to(self.device)
        self.free_servers = torch.ones((self.s)).to(self.device)
        self.cost = torch.tensor([0]).to(self.device)

        if isinstance(init_time, torch.Tensor):
            self.time_elapsed = init_time.float().to(self.device)
        else:
            self.time_elapsed = torch.tensor([0.]).to(self.device)

        self.time_weight_queue_len = torch.zeros(self.queues.size()).to(self.device)
        self.queue_len_dist = {}
        self.marg_queue_len_dist = [{} for _ in range(self.q)]
        self.terminated = False

        # service times is a list of lists containing tensors
        self.service_times = [[self.draw_service() for _ in range(int(self.queues[q]))] for q in range(self.q)]
        self.arrival_times = self.draw_inter_arrivals()
        
    def draw_service(self):
        service = self.service_dists(self.state, self.time_elapsed).to(self.device)
        return service

    def draw_inter_arrivals(self):
        interarrivals = self.inter_arrival_dists(self.state, self.time_elapsed).to(self.device)
        return interarrivals
        
    def reset(self, init_queues, time, seed = None):
        #self.episode += 1
        self.cost = torch.tensor([0]).to(self.device)
        self.time_elapsed = torch.tensor([time]).to(self.device)
        self.queues = init_queues.float().to(self.device)
        
        if seed is not None:
            self.state = np.random.RandomState(seed)

        self.service_times = [[self.draw_service() for _ in range(int(self.queues[q]))] for q in range(self.q)]
        self.arrival_times = self.draw_inter_arrivals()

    def step(self, action):
        #print(self.queues)
        #work = torch.sum(action * self.mu * self.network, 1) * (self.queues > 0).to(self.device)
        
        # Compliance with network
        action = action * self.network

        # action is zero if queues are zero
        #if self.f_preemptive:
        action = torch.minimum(action, self.queues)
        
        # work is action times mu
        work = action * self.mu

        # allocate work to jobs
        #allocated_work = work
        allocated_work, queue_nonzero_inds, num_allocated = allocator(work, self.service_times)

        if self.f_verbose:
            print(f"action:\t{action}")
            print(f"allocated work:\t{allocated_work}")
            print(f"queue_nonzero_inds:\t{queue_nonzero_inds}")

            print()
        
        # effective service times are service_time divided by mu
        #eff_service_times = torch.stack([torch.min(torch.stack(self.service_times[q]) / torch.stack(allocated_work[q])) for q in range(self.q)])
        
        
        eff_service_times = [torch.tensor([torch.inf])]*self.q
        for q in range(self.q):
            if num_allocated[q] > 0:
                eff_service_times[q] = torch.stack(self.service_times[q][:num_allocated[q]]) / torch.clip(torch.stack(allocated_work[q]), min = 1e-1)
                #eff_service_times[q] = torch.stack(self.service_times[q])[:num_allocated[q]] / allocated_work[q]
        
        min_eff_service_times = torch.stack([torch.min(eff_service_times[q]) for q in range(self.q)])
        
        # arrival times and service times are both q vectors
        event_times = torch.cat((self.arrival_times, min_eff_service_times))

        if self.f_verbose:
            print(f"service:\t\t{self.service_times}")
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
                    which_job[q] = int(torch.argmin(torch.stack(self.service_times[q][:num_allocated[q]]).detach() / torch.stack(allocated_work[q])).detach())
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
        self.time_weight_queue_len = self.time_weight_queue_len + event_time * self.queues
        
        # update time elapsed, cost, queues
        self.time_elapsed = self.time_elapsed + event_time
        current_cost = torch.matmul(event_time * self.queues, self.h)
        self.cost = self.cost + current_cost

        # update queues
        self.queues = F.relu(self.queues + delta_q)
        
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
                self.service_times[q][:num_allocated[q]] = list(torch.unbind(torch.stack(self.service_times[q][:num_allocated[q]]) - event_time.detach()* torch.stack(allocated_work[q]).detach()))
                # self.service_times[q][:num_allocated[q]] = list(torch.unbind(torch.stack(self.service_times[q][:num_allocated[q]]) - event_time* torch.stack(allocated_work[q])))
        # update arrival times
        self.arrival_times = self.arrival_times - event_time

        if self.f_verbose:
            print(f"new service times:\t\t{self.service_times}")
            print(f"new arrival times:\t\t{self.arrival_times}")
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
                new_arrival_times = self.draw_inter_arrivals()
                new_service_time = self.draw_service()
                
                # new arrival counter
                self.arrival_times = self.arrival_times + torch.nan_to_num((new_arrival_times) * delta_arrived, nan = torch.inf)
                which_arrival = torch.argmax(delta_arrived)
                
                # service time of the new arrival
                self.service_times[which_arrival].append(new_service_time)

                if self.f_verbose:
                    print('Arrival!')
                    print(f"new service times:\t\t{self.service_times}")
                    print(f"new arrival times:\t\t{self.arrival_times}")
                    print()
            
            if torch.sum(delta_left) > 0:
                # remove a served job
                self.service_times[which_queue][which_job[which_queue]] = self.service_times[which_queue][which_job[which_queue]].detach()
                popped_job = self.service_times[which_queue].pop(which_job[which_queue])
                del popped_job

                if self.f_verbose:
                    print('Service!')
                    #print(f"popped job:\t\t{popped_job}")
                    print(f"service times:\t\t{self.service_times}")
                    print()

        
        return self.get_observation(), current_cost, self.terminated, {}
    
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
