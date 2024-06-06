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
import matplotlib.pyplot as plt
import routing as rt
import json

def backpressure(queues, queue_event_options,
               network, mu, h):
    
    batch = network.size()[0]
    q = network.size()[2]

    A = queue_event_options[q:]

    # mu * torch.sum(-A.unsqueeze(0) * Q(x, i), 2)
    pr = mu * torch.sum(-A.unsqueeze(0) * queues * h.unsqueeze(0), 2)
    v, s_bar, q_bar = rt.pad(pr, queues, network = network)

    action = rt.linear_assignment_batch(v, s_bar, q_bar)
    return action

def max_weight(queues,
               network, mu, h):
    
    batch = network.size()[0]
    pr = h.unsqueeze(0).repeat(batch, 1, 1) * queues.unsqueeze(1) * mu
    v, s_bar, q_bar = rt.pad(pr, queues, network = network)

    action = rt.linear_assignment_batch(v, s_bar, q_bar)
    return action

def c_mu(queues,
         network, mu, h):

    batch = network.size()[0]
    pr = h.unsqueeze(0).repeat(batch, 1, 1) * mu
    v, s_bar, q_bar = rt.pad(pr, queues, network = network)

    action = rt.linear_assignment_batch(v, s_bar, q_bar)
    return action