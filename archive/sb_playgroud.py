import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import math
import torch.optim as optim
import orig_diff_des as des

from tqdm import trange
import routing as rt
import torch.distributions.one_hot_categorical as one_hot_sample

rho = 1
service_rate = 1
exp_arrival_1 = lambda state, t: state.exponential(1/0.9)
exp_arrival_2 = lambda state, t: 100000
exp_arrival_3 = lambda state, t: state.exponential(1/0.9)

exp_service = lambda state, t: state.exponential(1/service_rate)

inter_arrival_dists = [exp_arrival_1, exp_arrival_2, exp_arrival_3]
service_dists = [exp_service, exp_service, exp_service]

network = torch.tensor([[1,0,1],[0,1,0]]).float()

queue_event_options = torch.tensor([[1., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 1.],
                                    [-1., 1., 0.],
                                    [0., -1., 0.],
                                    [0., 0., -1.]]).float()

mu_diag = 1
mu_scaling = torch.tensor([[2, 0, 2], [0, 1, 0]]).float()

mu = network * mu_scaling
h = torch.tensor([1., 1., 1.])

init_queues = torch.tensor([[0.]*3]).float()
T = 1000
batch = 1

dq = des.PreemptiveDiffQueueSB(network, mu, h, inter_arrival_dists, service_dists, 
        init_queues = init_queues, queue_event_options = queue_event_options, 
                                batch = batch, temp = 1/2, seed = 42,f_hook = False, f_verbose = False)




import sys
sys.path.append('../stable-baselines3')
sys.path.append('../Gymnasium')

from stable_baselines3 import A2C, DQN


from stable_baselines3.common.callbacks import BaseCallback

class PrintQueuesCallback(BaseCallback):
    """
    A custom callback that prints the 'self.queues' attribute.
    """

    def __init__(self, verbose=0):
        super(PrintQueuesCallback, self).__init__(verbose)
        self.history = {"costs": [],
                        "actions": [],
                        "action_space_mats": dq.action_space_mats,
                        "eposide": [],
                        "queues": []}

    def _on_step(self) -> bool:
        """
        This method will be called for each environment step.
        """
        print(self.model.env.get_attr('episode')[0], self.model.env.get_attr('queues'), self.model.env.actions, self.model.env.get_attr('current_cost'))
        self.history["costs"].append(self.model.env.get_attr('current_cost')[0])
        self.history["actions"].append(self.model.env.actions[0])
        self.history["queues"].append(self.model.env.get_attr('queues')[0])
        self.history["eposide"].append(self.model.env.get_attr('episode')[0])

        return True



callback = PrintQueuesCallback()

model = A2C("MlpPolicy", dq, verbose=0)
model.learn(total_timesteps=200000, log_interval=1, callback = callback)

model.save("a2c")

import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(callback.history, f)