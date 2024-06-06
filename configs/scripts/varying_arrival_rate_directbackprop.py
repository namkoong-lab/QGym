import sys
sys.path.append('/user/hc3295/queue-learning')

from policies.direct_backprop import *
from utils.switchplot import *


import yaml
import argparse
from utils.switchplot import *
from main.trainer import Trainer

import torch
import torch.optim as optim

import json

import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', type=str)
parser.add_argument('-m', type=str)
parser.add_argument('-experiment_name', type=str)




args = parser.parse_args()

with open(f'../configs/env/{args.e}', 'r') as f:
    env_config = yaml.safe_load(f)

with open(f'../configs/model/{args.m}', 'r') as f:
    model_config = yaml.safe_load(f)

experiment_name = args.experiment_name

if model_config['env']['model_seed'] is not None:
        torch.manual_seed(model_config['env']['model_seed'])

name = env_config['name']
if env_config['network'] is None:
    env_config['network'] = np.load(f'../configs/env_data/{name}/{name}_network.npy')
env_config['network'] = torch.tensor(env_config['network']).float()


if env_config['mu'] is None:
    env_config['mu'] = np.load(f'../configs/env_data/{name}/{name}_mu.npy')
env_config['mu'] = torch.tensor(env_config['mu']).float()

orig_s, orig_q = env_config['network'].size()

if model_config['checkpoint'] is None:
    net = PriorityNet(orig_s, orig_q, model_config['param']['layers'], model_config['param']['width'], f_time = model_config['param']['f_time']).to(model_config['env']['device'])
else:
    net = torch.load(f"{model_config['checkpoint_dir']}_{model_config['checkpoint']}.pt")
policy = DirectBackpropPolicy(net)

env_config['network'] = env_config['network'].repeat_interleave(1, dim = 0)
env_config['mu'] = env_config['mu'].repeat_interleave(1, dim = 0)
env_config['server_pool_size'] = torch.tensor(env_config['server_pool_size']).to(model_config['env']['device'])



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
    




def draw_service(self, time):
    def service_dists(state, batch, t):
        return state.exponential(1, (batch, orig_q))
    service = torch.tensor(service_dists(self.state, self.batch, time)).to(self.device)
    return service

rho = 1.0
exp_arrival_1 = lambda state, t: state.exponential(1/(2.4*rho)) if t <= 100 else state.exponential(1/(0.4*rho))
exp_arrival_2 = lambda state, t: state.exponential(1/(0.6*rho)) if t <= 100 else state.exponential(1/(0.8*rho))
exp_arrival_3 = lambda state, t: state.exponential(1/(0.8*rho))
exp_arrival_4 = lambda state, t: state.exponential(1/(1.6*rho)) if t <= 100 else state.exponential(1/(0.8*rho))
exp_arrival_5 = lambda state, t: state.exponential(1/(0.6*rho))

def draw_inter_arrivals(self, time):

    interarrivals = torch.tensor([[exp_arrival_1(self.state, time), exp_arrival_2(self.state, time), exp_arrival_3(self.state, time), exp_arrival_4(self.state, time), exp_arrival_5(self.state, time)] for _ in range(self.batch)]).to(self.device)
    
    return interarrivals


optimizer = optim.Adam(net.parameters(), lr = model_config['opt']['lr'], betas = model_config['opt']['betas'])

trainer = Trainer(model_config, env_config, policy, optimizer, experiment_name = experiment_name, draw_service = draw_service, draw_inter_arrivals = draw_inter_arrivals)




for epoch in range(model_config['opt']['num_epochs']):
    if epoch % 25 == 0:
        trainer.test_epoch(epoch)

    plot_policy_switching_curve(policy_plot,
                                net = policy.network, 
                                model_config = model_config,
                                env_config = env_config,
                                fig_dir = f'{trainer.fig_dir}_{epoch}.png', 
                                device = "cpu", 
                                base_level = 5, 
                                q = env_config['network'].shape[-1],
                                max_queue = 50, 
                                inds = model_config['plot']['inds'],
                                val_inds = model_config['plot']['val_inds'])


    if model_config['env']['train_seed'] is None:
        model_config['env']['train_seed'] = int.from_bytes(os.urandom(4), 'big')


    policy.network = policy.network.to(model_config['env']['device'])

    trainer.train_epoch()

    with open(f'{trainer.loss_dir}/loss.json', 'w') as f:
        json.dump(trainer.test_loss, f)





