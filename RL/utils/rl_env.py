import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
print(project_root)
import numpy as np
import torch
from gym import spaces
from main.env import DiffDiscreteEventSystem
from typing import NamedTuple


class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor

class EnvState(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor
    service_times: torch.Tensor
    arrival_times: torch.Tensor

class RL_Wrapper_P_DiffDiscreteEventSystem(DiffDiscreteEventSystem):
    """
    This subclass of GymDiffDiscreteEventSystem is compatible with Stable Baselines 3.
    """
    def __init__(self, 
                 network:torch.Tensor, 
                 mu:torch.Tensor, 
                 h:torch.Tensor,
                 draw_service,
                 draw_inter_arrivals,
                 init_time,
                 queue_event_options = None,
                 straight_through_min = False,
                 batch:int = 1, 
                 temp:float = 1,
                 seed:int = 3003,
                 device:torch.device = torch.device('cpu'), 
                 f_hook:bool = False, 
                 f_verbose:bool = False,
                 time_f = False,
                 reward_scale = 1.0,
                 policy_name = 'WC',
                 action_map = None,
                 ):
        
        self.time_f = time_f
        self.policy_name = policy_name
        self.action_map = action_map
        self.seed = seed
        
        super().__init__(network = network,
                            mu = mu,
                            h = h,
                            draw_service= draw_service,
                            draw_inter_arrivals = draw_inter_arrivals,
                            init_time = init_time,
                            queue_event_options = queue_event_options,
                            straight_through_min = straight_through_min,
                            batch = batch,
                            temp = temp,
                            seed = seed,
                            device = device,
                            f_hook = f_hook,
                            f_verbose = f_verbose,
                            use_sb = True
                            )


        process_id = os.getpid()
        parent_process_id = os.getppid()
        print(f"Environment initialized in process: {process_id}, parent process: {parent_process_id}")

        if self.policy_name == 'WC' or self.policy_name == 'vanilla':
            self.action_space = spaces.Box(low=0, high=1, shape=(self.s, self.q), dtype=np.float32)
        elif self.policy_name == 'discrete':
            self.action_space = spaces.Discrete(len(self.action_map))
            # print all actions in action map:
            # self.action_space = spaces.MultiDiscrete([self.q] * self.s)
            # action_space = spaces.Box(low=0, high=1, shape=(self.s, self.q), dtype=np.float32)
            #self.action_space = spaces.Box(low=0, high=1, shape=(self.s, self.q), dtype=np.float32)

        if time_f:
            self.observation_space = spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(self.q + 1,),  # Add 1 for the time
                dtype=np.float32
            )
        
        else:
            self.observation_space = spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(self.q,),  
                dtype=np.float32
            )

        self.obs = None
        self.reward_scale = reward_scale

    def reset(self, init_queues=None, time=None, seed = None, options:dict = None):
        obs, even_state = super().reset(init_queues, time, seed = self.seed, options = None)
        queues = obs.queues
        return queues.cpu().numpy(), {}
    
    def reset_env_seed(self):
        seed = self.seed
        # print(f"seed: {seed}")  
        if seed is not None:
            self.state = np.random.RandomState(seed)


def load_rl_p_env(env_config, temp, batch, seed, policy_name, device):

    name = env_config['name']

    if 'env_type' in env_config:
        env_type = env_config['env_type']
    else:
        env_type = name

    if env_config['network'] is None:
        network_path = os.path.join(project_root, 'configs', 'env_data', env_type, f'{env_type}_network.npy')
        env_config['network'] = np.load(network_path)

    env_config['network'] = torch.tensor(env_config['network']).float()


    if env_config['mu'] is None:
        mu_path = os.path.join(project_root, 'configs', 'env_data', env_type, f'{env_type}_mu.npy')
        env_config['mu'] = np.load(mu_path)
    env_config['mu'] = torch.tensor(env_config['mu']).float()

    orig_s, orig_q = env_config['network'].size()


    network = env_config['network'].repeat_interleave(1, dim = 0)
    mu = env_config['mu'].repeat_interleave(1, dim = 0)
    # if 'server_pool_size' in env_config.keys():
    #     env_config['server_pool_size'] = torch.tensor(env_config['server_pool_size']).to(model_config['env']['device'])
    # else:
    #     env_config['server_pool_size'] = torch.ones(orig_s).to(model_config['env']['device'])

    queue_event_options = env_config['queue_event_options']
    if queue_event_options is not None:
        if queue_event_options == 'custom':
            queue_event_options_path = os.path.join(project_root, 'configs', 'env_data', env_type, f'{env_type}_delta.npy')
            queue_event_options = torch.tensor(np.load(queue_event_options_path))
        else:
            queue_event_options = torch.tensor(queue_event_options)



    lam_type = env_config['lam_type']
    lam_params = env_config['lam_params']
    h = torch.tensor(env_config['h'])

    if lam_params['val'] is None:
        lam_r_path = os.path.join(project_root, 'configs', 'env_data', env_type, f'{env_type}_lam.npy')
        lam_r = np.load(lam_r_path)
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
        

    if env_config['queue_event_options'] == 'custom':
        queue_event_options_path = os.path.join(project_root, 'configs', 'env_data', env_type, f'{env_type}_delta.npy')     
        env_config['queue_event_options'] = torch.tensor(np.load(queue_event_options_path))


    def draw_inter_arrivals(self, time):

        def inter_arrival_dists(state, batch, t):
            exps = state.exponential(1, (batch, orig_q))
            lam_rate = lam(t)
            return exps / lam_rate

        interarrivals = torch.tensor(inter_arrival_dists(self.state, self.batch, time)).to(self.device)
        return interarrivals
    
    def draw_service(self, time):
        def service_dists(state, batch, t):
            return state.exponential(1, (batch, orig_q))
        service = torch.tensor(service_dists(self.state, self.batch, time)).to(self.device)
        return service

    # def draw_service(self, time):
    #     def service_dists(state, batch, t):
    #         return state.exponential(1, (batch, orig_q))
    #     service = torch.tensor(service_dists(self.state, self.batch, time)).to(self.device)
    #     return service

    # rho = 1.0
    # exp_arrival_1 = lambda state, t: state.exponential(1/(2.4*rho)) if t <= 100 else state.exponential(1/(0.4*rho))
    # exp_arrival_2 = lambda state, t: state.exponential(1/(0.6*rho)) if t <= 100 else state.exponential(1/(0.8*rho))
    # exp_arrival_3 = lambda state, t: state.exponential(1/(0.8*rho))
    # exp_arrival_4 = lambda state, t: state.exponential(1/(1.6*rho)) if t <= 100 else state.exponential(1/(0.8*rho))
    # exp_arrival_5 = lambda state, t: state.exponential(1/(0.6*rho))

    # def draw_inter_arrivals(self, time):

    #     interarrivals = torch.tensor([[exp_arrival_1(self.state, time), exp_arrival_2(self.state, time), exp_arrival_3(self.state, time), exp_arrival_4(self.state, time), exp_arrival_5(self.state, time)] for _ in range(self.batch)]).to(self.device)
        
    #     return interarrivals

    dq = RL_Wrapper_P_DiffDiscreteEventSystem(network, mu, h, 
                                       draw_service= draw_service, draw_inter_arrivals = draw_inter_arrivals, init_time = 0, 
                                    queue_event_options= queue_event_options,
                                    batch = batch, 
                                    temp = temp, seed = seed,
                                    time_f = False,
                                    reward_scale = 1.0,
                                    policy_name= policy_name,
                                    action_map = None,
                                    device = torch.device(device))

    return dq