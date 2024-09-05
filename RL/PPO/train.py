import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RL_root = os.path.join(project_root, 'RL')
sys.path.append(project_root)
sys.path.append(RL_root)


import numpy as np
import torch
from torch import nn
import json
import yaml
from utils.rl_env import load_rl_p_env
from policies.WC_policy import WC_Policy
from policies.vanilla_policy import Vanilla_Policy
from utils.eval import parallel_eval
from trainer import CustomPPOTrainer
from utils.rollout_buffer import CustomRolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    config_file_name = sys.argv[1]  
    env_config_name = sys.argv[2]

    if not config_file_name.endswith('.yaml'):
        config_file_name += '.yaml'

    config_file_path = os.path.join(RL_root, 'policy_configs', config_file_name)
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f'env_config: {env_config_name}')

    env_config_path = os.path.join(project_root, 'configs', 'env', f'{env_config_name}.yaml')
    with open(env_config_path, 'r') as f:
        env_config = yaml.safe_load(f)

    name = env_config['name']
    print(f'name: {name}')
    
    env_type = env_config.get('env_type', name)

    ## Environment Parameters
    # load network
    if env_config['network'] is None:
        network_path = os.path.join(project_root, 'configs', 'env_data', env_type, f'{env_type}_network.npy')
        network = np.load(network_path)
    else:
        network = env_config['network']

    print(f'network: {network}')
    # load mu
    if env_config['mu'] is None:
        mu_path = os.path.join(project_root, 'configs', 'env_data', env_type, f'{env_type}_mu.npy')
        mu = np.load(mu_path)
    else:
        mu = env_config['mu']

    network = torch.tensor(network).float()
    print(f'network: {network}')
    mu = torch.tensor(mu).float()
    print(f'mu: {mu}')

    orig_s, orig_q = network.size()

    # repeat if server pools
    num_pool = env_config['num_pool']
    network = network.repeat_interleave(num_pool, dim = 0)
    mu = mu.repeat_interleave(num_pool, dim = 0)

    init_test_queues = torch.tensor([env_config['init_queues']]).float()

    # env hyperparameters
    device = config['env']['device']
    # use cuda
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_seed = config['env']['test_seed']
    train_seed = config['env']['train_seed']
    print(f'device: {device}')
    print(f'test_seed: {test_seed}')
    print(f'train_seed: {train_seed}')
    env_temp = config['env']['env_temp']
    randomize = config['env']['randomize']
    time_f = config['env']['time_f']
    policy_name = config['model']['policy_name']
    print(f'policy_name: {policy_name}')

    # training hyperparameters
    actors = config['training']['actors']
    normalize_advantage = config['training']['normalize_advantage']
    normalize_value = config['training']['normalize_value']
    normalize_reward = config['training']['normalize_reward']
    rescale_v = config['training']['rescale_v']
    truncation = config['training']['truncation']
    num_epochs = config['training']['num_epochs']
    amp_value = config['training']['amp_value']
    var_scaler = config['training']['var_scaler']
    per_iter_normal_obs = config['training']['per_iter_normal_obs']
    per_iter_normal_value = config['training']['per_iter_normal_value']

    # learning rates:
    lr = config['training']['lr']
    lr_policy = config['training']['lr_policy']
    lr_value = config['training']['lr_value']
    min_lr_policy = config['training']['min_lr_policy']
    min_lr_value = config['training']['min_lr_value']


    episode_steps = config['training']['episode_steps']
    gae_lambda = config['training']['gae_lambda']
    gamma = config['training']['gamma']
    target_kl = config['training']['target_kl']
    vf_coef = config['training']['vf_coef']
    ppo_batch_size = config['training']['batch_size']
    ppo_epochs = config['training']['ppo_epochs']
    train_batch = config['training']['train_batch']
    test_batch = config['training']['test_batch']
    clip_range_vf = config['training']['clip_range_vf']
    ent_coef = config['training']['ent_coef']
    bc = config['training']['behavior_cloning']

    # model hyperparameters:
    scale = config['model']['scale']
    # policy hyperparameters
    test_policy = config['policy']['test_policy']
    # total steps
    total_steps = num_epochs * episode_steps * actors
    eval_freq = episode_steps
    # reward_scale = reward_scale / episode_steps
    test_T = env_config['test_T']
    print('env_config_test_T', test_T)

    ############# Main Training Code: #############

    # Create a function that returns a new instance of the environment
    
    def make_env():
        return load_rl_p_env(env_config = env_config,
                       temp = env_temp, 
                       batch = 1,
                       seed = train_seed,
                       policy_name = policy_name,
                       device = torch.device(device))

    def make_test_env(seed):
        return load_rl_p_env(env_config = env_config,
                       temp = env_temp, 
                       batch = 1,
                       seed = seed,
                       policy_name = policy_name,
                       device = torch.device(device))
    
    # Train Env
    dq_raw = load_rl_p_env(env_config = env_config,
                       temp = env_temp, 
                       batch = 1,
                       seed = train_seed,
                       policy_name = policy_name,
                       device = torch.device(device))
    

    ### parallel training ###

    env_fns = [make_env for _ in range(actors)]
    # dq = SubprocVecEnv(env_fns, start_method='fork')
    raw_envs = [make_test_env(seed) for seed in range(train_seed, train_seed + actors)]
    dq = DummyVecEnv(env_fns)


    # Test Env
    dq_test_list = [make_test_env(seed) for seed in range(test_seed, test_seed + 100)]

    # model kwargs
    L = orig_q
    J = orig_s
    gmLJ = int(np.sqrt(L * J))
    pi_arch = [scale * L, scale * gmLJ, scale * J]
    vi_arch = [scale * L, scale * gmLJ, scale * J]
    print(f'pi_arch: {pi_arch}')



    # leakyrelu activation
    policy_kwargs = dict(
                    activation_fn=nn.Tanh,
                    network = network,
                    time_f = time_f,
                    randomize = randomize,
                    scale = scale,
                    rescale_v = rescale_v,
                    alpha = 0,
                    D = dq_raw.queue_event_options,
                    mu = mu,
                    net_arch=dict(pi=pi_arch, 
                                   vf=vi_arch))
    
    #target_kl = None
    # define sb model
    if policy_name == 'WC':
        policy = WC_Policy
    elif policy_name == 'vanilla':
        policy = Vanilla_Policy

    rollout_buffer_kwargs = dict(
                            q = orig_q,
                            normalize_advantage = normalize_advantage,
                            normalize_value = normalize_value,
                            normalize_reward = normalize_reward,
                            truncation = truncation,
                            var_scaler = var_scaler,
                            per_iter_normal_value = per_iter_normal_value,
    )

    model = CustomPPOTrainer(policy, dq, learning_rate=lr, lr_policy=lr_policy, lr_value=lr_value, min_lr_policy=min_lr_policy, amp_value = amp_value, min_lr_value=min_lr_value,n_steps=episode_steps, batch_size=ppo_batch_size, num_epochs = num_epochs, n_epochs=ppo_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=0.2, clip_range_vf=clip_range_vf, normalize_advantage=normalize_advantage, raw_env = raw_envs, normalize_value = normalize_value, rescale_v = rescale_v, ent_coef=ent_coef, actors = actors, vf_coef=vf_coef, max_grad_norm=1.0, use_sde=False, sde_sample_freq=-1, rollout_buffer_class=CustomRolloutBuffer, rollout_buffer_kwargs=rollout_buffer_kwargs, target_kl=target_kl, stats_window_size=100, tensorboard_log=None, policy_kwargs=policy_kwargs, verbose=1, seed=None, device=device, _init_setup_model=True)
    # def eval call back

    eval_env = dq_test_list 
    eval_callback = parallel_eval(model = model, eval_env = eval_env, eval_freq = eval_freq, eval_t = test_T, test_policy = test_policy, test_seed = test_seed, init_test_queues = init_test_queues, test_batch = test_batch, device = device, num_pool = num_pool, time_f = time_f, randomize = randomize, policy_name = policy_name, per_iter_normal_obs = per_iter_normal_obs, env_config_name = env_config_name, bc = bc, verbose = 1)

    eval_callback.pre_train_eval()
                                     
    # Train model
    model.learn(total_timesteps=total_steps, log_interval=1, callback=eval_callback)

    test_cost_list = eval_callback.test_costs
    final_cost_list = eval_callback.final_costs

    # Store the lists
    with open('test_cost_list.json', 'w') as f:
        json.dump(test_cost_list, f)


if __name__ == '__main__':
    main()




