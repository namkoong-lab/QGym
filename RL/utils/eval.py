import sys
sys.path.append('.../')
from stable_baselines3 import PPO
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange
from typing import NamedTuple
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.data import Dataset, DataLoader



class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor

class EnvState(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor
    service_times: torch.Tensor
    arrival_times: torch.Tensor


class BCD(Dataset):
    def __init__(self, num_samples, network):
        self.num_samples = num_samples
        self.network = network
        self.s = self.network.shape[0]
        self.q = self.network.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input data
        input_data = np.random.randint(0, 101, self.q)
        obs = torch.tensor(input_data)
        action_probs = F.softmax(torch.tensor(input_data).float(), dim=-1)
        action_probs = action_probs * self.network
        action_probs = torch.minimum(action_probs, obs.unsqueeze(0).repeat(1, self.s, 1))
        zero_mask = torch.all(action_probs == 0, dim=2).reshape(-1, self.s, 1).repeat(1, 1, self.q)
        action_probs = action_probs + zero_mask * self.network
        action_probs = action_probs / torch.sum(action_probs, dim=-1).reshape(-1, self.s, 1)
        output_data = action_probs
        input_tensor = torch.tensor(input_data, dtype=torch.float32).squeeze()
        output_tensor = torch.tensor(output_data, dtype=torch.float32).squeeze()
        
        return input_tensor, output_tensor


class parallel_eval(BaseCallback):
    def __init__(self, model, eval_env, eval_freq, eval_t, test_policy, test_seed, init_test_queues, test_batch, device, num_pool, time_f, policy_name, per_iter_normal_obs, env_config_name, bc, randomize = True, 
                 verbose=1):
        super(parallel_eval, self).__init__(verbose)
        self.model = model
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_t = eval_t
        print('eval_t', eval_t)
        self.test_policy = test_policy
        self.test_seed = test_seed
        self.init_test_queues = init_test_queues
        self.test_batch = test_batch
        self.device = device
        self.num_pool = num_pool
        self.randomize = randomize
        self.time_f = time_f
        self.policy_name = policy_name
        self.test_costs = []  
        self.final_costs = []
        self.per_iter_normal_obs = per_iter_normal_obs
        self.env_config_name = env_config_name
        self.bc = bc
        print(f'eval env config name: {self.env_config_name}')
        self.iter = 0

        self.lex_batch = []
        self.obs_batch = []
        self.state_batch = []
        self.total_cost_batch = []
        self.time_weight_queue_len_batch = []




    
    def behavior_cloning(self):

        print(f'---------------------behavior_cloning---------------------')
        if hasattr(self.model.policy, "log_std"):
            self.optimizer_policy = torch.optim.Adam([
                {'params': self.model.policy.log_std},
                {'params': self.model.policy.features_extractor.parameters()},
                {'params': self.model.policy.pi_features_extractor.parameters()},
                {'params': self.model.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.model.policy.action_net.parameters()}
            ], lr=3e-4)

        else:
            self.model.optimizer_policy = torch.optim.Adam([
                {'params': self.model.policy.features_extractor.parameters()},
                {'params': self.model.policy.pi_features_extractor.parameters()},
                {'params': self.model.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.model.policy.action_net.parameters()}
            ], lr=3e-4)

        # print(f'network: {self.eval_env[0].network}, shape: {self.eval_env[0].network[0].shape}')
        BCD_dataset = BCD(num_samples = 100000, network = self.eval_env[0].network[0])
        BCD_loader = DataLoader(BCD_dataset, batch_size = self.test_batch, shuffle = True)

        for i, (obs, target) in enumerate(BCD_loader):
            self.optimizer_policy.zero_grad()
            action, action_probs = self.model.policy.get_prob_act(obs)
            loss = F.mse_loss(action_probs, target)
            loss.backward()
            self.optimizer_policy.step()

    def pre_train_eval(self):
        print('pre_train_eval')
        if self.bc:
            self.behavior_cloning()
        # self.behavior_cloning()
        q_mean, q_std, t_mean, t_max, t_min, t_std = self.eval()

        self.model.policy.update_mean_std(mean_queue_length = q_mean, std_queue_length = q_std)
        print(f"mean_queue_length: {self.model.policy.mean_queue_length}")
        print(f"std_queue_length: {self.model.policy.std_queue_length}")

        return True

    def _on_step(self) -> bool:
        if self.per_iter_normal_obs:
            if (self.n_calls) % self.eval_freq == 0:
                q_mean, q_std, t_mean, t_max, t_min, t_std = self.eval()

                self.model.policy.update_mean_std(mean_queue_length = q_mean, std_queue_length = q_std)
                print(f"mean_queue_length: {self.model.policy.mean_queue_length}")
                print(f"std_queue_length: {self.model.policy.std_queue_length}")
        else:
            if (self.n_calls) % self.eval_freq == 0:
                self.eval()
                print(f"mean_queue_length: {self.model.policy.mean_queue_length}")
                print(f"std_queue_length: {self.model.policy.std_queue_length}")

        return True
    

    def eval(self):
        self.iter += 1
        print(f'iter: {self.iter}')

        lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch = self.construct_batch()
        test_dq_batch = self.eval_env

        with torch.no_grad():
            for tt in trange(self.eval_t):
                # print(tt)
                # print(f'---------------------')
                # print(f'obs:')
                # for obs in obs_batch:
                #     print(f'{obs}')                
                batch_queue = torch.cat([obs[0] for obs in obs_batch], dim = 0).reshape(self.test_batch,-1)
                # print(f'batch_queue: {batch_queue}')

                raw_actions, probs = self.model.predict(batch_queue)
                action = torch.tensor(raw_actions).float().to(self.device)
                
                for test_dq_idx in range(len(test_dq_batch)):
                    # step_time_start = time.time()
                    _, _, _, _, info = test_dq_batch[test_dq_idx].step(action[test_dq_idx])
                    # step_time_end = time.time()
                    # print(f'step time: {step_time_end - step_time_start}')
                    obs_batch[test_dq_idx], state_batch[test_dq_idx], cost, event_time  = info['obs'], info['state'], info['cost'], info['event_time']
                    total_cost_batch[test_dq_idx] = total_cost_batch[test_dq_idx] + cost
                    time_weight_queue_len_batch[test_dq_idx] = time_weight_queue_len_batch[test_dq_idx] + info['queues'] * info['event_time']

        # Test cost metrics
        # pdb.set_trace()
        test_cost_batch = [total_cost_batch[test_dq_idx] / state_batch[test_dq_idx].time for test_dq_idx in range(len(test_dq_batch))]
        test_cost = torch.mean(torch.concat(test_cost_batch))
        test_std = torch.std(torch.concat(test_cost_batch))
        test_queue_len = torch.mean(torch.concat([time_weight_queue_len_batch[test_dq_idx] / state_batch[test_dq_idx].time for test_dq_idx in range(len(test_dq_batch))]), dim = 0)
        test_queue_len = [float(_item) for _item in test_queue_len.to('cpu').detach().numpy().tolist()]
        
        print(f"queue lengths: \t{test_queue_len}")
        print(f"test cost: \t{test_cost}")
        print(f"test cost std: \t{test_std}")


        test_queue_len = torch.tensor(test_queue_len)

        q_mean = torch.mean(test_queue_len)
        q_std = torch.std(test_queue_len)
        t_mean = torch.mean(state_batch[0].time)
        t_max = torch.max(state_batch[0].time)
        t_min = torch.min(state_batch[0].time)
        t_std = torch.std(state_batch[0].time)
        
        return q_mean, q_std, t_mean, t_max, t_min, t_std

    def construct_batch(self):
        lex_batch = []
        obs_batch = []
        state_batch = []
        total_cost_batch = []
        time_weight_queue_len_batch = []


        for dq_idx in range(self.test_batch):

            dq = self.eval_env[dq_idx]
            lex = torch.zeros(dq.batch, dq.s, dq.q)
            obs, state = dq.reset(seed = dq.seed)
            obs = torch.tensor(obs).to(self.device)
            total_cost = torch.tensor([[0.]])
            time_weight_queue_len = torch.tensor([[0.]])

            lex_batch.append(lex)
            obs_batch.append(obs)
            state_batch.append(state)
            total_cost_batch.append(total_cost)
            time_weight_queue_len_batch.append(time_weight_queue_len)

        
        return lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch
