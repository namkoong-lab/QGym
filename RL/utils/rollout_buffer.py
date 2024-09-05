from stable_baselines3.common.buffers import RolloutBuffer
import numpy as np
import torch as th
from gymnasium import spaces


class CustomRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        self.q = kwargs.pop('q', None)
        self.normalize_advantage = kwargs.pop('normalize_advantage', False)
        self.normalize_value = kwargs.pop('normalize_value', False)
        self.normalize_reward = kwargs.pop('normalize_reward', False)
        self.truncation = kwargs.pop('truncation', False)
        self.var_scaler = kwargs.pop('var_scaler', 3.0)
        self.per_iter_normal_value = kwargs.pop('per_iter_normal_value', False)
        super().__init__(*args, **kwargs)
        self.iterations = 0
        self.rewards_mean = 0
        self.rewards_std = 1
        self.adv_std = 1
        self.returns_mean = 0
        self.returns_std = 1

        print("init custom rollout buffer...")
        print(f'buffer size: {self.buffer_size}')
        print(f'input q: {self.q}')
        print(f'normalize_advantage: {self.normalize_advantage}')
        print(f'normalize_value: {self.normalize_value}')
        print(f'normalize_reward: {self.normalize_reward}')
        print(f'truncation: {self.truncation}')

    def reset(self) -> None:
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        #super().add(obs, action, reward, episode_start, value, log_prob)
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        reward = reward.squeeze()
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).reshape(-1, self.q)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        self.iterations += 1
        print(f'compute_returns_and_advantage iteration: {self.iterations}')

        # normalize the reward values:
        rewards_mean = self.rewards.mean()
        rewards_std = self.rewards.std()

        if self.normalize_reward:
            if self.per_iter_normal_value:
                self.rewards_mean = rewards_mean
            else:
                if self.iterations == 1:
                    self.rewards_mean = rewards_mean

        ### normalize the rewards
        self.rewards = self.rewards - self.rewards_mean

        if self.truncation:
            print('truncating...')
            truncation_steps = np.full((self.n_envs, self.buffer_size), -1)
            for env_idx in range(self.n_envs):
                for step in range(self.buffer_size):
                    if np.all(self.observations[step, env_idx] == 0):
                        truncation_steps[env_idx, step] = 1

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # print(f'next_non_terminal: {dones}')
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                # replace the next_values with expected values 
                next_values = self.values[step + 1]
                   

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

            if self.truncation:
                for env_idx in range(self.n_envs):
                    if truncation_steps[env_idx, step] == 1:
                        last_gae_lam[env_idx] = 0
                

        self.returns = self.advantages + self.values

    
        # normalization:

        returns_mean = self.returns.mean()
        returns_std = self.returns.std() + 1e-8 
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8

        ### log rollouts results:
        print("rollout real stats")
        print(f'rewards mean: {rewards_mean}, rewards std: {rewards_std}')
        print(f'values mean: {self.values.mean()}, values std: {self.values.std()}')
        print(f'advantages mean before normalization: {adv_mean}, advantages std: {adv_std}')
        print(f'returns mean before normalization: {returns_mean}, returns std: {returns_std}')


        if self.normalize_advantage:
            if self.per_iter_normal_value:
                self.adv_std = adv_std
            else:
                if self.iterations == 1:
                    self.adv_std = adv_std

        if self.normalize_value:
            if self.per_iter_normal_value:
                self.returns_mean = returns_mean
                self.returns_std = returns_std * self.var_scaler
            else:
                if self.iterations == 1:
                    self.returns_mean = returns_mean
                    self.returns_std = returns_std * self.var_scaler

        # normalize the advantages and returns
        self.advantages = self.advantages / self.adv_std
        self.returns = (self.returns - self.returns_mean) / self.returns_std

        ### log normalization stats
        print("parameters used for normalization")
        print(f'reward mean: {self.rewards_mean}')
        print(f'advantage std: {self.adv_std}')
        print(f'returns mean: {self.returns_mean}, returns std: {self.returns_std}')

            
        return self.returns_mean, self.returns_std
        