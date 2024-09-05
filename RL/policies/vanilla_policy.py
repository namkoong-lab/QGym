import sys
sys.path.append('../')

import gym
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from gym import spaces
from stable_baselines3 import PPO
import numpy as np
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym
from typing import Union
import torch.distributions.one_hot_categorical as one_hot_sample
from typing import NamedTuple
from stable_baselines3.common.callbacks import BaseCallback
import yaml
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union


class Vanilla_Policy(ActorCriticPolicy):
    def __init__(self, *args, network: torch.Tensor = None, randomize: bool = False, time_f: bool = False, scale, rescale_v, alpha, mu, D, **kwargs):
        super(Vanilla_Policy, self).__init__(*args, **kwargs)
        # Store the custom parameter
        self.randomize = randomize
        self.network = network
        self.q = self.network.size()[1]
        self.s = self.network.size()[0]
        self.time_f = time_f
        self.mean_queue_length = 0
        self.std_queue_length = 1
        self.returns_mean = 0
        self.returns_std = 1
        self.printing = False
        self.rescale_v = rescale_v
        self.alpha = torch.tensor(alpha)
        self.mu = mu.unsqueeze(0)
        self.D = D

        print(f'self.alpha: {self.alpha}')
        print(f'self.mu: {self.mu}')
        print(f'self.D: {self.D}')        

    def update_mean_std(self, mean_queue_length, std_queue_length):
        self.mean_queue_length = mean_queue_length
        self.std_queue_length = std_queue_length

    def standardize_queues(self, queues):
        # Standardize the queue lengths
        standardization = (queues - self.mean_queue_length) / (self.std_queue_length + 1e-8)
        standardization.detach()

        return standardization.float()
    
    def update_rollout_stats(self, returns_mean, returns_std):
        print("update rollout stats")
        self.returns_mean = returns_mean
        self.returns_std = returns_std

    def rescale_values(self, values):
        scaled_values =  values * self.returns_std + self.returns_mean
        return scaled_values

    

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        obs = obs.view(-1, self.q)

        input_obs = self.standardize_queues(obs)
        features = self.extract_features(input_obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)        
        logits = self.action_net(latent_pi)
        action = logits.reshape((-1, *self.action_space.shape))
        action_probs = F.softmax(action, dim=-1)

        if self.rescale_v: 
            values = self.rescale_values(values)

        # randomize policy or not
        if self.randomize:
            action = one_hot_sample.OneHotCategorical(probs = action_probs).sample()
        else:
            action_indices = torch.argmax(action_probs, dim=-1)
            action = F.one_hot(action_indices, num_classes=action_probs.shape[-1])


        selected_probs = action * action_probs
        selected_probs_sum = selected_probs.sum(dim=-1)
        log_prob = torch.log(selected_probs_sum)
        log_prob = log_prob.sum(dim=1)

        return action, values, log_prob

    def get_prob_act(self, obs: torch.Tensor, deterministic: bool = False):
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)
        features = self.extract_features(input_obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        logits = self.action_net(latent_pi)
        action = logits.reshape((-1, *self.action_space.shape))
        action_probs = F.softmax(action, dim=-1)

        # randomize policy or not
        if self.randomize:
            action = one_hot_sample.OneHotCategorical(probs = action_probs).sample()
        else:
            action_indices = torch.argmax(action_probs, dim=-1)
            action = F.one_hot(action_indices, num_classes=action_probs.shape[-1])


        return action, action_probs



    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        obs_tensor = obs_tensor.view(-1, self.q)
        with torch.no_grad():
            action, action_probs = self.get_prob_act(obs_tensor, deterministic)

        return action, action_probs  # type: ignore[return-value]

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Preprocess the observation if needed
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)
        features = self.extract_features(input_obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        logits = self.action_net(latent_pi)
        action = logits.reshape((-1, *self.action_space.shape))


        if self.time_f:
            obs = obs[:, :-1]


        action_probs = F.softmax(action, dim=-1)
    
        actions = actions.reshape((-1, *self.action_space.shape))
        
        # Compute log probabilities of the provided actions

        selected_probs = actions * action_probs
        selected_probs_sum = selected_probs.sum(dim=-1) 
        log_prob = torch.log(selected_probs_sum)
        log_prob = log_prob.sum(dim=1)

        entropy = None

        return log_prob, entropy  
    
    def evaluate_values(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)
        features = super().extract_features(input_obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        return values
    
    def AMP(self, obs, action_repeat):
        '''
        Generates an approximating martingale process of a function f 
        from a batch of states, which follow a Markov Chain under the policy.

        Args:
            f: (lambda) a function mapping queue lengths (B,q) to (B,1) rewards/values
            x: (tensor (B,q)) a batch of queue length states
            policy: (lambda) policy mapping queue lengths (B,q) to (B,s,q) probabilities
            mu: (tensor (batch,s,q)) input dq.mu. note batch size may be different from B.
            D: (tensor (2q, q)) input dq.queue_event_options
    
        Returns:
            tensor (B,1): a batch of values
        '''

        # Batch size and num servers
        #print("AMP")
        #print(f'obs: {obs}')
        x = self.standardize_queues(obs)
        #print(f'x: {x}')
        B = x.size()[0]
        #print(f'B: {B}')
        #print(f'D: {self.D}, shape: {self.D.shape}')
        s = self.mu.size()[1]
        
        # Draw random actions from the policy
        # action: (A*B,s,q)
        # A = 10, B = 5, s = 2, q = 3
        features = self.extract_features(x)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        logits = self.action_net(latent_pi)
        action = logits.reshape((-1, *self.action_space.shape))
        action_probs = F.softmax(action, dim=-1)

        pr = action_probs.repeat_interleave(action_repeat, 0)
        pr_sample = one_hot_sample.OneHotCategorical(probs = pr)
        action = pr_sample.sample()

        #print(f'action: {action}, shape: {action.shape}')
        #print(f'mu:{self.mu[0].unsqueeze(0).repeat(B * action_repeat, 1, 1)}, shape: {self.mu[0].unsqueeze(0).repeat(B * action_repeat, 1, 1).shape}')

        # Multiply with mu
        # pmu: (A*B,q)
        # pmu = 50, 3
        #print(f'mu[0]', self.mu[0])
        pmu = action * self.mu[0].unsqueeze(0).repeat(B * action_repeat, 1, 1)

        # pmu_flat: (A*B,q)
        # pmu_flat = 50, 3
        pmu_flat = torch.sum(pmu, dim = 1)

        # Concatenate with arrival rates
        # Get probabilities by normalizing the rates
        # prob_transitions: (A*B,2q)
        # prob_transitions = 50, 6
        prob_transitions = torch.hstack((self.alpha.unsqueeze(0).repeat(B * action_repeat, 1), pmu_flat))
        prob_transitions /= prob_transitions.sum(dim = -1, keepdims = True)
        
        # Split by A, and then average
        # This will get transition probabilities, averaged over random actions: (1/A)\sum_i P(s'|s,a_i)
        # This is a monte carlo approximation of the true probabilities P(s'|s,a)
        # prob_transitions: (B*2q,1)
        # prob_transitions = 30, 1
        prob_transitions = torch.split(prob_transitions, action_repeat)
        prob_transitions = torch.mean(torch.stack(prob_transitions, dim = 0), dim = 1)
        prob_transitions = torch.cat(tuple(prob_transitions))
        #print(f'prob_transitions AMP: {prob_transitions}')

        # Get set of transition states by adding D, which is dq.queue_event_options
        # Px: (B*2q,q)
        # px = 30, 3
        Px = obs.unsqueeze(1) + self.D.unsqueeze(0).repeat(B,1,1)
        Px = torch.cat(tuple(Px))
        Px = torch.nn.ReLU()(Px)
        # Px = self.standardize_queues(Px)

        # Evaluate
        # Pfx: B x 1
        Pfx = torch.sum(torch.stack(torch.chunk(prob_transitions.unsqueeze(1) * self.predict_values(Px), B)), 1)

        return Pfx
    
    def predict_next_states(self, obs: torch.Tensor, action_repeat) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            
        obs = obs.view(-1, self.q)
        AMP_values = self.AMP(obs, action_repeat)

        return AMP_values
    
    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)
        features = super().extract_features(input_obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        if self.rescale_v: 
            values = self.rescale_values(values)
        return values

          
    

