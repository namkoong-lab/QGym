
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from typing import Union, List
from stable_baselines3.common.utils import explained_variance
from stable_baselines3 import PPO
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv


def cosine_lr_schedule(initial_lr, min_lr=1e-5, progress_remaining=1.0, warmup_proportion=0.03):
    """
    Computes the cosine decay of the learning rate with a linear warmup period at the beginning.
    
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :param progress_remaining: The progress remaining (from 1 to 0).
    :param warmup_proportion: The proportion of the total training time to be used for linear warmup.
    :return: The adjusted learning rate based on the cosine schedule with warmup.
    """
    # Ensure progress_remaining is between 0 and 1
    progress_remaining = np.clip(progress_remaining, 0, 1)

    if progress_remaining > (1 - warmup_proportion):
        # Warmup phase: linearly increase LR
        warmup_progress = (1 - progress_remaining) / warmup_proportion
        new_lr = min_lr + (initial_lr - min_lr) * warmup_progress
    else:
        # Adjusted progress considering warmup phase
        adjusted_progress = (progress_remaining - (1 - warmup_proportion)) / (1 - warmup_proportion)
        
        # Cosine decay phase
        cos_decay = 0.5 * (1 + np.cos(np.pi * adjusted_progress))
        decayed = (1 - min_lr / initial_lr) * cos_decay + min_lr / initial_lr
        new_lr = initial_lr * decayed

    return new_lr

class CustomPPOTrainer(PPO):
    def __init__(self, *args, normalize_value, lr_policy, lr_value, min_lr_policy, min_lr_value, amp_value, rescale_v, num_epochs, actors, raw_env = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_value = normalize_value
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.min_lr_policy = min_lr_policy
        self.min_lr_value = min_lr_value
        self.amp_value = amp_value
        self.rescale_v = rescale_v
        self.num_epochs = num_epochs
        self.raw_env = raw_env
        self.actors = actors
        self.training_iteration = 0
        self.policy_train_iter = 0
        self.value_train_iter = 0

        # check is self.policy has log_std:
        if hasattr(self.policy, "log_std"):
            self.optimizer_policy = th.optim.Adam([
                {'params': self.policy.log_std},
                {'params': self.policy.features_extractor.parameters()},
                {'params': self.policy.pi_features_extractor.parameters()},
                {'params': self.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.policy.action_net.parameters()}
            ], lr=self.lr_policy)

        else:
            self.optimizer_policy = th.optim.Adam([
                {'params': self.policy.features_extractor.parameters()},
                {'params': self.policy.pi_features_extractor.parameters()},
                {'params': self.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.policy.action_net.parameters()}
            ], lr=self.lr_policy)

        self.optimizer_value = th.optim.Adam([
            {'params': self.policy.vf_features_extractor.parameters()},
            {'params': self.policy.mlp_extractor.value_net.parameters()},
            {'params': self.policy.value_net.parameters()}
        ], lr=self.lr_value)

        ### print the architecture of the both networks:
        print('policy architecture:')
        print(self.policy)
        print('value architecture:')
        print(self.policy.value_net)

        #### Check if there's missing parameters: #####
        all_parameters = set(self.policy.parameters())

        # Collect parameters managed by each optimizer
        optimizer_policy_params = set(param for group in self.optimizer_policy.param_groups for param in group['params'])
        optimizer_value_params = set(param for group in self.optimizer_value.param_groups for param in group['params'])

        # Check for any missing parameters
        missing_params = all_parameters - (optimizer_policy_params | optimizer_value_params)
        assert len(missing_params) == 0, "Some parameters are not being optimized."


    def _update_learning_rate(self, policy_optimizer: th.optim.Optimizer, value_optimizer: th.optim.Optimizer) -> None:
        """
        Update the learning rates for policy and value optimizers separately
        using their respective cosine learning rate schedules based on the current progress remaining.
        """
        # Update policy optimizer learning rate
        lr_policy = cosine_lr_schedule(self.lr_policy, self.min_lr_policy, self._current_progress_remaining)
        for param_group in policy_optimizer.param_groups:
            param_group['lr'] = lr_policy
        self.logger.record("train/learning_rate/policy", lr_policy)

        # Update value optimizer learning rate
        lr_value = cosine_lr_schedule(self.lr_value, self.min_lr_value, self._current_progress_remaining)
        for param_group in value_optimizer.param_groups:
            param_group['lr'] = lr_value
        self.logger.record("train/learning_rate/value", lr_value)

    def train(self) -> None:
            """
            Update policy using the currently gathered rollout buffer.
            """
            # Switch to train mode (this affects batch norm / dropout)
            self.policy.set_training_mode(True)
            print('-----------------------------------------now training-----------------------------------------')
            training_time_start = time.time()
            self.training_iteration += 1    

            # Update optimizer learning rate
            # self._update_learning_rate(self.policy.optimizer)
            self._update_learning_rate(self.optimizer_policy, self.optimizer_value)
            # Compute current clip range
            clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
            clipping_alpha = 1.0 - self.training_iteration / self.num_epochs
            clip_range = max(0.01, clipping_alpha * clip_range)
            # print(f'clip_range: {clip_range}')
            # print(f'num_epochs: {self.num_epochs}')
            # print(f'current_training_iteration: {self.training_iteration}')
            # print(f'clipping_alpha: {clipping_alpha}')
            # print(f'clip_range: {clip_range}')
            
            # Optional: clip range for the value function
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []

            continue_training = True
            # train for n_epochs epochs
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                
                # Get all rollout data for policy training
                for rollout_data in self.rollout_buffer.get():
                
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                                        
                    advantages = rollout_data.advantages
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    policy_loss = policy_loss + self.ent_coef * entropy_loss

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Update Policy network
                    self.optimizer_policy.zero_grad()
                    policy_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer_policy.step()

                    self.policy_train_iter += 1

                # Do a complete pass on the rollout buffer for value training
                for rollout_data in self.rollout_buffer.get(self.batch_size):

                    values = self.policy.evaluate_values(rollout_data.observations)
                    values = values.flatten()
                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )

                    value_loss = F.mse_loss(rollout_data.returns, values_pred)                    
                    value_loss = self.vf_coef * value_loss
                    value_losses.append(value_loss.item())

                    # Update value network
                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer_value.step()

                    self.value_train_iter += 1


                if not continue_training:
                    break

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            training_time_end = time.time() 
            print(f'training_time: {training_time_end - training_time_start}')

            # Logs
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))

            # log lr rate for both
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train/policy_updates", self.policy_train_iter)
            self.logger.record("train/value_updates", self.value_train_iter)
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

    def construct_batch(self):
        lex_batch = []
        obs_batch = []
        state_batch = []
        total_cost_batch = []
        time_weight_queue_len_batch = []


        for dq_idx in range(self.actors):

            dq = self.raw_env[dq_idx]
            lex = th.zeros(dq.batch, dq.s, dq.q)
            obs, state = dq.reset(seed = dq.seed)
            total_cost = th.tensor([[0.]])
            time_weight_queue_len = th.tensor([[0.]])

            lex_batch.append(lex)
            obs_batch.append(obs)
            state_batch.append(state)
            total_cost_batch.append(total_cost)
            time_weight_queue_len_batch.append(time_weight_queue_len)

        
        return lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch

    # def collect_rollouts(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     rollout_buffer: RolloutBuffer,
    #     n_rollout_steps: int,
    # ) -> bool:
    #     """
    #     Collect experiences using the current policy and fill a ``RolloutBuffer``.
    #     The term rollout here refers to the model-free notion and should not
    #     be used with the concept of rollout used in model-based RL or planning.

    #     :param env: The training environment
    #     :param callback: Callback that will be called at each step
    #         (and at the beginning and end of the rollout)
    #     :param rollout_buffer: Buffer to fill with rollouts
    #     :param n_rollout_steps: Number of experiences to collect per environment
    #     :return: True if function returned with at least `n_rollout_steps`
    #         collected, False if callback terminated rollout prematurely.
    #     """
    #     assert self._last_obs is not None, "No previous observation was provided"
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)

    #     n_steps = 0
    #     rollout_buffer.reset()
    #     # Sample new weights for the state dependent exploration
    #     if self.use_sde:
    #         self.policy.reset_noise(env.num_envs)

    #     callback.on_rollout_start()
    #     # collect_roll_out_time_start = time.time()
    #     lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch = self.construct_batch()
    #     test_dq_batch = self.env

    #     while n_steps < n_rollout_steps:
    #         # start_time = time.time()
    #         if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.policy.reset_noise(env.num_envs)

    #         with th.no_grad():
    #             # Convert to pytorch tensor or to TensorDict
    #             self.policy.printing = False
    #             obs_tensor = obs_as_tensor(self._last_obs, self.device)
    #             actions, values, log_probs = self.policy(obs_tensor)
    #             self.policy.printing = False
    #         actions = actions.cpu().numpy()


    #         # Rescale and perform action
    #         clipped_actions = actions

    #         if isinstance(self.action_space, spaces.Box):
    #             if self.policy.squash_output:
    #                 # Unscale the actions to match env bounds
    #                 # if they were previously squashed (scaled in [-1, 1])
    #                 clipped_actions = self.policy.unscale_action(clipped_actions)
    #             else:
    #                 # Otherwise, clip the actions to avoid out of bound error
    #                 # as we are sampling from an unbounded Gaussian distribution
    #                 clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

    #         # Rescale and perform action
    #         new_obs, rewards, dones, _ = env.step(clipped_actions)
    #         new_obs = new_obs.squeeze() 
    #         # print(f'new_obs shape: {new_obs.shape}')
    #         # print(f'rewards shape: {rewards.shape}')     
    #         # print(f'dones: {dones}')  
    #         # print(f'n step: {n_steps} step_time: {dones}')


    #         # print(f'collect_time: {collect_time_end - start_time}')

    #         self.num_timesteps += env.num_envs

    #         # Give access to local variables
    #         callback.update_locals(locals())
    #         if not callback.on_step():
    #             return False

    #         # self._update_info_buffer(infos)
    #         n_steps += 1

    #         if isinstance(self.action_space, spaces.Discrete):
    #             # Reshape in case of discrete action
    #             actions = actions.reshape(-1, 1)


    #         # for idx, done in enumerate(dones):
    #         #     if (
    #         #         done
    #         #         and infos[idx].get("terminal_observation") is not None
    #         #         and infos[idx].get("TimeLimit.truncated", False)
    #         #     ):
    #         #         terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
    #         #         with th.no_grad():
    #         #             terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
    #         #         rewards[idx] += self.gamma * terminal_value
            
    #         # print(f'enumerate_time: {enumerate_time_end - collect_time_end}')
    #         rollout_buffer.add(
    #                 self._last_obs,  # type: ignore[arg-type]
    #                 actions,
    #                 rewards,
    #                 self._last_episode_starts,  # type: ignore[arg-type]
    #                 values,
    #                 log_probs,
    #             )
    #         self._last_obs = new_obs  # type: ignore[assignment]
    #         self._last_episode_starts = dones
    #         # end_time = time.time()

    #         # print(f'collect_per_roll_out_time: {end_time - start_time}')

    #     with th.no_grad():
    #         # Compute value for the last timestep
            
    #         values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        

    #     retunrs_mean, returns_std= rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    #     #print("update_rollout_stats for policy")
    #     #print(f'returns_mean: {retunrs_mean} returns_std: {returns_std}')
    #     self.policy.update_rollout_stats(retunrs_mean, returns_std)

    #     callback.update_locals(locals())

    #     callback.on_rollout_end()
    #     # collect_roll_out_time_end = time.time()

    #     # print(f'collect_roll_out_time: {collect_roll_out_time_end - collect_roll_out_time_start}')

    #     return True
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        # collect_roll_out_time_start = time.time()
        lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch = self.construct_batch()
        train_dq_batch = self.raw_env

        for idx, env in enumerate(train_dq_batch):
            env.reset_env_seed()    

        print(f'n_rollout_steps: {n_rollout_steps}')
        while n_steps < n_rollout_steps:
            # start_time = time.time()
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                self.policy.printing = False
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                self.policy.printing = False
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Rescale and perform action
            batch_rewards = []
            batch_obs = []
            for train_dq_idx in range(len(train_dq_batch)):
                new_obs, rewards, dones, truncated, _ = train_dq_batch[train_dq_idx].step(clipped_actions)
                new_obs = new_obs.squeeze() 
                batch_rewards.append(rewards)
                batch_obs.append(new_obs)
            rewards = np.array(batch_rewards)
            new_obs = np.array(batch_obs)
            rewards = rewards.squeeze()
            
            dones = [False for _ in range(self.actors)]

            # self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # print(f'enumerate_time: {enumerate_time_end - collect_time_end}')
            rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            # end_time = time.time()

            # print(f'collect_per_roll_out_time: {end_time - start_time}')

        with th.no_grad():
            # Compute value for the last timestep
            
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        
        dones = np.array(dones)
        retunrs_mean, returns_std= rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        #print("update_rollout_stats for policy")
        #print(f'returns_mean: {retunrs_mean} returns_std: {returns_std}')
        self.policy.update_rollout_stats(retunrs_mean, returns_std)

        callback.update_locals(locals())

        callback.on_rollout_end()
        # collect_roll_out_time_end = time.time()

        # print(f'collect_roll_out_time: {collect_roll_out_time_end - collect_roll_out_time_start}')

        return True 