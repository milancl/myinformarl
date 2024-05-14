import numpy as np
import argparse
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from algorithms.MAPPOPolicy import R_MAPPOPolicy
from utils.graph_buffer import GReplayBuffer
from utils.utils import get_grad_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check


class R_MAPPO:
    """
    Trainer class for MAPPO to update policies.
    args: (argparse.Namespace)
        Arguments containing relevant model, policy, and env information.
    policy: (R_MAPPO_Policy)
        Policy to update.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        args: argparse.Namespace,
        policy: R_MAPPOPolicy,
        device,
    ) -> None:
        self.device = device
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        # self._use_value_active_masks = args.use_value_active_masks
        # self._use_policy_active_masks = args.use_policy_active_masks

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(
        self,
        values: Tensor,
        value_preds_batch: Tensor,
        return_batch: Tensor,
        # active_masks_batch: Tensor,
    ) -> Tensor:
        """
        Calculate value function loss.
        values: (torch.Tensor)
            value function predictions.
        value_preds_batch: (torch.Tensor)
            "old" value  predictions from data batch (used for value clip loss)
        return_batch: (torch.Tensor)
            reward to go returns.
        active_masks_batch: (torch.Tensor)
            denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor)
            value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                self.value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def ppo_update(
        self, sample: Tuple, update_actor: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Update actor and critic networks.
        sample: (Tuple)
            contains data batch with which to update networks.
        update_actor: (bool)
            whether to update actor network.

        :return value_loss: (torch.Tensor)
            value function loss.
        :return critic_grad_norm: (torch.Tensor)
            gradient norm from critic update.
        ;return policy_loss: (torch.Tensor)
            actor(policy) loss value.
        :return dist_entropy: (torch.Tensor)
            action entropies.
        :return actor_grad_norm: (torch.Tensor)
            gradient norm from actor update.
        :return imp_weights: (torch.Tensor)
            importance sampling weights.
        """
        
        (
            share_obs_batch,
            local_obs_batch,
            node_obs_batch,
            adj_obs_batch,
            rnn_states_actor_batch,
            rnn_states_critic_batch,
            actions_batch,
            values_batch,
            done_masks_batch,
            cumulative_rewards_batch,
            advantages_batch,
            agent_ids_batch,
            old_action_log_probs_batch
        ) = [torch.from_numpy(e).to(torch.float32).to(self.device) for e in sample]

        # old_action_log_probs_batch = torch.from_numpy(old_action_log_probs_batch).to(torch.float32).to(self.device)
        # advantages_batch = torch.from_numpy(advantages_batch).to(torch.float32).to(self.device)
        # values_batch = torch.from_numpy(values_batch).to(torch.float32).to(self.device)
        # cumulative_rewards_batch = torch.from_numpy(cumulative_rewards_batch).to(torch.float32).to(self.device)
        agent_ids_batch = agent_ids_batch.to(torch.int)
        # adj_obs_batch = torch.from_numpy(adj_obs_batch).to(torch.float32).to(self.device)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs=share_obs_batch,
            local_obs=local_obs_batch,
            node_obs=node_obs_batch,
            adj_obs=adj_obs_batch,
            rnn_states_actor=rnn_states_actor_batch,
            rnn_states_critic=rnn_states_critic_batch,
            actions=actions_batch,
            masks=done_masks_batch,
            agent_ids=agent_ids_batch
        )
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * advantages_batch
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages_batch
        )

        policy_action_loss = -torch.sum(
            torch.min(surr1, surr2), dim=-1, keepdim=True
        ).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, values_batch, cumulative_rewards_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )

    def train(self, buffer: GReplayBuffer, update_actor: bool = True):
        """
        Perform a training update using minibatch GD.
        buffer: (SharedReplayBuffer)
            buffer containing training data.
        update_actor: (bool)
            whether to update actor network.

        :return train_info: (dict)
            contains information regarding
            training update (e.g. loss, grad norms, etc).
        """
        if self._use_valuenorm:
            advantages = buffer.cumulative_rewards[
                :-1
            ] - self.value_normalizer.denormalize(buffer.values[:-1])
        else:
            advantages = buffer.cumulative_rewards[:-1] - buffer.values[:-1]
        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample, update_actor)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Convert networks to training mode"""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """Convert networks to eval mode"""
        self.policy.actor.eval()
        self.policy.critic.eval()
