"""
Modules to compute actions. Basically the final layer in the network.
This imports modified probability distribution layers wherein we can give action
masks to re-normalise the probability distributions
"""
from .distributions import Categorical
import torch
import torch.nn as nn
from typing import Optional


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    action_space: (gym.Space) action space.
    inputs_dim: int
        Dimension of network input.
    gain: float
        Gain of the output layer of the network.
    """

    def __init__(
        self, action_space, inputs_dim: int, gain: float
    ):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False

        action_dim = action_space
        self.action_out = Categorical(inputs_dim, action_dim, gain)

        

    def forward(
        self,
        x: torch.tensor,
        available_actions: Optional[torch.tensor] = None,
        deterministic: bool = False,
    ):
        """
        Compute actions and action logprobs from given input.
        x: torch.Tensor
            Input to network.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: bool
            Whether to sample from action distribution or return the mode.

        :return actions: torch.Tensor
            actions to take.
        :return action_log_probs: torch.Tensor
            log probabilities of taken actions.
        """

        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs

    def get_probs(
        self, x: torch.Tensor, available_actions: Optional[torch.tensor] = None
    ):
        """
        Compute action probabilities from inputs.
        x: torch.Tensor
            Input to network.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)

        :return action_probs: torch.Tensor
        """

        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(
        self,
        x: torch.tensor,
        action: torch.tensor,
        available_actions: Optional[torch.tensor] = None,
        active_masks: Optional[torch.tensor] = None,
    ):
        """
        Compute log probability and entropy of given actions.
        x: torch.Tensor
            Input to network.
        action: torch.Tensor
            Actions whose entropy and log probability to evaluate.
        available_actions: torch.Tensor
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: torch.Tensor
            Denotes whether an agent is active or dead.

        :return action_log_probs: torch.Tensor
            log probabilities of the input actions.
        :return dist_entropy: torch.Tensor
            action distribution entropy for the given inputs.
        """

        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            dist_entropy = (
                action_logits.entropy() * active_masks.squeeze(-1)
            ).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy
