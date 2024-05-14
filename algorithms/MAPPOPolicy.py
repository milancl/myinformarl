import torch
from torch import Tensor
from typing import Tuple
from algorithms.graph_actor_critic import GraphActor, GraphCritic


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks
    to compute actions and value function predictions.

    args: (argparse.Namespace)
        Arguments containing relevant model and policy information.
    obs_space: (gym.Space)
        Observation space.
    share_obs_space: (gym.Space)
        Value function input space
        (centralized input for MAPPO, decentralized for IPPO).
    action_space: (gym.Space) a
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        # args: argparse.Namespace,
        config,
        local_obs_space,
        share_obs_space,
        node_obs_space,
        act_space,
        device,
    ) -> None:
        self.device = device
        self.lr = config.lr
        self.critic_lr = config.critic_lr
        self.opti_eps = config.opti_eps
        self.weight_decay = config.weight_decay

        self.local_obs_space = local_obs_space
        self.share_obs_space = share_obs_space
        self.node_obs_space = node_obs_space

        self.act_space = act_space

        # print(self.local_obs_space, self.share_obs_space, self.node_obs_space)

        self.actor = GraphActor(
            config=config,
            observation_shape=self.local_obs_space[0],
            x_agg_shape=self.node_obs_space[1],
            x_agg_out=16,
            action_space=act_space,
            device=self.device,
        )

        self.critic = GraphCritic(
            config=config,
            observation_shape=self.share_obs_space[0],
            x_agg_shape=self.node_obs_space[1],
            x_agg_out=16,
            device=self.device,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_actions(
        self,
        share_obs,
        local_obs,
        node_obs,
        adj_obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        agent_ids,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute actions and value function predictions for the given inputs.
        share_obs (np.ndarray):
            Centralized input to the critic.
        obs (np.ndarray):
            Local agent inputs to the actor.
        rnn_states_actor: (np.ndarray)
            If actor is RNN, RNN states for actor.
        rnn_states_critic: (np.ndarray)
            If critic is RNN, RNN states for critic.
        masks: (np.ndarray)
            Denotes points at which RNN states should be reset.
        available_actions: (np.ndarray)
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            Whether the action should be mode of
            distribution or should be sampled.

        :return values: (torch.Tensor)
            value function predictions.
        :return actions: (torch.Tensor)
            actions to take.
        :return action_log_probs: (torch.Tensor)
            log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor)
            updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor)
            updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor.forward(
            local_obs=local_obs,
            node_obs=node_obs,
            adj=adj_obs,
            agent_ids=agent_ids,
            rnn_states=rnn_states_actor,
            masks=masks,
        )

        values, rnn_states_critic = self.critic.forward(
            node_obs=node_obs, adj=adj_obs, rnn_states=rnn_states_critic, masks=masks
        )
        return (values, actions, action_log_probs, rnn_states_actor, rnn_states_critic)

    def get_values(self, node_obs, adj_obs, rnn_states_critic, masks) -> Tensor:
        """
        Get value function predictions.
        share_obs (np.ndarray):
            centralized input to the critic.
        rnn_states_critic: (np.ndarray)
            if critic is RNN, RNN states for critic.
        masks: (np.ndarray)
            denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic.forward(
            node_obs=node_obs, adj=adj_obs, rnn_states=rnn_states_critic, masks=masks
        )
        return values

    def evaluate_actions(
        self,
        share_obs,
        local_obs,
        node_obs,
        adj_obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        masks,
        agent_ids
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get action logprobs / entropy and
        value function predictions for actor update.
        share_obs (np.ndarray):
            centralized input to the critic.
        obs (np.ndarray):
            local agent inputs to the actor.
        rnn_states_actor: (np.ndarray)
            if actor is RNN, RNN states for actor.
        rnn_states_critic: (np.ndarray)
            if critic is RNN, RNN states for critic.
        action: (np.ndarray)
            actions whose log probabilites and entropy to compute.
        masks: (np.ndarray)
            denotes points at which RNN states should be reset.
        available_actions: (np.ndarray)
            denotes which actions are available to agent
            (if None, all actions available)
        active_masks: (torch.Tensor)
            denotes whether an agent is active or dead.

        :return values: (torch.Tensor)
            value function predictions.
        :return action_log_probs: (torch.Tensor)
            log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            local_obs=local_obs,
            node_obs=node_obs,
            adj=adj_obs,
            rnn_states=rnn_states_actor,
            actions=actions,
            masks=masks,
            agent_ids=agent_ids
        )

        values, _ = self.critic.forward(
            node_obs=node_obs, adj=adj_obs, rnn_states=rnn_states_critic, masks=masks
        )
        return values, action_log_probs, dist_entropy

    def act(
        self, local_obs, node_obs, adj_obs, rnn_states_actor, masks, agent_ids, deterministic=False
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute actions using the given inputs.
        local_obs (np.ndarray):
            local agent inputs to the actor.
        rnn_states_actor: (np.ndarray)
            if actor is RNN, RNN states for actor.
        masks: (np.ndarray)
            denotes points at which RNN states should be reset.
        available_actions: (np.ndarray)
            denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            whether the action should be mode of
            distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor.forward(
            local_obs=local_obs,
            node_obs=node_obs,
            adj=adj_obs,
            rnn_states=rnn_states_actor,
            masks=masks,
            agent_ids=agent_ids,
            deterministic=deterministic
        )
        return actions, rnn_states_actor
