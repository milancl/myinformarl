import argparse
from typing import Tuple, List

import gym
import torch
from torch import Tensor
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.utils.gnn import GNN
from utils.utils import split_batch


class GraphActor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    """

    def __init__(
        self, config, observation_shape, x_agg_shape, x_agg_out, action_space, device
    ) -> None:
        super(GraphActor, self).__init__()

        self.device = device

        # actor gnn does not aggregate global graph features
        self.gnn = GNN(
            in_features=x_agg_shape,
            out_features=x_agg_out,
            aggregate=False,
            sensing_radius=config.max_edge_dist,
            n_agents=config.num_agents,
            mask_agent_to_other=config.mask_agent_to_other
        )

        mlp_input_dim = self.gnn.out_features + observation_shape

        self.mlp = MLPBase(args=config, input_dim=mlp_input_dim)

        self.rnn = RNNLayer(
            inputs_dim=config.hidden_size,
            outputs_dim=config.hidden_size,
            recurrent_N=config.recurrent_N,
        )

        self.act = ACTLayer(
            action_space=action_space, inputs_dim=config.hidden_size, gain=config.gain
        )

        self.to(self.device)

    def forward(
        self, local_obs, node_obs, adj, agent_ids, rnn_states, masks, deterministic=False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        """
        # print("Actor")
        adj = torch.from_numpy(adj).to(torch.float32).to(self.device)
        local_obs = torch.from_numpy(local_obs).to(torch.float32).to(self.device)
        node_obs = torch.from_numpy(node_obs).to(torch.float32).to(self.device)
        agent_ids = torch.from_numpy(agent_ids).to(torch.int).to(self.device)
        rnn_states = torch.from_numpy(rnn_states).to(torch.float32).to(self.device)
        masks = torch.from_numpy(masks).to(torch.float32).to(self.device)

        agg_nb_features = self.gnn(node_obs, adj, agent_ids=agent_ids)
        # print("GNN Actor Done")
        # print(local_obs.shape, node_obs.shape, agg_nb_features.shape)
        actor_features = torch.cat([local_obs, agg_nb_features], dim=1)
        actor_features = self.mlp(actor_features)

        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, deterministic=deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, local_obs, node_obs, adj, agent_ids, rnn_states, masks, actions
    ):
        """
        Compute log probability and entropy of given actions.
        """
        # print(local_obs.shape)

        gen = split_batch(local_obs, node_obs, adj, agent_ids, 32)

        outs = []

        for (mini_local_obs, mini_node_obs, mini_adj, mini_agent_ids) in gen:
            nbd_features = self.gnn(mini_node_obs, mini_adj, mini_agent_ids)
            actor_features = torch.cat([mini_local_obs, nbd_features], dim=1)
            actor_features = self.mlp(actor_features)
            outs.append(actor_features)
        actor_features = torch.cat(outs, dim=0)


        # nbd_features = self.gnn(node_obs, adj, agent_ids)
        # actor_features = torch.cat([local_obs, nbd_features], dim=1)
        # actor_features = self.mlp(actor_features)

        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            x=actor_features,
            action=actions,
        )

        return action_log_probs, dist_entropy


class GraphCritic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    """

    def __init__(
        self, config, observation_shape, x_agg_shape, x_agg_out, device
    ) -> None:
        super(GraphCritic, self).__init__()

        self.device = device

        # critic gnn aggregates global graph features
        self.gnn = GNN(
            in_features=x_agg_shape,
            out_features=x_agg_out,
            aggregate=True,
            sensing_radius=config.max_edge_dist,
            n_agents=config.num_agents,
            mask_agent_to_other=config.mask_agent_to_other
        )

        self.mlp = MLPBase(args=config, input_dim=self.gnn.out_features)

        self.rnn = RNNLayer(
            inputs_dim=config.hidden_size,
            outputs_dim=config.hidden_size,
            recurrent_N=config.recurrent_N,
        )

        # self.v_out = nn.init.orthogonal_(nn.Linear(config.hidden_size, 1))
        self.v_out = nn.Linear(config.hidden_size, 1)
        nn.init.orthogonal_(self.v_out.weight)

        self.to(device)

    def forward(self, node_obs, adj, rnn_states, masks):
        # print("Critic")
        if not isinstance(node_obs, torch.Tensor):
            node_obs = torch.from_numpy(node_obs).to(torch.float32).to(self.device)
            adj = torch.from_numpy(adj).to(torch.float32).to(self.device)
            rnn_states = torch.from_numpy(rnn_states).to(torch.float32).to(self.device)
            masks = torch.from_numpy(masks).to(torch.float32).to(self.device)

        # print(node_obs.shape, adj.shape, rnn_states.shape, masks.shape)

        gen = split_batch(None, node_obs, adj, None, 32)

        outs = []

        for (mini_node_obs, mini_adj) in gen:
            nbd_features = self.gnn(mini_node_obs, mini_adj)
            critic_features = torch.cat([nbd_features], dim=1)
            critic_features = self.mlp(critic_features)
            outs.append(critic_features)
        critic_features = torch.cat(outs, dim=0)

        # agg_nb_features = self.gnn(node_obs, adj)

        # critic_features = self.mlp(agg_nb_features)


        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
