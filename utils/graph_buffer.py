import torch
import numpy as np
from utils.utils import _flatten


class GReplayBuffer(object):
    def __init__(
        self,
        config,
        local_obs_shape,
        node_obs_shape,
        share_obs_shape,
        adj_obs_shape,
    ) -> None:
        self.max_len = config.episode_length + 1
        self.episode_len = config.episode_length
        self.pointer = 0
        self.threads = config.n_rollout_threads
        self.n_agents = config.num_agents
        self.n_recurrent = config.recurrent_N
        self.hidden_size = config.hidden_size
        self.gamma = config.gamma
        self.max_batch_size = config.max_batch_size

        self.local_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *local_obs_shape)
        )

        self.node_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *node_obs_shape)
        )

        self.share_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *share_obs_shape)
        )

        self.adj_obs = np.zeros(
            (self.max_len, self.threads, self.n_agents, *adj_obs_shape)
        )

        self.actions = np.zeros((self.episode_len, self.threads, self.n_agents, 1))

        self.action_log_probs = np.zeros(
            (self.episode_len, self.threads, self.n_agents, 1)
        )

        self.agent_ids = np.zeros((self.max_len, self.threads, self.n_agents, 1))

        self.rewards = np.zeros((self.episode_len, self.threads, self.n_agents, 1))

        self.values = np.zeros((self.max_len, self.threads, self.n_agents, 1))

        self.cumulative_rewards = np.zeros(
            (self.max_len, self.threads, self.n_agents, 1)
        )

        self.done_masks = np.ones((self.max_len, self.threads, self.n_agents, 1))

        self.rnn_states_actor = np.zeros(
            (
                self.max_len,
                self.threads,
                self.n_agents,
                self.n_recurrent,
                self.hidden_size,
            )
        )

        self.rnn_states_critic = np.zeros(
            (
                self.max_len,
                self.threads,
                self.n_agents,
                self.n_recurrent,
                self.hidden_size,
            )
        )

    def insert(
        self,
        local_obs,
        node_obs,
        share_obs,
        adj_obs,
        rewards,
        actions,
        values,
        dones,
        rnn_states_actor,
        rnn_states_critic,
        agent_ids,
        action_log_probs,
    ):
        self.done_masks[self.pointer + 1] = np.ones((self.threads, self.n_agents, 1))
        self.done_masks[self.pointer + 1][dones] = np.zeros(((dones).sum(), 1))

        # if not ((rnn_states_actor is None) or (rnn_states_critic is None)):
        rnn_states_actor[dones] = np.zeros(
            ((dones).sum(), self.n_recurrent, self.hidden_size)
        )
        rnn_states_critic[dones] = np.zeros(
            ((dones).sum(), *self.rnn_states_critic.shape[3:])
        )
        self.rnn_states_actor[self.pointer + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.pointer + 1] = rnn_states_critic.copy()

        self.local_obs[self.pointer + 1] = local_obs.copy()
        self.node_obs[self.pointer + 1] = node_obs.copy()
        self.share_obs[self.pointer + 1] = share_obs.copy()
        self.adj_obs[self.pointer + 1] = adj_obs.copy()
        self.rewards[self.pointer] = rewards.copy()
        self.actions[self.pointer] = actions.copy()
        self.agent_ids[self.pointer + 1] = agent_ids.copy()
        self.values[self.pointer] = values.copy()
        self.action_log_probs[self.pointer] = action_log_probs.copy()

        self.pointer = (self.pointer + 1) % self.episode_len

    def cum_reward(self, next_val_pred):
        self.cumulative_rewards[-1] = next_val_pred
        for step in range(self.episode_len)[::-1]:
            self.cumulative_rewards[step] = (
                self.cumulative_rewards[step + 1]
                * self.gamma
                * self.done_masks[step + 1]
                + self.rewards[step]
            )

    def generator(self, advantages, num_mini_batch):
        batch_size = self.threads * self.n_agents
        assert batch_size >= num_mini_batch
        chunk_len = batch_size // num_mini_batch
        batch_perm = np.arange(batch_size)
        np.random.shuffle(batch_perm)

        # (max_len, threads, n_agents, *data_shape) --> (max_len, batch_size (threads * n_agents), *data_shape)
        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        local_obs = self.local_obs.reshape(-1, batch_size, *self.local_obs.shape[3:])
        node_obs = self.node_obs.reshape(-1, batch_size, *self.node_obs.shape[3:])
        adj_obs = self.adj_obs.reshape(-1, batch_size, *self.adj_obs.shape[3:])
        rnn_states_actor = self.rnn_states_actor.reshape(
            -1, batch_size, *self.rnn_states_actor.shape[3:]
        )
        rnn_states_critic = self.rnn_states_critic.reshape(
            -1, batch_size, *self.rnn_states_critic.shape[3:]
        )
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])

        action_log_probs = self.action_log_probs.reshape(
            -1, batch_size, self.action_log_probs.shape[-1]
        )
        values = self.values.reshape(-1, batch_size, 1)
        done_masks = self.done_masks.reshape(-1, batch_size, 1)
        cumulative_rewards = self.cumulative_rewards.reshape(-1, batch_size, 1)

        advantages = advantages.reshape(-1, batch_size, 1)

        agent_ids = self.agent_ids.reshape(-1, batch_size, 1)

        for chunk in range(0, batch_size, chunk_len):
            share_obs_batch = []
            local_obs_batch = []
            node_obs_batch = []
            adj_obs_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            values_batch = []
            done_masks_batch = []
            cumulative_rewards_batch = []
            advantages_batch = []
            agent_ids_batch = []
            old_action_log_probs_batch = []

            for offset in range(chunk_len):
                idx = batch_perm[chunk + offset]
                share_obs_batch.append(share_obs[:-1, idx])
                local_obs_batch.append(local_obs[:-1, idx])
                node_obs_batch.append(node_obs[:-1, idx])
                adj_obs_batch.append(adj_obs[:-1, idx])
                rnn_states_actor_batch.append(rnn_states_actor[0:1, idx])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, idx])
                actions_batch.append(actions[:, idx])
                values_batch.append(values[:-1, idx])
                done_masks_batch.append(done_masks[:-1, idx])
                cumulative_rewards_batch.append(cumulative_rewards[:-1, idx])
                advantages_batch.append(advantages[:, idx])
                agent_ids_batch.append(agent_ids[:-1, idx])
                old_action_log_probs_batch.append(action_log_probs[:, idx])

            # (chunk_len, episode_length, *data_shape) --> (episode_length, chunk_len, *data_shape)
            share_obs_batch = np.stack(share_obs_batch, 1)
            local_obs_batch = np.stack(local_obs_batch, 1)
            node_obs_batch = np.stack(node_obs_batch, 1)
            adj_obs_batch = np.stack(adj_obs_batch, 1)
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(
                chunk_len, *self.rnn_states_actor.shape[3:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                chunk_len, *self.rnn_states_critic.shape[3:]
            )
            actions_batch = np.stack(actions_batch, 1)
            values_batch = np.stack(values_batch, 1)
            agent_ids_batch = np.stack(agent_ids_batch, 1)
            done_masks_batch = np.stack(done_masks_batch, 1)
            cumulative_rewards_batch = np.stack(cumulative_rewards_batch, 1)
            advantages_batch = np.stack(advantages_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)

            # (episode_length, chunk_len, *data_shape) --> (episode_length * chunk_len, *data_shape)
            share_obs_batch = _flatten(self.max_len - 1, chunk_len, share_obs_batch)
            local_obs_batch = _flatten(self.max_len - 1, chunk_len, local_obs_batch)
            node_obs_batch = _flatten(self.max_len - 1, chunk_len, node_obs_batch)
            adj_obs_batch = _flatten(self.max_len - 1, chunk_len, adj_obs_batch)
            actions_batch = _flatten(self.max_len - 1, chunk_len, actions_batch)
            old_action_log_probs_batch = _flatten(
                self.max_len - 1, chunk_len, old_action_log_probs_batch
            )
            values_batch = _flatten(self.max_len - 1, chunk_len, values_batch)
            done_masks_batch = _flatten(self.max_len - 1, chunk_len, done_masks_batch)
            cumulative_rewards_batch = _flatten(
                self.max_len - 1, chunk_len, cumulative_rewards_batch
            )
            advantages_batch = _flatten(self.max_len - 1, chunk_len, advantages_batch)
            agent_ids_batch = _flatten(self.max_len - 1, chunk_len, agent_ids_batch)

            # print(local_obs_batch.shape)

            # num_minibatches = local_obs_batch.shape[0] // self.max_batch_size + 1
            # for i in range(num_minibatches):
            #     yield (
            #         local_obs_batch[
            #             i * self.max_batch_size : (i + 1) * self.max_batch_size
            #         ],
            #         node_obs_batch[
            #             i * self.max_batch_size : (i + 1) * self.max_batch_size
            #         ],
            #         adj_obs_batch[
            #             i * self.max_batch_size : (i + 1) * self.max_batch_size
            #         ],
            #         agent_ids_batch[
            #             i * self.max_batch_size : (i + 1) * self.max_batch_size
            #         ],
            #     )
            yield (
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
                old_action_log_probs_batch,
            )
