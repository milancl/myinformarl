import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax

class GNN(nn.Module):
    def __init__(self, in_features, out_features, aggregate, sensing_radius, mask_agent_to_other, n_agents) -> None:
        super().__init__()

        self.out_features = out_features

        self.sensing_radius = sensing_radius

        self.batched_aggregation = UniMPGNN(
            in_features,  # (rel_pos, vel, rel_goal) * self.dim + entity embedding
            out_features,
            aggregate
        )

        self.mask_agent_to_other = mask_agent_to_other
        self.n_agents = n_agents    

    def forward(self, node_obs: torch.Tensor, adj: torch.Tensor, agent_ids=None):
        # adj = (threads * n_agent, entities, entities)
        # node_obs = (threads * n_agent, n_entities, x_j_shape)
        data = []
        for i in range(adj.shape[0]):
            edge_index, edge_attr = self.parse_adj(adj[i])
            data.append(
                Data(x=node_obs[i], edge_index=edge_index, edge_attr=edge_attr.unsqueeze(1))
            )
        
        loader = DataLoader(data, shuffle=False, batch_size=adj.shape[0])
        batch = next(iter(loader))
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # batch = data.batch
        
        return self.batched_aggregation(batch, adj.shape, agent_ids=agent_ids)
        # return (threads * n_agent, x_agg_out)


    def parse_adj(self, adj: torch.Tensor):
        # adj is NxN, N = self.agents + self.landmarks + self.obstacles
        # self.landmarsk == self.agents

        assert adj.dim() == 2
        
        masks = ((adj < self.sensing_radius) * (adj > 0)).to(torch.float32)

        adj_masked = masks * adj

        if self.mask_agent_to_other:
            adj_masked[:self.n_agents, self.n_agents:] = 0.0

        #nonzero edge indexes
        nz_edge_indexes = adj_masked.nonzero(as_tuple=True)

        #nonzero edge attributes
        nz_edge_attrs = adj_masked[nz_edge_indexes]

        return torch.stack(nz_edge_indexes, dim=0), nz_edge_attrs



class UniMPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1):
        super(UniMPLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.w1 = nn.Linear(in_channels, out_channels)
        self.w2 = nn.Linear(in_channels, out_channels)
        self.w3 = nn.Linear(in_channels, out_channels)
        self.w4 = nn.Linear(in_channels, out_channels)
        self.w5 = nn.Linear(1, out_channels)  

        self.sqrt_d = torch.sqrt(torch.tensor(out_channels, dtype=torch.float32))

    def forward(self, x, edge_index, edge_attr, size=None):
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_index_i, edge_attr, size_i):
        x_i_transformed = self.w3(x_i)
        x_j_transformed = self.w4(x_j)
        edge_attr_transformed = self.w5(edge_attr)

        scores = (x_i_transformed * (x_j_transformed + edge_attr_transformed)).sum(dim=-1)
        scores = scores / self.sqrt_d
        alpha = softmax(scores, edge_index_i, num_nodes=size_i)

        return alpha.view(-1, 1) * self.w2(x_j)

    def update(self, aggr_out, x):
        return self.w1(x) + aggr_out



class UniMPGNN(nn.Module):
    def __init__(self, in_channels, out_channels, aggregate, heads=1):
        super(UniMPGNN, self).__init__()
        self.layer = UniMPLayer(in_channels, out_channels, heads)
        self.out_channels = out_channels
        self.aggregate = aggregate

    def forward(self, batch_data, adj_size, agent_ids=None):
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr

        # print("UniMPGNN forward:", x.shape, edge_index.shape, edge_attr.shape)

        # batched graph data
        node_out = self.layer(x, edge_index, edge_attr, size=(x.size(0), x.size(0)))
        # print("node_out:", node_out.shape)
        
        if self.aggregate:
            # pool node features per graph for graph-level predictions 
            # (aggregation of all entities)
            return global_mean_pool(node_out, batch_data.batch)
        
        else:
            # get the ith row of the matrix features, where i is the agent
            # for which we are interested to extract the feature vector

            node_out = node_out.reshape((*adj_size[:-1], self.out_channels))

            agent_ids = agent_ids.squeeze()

            # print(agent_ids, agent_ids.shape)

            return node_out[torch.arange(node_out.shape[0]), agent_ids, :]
        
            # for i in node_out[0]:
                # out.append(node_out[i][agent_ids[i]])

        # node_out = (threads * n_agent, n_entities, x_j_shape)
        # (threads * n_agent, x_j_shape)
        

