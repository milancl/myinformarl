import torch
import torch.nn as nn
from .util import init, get_clones
import argparse
from typing import List, Tuple, Union, Optional

"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        layer_N: int,
    ):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = nn.ReLU()
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain("relu")

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        input_dim: int
        # obs_shape: Union[List, Tuple],
        # override_obs_dim: Optional[int] = None,
    ):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        # override_obs_dim is only used for graph-based models
        # if override_obs_dim is None:
            # obs_dim = obs_shape[0]
        # else:
            # print("Overriding Observation dimension")
            # obs_dim = override_obs_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        self.mlp = MLPLayer(
            input_dim,
            self.hidden_size,
            self._layer_N,
        )

    def forward(self, x: torch.tensor):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
