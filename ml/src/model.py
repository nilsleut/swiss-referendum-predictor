"""
MLP model for Swiss referendum turnout regression.
"""

import copy
from typing import List

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    options: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
    }
    if name not in options:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(options)}")
    return options[name]


class MLP(nn.Module):
    """Fully-connected MLP for turnout regression.

    Architecture per hidden layer: Linear → [BatchNorm] → Activation → [Dropout]
    Final layer is a single linear unit (no activation) for unbounded regression.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int = 1,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(copy.deepcopy(_make_activation(activation)))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
