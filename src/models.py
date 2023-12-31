from typing import Tuple, List
import numpy as np
import torch


class PolicyNetwork(torch.nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_hiddens: List[int],
            num_outputs: int
        ) -> None:
        super(PolicyNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)
        self.log_sigma = torch.nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        for layer in self.layers:
            x = layer(x)
        return x, torch.clamp(self.log_sigma, min=-2.0, max=0.0)


class ValueNetwork(torch.nn.Module):
    def __init__(self,
            num_inputs: int,
            num_hiddens: List[int],
            num_outputs: int
        ) -> None:
        super(ValueNetwork, self).__init__()
        self.layers = torch.nn.ModuleList()
        for h in num_hiddens:
            self.layers.append(torch.nn.Linear(num_inputs, h))
            self.layers.append(torch.nn.Tanh())
            num_inputs = h
        action_layer = torch.nn.Linear(num_inputs, num_outputs)
        action_layer.weight.data.mul_(0.1)
        action_layer.bias.data.mul_(0.0)
        self.layers.append(action_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PolicyLR(torch.nn.Module):
    def __init__(
            self,
            n: int,
            m: int,
            k: int,
            scale: float=1.0
        ) -> None:
        super().__init__()

        L = scale*torch.randn(n, k, dtype=torch.float32, requires_grad=True)
        R = scale*torch.randn(k, m, dtype=torch.float32, requires_grad=True)

        self.L = torch.nn.Parameter(L)
        self.R = torch.nn.Parameter(R)

        self.log_sigma = torch.nn.Parameter(torch.zeros(1))

    def forward(
            self,
            indices: Tuple[np.ndarray, np.ndarray]
        ) -> Tuple[torch.Tensor]:
        rows, cols = indices
        prod = self.L[rows, :] * self.R[:, cols].T
        res = torch.sum(prod, dim=-1)
        return res, torch.clamp(self.log_sigma, min=-2.5, max=0.0)


class ValueLR(torch.nn.Module):
    def __init__(
            self,
            n: int,
            m: int,
            k: int,
            scale: float=1.0
        ) -> None:
        super().__init__()

        L = scale*torch.randn(n, k, dtype=torch.float32, requires_grad=True)
        R = scale*torch.randn(k, m, dtype=torch.float32, requires_grad=True)

        self.L = torch.nn.Parameter(L)
        self.R = torch.nn.Parameter(R)

    def forward(
            self,
            indices: Tuple[np.ndarray, np.ndarray]
        ) -> Tuple[torch.Tensor]:
        rows, cols = indices
        prod = self.L[rows, :] * self.R[:, cols].T
        return torch.sum(prod, dim=-1)
