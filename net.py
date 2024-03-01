import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    #x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x

def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    activation: str = 'relu',
    final_activation: str = None
) -> torch.nn.modules.container.Sequential:


    if activation == 'tanh':
        act_f = nn.Tanh()
    elif activation == 'relu':
        act_f = nn.ReLU()

    layers = [nn.Linear(input_dim, hidden_dim), act_f]
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act_f)

    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())
    else:
        layers = layers

    return nn.Sequential(*layers)

# def MLP(
#     input_dim: int,
#     hidden_dim: int,
#     depth: int,
#     output_dim: int,
#     final_activation: str = None
# ) -> torch.nn.modules.container.Sequential:

#     layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
#     for _ in range(depth -1):
#         layers.append(nn.Linear(hidden_dim, hidden_dim))
#         layers.append(nn.ReLU())
#     layers.append(nn.Linear(hidden_dim, output_dim))
#     if final_activation == 'relu':
#         layers.append(nn.ReLU())
#     elif final_activation == 'tanh':
#         layers.append(nn.Tanh())
#     else:
#         layers = layers
#     return nn.Sequential(*layers)



class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, state_dim: int, hidden_dim: int, depth: int
    ) -> None:
        super().__init__()
        self._net = MLP(state_dim, hidden_dim, depth, 1)

    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)



class QMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, 
        state_dim: int, action_dim: int, hidden_dim: int, depth:int
    ) -> None:
        super().__init__()
        self._net = MLP((state_dim + action_dim), hidden_dim, depth, 1)

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=1)
        return self._net(sa)

class DoubleQMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, 
        state_dim: int, action_dim: int, hidden_dim: int, depth:int
    ) -> None:
        super().__init__()
        self._net1 = MLP((state_dim + action_dim), hidden_dim, depth, 1)
        self._net2 = MLP((state_dim + action_dim), hidden_dim, depth, 1)


    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=1)
        return self._net1(sa), self._net2(sa)

class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, 
        state_dim: int, hidden_dim: int, depth: int, action_dim: int, pi_activation_f = 'relu'
    ) -> None:
        super().__init__()
        if pi_activation_f == 'relu':
            print('using relu as activation function!!!')
        elif pi_activation_f == 'tanh':
            print('using tanh as activation function!!!') 
        self._net = MLP(state_dim, hidden_dim, depth, (2 * action_dim), pi_activation_f, 'tanh')
        self._log_std_bound = (-5., 0.)
        for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist
    
    def predict(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist, mu, std
