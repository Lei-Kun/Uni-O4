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
    final_activation: str = None
) -> torch.nn.modules.container.Sequential:

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())
    else:
        layers = layers
    return nn.Sequential(*layers)



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

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

# class GaussPolicyMLP(nn.Module):
#     _net: torch.nn.modules.container.Sequential
#     _log_std_bound: tuple

#     def __init__(
#         self, 
#         state_dim: int, hidden_dim: int, depth: int, action_dim: int, 
#     ) -> None:
#         super().__init__()
#         self._net = MLP(state_dim, hidden_dim, depth, (2 * action_dim))
#         self._log_std_bound = (-5., 0.)
#         for name, p in self.named_parameters():
#                 if 'weight' in name:
#                     if len(p.size()) >= 2:
#                         nn.init.orthogonal_(p, gain=1)
#                 elif 'bias' in name:
#                     nn.init.constant_(p, 0)

#     def forward(
#         self, s: torch.Tensor
#     ) -> torch.distributions:

#         mu, log_std = self._net(s).chunk(2, dim=-1)
#         #log_std = soft_clamp(log_std, self._log_std_bound)
#         std = log_std.exp()

#         dist = Normal(mu, std)
#         return dist
    
class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, 
        state_dim: int, hidden_dim: int, depth: int, action_dim: int, activation_name: str = 'elu', para_std = True
    ) -> None:
        super().__init__()
        self.para_std = para_std
        activation = get_activation(activation_name)

        layers = [nn.Linear(state_dim, hidden_dim[0]), activation]

        for l in range(len(hidden_dim)):
            if l == len(hidden_dim) - 1:
                if para_std:
                    layers.append(nn.Linear(hidden_dim[l], action_dim))   
                else:   
                    layers.append(nn.Linear(hidden_dim[l], action_dim * 2))
            else:
                layers.append(nn.Linear(hidden_dim[l], hidden_dim[l + 1]))
                layers.append(activation)

        self._net = nn.Sequential(*layers)
        self._log_std_bound = (-10., 2.)

        if para_std:
            self.std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:
        if self.para_std:
            mu = self._net(s)
            log_std = self.std
        else:
            mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()
        dist = Normal(mu, std)
        return dist
    def sample_a_logprob(self, s: torch.Tensor):

        dist = self.forward(s=s)
        a = dist.sample() 
        a_logprob = dist.log_prob(a) 
        return a, a_logprob
    def sample(
        self, s: torch.Tensor
    ):
        dist = self.forward(s=s)
        return dist.sample()
    
    def mean(self, s):
        if self.para_std:
            mu = self._net(s)
            log_std = self.std
        else:
            mu, log_std = self._net(s).chunk(2, dim=-1)
        return mu
