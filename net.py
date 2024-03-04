import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple

import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x
# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    activation: int,
    final_activation: str
) -> torch.nn.modules.container.Sequential:


    if activation == 'tanh':
        act_f = nn.Tanh()
    elif activation == 'relu':
        act_f = nn.ReLU()
    layer = nn.Linear(input_dim, hidden_dim)
    #orthogonal_init(layer)
    layers = [layer, act_f]
    for _ in range(depth -1):
        layer = nn.Linear(hidden_dim, hidden_dim)
        #orthogonal_init(layer)
        layers.append(layer)
        layers.append(act_f)
    layer = nn.Linear(hidden_dim, output_dim)
    #orthogonal_init(layer)
    layers.append(layer)
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)



class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, args
    ) -> None:
        super().__init__()
        self._net = MLP(args.state_dim, args.v_hidden_dim, args.depth, 1, 'relu', 'relu')
    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)
    
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
    
class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, 
        args, activation_name: str = 'elu', para_std = True
    ) -> None:
        super().__init__()
        self.para_std = para_std
        activation = get_activation(activation_name)

        layers = [nn.Linear(args.state_dim, args.pi_hidden_dim[0]), activation]

        for l in range(len(args.pi_hidden_dim)):
            if l == len(args.pi_hidden_dim) - 1:
                if para_std:
                    layers.append(nn.Linear(args.pi_hidden_dim[l], args.action_dim))   
                else:   
                    layers.append(nn.Linear(args.pi_hidden_dim[l], args.action_dim * 2))
            else:
                layers.append(nn.Linear(args.pi_hidden_dim[l], args.pi_hidden_dim[l + 1]))
                layers.append(activation)

        self._net = nn.Sequential(*layers)
        self._log_std_bound = (-10., 1.5)

        if para_std:
            self.std = nn.Parameter(torch.zeros(args.action_dim))
        
    def get_dist(
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

        dist = self.get_dist(s=s)
        a = dist.mean
        a_logprob = dist.log_prob(a) 
        return a, a_logprob
    def sample(
        self, s: torch.Tensor
    ):
        dist = self.get_dist(s=s)
        return dist.sample()
    
    def mean(self, s):
        if self.para_std:
            mu = self._net(s)
            log_std = self.std
        else:
            mu, log_std = self._net(s).chunk(2, dim=-1)
        return mu
