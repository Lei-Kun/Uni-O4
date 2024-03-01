import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple
from offline_buffer import OnlineReplayBuffer
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



class ValueReluMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, args
    ) -> None:
        super().__init__()
        self._net = MLP(args.state_dim, args.v_hidden_width, args.v_depth, 1, 'relu', 'relu')
    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)
    
class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, args
    ) -> None:
        super().__init__()
        self._net = MLP(args.state_dim, args.v_hidden_width, args.v_depth, 1, 'relu')
    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)

class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, args
    ) -> None:
        super().__init__()
        if args.use_tanh:
            self._net = MLP(args.state_dim, args.hidden_width, args.depth, (2 * args.action_dim), 'tanh', 'tanh')
        else:
            self._net = MLP(args.state_dim, args.hidden_width, args.depth, (2 * args.action_dim), 'relu', 'tanh')
        self._log_std_bound = (-5., args.std_upper_bound)

    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:
        mu, _ = self._net(s).chunk(2, dim=-1)
        return mu
    
    def get_dist(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist


class GaussPolicyMLP_(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(self, args,
        scale_min=1e-4, scale_max=1.,):
        super(GaussPolicyMLP_, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, 2 * args.action_dim)
        #self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)
        self._log_std_bound = (-5., 0.)


    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mu, _ = self.activate_func(self.mean_layer(s)).chunk(2, dim=-1)


        return mu

    def get_dist(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mu, log_std = self.activate_func(self.mean_layer(s)).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist
    



class ValueLearner:
    _device: torch.device
    _value: ValueReluMLP
    _optimizer: torch.optim
    _batch_size: int

    def __init__(
        self, 
        args,
        value_lr: float, 
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = args.device
        self._value = ValueReluMLP(args).to(args.device)
        self._optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=value_lr,
            )
        self._batch_size = batch_size


    def __call__(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._value(s)


    def update(
        self, replay_buffer: OnlineReplayBuffer
    ) -> float:
        s, _, _, _, _, _, Return, _ = replay_buffer.sample(self._batch_size)
        value_loss = F.mse_loss(self._value(s), Return)

        self._optimizer.zero_grad()
        value_loss.backward()
        self._optimizer.step()

        return value_loss.item()


    def save(
        self, path: str
    ) -> None:
        torch.save(self._value.state_dict(), path)
        print('Value parameters saved in {}'.format(path))


    def load(
        self, path: str
    ) -> None:
        self._value.load_state_dict(torch.load(path, map_location=self._device))
        print('Value parameters loaded')