import torch
import torch.nn.functional as F

from net import ValueMLP, QMLP, DoubleQMLP
from buffer import OnlineReplayBuffer

import torch.nn as nn


class ValueLearner:
    _device: torch.device
    _value: ValueMLP
    _optimizer: torch.optim
    _batch_size: int

    def __init__(
        self, 
        device: torch.device, 
        state_dim: int, 
        hidden_dim: int, 
        depth: int, 
        value_lr: float, 
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = device
        self._value = ValueMLP(state_dim, hidden_dim, depth).to(device)
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



class QLearner:
    _device: torch.device
    _Q: QMLP
    _optimizer: torch.optim
    _target_Q: QMLP
    _total_update_step: int
    _target_update_freq: int
    _tau: float
    _gamma: float
    _batch_size: int

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int
    ) -> None:
        super().__init__()
        self._device = device
        self._Q = QMLP(state_dim, action_dim, hidden_dim, depth).to(device)
        self._optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=Q_lr,
            )

        self._target_Q = QMLP(state_dim, action_dim, hidden_dim, depth).to(device)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau

        self._gamma = gamma
        self._batch_size = batch_size


    def __call__(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        return self._Q(s, a)


    def loss(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> torch.Tensor:
        raise NotImplementedError


    def update(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> float:
        Q_loss = self.loss(replay_buffer, pi)
        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return Q_loss.item()


    def save(
        self, path: str
    ) -> None:
        torch.save(self._Q.state_dict(), path)
        print('Q function parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')



class QSarsaLearner(QLearner):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int
    ) -> None:
        super().__init__(
        device = device,
        state_dim = state_dim,
        action_dim = action_dim,
        hidden_dim = hidden_dim,
        depth = depth,
        Q_lr = Q_lr,
        target_update_freq = target_update_freq,
        tau = tau,
        gamma = gamma,
        batch_size = batch_size
        )


    def loss(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> torch.Tensor:
        s, a, r, s_p, a_p, not_done, _, _ = replay_buffer.sample(self._batch_size)
        with torch.no_grad():
            target_Q_value = r + not_done * self._gamma * self._target_Q(s_p, a_p)
        
        Q = self._Q(s, a)
        loss = F.mse_loss(Q, target_Q_value)
        
        return loss



class QPiLearner(QLearner):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int
    ) -> None:
        super().__init__(
        device = device,
        state_dim = state_dim,
        action_dim = action_dim,
        hidden_dim = hidden_dim,
        depth = depth,
        Q_lr = Q_lr,
        target_update_freq = target_update_freq,
        tau = tau,
        gamma = gamma,
        batch_size = batch_size
        )


    def loss(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> torch.Tensor:
        s, a, r, s_p, _, not_done, _, _ = replay_buffer.sample(self._batch_size)
        a_p = pi.select_action(s_p, is_sample=True)
        with torch.no_grad():
            target_Q_value = r + not_done * self._gamma * self._target_Q(s_p, a_p)
        
        Q = self._Q(s, a)
        loss = F.mse_loss(Q, target_Q_value)
        return loss
    
class IQL_Q_V(nn.Module):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        q_hidden_dim: int,
        q_depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int,
        v_hidden_dim: int,
        v_depth: int,
        v_lr: float,
        omega: float,
        is_double_q: bool
    ) -> None:
        
        super().__init__()
        self._device = device
        self._omega = omega
        self._is_double_q = is_double_q
        #for q
        if is_double_q:
            print('using double q learning')
            self._Q = DoubleQMLP(state_dim, action_dim, q_hidden_dim, q_depth).to(device)
            self._target_Q = DoubleQMLP(state_dim, action_dim, q_hidden_dim, q_depth).to(device)
        else:
            self._Q = QMLP(state_dim, action_dim, q_hidden_dim, q_depth).to(device)
            self._target_Q = QMLP(state_dim, action_dim, q_hidden_dim, q_depth).to(device)

        self._q_optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=Q_lr,
            )
        
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau
        self._gamma = gamma
        self._batch_size = batch_size
        #for v
        self._value = ValueMLP(state_dim, v_hidden_dim, v_depth).to(device)
        self._v_optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=v_lr,
            )

    def minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._Q(s, a)
        return torch.min(Q1, Q2)

    def target_minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._target_Q(s, a)
        return torch.min(Q1, Q2)
    
    def expectile_loss(self, loss: torch.Tensor)->torch.Tensor:
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss**2)
    def update(self, replay_buffer: OnlineReplayBuffer) -> float:
        s, a, r, s_p, _, not_done, _, _ = replay_buffer.sample(self._batch_size)
        # Compute value loss
        with torch.no_grad():
            self._target_Q.eval()
            if self._is_double_q:
                target_q = self.target_minQ(s, a)
            else:
                target_q = self._target_Q(s, a)
        value = self._value(s)
        value_loss = self.expectile_loss(target_q - value).mean()

        #update v
        self._v_optimizer.zero_grad()
        value_loss.backward()
        self._v_optimizer.step()

        # Compute critic loss
        with torch.no_grad():
            self._value.eval()
            next_v = self._value(s_p)
            
        target_q = r + not_done * self._gamma * next_v
        if self._is_double_q: 
            current_q1, current_q2 = self._Q(s, a)
            q_loss = ((current_q1 - target_q)**2 + (current_q2 - target_q)**2).mean()
        else:
            Q = self._Q(s, a)
            q_loss = F.mse_loss(Q, target_q)

        #update q and target q
        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)


        return q_loss, value_loss
        
    def get_advantage(self, s, a)->torch.Tensor:
        if self._is_double_q:
            return self.minQ(s, a) - self._value(s)
        else:
            return self._Q(s, a) - self._value(s)
    
    def save(
        self, q_path: str, v_path: str
    ) -> None:
        torch.save(self._Q.state_dict(), q_path)
        print('Q function parameters saved in {}'.format(q_path))
        torch.save(self._value.state_dict(), v_path)
        print('Value parameters saved in {}'.format(v_path))

    def load(
        self, q_path: str, v_path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(q_path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')
        self._value.load_state_dict(torch.load(v_path, map_location=self._device))
        print('Value parameters loaded')
