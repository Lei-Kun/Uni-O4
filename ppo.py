import gym
import torch
import numpy as np
from copy import deepcopy

from buffer import OnlineReplayBuffer
from net import GaussPolicyMLP
from critic import ValueLearner, QLearner
from util import orthogonal_initWeights, log_prob_func



class ProximalPolicyOptimization:
    _device: torch.device
    _policy: GaussPolicyMLP
    _optimizer: torch.optim
    _policy_lr: float
    _old_policy: GaussPolicyMLP
    _scheduler: torch.optim
    _clip_ratio: float
    _entropy_weight: float
    _decay: float
    _omega: float
    _batch_size: int


    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
        is_iql:bool
    ) -> None:
        super().__init__()
        self._is_iql = is_iql
        self._device = device
        self._policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
        #orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=policy_lr
            )
        self._policy_lr = policy_lr
        self._old_policy = deepcopy(self._policy)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=2,
            gamma=0.98
            )
        
        self._clip_ratio = clip_ratio
        self._entropy_weight = entropy_weight
        self._decay = decay
        self._omega = omega
        self._batch_size = batch_size


    def weighted_advantage(
        self,
        advantage: torch.Tensor
    ) -> torch.Tensor:
        if self._omega == 0.5:
            return advantage
        else:

            weight = torch.where(advantage > 0, self._omega, (1 - self._omega))
            weight.to(self._device)
            return weight * advantage


    def loss(
        self, 
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
        is_linear_decay, clip_ratio_now
    ) -> torch.Tensor:
        # -------------------------------------Advantage-------------------------------------
        s, a, _, _, _, _, _, advantage = replay_buffer.sample(self._batch_size)
        old_dist = self._old_policy(s)
        # -------------------------------------Advantage-------------------------------------
        new_dist = self._policy(s)
        
        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()
        
        advantage = self.weighted_advantage(advantage)

        loss1 =  ratio * advantage 

        if is_clip_decay:
            if is_linear_decay:
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage 

        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss


    def update(
        self, 
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
        is_lr_decay: bool,
        iql =  None,
        is_linear_decay =  None,
        bppo_lr_now =  None, 
        clip_ratio_now =  None
    ) -> float:
        policy_loss = self.loss(replay_buffer, Q, value, is_clip_decay, iql, is_linear_decay, clip_ratio_now)
        
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
        self._optimizer.step()
        
        if is_lr_decay:
            self._scheduler.step()
        if is_linear_decay:
            for p in self._optimizer.param_groups:
                p['lr'] = bppo_lr_now    
        return policy_loss.item()


    def select_action(
        self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:    
            action = dist.mean
        # clip 
        action = action.clamp(-1., 1.)
        return action


    def evaluate(
        self,
        env_name: str,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        eval_episodes: int=10
        ) -> float:
        env = gym.make(env_name)
        env.seed(seed)

        total_reward = 0
        for _ in range(eval_episodes):
            s, done = env.reset(), False
            while not done:
                s = torch.FloatTensor((np.array(s).reshape(1, -1) - mean) / std).to(self._device)
                a = self.select_action(s, is_sample=False).cpu().data.numpy().flatten()
                s, r, done, _ = env.step(a)
                total_reward += r
        
        avg_reward = total_reward / eval_episodes
        return avg_reward
    def evaluate_without_reward(
        self,
        env_name: str,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        Q,
        fist_q = False,
        eval_episodes: int=10
        ) -> float:
        env = gym.make(env_name)
        env.seed(seed)

        total_reward = 0
        total_q = []
        first_q = []
        for _ in range(eval_episodes):
            s, done = env.reset(), False
            episode_q = []
            i_q = 0
            while not done:
                s = torch.FloatTensor((np.array(s).reshape(1, -1) - mean) / std).to(self._device)
                a = self.select_action(s, is_sample=False).cpu().data.numpy().flatten()
                Q_value = Q(s, torch.FloatTensor(a.reshape(1, -1)).to(self._device))

                if i_q == 0:
                    first_q.append(Q_value)
                    i_q += 1
                episode_q.append(Q_value)
                s, r, done, _ = env.step(a)
                total_reward += r
            total_q.append(torch.cat(episode_q, dim=0).flatten())
        first_q = torch.cat(first_q, dim=0).mean()
        total_q = torch.cat(total_q, dim=0).mean()
        avg_reward = total_reward / eval_episodes
        d4rl_score = env.get_normalized_score(avg_reward) * 100

        if fist_q:
            print('using first q evaluation')
            return d4rl_score, first_q
        else:
            return d4rl_score, total_q

    def save(
        self, path: str
    ) -> None:
        torch.save(self._policy.state_dict(), path)
        print('Policy parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        self._old_policy.load_state_dict(self._policy.state_dict())
        #self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._policy_lr)
        print('Policy parameters loaded')

    def set_old_policy(
        self,
    ) -> None:
        self._old_policy.load_state_dict(self._policy.state_dict())

