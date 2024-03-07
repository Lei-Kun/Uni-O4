import gym
import torch
import numpy as np
from buffer import OnlineReplayBuffer
from net import GaussPolicyMLP
from util import log_prob_func, orthogonal_initWeights
import os
from copy import deepcopy
import torch.nn.functional as F
from critic import QPiLearner
from critic import QPiLearner
from tqdm import tqdm
from torch.distributions import Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

import copy
class BehaviorCloning:
    _device: torch.device
    _policy: GaussPolicyMLP
    _optimizer: torch.optim
    _policy_lr: float
    _batch_size: int
    _policy_id: int
    _num_policy: int
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        batch_size: int,
        policy_id: int,
        num_policy: int
    ) -> None:
        super().__init__()
        self._device = device
        self._num_policy = num_policy
        self._policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
        orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr = policy_lr
        )
        self._lr = policy_lr
        self._batch_size = batch_size
        self._policy_id = policy_id

    def loss(self, 
        s: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        
        dist = self._policy(s)
        log_prob = log_prob_func(dist, a) 
        loss = (-log_prob).mean()

        return loss
    def joint_loss(self, 
        s: torch.Tensor,
        a: torch.Tensor,
        policies: list,
        alpha: float,
        bc_kl: str,
        all_prob_a: list,
        pre_id: int,
    ) -> torch.Tensor:
        # input_shape (batch_size, state/action dim) -> prob (batch_size,1)

        #calculate max prob of state, action pair

        if bc_kl == 'pi':
            if len(policies) !=1:
                for bc in policies:

                    a = bc.select_action(s,is_sample=True)
                    prob_a = bc.get_log_prob(s,a) # tensor: (batch size,1)
                    all_prob_a.append(prob_a)
            else:

                a = policies[0].select_action(s,is_sample=True)
                prob_a = policies[0].get_log_prob(s,a)# only one policy
                all_prob_a[pre_id] = prob_a

            all_prob_a_t = torch.cat(all_prob_a,dim=1) #tensor: (batch size, num_policies)
            max_prob_a, i_max_prob_a = all_prob_a_t.max(-1) # tensor: (batch_size)

        elif bc_kl == 'data':
            if len(policies) !=1:
                for bc in policies:

                    dist = bc._policy(s)
                    prob_a = log_prob_func(dist, a) 
                    all_prob_a.append(prob_a)
            else:

                dist = policies[0]._policy(s)
                prob_a = log_prob_func(dist, a) 
                all_prob_a[pre_id] = prob_a

            all_prob_a_t = torch.cat(all_prob_a,dim=1) #tensor: (batch size, num_policies)
            max_prob_a, _ = all_prob_a_t.max(-1) # tensor: (batch_size)
        
        #\pi_i's log_prob
        dist = self._policy(s)
        log_prob = log_prob_func(dist, a) 
        loss = -log_prob - alpha*(log_prob-max_prob_a.detach())

        return loss.mean(), all_prob_a
    
    def get_log_prob(self,s:torch.Tensor,a:torch.Tensor) -> torch.Tensor:

        dist = self._policy(s)
        log_prob = log_prob_func(dist, a) 

        return log_prob
    
    def get_dist(self,s:torch.Tensor) -> torch.Tensor:
        return self._policy(s)
    
    def get_mu_std(self,s:torch.Tensor) -> torch.Tensor:
        _, mean, std = self._policy.predict(s)
        return mean, std

    def update(
        self, s: torch.Tensor,
        a: torch.Tensor,
        policies: list,
        alpha: float,
        bc_kl: str=None,
        is_single_train: bool=False,
        all_prob_a: list=None,
        pre_id: int=None,
        ) -> float:

        if is_single_train:
            policy_loss = self.loss(s,a)
            all_prob_a = 0
        else:
            policy_loss, all_prob_a = self.joint_loss(s,a,policies=policies,alpha=alpha,bc_kl=bc_kl, all_prob_a=all_prob_a, pre_id=pre_id)

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        return policy_loss.item(), all_prob_a

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

    def offline_evaluate(
        self,
        env_name: str,
        seed: int,
        mean: np.ndarray = 0.,
        std: np.ndarray = 1.,
        eval_episodes: int=10,
        belief: torch.Tensor=None
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
        d4rl_score = env.get_normalized_score(avg_reward) * 100
        return d4rl_score
    def save_policy(self, path: str,
        save_id: int
    ) -> None:
        path = path + 'policy_{}.jit'.format(save_id)
        body_model = copy.deepcopy(self._policy).to('cpu')
        traced_script_body_module = torch.jit.script(body_model)
        traced_script_body_module.save(path)


    def save(
        self, path: str,
        save_id: int
    ) -> None:
        path = path + '_{}.pt'.format(save_id)
        torch.save(self._policy.state_dict(), path)
        print('Behavior policy parameters saved in {}'.format(path))

    def load(
        self, path: str, i: int=None
    ) -> None:
        path = path + '_{}.pt'.format(self._policy_id)
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        print('Seccessfully load pi_{} at {}'.format(self._policy_id,path))

class BC_ensemble():
    def __init__(self, 
        num_policy: int,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        batch_size: int,
        bc_kl: str
        ) -> None:
        super().__init__()

        self.num = num_policy
        self.batch_size = batch_size
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bc_kl = bc_kl

        ensemble = []
        for i in range(self.num):
            bc = BehaviorCloning(device=device,state_dim=state_dim,hidden_dim=hidden_dim,depth=depth,action_dim=action_dim,policy_lr=policy_lr,batch_size=batch_size,policy_id=i,num_policy=num_policy)
            ensemble.append(bc)
        self.ensemble = ensemble
        

    def get_ensemble(self,) -> list:
        return self.ensemble

    def joint_train(self, replay_buffer: OnlineReplayBuffer, alpha: float) -> float:
        s, a, _, _, _, _, _, _ = replay_buffer.sample(self.batch_size)
        #jointly train each behavior policy
        losses = []
        all_prob_a = []

        p_is = np.arange(0, self.num)
        # shuffle pi's order
        np.random.shuffle(p_is)
        
        for i, p_i in enumerate(p_is):
            bc = self.ensemble[p_i]
            if i == 0:
                others = deepcopy(self.ensemble)
                del others[p_i]
                first_pi = p_i
                pre_id =p_i
            else:
                others = [self.ensemble[p_is[i-1]]]
                if p_i > first_pi:
                    pre_id = p_i -1
                else:
                    pre_id = p_i
            each_loss, all_prob_a = bc.update(s=s,a=a,policies=others,alpha=alpha,bc_kl=self.bc_kl, all_prob_a=all_prob_a, pre_id=pre_id)
            losses.append(each_loss)

        return np.array(losses)
    
    def evaluation(self,
        env_name: str,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        eval_episodes: int=10) -> list:
        scores = []
        for i in range(self.num):
            bc = self.ensemble[i]
            each_score = bc.offline_evaluate(env_name, seed, mean, std,eval_episodes=eval_episodes)
            scores.append(each_score)
        return np.array(scores)
    
    def ensemble_save(self, path: str, save_id: list) -> None:
        for i in save_id:
            bc = self.ensemble[i]
            bc.save(path,i)
    def ensemble_save_policy(self, path: str, save_id: list) -> None:
        for i in save_id:
            bc = self.ensemble[i]
            bc.save_policy(path,i)
    def load_pi(self, path: str) -> None:
        for i in range(self.num):
            self.ensemble[i].load(path)
                    
    def ope_dynamics_eval(self, args, dynamics_eval, q_eval, dynamics, eval_buffer, env, mean, std):
        best_mean_qs =  []
        for bc in self.ensemble:
            best_mean_q, _ = dynamics_eval(args, bc, q_eval, dynamics, eval_buffer, env, mean, std)
            best_mean_qs.append(best_mean_q)
        return np.array(best_mean_qs)
    