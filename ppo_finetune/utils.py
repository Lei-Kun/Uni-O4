import torch
import torch.nn as nn
from torch.distributions import Distribution
import numpy as np
import tqdm
CONST_EPS = 1e-10
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x


def orthogonal_initWeights(
    net: nn.Module,
    ) -> None:
    for e in net.parameters():
        if len(e.size()) >= 2:
            nn.init.orthogonal_(e)


def log_prob_func(
    dist: Distribution, action: torch.Tensor
    ) -> torch.Tensor:
    log_prob = dist.log_prob(action)

    return log_prob.sum(-1, keepdim=True)


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.ones(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations)), desc='split the buffer to trajectories'):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs
def normalize(observations, actions, rewards, masks, dones_float, next_observations):

    trajs = split_into_trajectories(observations, actions, rewards, masks, dones_float, next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    rewards *= 1000.0

    return rewards


def evaluate_policy(args, env, agent, state_norm, offline_eval=False):
    if offline_eval:
        print('offline evaluation')
        times = 10
    else:
        times = 10
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        
        done = False
        episode_reward = 0
        while not done:
            action = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    if offline_eval:
        avg_reward = evaluate_reward / times
        d4rl_score = env.get_normalized_score(avg_reward) * 100
        return d4rl_score
    else:
        return evaluate_reward / times
    
def load_config(logdir, args):
    import os
    import glob

    config_path = os.path.join(logdir, 'config.txt')

    loaded_config = {}
    with open(config_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.strip().split(':')
            loaded_config[key.strip()] = value.strip()
    args.hidden_width = int(loaded_config['bc_hidden_dim']) 
    args.depth = int(loaded_config['bc_depth'])
    if args.scale_strategy == None:
        args.v_hidden_width = int(loaded_config['v_hidden_dim'] )
        args.v_depth = int(loaded_config['v_depth'])
    args.use_state_norm = bool(loaded_config['is_state_norm'])
    if loaded_config['pi_activation_f'] == 'tanh':
        args.use_tanh = True
    args.is_shuffle = bool(loaded_config['is_shuffle'])
    
    return args

def get_top_x_indices(arr, x):
    max_values = sorted(arr, reverse=True)[:x]  # 获取最大的两个数
    indices = [i for i, num in enumerate(arr) if num in max_values]  # 获取最大数的位置
    return indices
def get_values_by_indices(arr, indices):
    values = [arr[idx] for idx in indices]
    return values
