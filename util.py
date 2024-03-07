import torch
import torch.nn as nn
from torch.distributions import Distribution
import numpy as np
from tqdm import tqdm
import os

CONST_EPS = 1e-10


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
    if len(log_prob.shape) == 1:
        return log_prob
    else:
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

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

class histroy_obs(object):
    def __init__(self, history_len: int = 30, num_obs: int = 70):
        self.num_obs = num_obs
        self.obs_history_length = history_len
        self.num_obs_history = self.obs_history_length * num_obs
        self.obs_history = np.zeros(self.num_obs_history)

    def step(self, obs):
        self.obs_history = np.concatenate((self.obs_history[self.num_obs:], obs))
        return self.obs_history

    def reset(self,):
        self.obs_history[ :] = 0