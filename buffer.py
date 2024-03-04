import os
import torch
import h5py
import numpy as np
from tqdm import tqdm
from utils import CONST_EPS
from copy import deepcopy
from utils import RewardScaling, normalize
from reward_aliengo_new import reward_aliengo

class load_dataset():
    def __init__(self, ) -> None:
        super().__init__()

    def compute_reward(self, observations, actions):
        rewards = []
        reward_calculator = reward_aliengo(action_dim=12)
        for i in range(len(observations)-1):
            reward_calculator.load_data(observations[i], actions[i])
            reward = reward_calculator.calculate_reward()
            rewards.append(reward)
        return rewards

    def load_file(self, path: str, traj_length: float = 1000):
        dataset = {'observations': [], 'actions': [], 'terminals': [], 'rewards': [], 'timeouts': []}
        file_id = 0
        total_returns = []
        for files in os.listdir(path):
            print(files)
            for filename in os.listdir(os.path.join(path, files)):
                print(filename)
                if filename.endswith(".hdf5"):

                    with h5py.File(os.path.join(path, files, filename), 'r') as file:
                        actions = np.array(file['actions'])
                        actions = actions[:, :12]
                        observations = np.array(file['states'])
                        # terminals = np.array(file['dones'])
                     
                        # np.savetxt(r'test_{}'.format(str(filename)), [np.max(actions,axis=0), np.min(actions, axis=0),
                        #                                                 np.mean(actions, axis=0), np.std(actions, axis=0)], fmt='%.3f')                        
                        reward = self.compute_reward(observations, actions)
                        traj_return_mean = np.sum(reward)
                        total_returns.append(traj_return_mean)
                        print('max: {}, min: {}, mean: {}, std: {}'.format(np.max(reward,axis=0), np.min(reward, axis=0), np.mean(reward, axis=0), np.std(reward, axis=0)))
    
                        terminals = [False]*(len(observations)-1) + [True] 
                        dy_length = 0
                        for i in range(len(actions)-1):
                            dy_length += 1
                            dataset['actions'].append(actions[i])
                            dataset['observations'].append(observations[:, :58][i])
                            dataset['terminals'].append(terminals[i])
                            dataset['rewards'].append(reward[i])
                            if dy_length % traj_length == 0:
                                dataset['timeouts'].append(True)
                                dy_length = 0
                            else:
                                dataset['timeouts'].append(False)

        dataset['actions'] = np.array(dataset['actions'])
        dataset['observations'] = np.array(dataset['observations'])
        dataset['terminals'] = np.array(dataset['terminals'])
        dataset['rewards'] = np.array(dataset['rewards'])
        dataset['timeouts'] = np.array(dataset['timeouts'])

        print('mean_return: {}'.format(np.mean(total_returns)))
        
        return dataset
    
class OnlineReplayBuffer:
    _device: torch.device
    _state: np.ndarray
    _action: np.ndarray
    _reward: np.ndarray
    _next_state: np.ndarray
    _next_action: np.ndarray
    _not_done: np.ndarray
    _return: np.ndarray
    _size: int

    def __init__(
        self, 
        device: torch.device, 
        state_dim: int, action_dim: int, max_size: int, percentage: float
    ) -> None:
        self._percentage = percentage
        self._device = device


        self._state = np.zeros((max_size, state_dim))
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._next_state = np.zeros((max_size, state_dim))
        self._next_action = np.zeros((max_size, action_dim))
        self._not_done = np.zeros((max_size, 1))
        self._return = np.zeros((max_size, 1))
        self._advantage = np.zeros((max_size, 1))

        self._size = 0


    def store(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_p: np.ndarray,
        a_p: np.ndarray,
        not_done: bool
    ) -> None:

        self._state[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._next_state[self._size] = s_p
        self._next_action[self._size] = a_p
        self._not_done[self._size] = not_done
        self._size += 1


    def compute_return(
        self, gamma: float
    ) -> None:

        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc='Computing the returns'):
            self._return[i] = self._reward[i] + gamma * pre_return * self._not_done[i]
            pre_return = self._return[i]

        print('max_return: {}; min_return: {}; mean_return: {}'.format(np.max(self._return), np.min(self._return), np.mean(self._return)))

    def compute_advantage(
        self, gamma:float, lamda: float, value
    ) -> None:
        delta = np.zeros_like(self._reward)

        pre_value = 0
        pre_advantage = 0

        for i in tqdm(reversed(range(self._size)), 'Computing the advantage'):
            current_state = torch.FloatTensor(self._state[i]).to(self._device)
            current_value = value(current_state).cpu().data.numpy().flatten()

            delta[i] = self._reward[i] + gamma * pre_value * self._not_done[i] - current_value
            self._advantage[i] = delta[i] + gamma * lamda * pre_advantage * self._not_done[i]

            pre_value = current_value
            pre_advantage = self._advantage[i]

        self._advantage = (self._advantage - self._advantage.mean()) / (self._advantage.std() + CONST_EPS)


    def shuffle(self,):
        indices = np.arange(self._state.shape[0])
        np.random.shuffle(indices)
        self._state = self._state[indices]
        self._action = self._action[indices]
        self._reward = self._reward[indices]
        self._next_state = self._next_state[indices]
        self._next_action = self._next_action[indices]
        self._not_done = self._not_done[indices]
        self._return = self._return[indices]
        self._advantage = self._advantage[indices]

    def sample_all(self,):
        
        return {
            "observations": self._state[:self._size].copy(),
            "actions": self._action[:self._size].copy(),
            "next_observations": self._next_state[:self._size].copy(),
            "terminals": 1. - self._not_done[:self._size].copy(),
            "rewards": self._reward[:self._size].copy()
        }
    def sample_sequence(self, length = 512):
        
        return {
            "observations": self._state[:length].copy(),
            "actions": self._action[:length].copy(),
            "next_observations": self._next_state[:length].copy(),
            "terminals": 1. - self._not_done[:length].copy(),
            "rewards": self._reward[:length].copy()
        }
    def sample_aug_all(self,):
        self._state = np.concatenate((self._state, self._aug_state), axis=0)
        self._action = np.concatenate((self._action, self._action), axis = 0)
        self._next_state = np.concatenate((self._next_state, self._aug_next_state), axis = 0)
        self._not_done = np.concatenate((self._not_done, self._not_done), axis = 0)
        self._reward = np.concatenate((self._reward, self._reward), axis = 0)
        indices = np.arange(self._state.shape[0])
        np.random.shuffle(indices)
        return {
            "observations": self._state[indices].copy(),
            "actions": self._action[indices].copy(),
            "next_observations": self._next_state[indices].copy(),
            "terminals": 1. - self._not_done[indices].copy(),
            "rewards": self._reward[indices].copy()
        }
    
    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, int(self._size * self._percentage), size=batch_size)

        return (
            torch.FloatTensor(self._state[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._next_state[ind]).to(self._device),
            torch.FloatTensor(self._next_action[ind]).to(self._device),
            torch.FloatTensor(self._not_done[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._advantage[ind]).to(self._device)
        )
    def sample_aug_state(self, batch_size: int):
        self._state = np.concatenate((self._state, self._aug_state), axis=0)
        ind = np.random.randint(0, int(self._size * 2), size=batch_size)
        return (
            torch.FloatTensor(self._state[ind]).to(self._device)
        )


    def augmentaion(self, alpha = 0.75, beta = 1.25):
        z = np.random.uniform(low=alpha, high=beta, size=self._state.shape)
        self._aug_state = deepcopy(self._state) * z
        self._aug_next_state = deepcopy(self._next_state) * z


    def sample_percentage_eval(
        self, batch_size: int
    ) -> tuple:
        ind = np.random.randint(int(self._size *  self._percentage), self._size, size=batch_size)

        return (
            torch.FloatTensor(self._state[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._next_state[ind]).to(self._device),
            torch.FloatTensor(self._next_action[ind]).to(self._device),
            torch.FloatTensor(self._not_done[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._advantage[ind]).to(self._device)
        )

class OfflineReplayBuffer(OnlineReplayBuffer):

    def __init__(
        self, device: torch.device, 
        state_dim: int, action_dim: int, max_size: int, percentage: float = 1.
    ) -> None:
        super().__init__(device, state_dim, action_dim, max_size, percentage)


    def load_dataset(
        self, dataset: dict, clip = False
    ) -> None:
        if clip:
            dataset['actions'] = np.clip(dataset['actions'], -32., 32.)
        self._state = dataset['observations'][:-1, :]

        self._action = dataset['actions'][:-1, :]
        self._reward = dataset['rewards'].reshape(-1, 1)[:-1, :]
        self._next_state = dataset['observations'][1:, :]
        self._next_action = dataset['actions'][1:, :]
        self._not_done = 1. - (dataset['terminals'].reshape(-1, 1)[:-1, :])# | dataset['timeouts'].reshape(-1, 1)[:-1, :])
        self._size = len(dataset['actions']) - 1

    def reward_normalize(self, gamma = 0.99, scaling = 'dynamic'): # dynamic/normsal/number
        if scaling == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma)
            rewards = self._reward.flatten()
            for i, not_done in enumerate(self._not_done.flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self._reward = rewards.reshape(-1, 1)
            return reward_norm
        elif scaling == 'normal':
            print('use normal reward scaling')
            normalized_rewards = normalize(self._state, self._action, deepcopy(self._reward.flatten()), self._not_done.flatten(), 1 - self._not_done.flatten(), self._next_state)
            self._reward = normalized_rewards.reshape(-1, 1)

        elif scaling == 'number':
            print('use a fixed number reward scaling')
            self._reward = self._reward * 0.1
        else:
            print('donnot use any reward scaling')
            self._reward = self._reward 


    def normalize_state(
        self
    ) -> tuple:

        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)

if __name__ == '__main__':
    path = 'dataset/stand_offline'
    data_loader = load_dataset()
    dataset = data_loader.load_file(path)
    print('max_action:{}'.format(dataset['observations'].shape[1]))