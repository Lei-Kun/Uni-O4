import h5py
import os
import numpy as np
import time
import torch
from reward_aliengo_new import reward_aliengo
from tqdm import tqdm
# class HDF5_recorder:
#     def __init__(self):
#         self.folder_name = "dataset/"+time.strftime("%Y%m%d-%H%M%S")
#         os.makedirs(self.folder_name, exist_ok=True)
#         self._reset_data()
#         print("HDF5 init")

#     def _reset_data(self):
#         self.action_data = []
#         self.states_data = []
#         self.timestamp = time.strftime("%Y%m%d-%H%M%S")
#         print("HDF5 reset data")
#         print("———————————————————————————————————————————————————————————")

#     def record_step(self, states, action):
#         self.action_data.append(action)
#         self.states_data.append(states)

#     def save_file(self):
#         if((self.action_data==[]) or (self.states_data==[])):
#             return
#         actions = self.action_data
#         states = self.states_data
#         print("action:", actions)
#         print("states:", states)
#         with h5py.File("{}/{}.hdf5".format(self.folder_name, self.timestamp), 'w') as f:
#             f.create_dataset("actions", data=np.array(actions))
#             f.create_dataset("states", data=np.array(states))
#             print("save file: {}.hdf5".format(self.timestamp))
#         self._reset_data()

class HDF5_recorder:
    def __init__(self, max_steps, state_dim, obs_dim, action_dim, history_length = 5, device = 'cuda:0'):
        self.device = device
        self.folder_name = "dataset/"+time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.folder_name, exist_ok=True)
        self._reset_data()
        print("HDF5 init")
        #state and next_state are history observations
        self.s = np.zeros((max_steps, obs_dim * history_length))
        self.a = np.zeros((max_steps, action_dim))
        self.s_ = np.zeros((max_steps, obs_dim * history_length))
        self.r = np.zeros((max_steps, 1))
        self.a_logprob = np.zeros((max_steps, action_dim))
        self.dw = np.zeros((max_steps, 1))
        self.done = np.zeros((max_steps, 1))

        #pure state which contains the realsense sensor's data and the robot's obs  
        self.obs = np.zeros((max_steps, state_dim))
        self.obs_ = np.zeros((max_steps, state_dim))
        self.count = 0


    def _reset_data(self):
        self.count = 0
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("HDF5 reset data")
        print("———————————————————————————————————————————————————————————")

    def record_step(self, s, a, s_, a_logprob, dw, done, obs, obs_):
        self.s[self.count] = s
        self.a[self.count] = a
        self.s_[self.count] = s_
        self.a_logprob[self.count] = a_logprob
        self.dw[self.count] = dw
        self.done[self.count] = done

        self.obs[self.count] = obs
        self.obs_[self.count] = obs_
        # self.count += 1 # count is recorder by outer loop
    
    def compute_reward(self,reward_scaling):
        reward_calculator = reward_aliengo(action_dim=12)
        # print('obs_: {}, action: {}'.format(self.obs_, self.a))
        print('count num: {}'.format(self.count))
        for i in range(self.count):
            reward_calculator.load_data(self.obs_[i], self.a[i])
            reward = reward_calculator.calculate_reward()
            # print('reward: {}'.format(reward))
            self.r[i] = reward
            if self.done[i]:
                print('A trajectory finished, initialize a new reward caculator')
                reward_calculator = reward_aliengo(action_dim=12)
        rewards = self.r.flatten()

        # print('rewards:', rewards)
        for i, not_done in enumerate(1. - self.done.flatten()):
            if not not_done:
                reward_scaling.reset()
            else:
                rewards[i] = reward_scaling(rewards[i])
        self.r = rewards.reshape(-1, 1)
        print('finished the reward computation')
        return np.sum(self.r)/np.sum(self.done)

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.a, dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)
        # for i in range(5):
        #     s[:, i*58 +9] = 0.03
        #     s_[:, i*58 +9] = 0.03



        return s, a, a_logprob, r, s_, dw, done
    def save_file(self):
        if self.count == 0:
            return
        else:
            with h5py.File("{}/{}.hdf5".format(self.folder_name, self.timestamp), 'w') as f:
                f.create_dataset("actions", data=np.array(self.a[:self.count, :]))
                f.create_dataset("states", data=np.array(self.obs[:self.count, :]))
                f.create_dataset("next_states", data=np.array(self.obs_[:self.count, :]))
                f.create_dataset("done", data=np.array(self.done[:self.count, :]))
                print("save file: {}.hdf5".format(self.timestamp))
            # self._reset_data()