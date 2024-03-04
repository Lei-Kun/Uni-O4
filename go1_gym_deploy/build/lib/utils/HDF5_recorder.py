import h5py
import os
import numpy as np
import time

class HDF5_recorder:
    def __init__(self):
        self.folder_name = "dataset/"+time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.folder_name, exist_ok=True)
        self._reset_data()
        print("HDF5 init")

    def _reset_data(self):

        self.action_data = []
        self.states_data = []
        self.next_states_data = []
        self.dones = []
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("HDF5 reset data")
        print("———————————————————————————————————————————————————————————")

    def record_step(self, state, action, next_state, done):
        print('recoder step')
        print('state: {}'.format(state) )
        print('action: {}'.format(action) )
        self.action_data.append(action)
        self.states_data.append(state)
        self.next_states_data.append(next_state)
        self.dones.append(done)
    def save_file(self):
        if((self.action_data==[]) or (self.states_data==[])):
            return
        actions = np.array(self.action_data)
        states = np.array(self.states_data)
        next_states = np.array(self.next_states_data)
        dones = np.array(self.dones)
        with h5py.File("{}/{}.hdf5".format(self.folder_name, self.timestamp), 'w') as f:
            f.create_dataset("actions", data=actions)
            assert states.shape[1] == 76, 'the shape is not aligned'
            f.create_dataset("states", data=states)
            f.create_dataset("next_states", data=next_states)
            f.create_dataset("dones", data=dones)
            print("save file: {}.hdf5".format(self.timestamp))
        self._reset_data()
