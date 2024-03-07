
import numpy as np
import torch


class reward_aliengo(object):
    def __init__(self, action_dim, obs_dim = 51):
        super().__init__()
        self.obs_dim = obs_dim
        self.action = torch.zeros((action_dim))
        self.last_action = torch.zeros((action_dim))
        self.cmd =  torch.zeros((15))
        self.dof_vel = torch.zeros((12))
        self.dof = torch.zeros((12))

    def load_data(self, state, action):
        state, action = torch.FloatTensor(state), torch.FloatTensor(action)
        self.last_last_action = self.last_action
        self.last_action = self.action
        self.action = action

        self.last_dof_vel = self.dof_vel
        self.projected_gravity = state[:3]
        self.cmd = state[3:18]
        self.dof = state[18:30]
        self.dof_vel = state[30:42]

        root_states, root_ang_states = [], []

        for i in range(9):
            root_states.append(state[i+58])
        self.root_states = torch.stack(root_states)
        
        for i in range(9):
            root_ang_states.append(state[i+67])
        self.root_ang_states = torch.stack(root_ang_states)

        self.rpy = self.root_ang_states[:3]
        self.base_ang_vel = self.root_ang_states[3:6]
        self.base_ang_acc = self.root_ang_states[6:9]

        
    def calculate_reward(self,):
        positive_reward = (0.8* self._reward_tracking_lin_vel()+0.7* self._reward_tracking_ang_vel())
        negative_reward = (-0.02)*self._reward_lin_vel_z()+(-0.001)* self._reward_ang_vel_xy()+(-0.00005)*self._reward_torques()+(-2.5e-7)*self._reward_dof_acc()+(-0.01)*self._reward_action_rate()+(-1e-4)*self._reward_dof_vel()+(-0.1)*(self._reward_action_smoothness_1()+self._reward_action_smoothness_2())
        reward = positive_reward * torch.exp(negative_reward)/10
        return reward.numpy()


    # ------------ reward functions----------------

    # def _reward_base_height(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     height_error = torch.square(self.env.cfg.rewards.base_height_target - self.env.root_states[:, 0, 2])

    #     return height_error
    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.cmd[:2] - self.root_states[3:5]))
        return torch.exp(-lin_vel_error / 0.25)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.cmd[2] - self.base_ang_vel[2])
        return torch.exp(-ang_vel_error / 0.25)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.root_states[5])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:2]))

    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     # return torch.sum(torch.square(self.env.projected_gravity[:, :2]))
    #     return torch.sum(torch.square(self.rpy[0])+torch.square(self.rpy[1]))

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.action))

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / 0.02))

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_action - self.action))

    # def _reward_dof_pos(self):
    #     # Penalize dof positions
    #     return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=0)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square((self.action - self.last_action)*0.5)
        diff = diff * (self.last_action != 0)  # ignore first step
        return torch.sum(diff, dim=0)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.action - 2 * self.last_action + self.last_last_action)
        diff = diff * (self.last_action != 0)  # ignore first step
        diff = diff * (self.last_last_action != 0)  # ignore second step
        return torch.sum(diff, dim=0)


    # def _reward_orientation_control(self):
    #     # Penalize non flat base orientation
    #     roll_pitch_commands = self.cmd[10:12]
    #     quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
    #                                      torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
    #     quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
    #                                       torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

    #     desired_base_quat = quat_mul(quat_roll, quat_pitch)
    #     desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

    #     return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]))
