
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
from ml_logger import logger

from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR
import glob
import os
def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

class eval_go1(object):
    def __init__(
        self, headless = False) -> None:
        super().__init__()

        label = "gait-conditioned-agility/2023-08-04/train"
        self.env = self.load_env(label, headless=headless)

    def load_env(self, label, headless=False):
        dirs = glob.glob(f"runs/{label}/*")
        logdir = sorted(dirs)[-1]

        with open(logdir + "/parameters.pkl", 'rb') as file:
            pkl_cfg = pkl.load(file)
            print("cfg keys:",pkl_cfg.keys())
            cfg = pkl_cfg["Cfg"]
            print(cfg.keys())

            for key, value in cfg.items():
                if hasattr(Cfg, key):
                    for key2, value2 in cfg[key].items():
                        setattr(getattr(Cfg, key), key2, value2)

        # turn off DR for evaluation script
        # Cfg.env.num_observations = 51
        Cfg.domain_rand.push_robots = False
        Cfg.domain_rand.randomize_friction = False
        Cfg.domain_rand.randomize_gravity = False
        Cfg.domain_rand.randomize_restitution = False
        Cfg.domain_rand.randomize_motor_offset = False
        Cfg.domain_rand.randomize_motor_strength = False
        Cfg.domain_rand.randomize_friction_indep = False
        Cfg.domain_rand.randomize_ground_friction = False
        Cfg.domain_rand.randomize_base_mass = False
        Cfg.domain_rand.randomize_Kd_factor = False
        Cfg.domain_rand.randomize_Kp_factor = False
        Cfg.domain_rand.randomize_joint_friction = False
        Cfg.domain_rand.randomize_com_displacement = False

        Cfg.env.num_recording_envs = 1
        Cfg.env.num_envs = 1
        Cfg.terrain.num_rows = 5
        Cfg.terrain.num_cols = 5
        Cfg.terrain.border_size = 0
        Cfg.terrain.center_robots = True
        Cfg.terrain.center_span = 1
        Cfg.terrain.teleport_robots = True

        Cfg.domain_rand.lag_timesteps = 6
        Cfg.domain_rand.randomize_lag_timesteps = True
        Cfg.control.control_type = "actuator_net"

        from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

        env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
        env = HistoryWrapper(env)

        # load policy
        from ml_logger import logger
        from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

        # policy = load_policy(logdir)

        return env


    def play_go1(self, policy, mean, std, eval_episode_num = 5):
        all_returns = []
        for bppo in policy:
            pi_returns = []
            policy = bppo._policy


            num_eval_steps = 250
            gaits = {"pronking": [0, 0, 0],
                    "trotting": [0.5, 0, 0],
                    "bounding": [0, 0.5, 0],
                    "pacing": [0, 0, 0.5]}

            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.5, 0.0, 0.0
            body_height_cmd = 0.0
            step_frequency_cmd = 3.0
            gait = torch.tensor(gaits["trotting"])
            footswing_height_cmd = 0.08
            pitch_cmd = 0.0
            roll_cmd = 0.0
            stance_width_cmd = 0.25

            measured_x_vels = np.zeros(num_eval_steps)
            target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
            joint_positions = np.zeros((num_eval_steps, 12))
            list = [0.2, 0.4, 0.6, 0.8,]
            for _ in range(eval_episode_num):
                obs = self.env.reset()
                returns = 0
                x_vel_cmd, y_vel_cmd, yaw_vel_cmd = np.random.uniform(-2., 2.), 0, np.random.uniform(-2., 2.)
                for i in tqdm(range(num_eval_steps)):
                    with torch.no_grad():
                        dist = policy((obs['obs_history'] - mean) / std)
                        actions = dist.mean
                    self.env.commands[:, 0] = x_vel_cmd
                    self.env.commands[:, 1] = y_vel_cmd
                    self.env.commands[:, 2] = yaw_vel_cmd
                    self.env.commands[:, 3] = body_height_cmd
                    self.env.commands[:, 4] = step_frequency_cmd
                    self.env.commands[:, 5:8] = gait
                    self.env.commands[:, 8] = 0.5
                    self.env.commands[:, 9] = footswing_height_cmd
                    self.env.commands[:, 10] = pitch_cmd
                    self.env.commands[:, 11] = roll_cmd
                    self.env.commands[:, 12] = stance_width_cmd
                    obs, rew, done, info = self.env.step(actions)
                    returns += rew.cpu().detach().numpy()
                    measured_x_vels[i] = self.env.base_lin_vel[0, 0]
                    joint_positions[i] = self.env.dof_pos[0, :].cpu()

                    
                pi_returns.append(returns)
            all_returns.append(np.array(pi_returns).mean())
            # plot target and measured forward velocity
            # from matplotlib import pyplot as plt
            # fig, axs = plt.subplots(2, 1, figsize=(12, 5))
            # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
            # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
            # axs[0].legend()
            # axs[0].set_title("Forward Linear Velocity")
            # axs[0].set_xlabel("Time (s)")
            # axs[0].set_ylabel("Velocity (m/s)")

            # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
            # axs[1].set_title("Joint Positions")
            # axs[1].set_xlabel("Time (s)")
            # axs[1].set_ylabel("Joint Position (rad)")

            #plt.tight_layout()
            #plt.show()
        return np.array(all_returns).flatten()
    
    def prove_real_data(self, buffer):
        all_returns = []


        #policy = GaussPolicyMLP(57, [512, 256, 128], 12)
        num_eval_steps = 200
        gaits = {"pronking": [0, 0, 0],
                "trotting": [0.5, 0, 0],
                "bounding": [0, 0.5, 0],
                "pacing": [0, 0, 0.5]}

        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.5, 0.0, 0.0
        body_height_cmd = 0.0
        step_frequency_cmd = 3.0
        gait = torch.tensor(gaits["trotting"])
        footswing_height_cmd = 0.08
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25

        measured_x_vels = np.zeros(num_eval_steps)
        target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
        joint_positions = np.zeros((num_eval_steps, 12))

        obs = self.env.reset()
        
        returns = 0
        transitions = buffer.sample_sequence(num_eval_steps)
        actions = transitions['actions']
        actions = torch.FloatTensor(actions)
        print(actions)
        for i in tqdm(range(num_eval_steps)):
            action = actions[i]


            self.env.commands[:, 0] = x_vel_cmd
            self.env.commands[:, 1] = y_vel_cmd
            self.env.commands[:, 2] = yaw_vel_cmd
            self.env.commands[:, 3] = body_height_cmd
            self.env.commands[:, 4] = step_frequency_cmd
            self.env.commands[:, 5:8] = gait
            self.env.commands[:, 8] = 0.5
            self.env.commands[:, 9] = footswing_height_cmd
            self.env.commands[:, 10] = pitch_cmd
            self.env.commands[:, 11] = roll_cmd
            self.env.commands[:, 12] = stance_width_cmd

            obs, rew, done, info = self.env.step(action.unsqueeze(0))
            returns += rew
            measured_x_vels[i] = self.env.base_lin_vel[0, 0]
            joint_positions[i] = self.env.dof_pos[0, :].cpu()
        all_returns.append(returns)
        # plot target and measured forward velocity
        # from matplotlib import pyplot as plt
        # fig, axs = plt.subplots(2, 1, figsize=(12, 5))
        # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
        # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
        # axs[0].legend()
        # axs[0].set_title("Forward Linear Velocity")
        # axs[0].set_xlabel("Time (s)")
        # axs[0].set_ylabel("Velocity (m/s)")

        # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
        # axs[1].set_title("Joint Positions")
        # axs[1].set_xlabel("Time (s)")
        # axs[1].set_ylabel("Joint Position (rad)")

        #plt.tight_layout()
        #plt.show()
        return all_returns
        

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(policy, headless=False)
