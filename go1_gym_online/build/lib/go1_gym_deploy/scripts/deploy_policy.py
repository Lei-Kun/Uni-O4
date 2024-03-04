import glob
import pickle as pkl
import lcm
import sys

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner, CONT
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *

import pathlib
print(CONT)
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[-1]
    with open(logdir+"/parameters_cpu.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(logdir+"/parameters_cpu.pkl")
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())


    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy(logdir)

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

# def run_sample(dirs, experiment_name, max_vel=1.0, max_yaw_vel=1.0, max_steps = 2048, state_dim=76, obs_dim = 58, action_dim=12, device = 'cuda:0'):
#     # load agent
    
#     logdir = sorted(dirs)[-1]
#     with open(logdir+"/parameters_cpu.pkl", 'rb') as file:
#         pkl_cfg = pkl.load(file)
#         print(logdir+"/parameters_cpu.pkl")
#         print(pkl_cfg.keys())
#         cfg = pkl_cfg["Cfg"]
#         print(cfg.keys())


#     se = StateEstimator(lc)

#     control_dt = 0.02
#     command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

#     hardware_agent = LCMAgent(cfg, se, command_profile)
#     se.spin()

#     from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
#     hardware_agent = HistoryWrapper(hardware_agent)

#     # policy = load_policy(logdir)

#     # load runner
#     root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
#     pathlib.Path(root).mkdir(parents=True, exist_ok=True)
#     deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None, log_root=f"{root}/{experiment_name}",
#                                          max_steps=max_steps, state_dim=state_dim, obs_dim = obs_dim, action_dim=action_dim, device=device)
#     deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
#     deployment_runner.add_command_profile(command_profile)
#     return deployment_runner


#     deployment_runner.add_policy(policy)
#     deployment_runner.add_command_profile(command_profile)

#     # if len(sys.argv) >= 2:
#     #     max_steps = int(sys.argv[1])
#     # else:
#     #     max_steps = 10000000
#     print(f'max steps {max_steps}')

#     replay_buffer = deployment_runner.run(max_steps=max_steps, logging=True)
#     return replay_buffer


# def load_policy(logdir):
#     body = torch.jit.load(logdir + '/best1/pipolicy_0.jit')
#     import os
#     adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest_cpu.jit')

#     def policy(obs, info):
#         i = 0
#         latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
#         action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
#         info['latent'] = latent
#         return action
#     return policy


def load_policy(logdir):
    from bppo import BehaviorCloning
    bc = BehaviorCloning("cuda:0", 58 * 5, [512, 256, 128], 3, 12, 1e-4, 512)
    bc.load(logdir + '/best_iql/pi_1.pt')

    def policy(obs, info):
        dist  = bc._policy(obs["obs_history"].to('cuda:0'))
        a = dist.sample() 
        a_logprob = dist.log_prob(a) 
        info['latent'] = 0
        return a, a_logprob
    return policy

# def load_policy(logdir):
#     from go1_gym_deploy.scripts.actor_critic import ActorCritic
#     actor_critic = ActorCritic(58, 0, 5 * 58, 12) # num obs; num privileged obs; num history; num actions
#     weights = torch.load(logdir + '/best1/pipolicy_0.pt')
#     actor_critic.load_state_dict(state_dict=weights)

#     def sample_policy(obs, info):
#         i = 0
#         action = actor_critic.deploy_sample(obs["obs_history"].to('cpu'))
#         info['latent'] = 0
#         return action
#     return sample_policy

if __name__ == '__main__':
    label = "gait-conditioned-agility/1000wan_ft/train"

    experiment_name = "example_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, max_vel=2.0, max_yaw_vel=4.)
