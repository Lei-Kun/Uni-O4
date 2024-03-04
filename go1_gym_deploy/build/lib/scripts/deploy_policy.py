import glob
import pickle as pkl
import lcm
import sys

from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, args, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    print('file:', dirs)
    logdir = sorted(dirs)[-1]
    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(logdir+"/parameters.pkl")
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
    if args.deploy_policy == 'sim':
        policy = load_policy_sim(logdir)
    elif args.deploy_policy == 'offline':
        policy = load_policy_offline(logdir)
    elif args.deploy_policy == 'online':
        policy = load_policy_online(logdir)
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
        max_steps = 10000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

# def load_policy(logdir):
#     body = torch.jit.load(logdir + '/checkpoints/body_latest_cpu.jit')
#     import os
#     adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest_cpu.jit')

#     def policy(obs, info):
#         i = 0
#         latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
#         action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
#         info['latent'] = latent
#         return action
#     return policy

# 1-----for online fine-tuned policy deployment-----
def load_policy_online(logdir):
    from bppo import BehaviorCloning
    bc = BehaviorCloning("cuda:0", 58 * 5, [512, 256, 128], 3, 12, 1e-4, 512)
    bc.load(logdir + '/online_finetuned/pi_latest.pt')

    def policy(obs, info):
        action = bc._policy.mean(obs["obs_history"].to('cuda:0'))
        info['latent'] = 0
        return action
    return policy

# 1-----for offline fine-tuned policy deployment-----
def load_policy_offline(logdir):
    from bppo import BehaviorCloning
    bc = BehaviorCloning("cuda:0", 58 * 5, [512, 256, 128], 3, 12, 1e-4, 512)
    bc.load(logdir + '/offline_finetuned/pi_0.pt')

    def policy(obs, info):
        action = bc._policy.mean(obs["obs_history"].to('cuda:0'))
        info['latent'] = 0
        return action
    return policy

# 0-----collect data using pre-trained policy in simulator (1,000,000 steps)-----
def load_policy_sim(logdir):
    from actor_critic import ActorCritic
    actor_critic = ActorCritic(58, 0, 5 * 58, 12) # num obs; num privileged obs; num history; num actions
    load_dir = logdir + '/checkpoints/ac_weights_000300.pt'
    print('policy loaded from: {}'.format(str(load_dir)))
    weights = torch.load(logdir + '/checkpoints/ac_weights_000300.pt')

    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to('cuda:0')
    def sample_policy(obs, info):
        i = 0
        action = actor_critic.mean(obs["obs_history"].to('cuda:0'))
        info['latent'] = 0
        return action
    return sample_policy

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--deploy_policy", type=str, default='sim', help="choice: sim/offline/online trained policy")

    args = parser.parse_args()
    label = "gait-conditioned-agility/1000wan_ft/train"
    experiment_name = "example_experiment"
    load_and_run_policy(label, experiment_name=experiment_name, args=args, max_vel=1.5, max_yaw_vel=1.5)
