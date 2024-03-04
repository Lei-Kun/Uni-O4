import torch
import numpy as np
from tensorboardX import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo import PPO
import os
import glob
from tqdm import tqdm
import time
from collections import deque
from buffer import OfflineReplayBuffer
from go1_gym_online.go1_gym_deploy.scripts.deploy_policy import run_sample
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--pi_hidden_dim", type=int, default=[512, 256, 128], help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--v_hidden_dim", type=int, default=512, help="The number of neurons in hidden layers of the neural network")
    
    parser.add_argument("--depth", type=int, default=2, help="The number of layer in MLP")
    parser.add_argument("--lr_a", type=float, default=2e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=30, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=bool, default=False, help="tanh activation function")
    parser.add_argument("--is_from_bppo", type=bool, default=True, help="training from scratch or fine-tune from bppo")
    parser.add_argument("--is_load_value", type=bool, default=True, help="training from scratch or fine-tune from bppo")
    parser.add_argument("--is_shuffle", type=bool, default=False, help="shuffle the dataset")
    
    parser.add_argument("--env_name", type=str, default='go1_realworld', help="training env")
    parser.add_argument("--seed", type=int, default=1, help="run with a fixed seed")
    parser.add_argument("--pi_load_path", type=str, default='/home/lk/mobile-main/logs_clip', help="training env")
    parser.add_argument("--v_load_path", type=str, default='/home/lk/mobile-main/logs_clip', help="training env")
    parser.add_argument("--r_scale", default=0.1 , type=float, help='the weight of Q loss')
    parser.add_argument("--gpu", default=0, type=int, help='id of gpu')
    parser.add_argument("--path", default='logs', type=str, help='save dir')
    parser.add_argument("--is_clip_value", default=True, type=bool, help='Asynchronous Update: train critic then update policy')
    parser.add_argument("--scale_strategy", default='dynamic', type=str, help='reward scaling technique: dynamic/normal/number(0.1)')
    parser.add_argument("--from_scratch", default=False, type=bool, help='training from the initial policy (offline finetuned) of the checkpoints or latest training')
    parser.add_argument("--date", default='0808', type=str, help='reward scaling technique: dynamic/normal/number(0.1)')
    args = parser.parse_args()
    if args.scale_strategy == 'dynamic':
        args.use_reward_scaling = True
        args.r_scale = 1.

    dirs = glob.glob(f"checkpoints/*")
    logdir = sorted(dirs)[-1]

    if not args.from_scratch:
        lrs = np.loadtxt(os.path.join('{}'.format(logdir), 'best_bppo.csv'), delimiter=',')
        args.lr_a = lrs[0]
        args.lr_c = lrs[1]
        print('load from previous checkpoint: {} \ lr_a: {}, lr_c: {}'.format(logdir, args.lr_a, args.lr_c))
    seed=args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    args.state_dim = 58 * 5
    args.action_dim = 12
    args.max_episode_steps = args.batch_size  # Maximum number of steps per episode
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))
    print("max_episode_steps={}".format(args.max_episode_steps))
    print('v_hidden_width: {}'.format(args.v_hidden_dim))



    path = os.path.join(args.path, args.env_name, str(args.seed))
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

    os.makedirs(os.path.join(path, current_time))
    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    if args.use_reward_scaling:
        buffer_save_path = os.path.join('dataset_{}.pt'.format(args.date))
        dataset = torch.load(buffer_save_path)
        offline_buffer = OfflineReplayBuffer(device, args.state_dim, args.action_dim, len(dataset['actions']) - 1, 1.)
        offline_buffer.load_dataset(dataset=dataset)
        reward_scaling = offline_buffer.reward_normalize(args.gamma, args.scale_strategy)
        
    agent = PPO(args, device)


    print('checkpoint path', logdir)
    if args.from_scratch:
        v_path = 'checkpoints/offline_finetuned/value_{}.pt'.format(args.date)
        pi_path = 'checkpoints/offline_finetuned/pi_0.pt'
    else:
        v_path = os.path.join(logdir, 'value_latest.pt')
        pi_path = os.path.join(logdir, 'pi_latest.pt')
    agent.load_pi_value(pi_path=pi_path, value_path=v_path)
    
    label = "gait-conditioned-agility/1000wan_ft/train"
    dirs = glob.glob(f"go1_gym_online/runs/{label}/*")
    experiment_name = "example_experiment"
    deployment_runner = run_sample(dirs, experiment_name, max_vel=1.0, max_yaw_vel=1.0, max_steps = args.max_episode_steps, device=device)

    file = 'checkpoints/pretrained_pi_v_{}'.format(current_time)

    os.makedirs(file, exist_ok=True)
    actor_losses, critic_losses, episode_rewards = [], [], []
    grad_steps = int(args.max_train_steps / args.batch_size)
    print(grad_steps)

    with tqdm(total=grad_steps) as pbar:
        iterations = 0
        while total_steps < args.max_train_steps:

            total_steps += args.batch_size
            deployment_runner.add_policy(agent.actor.sample_a_logprob)
            replay_buffer = deployment_runner.run(max_steps=args.max_episode_steps, logging=True)
            episode_reward = replay_buffer.compute_reward(reward_scaling)
            print('episode reward', episode_reward)
            episode_rewards.append(episode_reward)
            actor_loss, critic_loss = agent.update(replay_buffer, total_steps)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            replay_buffer._reset_data()
            replay_buffer.count = 0
            agent.save_pi_value('{}/pi_latest.pt'.format(file), '{}/value_latest.pt'.format(file))
            np.savetxt(os.path.join('{}'.format(file), 'best_bppo.csv'), [agent.lr_a_now, agent.lr_c_now], fmt='%f', delimiter=',')
            if total_steps % args.save_freq == 0:
                agent.save_pi_value('{}/pi_{}.pt'.format(file, total_steps), '{}/value_{}.pt'.format(file, total_steps))
            iterations += 1
            pbar.update(1)
            print("evaluate_num:{} \t evaluate_reward:{}".format(iterations, np.mean(episode_rewards[int(-args.evaluate_freq):])))
            print('actor_loss: {}, critic_loss: {}'.format(np.mean(actor_losses[int(-args.evaluate_freq):]), np.mean(critic_losses[int(-args.evaluate_freq):])))

        if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                # evaluate_rewards.append(evaluate_reward)
                