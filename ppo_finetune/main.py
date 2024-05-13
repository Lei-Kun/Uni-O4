import torch
import numpy as np
from tensorboardX import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from online_buffer import ReplayBuffer
from ppo import PPO
import os
import d4rl
from utils import evaluate_policy, load_config
from tqdm import tqdm
import time
from net import ValueLearner
from collections import deque
from offline_buffer import OfflineReplayBuffer
import glob
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    # base hyperparameters and settings for online ppo
    parser.add_argument("--env_name", type=str, default='walker2d-medium-replay-v2', help="training env")
    parser.add_argument("--seed", type=int, default=1, help="run with a fixed seed")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--v_hidden_width", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--gpu", default=3, type=int, help='id of gpu')
    parser.add_argument("--path", default='logs', type=str, help='save dir')
    parser.add_argument("--depth", type=int, default=3, help="The number of layer in MLP")
    parser.add_argument("--v_depth", type=int, default=3, help="The number of layer in MLP")
    parser.add_argument("--lr_a", type=float, default=2e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.05, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=30, help="PPO parameter")

    # tricks usually used in online ppo
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")# if use here, please retrain the value function
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=bool, default=False, help="Trick 10: tanh activation function")
    # tricks usually used in online ppo

    # setup for fine-tuning
    parser.add_argument("--is_from_bppo", type=bool, default=True, help="training from scratch or fine-tune from bppo")
    parser.add_argument("--is_load_value", type=bool, default=True, help="training from scratch or fine-tune from bppo")
    parser.add_argument("--is_shuffle", type=bool, default=False, help="shuffle the dataset")
    parser.add_argument("--pi_load_path", type=str, default='pi_', help="training env")
    parser.add_argument("--v_load_path", type=str, default='/home/lk/mobile-main/logs_clip', help="training env")
    parser.add_argument("--r_scale", default=1., type=float, help='the weight of Q loss')
    parser.add_argument("--is_clip_value", default=True, type=bool, help='Asynchronous Update: train critic then update policy')
    parser.add_argument("--scale_strategy", default=None, type=str, help='reward scaling technique: dynamic/normal/number(0.1)')
    parser.add_argument("--is_decay_pi", default=False, type=bool, help='decay the update of target policy')
    parser.add_argument("--tau", default=5e-3, type=float)
    parser.add_argument("--std_upper_bound", default=0, type=float)

    args = parser.parse_args()
    if args.scale_strategy == 'dynamic':
        args.use_reward_scaling = True
    elif args.scale_strategy == 'number':
        args.r_scale = 0.1
    
    # align config params with offline phase, e.g., use_state_norm, use_tanh etc.
    load_path = os.path.join('../logs', args.env_name, str(args.seed))
    config_path = os.path.join(load_path, args.pi_load_path)
    dirs = glob.glob(f"{config_path}/*")
    logdir = sorted(dirs)[-1]
    print(logdir)
    args = load_config(logdir, args)

    env_name=args.env_name
    number=1
    seed=args.seed

    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))
    print('v_hidden_width: {}'.format(args.v_hidden_width))
    path = os.path.join(args.path, args.env_name, str(args.seed))
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

    # summarywriter logger
    comment = args.env_name + '_' + str(args.seed)
    logger_path = os.path.join(path, current_time)
    logger = SummaryWriter(log_dir=logger_path, comment=comment)
    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    
    replay_buffer = ReplayBuffer(args, device=device)
    agent = PPO(args, device)
    if args.is_from_bppo:
        # set offline replay buffer
        dataset = env.get_dataset()
        offline_buffer = OfflineReplayBuffer(device, args.state_dim, args.action_dim, len(dataset['actions'])-1)
        offline_buffer.load_dataset(dataset=dataset, clip=True)
        if args.use_reward_scaling:  # Trick 4:reward scaling
            reward_scaling = offline_buffer.reward_normalize(args.gamma, args.scale_strategy)
        elif args.scale_strategy == 'number':
            offline_buffer.reward_normalize(args.gamma, args.scale_strategy)
            
        offline_buffer.compute_return(args.gamma)
        print('return : {}'.format(offline_buffer._return[0: 10]))
    
        if args.is_shuffle:
            print('shuffle the dataset')
            offline_buffer.shuffle()
        if args.use_state_norm:
            mean, std = offline_buffer.normalize_state()
        else:
            mean, std = 0., 1.
        
        print('mean: {}; std: {}'.format(mean, std))
        state_norm = Normalization(shape=args.state_dim, mean=mean, std=std)  # Trick 2:state normalization
       
        # if use reward scaling strategy, value function retraining is needed
        if args.scale_strategy == 'dynamic' or args.scale_strategy == 'number':
            value = ValueLearner(args, value_lr = 1e-4, batch_size = 512)
            if args.v_hidden_width == 256:
                v_path = os.path.join(path, 'value_256_{}.pt'.format(args.scale_strategy))
            else:
                v_path = os.path.join(path, 'value_{}.pt'.format(args.scale_strategy))
            if os.path.exists(v_path):
                value.load(v_path)
            else:
                for step in tqdm(range(int(1e6)), desc='value upladating ......'): 
                    value_loss = value.update(offline_buffer)
                    if step % int(2e4) == 0:
                        print(f"Step: {step}, Loss: {value_loss:.4f}")
                        logger.add_scalar('value_loss', value_loss, global_step=(step+1))
                value.save(v_path)
        else:
            v_path = os.path.join(load_path, 'value.pt')
        # load pi and value function from offline phase 
        agent.load_pi_value(pi_path=logdir, value_path=v_path)

        evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm, True)
        #evaluate_reward = agent.offline_evaluate(args.env_name, args.seed, mean, std)
        print('initial score from bppo:{}'.format(evaluate_reward))
    else:
        state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization

    # Build a tensorboard
    comment = args.env_name + '_' + str(args.seed)
    logger_path = os.path.join(path, current_time)
    writer = SummaryWriter(log_dir=logger_path, comment=comment)
    
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    total_episode_r =  deque(maxlen=10)
    episode_reward = 0
    scores = []
    d4rl_scores = []
    actor_losses, critic_losses, values, first_values = [], [], [], []
    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        total_episode_r.append(episode_reward)
        episode_reward = 0

        while not done:
            episode_steps += 1
            action, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)[0]
            episode_reward += r
            
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, action, a_logprob, r * args.r_scale, s_, dw, done)
            s = s_
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                actor_loss, critic_loss = agent.update(replay_buffer, total_steps)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                replay_buffer.count = 0

            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                avg_return, d4rl_score, value_score, first_score = agent.off_evaluate(args.env_name, args.seed, state_norm.running_ms.mean, state_norm.running_ms.std, eval_episodes=3)
                
                scores.append(avg_return)
                d4rl_scores.append(d4rl_score)
                values.append(value_score)
                first_values.append(first_score)

                print("evaluate_num:{} \t evaluate_reward:{} \t d4rl_score: {}".format(evaluate_num, avg_return, d4rl_score))
                print('collecting performance: {}, actor_loss: {}, critic_loss: {}'.format(np.mean(total_episode_r), np.mean(actor_losses[int(-args.evaluate_freq):]), np.mean(critic_losses[int(-args.evaluate_freq):])))
            
                writer.add_scalar('step_rewards_{}'.format(env_name), scores[-1], global_step=total_steps)
                writer.add_scalar('actor_loss_{}'.format(env_name), np.mean(actor_losses[int(-args.evaluate_freq):]), global_step=total_steps)
                writer.add_scalar('critic_loss_{}'.format(env_name), np.mean(critic_losses[int(-args.evaluate_freq):]), global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.savetxt(os.path.join(path, current_time, 'avg_returns'), scores, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(path, current_time, 'd4rl_scores'), d4rl_scores, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(path, current_time, 'values'), values, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(path, current_time, 'first_values'), first_values, fmt='%f', delimiter=',')
    np.savetxt(os.path.join(path, current_time, 'avg_returns'), scores, fmt='%f', delimiter=',')
    np.savetxt(os.path.join(path, current_time, 'd4rl_scores'), d4rl_scores, fmt='%f', delimiter=',')

    np.savetxt(os.path.join(path, current_time, 'values'), values, fmt='%f', delimiter=',')
    np.savetxt(os.path.join(path, current_time, 'first_values'), first_values, fmt='%f', delimiter=',')
    
