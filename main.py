import gym
import d4rl
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
from BC_ensemble import BC_ensemble
from transition_model.configs import loaded_args
from buffer import OfflineReplayBuffer
from critic import ValueLearner, QPiLearner, QSarsaLearner, IQL_Q_V
from abppo import AdaptiveBehaviorProximalPolicyOptimization
from dynamics_eval import dynamics_eval, train_dynamics # The code base of transition model is from mobile https://github.com/yihaosun1124/mobile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env", default="hopper-medium-v2")        
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--gpu", default=0, type=int)             
    parser.add_argument("--log_freq", default=int(2e4), type=int)
    parser.add_argument("--path", default="logs", type=str)
    # For Value
    parser.add_argument("--v_steps", default=int(2e6), type=int) 
    parser.add_argument("--v_hidden_dim", default=512, type=int)
    parser.add_argument("--v_depth", default=3, type=int)
    parser.add_argument("--v_lr", default=1e-4, type=float)
    parser.add_argument("--v_batch_size", default=512, type=int)
    # For Q
    parser.add_argument("--q_bc_steps", default=int(2e6), type=int) 
    parser.add_argument("--q_pi_steps", default=10, type=int) 
    parser.add_argument("--q_hidden_dim", default=1024, type=int)
    parser.add_argument("--q_depth", default=2, type=int)       
    parser.add_argument("--q_lr", default=1e-4, type=float) 
    parser.add_argument("--q_batch_size", default=512, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    parser.add_argument("--is_offpolicy_update", default=False, type=bool)
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=int(4e5), type=int) 
    parser.add_argument("--save_num", default=int(4), type=int) 
    parser.add_argument("--bc_hidden_dim", default=256, type=int)
    parser.add_argument("--bc_depth", default=3, type=int)
    parser.add_argument("--bc_lr", default=1e-4, type=float)
    parser.add_argument("--bc_batch_size", default=512, type=int)
    parser.add_argument("--pi_activation_f", default='relu', type=str)
    parser.add_argument("--is_filter_bc", default=False, type=bool)
    # For BPPO 
    parser.add_argument("--alpha_bc", default=0.1, type=float) 
    parser.add_argument("--num_policies", default=int(4), type=int) 
    parser.add_argument("--bppo_steps", default=int(10000), type=int)
    parser.add_argument("--bppo_hidden_dim", default=256, type=int)
    parser.add_argument("--bppo_depth", default=3, type=int)
    parser.add_argument("--bppo_lr", default=1e-4, type=float)  
    parser.add_argument("--bppo_batch_size", default=512, type=int)
    parser.add_argument("--clip_ratio", default=0.25, type=float)
    parser.add_argument("--entropy_weight", default=0.0, type=float)
    parser.add_argument("--decay", default=0.96, type=float)
    parser.add_argument("--omega", default=0.7, type=float)
    parser.add_argument("--is_clip_decay", default=True, type=bool)  
    parser.add_argument("--is_bppo_lr_decay", default=False, type=bool)       
    parser.add_argument("--is_update_old_policy", default=True, type=bool)
    parser.add_argument("--is_state_norm", default=False, type=bool)
    parser.add_argument("--is_eval_state_norm", default=False, type=bool)
    parser.add_argument("--is_linear_decay", default=True, type=bool)

    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--is_iql", default=False, type=bool)
    parser.add_argument("--is_double_q", default=True, type=bool)
    parser.add_argument("--is_shuffle", default=True, type=bool)
    parser.add_argument("--percentage", default=1.0, type=float)

    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--task", default="hopper-medium-v2")   
    parser.add_argument("--algo-name", type=str, default="mobile")
    parser.add_argument("--rollout_step", default=1000, type=int)
    parser.add_argument("--kl_bc", default='data', type=str, help='(trpo, pi, data)kl penalty for joint bc training: trpo: kl the distribution given s; \
                        pi: kl the probability of the action sampled from pi; data: kl the probability of the action sampled from offline dataset')
    parser.add_argument("--kl_type", default='heuristic', type=str, help='choice=(distribution, heuristic) kl type for joint bc training')
    parser.add_argument("--is_kl_update", default=False, type=bool)
    parser.add_argument("--kl_strategy", default='max', type=str)
    parser.add_argument("--alpha_bppo", default=0.1, type=float)
    parser.add_argument("--scale_strategy", default=None, type=str, help='reward scaling technique: dynamic/normal/number(0.1)')


    parser.add_argument("--eval_freq", default=int(500), type=int)
    parser.add_argument("--is_clip_action", default=False, type=bool)

    parser.add_argument("--eval_episode", default=10, type=int)
    known_args, _ = parser.parse_known_args()
    default_args = loaded_args[known_args.env]
    for arg_key, default_value in default_args.items():
        print(arg_key)
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))
        
    args = parser.parse_args()

    args.rollout_length = args.rollout_step
    args.rollout_batch_size = 512
    #bppo policy's weight loads from bc policy
    args.bppo_hidden_dim = args.bc_hidden_dim
    args.bppo_depth = args.bc_depth
    print(f'------current env {args.env} and current seed {args.seed} and rollout_length {args.rollout_length}------')
    # path
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    path = os.path.join(args.path, args.env, str(args.seed))
    if 'antmaze' in args.env:
        bc_path = os.path.join(path, 'bc/{}'.format(args.alpha_bc))
    else:
        bc_path = os.path.join(path, 'bc_/{}'.format(args.alpha_bc))
    # bc_path = os.path.join(path, 'bc_{}_{}/{}'.format(args.bc_hidden_dim, args.bc_depth, args.alpha_bc))
    os.makedirs(bc_path, exist_ok=True)
    # save args
    os.makedirs(os.path.join(path, 'pi', current_time), exist_ok=True)
    config_path = os.path.join(path, 'pi', current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")

    env = gym.make(args.env)
    # seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # dim of state and action
    state_dim = env.observation_space.shape[0]                  
    action_dim = env.action_space.shape[0]
    print('state_dim: {}; action_dim: {}'.format(state_dim, action_dim))
    # device
    device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    args.device = device

    # offline dataset to replay buffer
    dataset = env.get_dataset()
    replay_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']) - 1, percentage=args.percentage)
    replay_buffer.load_dataset(dataset=dataset, clip=args.is_clip_action, env_name=args.env)
    replay_buffer.reward_normalize(args.gamma, args.scale_strategy)
    replay_buffer.compute_return(args.gamma)

    if args.is_shuffle:
        print('shuffle the dataset')
        replay_buffer.shuffle()
    if args.is_state_norm:
        mean, std = replay_buffer.normalize_state()
    else:
        mean, std = 0., 1.

    eval_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']) - 1, percentage=args.percentage)
    eval_buffer.load_dataset(dataset=dataset, clip=args.is_clip_action, env_name=args.env)
    eval_buffer.reward_normalize(args.gamma, args.scale_strategy)
    eval_buffer.compute_return(args.gamma)
    if args.is_eval_state_norm:
        _, _ = eval_buffer.normalize_state()
    eval_buffer.augmentaion()
    if 'antmaze' in args.env:
        bc_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']) - 1, percentage=args.percentage)
        if args.is_filter_bc:
            bc_buffer.load_filter_dataset(dataset=dataset, gamma=args.gamma, clip=args.is_clip_action, env_name=args.env)
            if args.is_state_norm:
                bc_buffer._state = (bc_buffer._state - mean) / std
                bc_buffer._next_state = (bc_buffer._next_state - mean) / std
    else:
        bc_buffer = replay_buffer

    # summarywriter logger
    comment = args.env + '_' + str(args.seed)
    logger_path = os.path.join(path, current_time)
    logger = SummaryWriter(log_dir=logger_path, comment=comment)

    loggers = []
    for i in range(args.num_policies):
        logger_path_p = os.path.join(logger_path, '_{}'.format(i))
        logger_p = SummaryWriter(log_dir=logger_path_p, comment=comment)
        loggers.append(logger_p)
    
    # initilize
    if args.is_iql:
        Q_bc, value = None, None
        iql = IQL_Q_V(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_lr, args.target_update_freq, args.tau, args.gamma, args.q_batch_size, args.v_hidden_dim, args.v_depth, args.v_lr,args.omega, args.is_double_q)
    else:
        iql = None
        value = ValueLearner(device, state_dim, args.v_hidden_dim, args.v_depth, args.v_lr, args.v_batch_size)
        Q_bc = QSarsaLearner(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_lr, args.target_update_freq, args.tau, args.gamma, args.q_batch_size)
    if args.is_offpolicy_update:
        Q_pi = QPiLearner(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_lr, args.target_update_freq, args.tau, args.gamma, args.q_batch_size)
    
    
    ensemble_bc = BC_ensemble(args.num_policies, device, state_dim, args.bc_hidden_dim, args.bc_depth, action_dim, args.bc_lr, args.bc_batch_size, bc_kl=args.kl_bc, kl_type=args.kl_type, pi_activation_f=args.pi_activation_f)

    abppo = AdaptiveBehaviorProximalPolicyOptimization(device, state_dim, args.bppo_hidden_dim, args.bppo_depth, action_dim, args.bppo_lr, 
                                                       args.clip_ratio, args.entropy_weight, args.decay, args.omega, args.bppo_batch_size, args.num_policies, args.gamma, args.is_iql, args.is_kl_update, args.kl_strategy, args.alpha_bppo, args.is_clip_action, pi_activation_f=args.pi_activation_f)


    if args.is_iql:
        # Q_bc training
        if 'antmaze' in args.env:
            Q_bc_path = os.path.join(path, 'Q_bc_{}{}{}.pt'.format(args.scale_strategy, str(args.omega), str(3)))
            value_path = os.path.join(path, 'value_{}{}{}.pt'.format(args.scale_strategy, str(args.omega), str(3)))
        else:
            Q_bc_path = os.path.join(path, 'Q_bc.pt')
            value_path = os.path.join(path, 'value.pt')
        if os.path.exists(Q_bc_path):
            iql.load(Q_bc_path, value_path)
        else:
            for step in tqdm(range(int(args.v_steps)), desc='value and q upladating ......'): 
                Q_bc_loss, value_loss = iql.update(replay_buffer=replay_buffer)
                if step % int(args.log_freq) == 0:
                    print(f"Step: {step}, Loss: {Q_bc_loss:.4f}")
                    logger.add_scalar('Q_bc_loss', Q_bc_loss, global_step=(step+1))
                    print(f"Step: {step}, L    Q = Q_bcoss: {value_loss:.4f}")
                    logger.add_scalar('value_loss', value_loss, global_step=(step+1))
            iql.save(Q_bc_path, value_path)
        q_eval = iql.minQ
    else:
        # value training 
        value_path = os.path.join(path, 'value.pt')
        if os.path.exists(value_path):
            value.load(value_path)
        else:
            for step in tqdm(range(int(args.v_steps)), desc='value upladating ......'): 
                value_loss = value.update(replay_buffer)
                
                if step % int(args.log_freq) == 0:
                    print(f"Step: {step}, Loss: {value_loss:.4f}")
                    logger.add_scalar('value_loss', value_loss, global_step=(step+1))

            value.save(value_path)

        # Q_bc training
        Q_bc_path = os.path.join(path, 'Q_bc.pt')
        if os.path.exists(Q_bc_path):
            Q_bc.load(Q_bc_path)
        else:
            for step in tqdm(range(int(args.q_bc_steps)), desc='Q_bc updating ......'): 
                Q_bc_loss = Q_bc.update(replay_buffer, pi=None)

                if step % int(args.log_freq) == 0:
                    print(f"Step: {step}, Loss: {Q_bc_loss:.4f}")
                    logger.add_scalar('Q_bc_loss', Q_bc_loss, global_step=(step+1))

            Q_bc.save(Q_bc_path)
        q_eval = Q_bc
        
    Q = Q_bc
    #train dynamics
    dynamics =  train_dynamics(args, env, eval_buffer)

    # bc training
    if 'antmaze' in args.env:
        best_bc_path = os.path.join(bc_path, 'bc_3_{}.pt'.format(1))
    else:
        best_bc_path = os.path.join(bc_path, 'bc_last_{}.pt'.format(1))
    if not os.path.exists(best_bc_path):
        save_id = 0
        save_freq = int(args.bc_steps / args.save_num)
        best_bc_scores = np.zeros(args.num_policies)
        best_bc_meta_score = 0
        for step in tqdm(range(int(args.bc_steps)), desc='bc updating ......'):
            bc_losses = ensemble_bc.joint_train(bc_buffer, alpha=args.alpha_bc)

            if step % int(args.log_freq) == 0:

                current_bc_score = ensemble_bc.evaluation(args.env, args.seed, mean, std)
                mean_loss, mean_bc_score = bc_losses.mean(), current_bc_score.mean()
                print(f"Step: {step}, Loss: {mean_loss:.4f}, Score: {mean_bc_score:.4f}")
                for i in range(args.num_policies):
                    logger.add_scalar('bc_loss_{}'.format(i), bc_losses[i], global_step=(step+1))
                    logger.add_scalar('bc_score{}'.format(i), current_bc_score[i], global_step=(step+1))
            if (step+1) % int(save_freq) == 0:
                index = [i for i in range(args.num_policies)]
                save_path = os.path.join(bc_path, 'bc_{}'.format(save_id))
                ensemble_bc.ensemble_save(save_path,index)
                save_id += 1
        index = [i for i in range(args.num_policies)]
        save_path = os.path.join(bc_path, 'bc_last')
        ensemble_bc.ensemble_save(save_path,index)

    # bppo training
    if 'antmaze' in args.env:
        best_bc_path = os.path.join(bc_path, 'bc_3')
    else:
        best_bc_path = os.path.join(bc_path, 'bc_last') 
    abppo.load_bc(best_bc_path)
    best_bppo_scores = abppo.off_evaluate(args.env, args.seed, mean, std , args.eval_episode)
    meta_score = abppo.mixed_offline_evaluate(args.env, args.seed, mean, std)
    best_mean_qs = abppo.ope_dynamics_eval(args, dynamics_eval, q_eval, dynamics, eval_buffer, env, mean, std)
    best_bppo_path = os.path.join(path, 'pi', current_time)
    print('meta_score: {}'.format(meta_score))
    print('rollout trajectory q mean:{}'.format(best_mean_qs))
    print('best_bppo_score:',best_bppo_scores,'-------------------------')
    update_num = 0
    success_num = 0
    current_bppo_score = 0
    scores, meta_scores = [], []
    current_bppo_scores = [0 for i in range(args.num_policies)]
    scores.append(best_bppo_scores)
    meta_scores.append(meta_score)
    for step in tqdm(range(int(args.bppo_steps)), desc='bppo updating ......'):
        if args.is_linear_decay:
            bppo_lr_now = args.bppo_lr * (1 - step / args.bppo_steps)
            q_lr_now = args.q_lr * (1 - step / args.bppo_steps)
            clip_ratio_now = args.clip_ratio * (1 - step / args.bppo_steps)
        else:
            bppo_lr_now = None
            q_lr_now = None
            clip_ratio_now = None
        if step > 200:
            args.is_clip_decay = False
            args.is_bppo_lr_decay = False
        losses = abppo.joint_train(replay_buffer, value, args.is_clip_decay, args.is_bppo_lr_decay, is_linear_decay=args.is_linear_decay \
                                   , bppo_lr_now= bppo_lr_now, clip_ratio_now= clip_ratio_now, Q=Q, iql=iql)

        if (step+1) % args.eval_freq == 0:
            current_bppo_scores = abppo.off_evaluate(args.env, args.seed, mean, std, args.eval_episode)
            meta_score = abppo.mixed_offline_evaluate(args.env, args.seed, mean, std)
            scores.append(current_bppo_scores)
            meta_scores.append(meta_score)
        
        if (step+1)% args.eval_step == 0:

            current_mean_qs = abppo.ope_dynamics_eval(args, dynamics_eval, q_eval, dynamics, eval_buffer, env, mean, std)
            print('meta_score: {}'.format(meta_score))
            print('rollout trajectory q mean:{}'.format(current_mean_qs))
            print(f"Step: {step}, Score: ", current_bppo_scores)
        
            index = np.where(current_mean_qs > best_mean_qs)[0]

            if len(index) != 0:
                update_num += len(index)
                best_mean_qs[index] = current_mean_qs[index]
                if args.is_update_old_policy:
                    for i_d in index:
                        abppo.replace(index=index)
                        print('------------------------------update behavior policy {}----------------------------------------'.format(i_d))     
        np.savetxt(os.path.join(best_bppo_path, 'each_scores.csv'), scores, fmt='%f', delimiter=',')
        np.savetxt(os.path.join(best_bppo_path, 'meta_score.csv'), meta_scores, fmt='%f', delimiter=',')
    np.savetxt(os.path.join(best_bppo_path, 'last_ope_score.csv'), current_mean_qs, fmt='%f', delimiter=',') 
    np.savetxt(os.path.join(best_bppo_path, 'best_ope_score.csv'), best_mean_qs, fmt='%f', delimiter=',') 
    os.makedirs(os.path.join(best_bppo_path, 'last'), exist_ok=True)
    abppo.ensemble_save(os.path.join(best_bppo_path, 'last'), [i for i in range(args.num_policies)])
    logger.close()
