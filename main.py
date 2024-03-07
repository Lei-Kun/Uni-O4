import isaacgym
assert isaacgym
import gym
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import h5py
from tensorboardX import SummaryWriter
from BC_ensemble import BC_ensemble
from configs import loaded_args
from critic import ValueLearner, QPiLearner, QSarsaLearner, IQL_Q_V
from abppo import AdaptiveBehaviorProximalPolicyOptimization
from buffer import OfflineReplayBuffer, load_dataset
from scripts.play_eval import eval_go1
from dynamics_eval import train_dynamics, dynamics_eval
from util import histroy_obs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env", default="hopper-medium-v2")        
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--gpu", default=0, type=int)             
    parser.add_argument("--log_freq", default=int(6e3), type=int)
    parser.add_argument("--path", default="logs", type=str)
    # For Value
    parser.add_argument("--v_steps", default=int(5e5), type=int) 
    parser.add_argument("--v_hidden_dim", default=512, type=int)
    parser.add_argument("--v_depth", default=3, type=int)
    parser.add_argument("--v_lr", default=1e-4, type=float)
    parser.add_argument("--v_batch_size", default=512, type=int)
    # For Q
    parser.add_argument("--q_bc_steps", default=int(5e5), type=int) 
    parser.add_argument("--q_pi_steps", default=10, type=int) 
    parser.add_argument("--q_hidden_dim", default=1024, type=int)
    parser.add_argument("--q_depth", default=2, type=int)       
    parser.add_argument("--q_lr", default=1e-4, type=float) 
    parser.add_argument("--q_batch_size", default=512, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    # parser.add_argument("--tau", default=0.005, type=float)
    # parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--is_offpolicy_update", default=False, type=bool)
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=int(5e5), type=int) 
    parser.add_argument("--save_num", default=int(4), type=int) 
    parser.add_argument("--bc_hidden_dim", default=[512, 256, 128], type=int)
    parser.add_argument("--bc_depth", default=2, type=int)
    parser.add_argument("--bc_lr", default=1e-4, type=float)
    parser.add_argument("--bc_batch_size", default=512, type=int)
    # For BPPO 
    parser.add_argument("--alpha_bc", default=0.0, type=float) 
    parser.add_argument("--num_policies", default=int(4), type=int) 
    parser.add_argument("--bppo_steps", default=int(4e2), type=int)
    parser.add_argument("--bppo_hidden_dim", default=[512, 256, 128], type=int)
    parser.add_argument("--bppo_depth", default=2, type=int)
    parser.add_argument("--bppo_lr", default=1e-4, type=float)  
    parser.add_argument("--bppo_batch_size", default=512, type=int)
    parser.add_argument("--clip_ratio", default=0.25, type=float)
    parser.add_argument("--entropy_weight", default=0.0, type=float)
    parser.add_argument("--decay", default=0.96, type=float)
    parser.add_argument("--omega", default=0.9, type=float)
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

    parser.add_argument("--is_kl_update", default=False, type=bool)
    parser.add_argument("--kl_strategy", default='max', type=str)
    parser.add_argument("--alpha_bppo", default=0.1, type=float)
    parser.add_argument("--scale_strategy", default=None, type=str, help='reward scaling technique: dynamic/normal/number(0.1)')

    parser.add_argument("--is_clip_action", default=False, type=bool)
    parser.add_argument("--data_load_path", default="dataset_0808", type=str)
    parser.add_argument("--load_from_wd3", default=True, type=bool)

    parser.add_argument("--finetune_qv_dynamics", default=False, type=bool)

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
    bc_path = os.path.join(path, 'bc/{}'.format(args.alpha_bc))

    os.makedirs(bc_path, exist_ok=True)
    os.makedirs(os.path.join(path, current_time))
    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # device
    device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    args.device = device

#----------------------------------------load real data-------------------------------
    buffer_save_path = os.path.join(args.data_load_path +'.pt')
    if not os.path.exists(buffer_save_path):
        data_loader = load_dataset()
        dataset_real = data_loader.load_file(args.data_load_path)
        # with h5py.File('dataset_real_{}.hdf5'.format(args.data_load_path), 'w') as f:
        #     for key, value in dataset_real.items():
        #         f.create_dataset(key, data=value)

        torch.save(dataset_real, buffer_save_path)
    else:
        print('load buffer')
        dataset_real = torch.load(buffer_save_path)

    print('real dataset: max: {}, min: {}, mean: {}, std: {}'.format(np.max(dataset_real['rewards']), np.min(dataset_real['rewards']), np.mean(dataset_real['rewards']), np.std(dataset_real['rewards'])))

    histroy_obs_revise = histroy_obs(history_len=5, num_obs=dataset_real["observations"].shape[1])
    obs_historys = []
    for i in tqdm(range(len(dataset_real["observations"]))):
        obs_history = histroy_obs_revise.step(obs = dataset_real["observations"][i])
        obs_historys.append(obs_history)
        if dataset_real['terminals'][i]:
            histroy_obs_revise.reset()
    dataset_real["observations"] = np.array(obs_historys)
    dataset = dataset_real

    print('total step: {}'.format(len(dataset["actions"])))
    state_dim = dataset['observations'].shape[1]                  
    action_dim = dataset['actions'].shape[1]
    print('state_dim: {}, action_dim: {}'.format(state_dim, action_dim))

    replay_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']) - 1, percentage=args.percentage)
    replay_buffer.load_dataset(dataset=dataset)
    replay_buffer.reward_normalize(args.gamma, args.scale_strategy)
    replay_buffer.compute_return(args.gamma)

    args.obs_shape = state_dim 
    args.action_dim = action_dim

    if args.is_shuffle:
        print('shuffle the dataset')
        replay_buffer.shuffle()
    if args.is_state_norm:
        mean, std = replay_buffer.normalize_state()
    else:
        mean, std = [0.],[ 1.]

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
    
    
    ensemble_bc = BC_ensemble(args.num_policies, device, state_dim, args.bc_hidden_dim, args.bc_depth, action_dim, args.bc_lr, args.bc_batch_size, bc_kl=args.kl_bc)
    abppo = AdaptiveBehaviorProximalPolicyOptimization(device, 290, args.bppo_hidden_dim, args.bppo_depth, 12, args.bppo_lr, 
                                                       args.clip_ratio, args.entropy_weight, args.decay, args.omega, args.bppo_batch_size, args.num_policies, args.gamma, args.is_iql, args.is_kl_update, args.kl_strategy, args.alpha_bppo, args.is_clip_action)
    
    go1_eval = eval_go1(headless=False)
    go1_eval.prove_real_data(replay_buffer)
    
    # bc training
    best_bc_path = os.path.join(bc_path, 'bc_best_{}.pt'.format(2))
    print(os.path.exists(best_bc_path))
    if not args.load_from_wd3:
        if not os.path.exists(best_bc_path):
            save_id = 0
            save_freq = int(args.bc_steps / args.save_num)
            best_bc_scores = np.zeros(args.num_policies)
            best_bc_meta_score = 0
            for step in tqdm(range(int(args.bc_steps)), desc='bc updating ......'):
                bc_losses = ensemble_bc.joint_train(replay_buffer,alpha=args.alpha_bc)

                if step % int(args.log_freq) == 0:

                    current_bc_score = go1_eval.play_go1(ensemble_bc.ensemble, torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device))
                    
                    index = np.where(current_bc_score>best_bc_scores)[0]
                    if len(index) !=  0:
                        best_bc_path = os.path.join(bc_path, 'bc_best')
                        best_bc_scores[index] = current_bc_score[index]
                        ensemble_bc.ensemble_save(best_bc_path,index)
                        np.savetxt(os.path.join(bc_path, 'best_bc.csv'), [best_bc_scores], fmt='%f', delimiter=',')

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
    if args.is_iql:
        # Q_bc training
        Q_bc_path = os.path.join(path, 'Q_bc_{}.pt'.format(args.omega))
        value_path = os.path.join(path, 'value_{}.pt'.format(args.omega))
        if os.path.exists(Q_bc_path):
            iql.load(Q_bc_path, value_path)

        if not os.path.exists(Q_bc_path) or args.finetune_qv_dynamics:
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

    # train dynamics
    dynamics =  train_dynamics(args, replay_buffer)

    bc_path = os.path.join(path, 'best1')
    # bppo training
    best_bc_path = os.path.join(bc_path, 'pi')
    if args.load_from_wd3:
        import glob
        label = "gait-conditioned-agility/2023-08-04/train"
        dirs = glob.glob(f"runs/{label}/*")
        logdir = sorted(dirs)[-1]
        net_path = logdir + '/checkpoints/actor_weights_000300.pt'
        std_path = logdir + '/checkpoints/std_weights_000300.pt'
        abppo.load_from_wd3(net_path, std_path)
    else:
        abppo.load_bc(best_bc_path)
        print('load_path: {}'.format(best_bc_path))
        abppo.ensemble_save_body(best_bc_path, [i for i in range(args.num_policies)])
    #abppo.ensemble_save_policy(best_bc_path, [i for i in range(args.num_policies)])
    best_bppo_scores =  go1_eval.play_go1(abppo.bppo_ensemble, torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device), 2)
    
    best_mean_qs = abppo.ope_dynamics_eval(args, dynamics_eval, q_eval, dynamics, replay_buffer, mean, std)
    best_bppo_path = os.path.join(path, current_time)
    print('best_bppo_score:',best_bppo_scores,'-------------------------')
    scores = []
    current_bppo_scores = [0 for i in range(args.num_policies)]

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

        if (step+1) % 500 == 0:
            current_bppo_scores =  go1_eval.play_go1(abppo.bppo_ensemble, torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device), 2)

            scores.append(current_bppo_scores)
        if (step+1)% args.eval_step == 0:
            
            current_mean_qs = abppo.ope_dynamics_eval(args, dynamics_eval, q_eval, dynamics, replay_buffer,  mean, std)
            print('rollout trajectory q mean:{}'.format(current_mean_qs))
            print(f"Step: {step}, Score: ", current_bppo_scores)

            
            index = np.where(current_mean_qs > best_mean_qs)[0]  

            if len(index) != 0:
                os.makedirs(os.path.join(best_bppo_path, 'best'), exist_ok=True)
                abppo.ensemble_save(os.path.join(best_bppo_path, 'best'), index)
                best_mean_qs[index] = current_mean_qs[index]
                np.savetxt(os.path.join(path, current_time, 'best_bppo.csv'), [best_bppo_scores], fmt='%f', delimiter=',')

                if args.is_update_old_policy:
                    os.makedirs(os.path.join(best_bppo_path, 'best'), exist_ok=True)
                    abppo.ensemble_save(os.path.join(best_bppo_path, 'best'), index)
                    for i_d in index:
                        abppo.replace(index=index)
                        print('------------------------------update behavior policy {}----------------------------------------'.format(i_d))     

    os.makedirs(os.path.join(best_bppo_path, 'last'), exist_ok=True)
    abppo.ensemble_save(os.path.join(best_bppo_path, 'last'), [i for i in range(args.num_policies)])
    logger.close()
