import isaacgym
assert isaacgym
import gym
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter
import h5py
from bppo import BehaviorCloning
from critic import ValueLearner, QPiLearner, QSarsaLearner, IQL_Q_V
from buffer_aliengo import OfflineReplayBuffer, load_dataset
from scripts.play_eval import eval_go1
from util import histroy_obs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env", default="hopper-medium-v2")        
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--gpu", default=0, type=int)             
    parser.add_argument("--log_freq", default=int(5e3), type=int)
    parser.add_argument("--path", default="logs", type=str)
    # For Value
    parser.add_argument("--v_steps", default=int(5e5), type=int) 
    parser.add_argument("--v_hidden_dim", default=1024, type=int)
    parser.add_argument("--v_depth", default=2, type=int)
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
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--is_offpolicy_update", default=False, type=bool)
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=int(5e5), type=int) 
    parser.add_argument("--save_num", default=int(4), type=int) 
    parser.add_argument("--bc_hidden_dim", default=[512, 256, 256], type=int)
    parser.add_argument("--bc_depth", default=3, type=int)
    parser.add_argument("--bc_lr", default=1e-4, type=float)
    parser.add_argument("--bc_batch_size", default=256, type=int)
    # For BPPO 
    parser.add_argument("--bppo_steps", default=int(4e2), type=int)
    parser.add_argument("--bppo_hidden_dim", default=[512, 256, 256], type=int)
    parser.add_argument("--bppo_depth", default=3, type=int)
    parser.add_argument("--bppo_lr", default=1e-4, type=float)  
    parser.add_argument("--bppo_batch_size", default=256, type=int)
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
    parser.add_argument("--is_shuffle", default=False, type=bool)
    parser.add_argument("--percentage", default=1.0, type=float)

    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--task", default="hopper-medium-v2")   
    parser.add_argument("--algo-name", type=str, default="mobile")
    parser.add_argument("--rollout_step", default=1000, type=int)
    parser.add_argument("--scale_strategy", default=None, type=str, help='reward scaling technique: dynamic/normal/number(0.1)')

    parser.add_argument("--data_load_path", default="dataset_0728", type=str)

    args = parser.parse_args()

    args.rollout_length = args.rollout_step
    args.rollout_batch_size = 256
    print(f'------current env {args.env} and current seed {args.seed} and rollout_length {args.rollout_length}------')
    # path
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    path = os.path.join(args.path, args.env, str(args.seed))
    os.makedirs(os.path.join(path, current_time))
    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")




    # env = gym.make(args.env)
    # # seed
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # dim of state and action
    # state_dim = env.observation_space.shape[0]                  
    # action_dim = env.action_space.shape[0]
    #args.max_action = np.max(dataset['actions'])
    # device
    device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    args.device = device

    # offline dataset to replay buffer
    # dataset = env.get_dataset()

#--------------------------------------load sim data-----------------------------------
    # dataset = {'observations': [], 'actions': [], 'terminals': [], 'rewards': [], 'timeouts': [], 'obs': []} # here observations represent obs_history
    # with h5py.File('dataset6.hdf5', 'r') as file:
    #     dataset['observations'] = np.array(file['observations'])
    #     dataset['actions'] = np.array(file['actions'])
    #     dataset['terminals'] = np.array(file['terminals'])
    #     dataset['rewards'] = np.array(file['rewards'])
    #     dataset['timeouts'] = np.array(file['timeouts'])
    # print('parameter loaded')
    # obs_historys = []
    # rewards = dataset['rewards']
    # print('max: {}, min: {}, mean: {}, std: {}'.format(np.max(rewards,axis=0), np.min(rewards, axis=0), np.mean(rewards, axis=0), np.std(rewards, axis=0)))
    
    # dataset["observations"] = np.array([row[3: 6].tolist() + row[18: 66].tolist() for row in dataset["observations"]])

    # histroy_obs_revise = histroy_obs(history_len=30, num_obs=dataset["observations"].shape[1])
    # for i in tqdm(range(len(dataset["observations"]))):
    #     obs_history = histroy_obs_revise.step(obs = dataset["observations"][i])
    #     obs_historys.append(obs_history)
    #     if dataset['terminals'][i]:
    #         histroy_obs_revise.reset()
    # dataset["observations"] = np.array(obs_historys)

#----------------------------------------load real data-------------------------------
    buffer_save_path = os.path.join(args.data_load_path +'.pt')
    if not os.path.exists(buffer_save_path):
        data_loader = load_dataset()
        dataset_real = data_loader.load_file(args.data_load_path)
        # with h5py.File('dataset_real_{}.hdf5'.format(args.data_load_path), 'w') as f:
        #     for key, value in dataset.items():
        #         f.create_dataset(key, data=value)

        torch.save(dataset_real, buffer_save_path)
    else:
        print('load buffer')
        dataset_real = torch.load(buffer_save_path)
    dataset = dataset_real
    # dataset['observations'] = np.concatenate((dataset["observations"], dataset_real['observations']))
    # dataset['actions'] = np.concatenate((dataset["actions"], dataset_real['actions']))
    # dataset['rewards'] = np.concatenate((dataset["rewards"].flatten(), dataset_real['rewards']))
    # dataset['terminals'] = np.concatenate((dataset["terminals"].flatten(), dataset_real['terminals']))
    # dataset['timeouts'] = np.concatenate((dataset["timeouts"].flatten(), dataset_real['timeouts']))
    # rewards = dataset['rewards']
    # print('max: {}, min: {}, mean: {}, std: {}'.format(np.max(rewards,axis=0), np.min(rewards, axis=0), np.mean(rewards, axis=0), np.std(rewards, axis=0)))
    print('total step: {}'.format(len(dataset["actions"])))
    state_dim = dataset['observations'].shape[1]                  
    action_dim = dataset['actions'].shape[1]


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
    

    # initilize
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
    bc = BehaviorCloning(device, state_dim, args.bc_hidden_dim, args.bc_depth, action_dim, args.bc_lr, args.bc_batch_size)
    
    
    go1_eval = eval_go1(headless=False)
    go1_eval.prove_real_data(replay_buffer)
    # bc training
    save_freq = int(args.bc_steps / args.save_num)
    best_bc_path = os.path.join(path, 'bc_3.pt')
    if os.path.exists(best_bc_path):
        bc.load(best_bc_path)
        eval_returns =  go1_eval.play_go1([bc], torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device))
        print('eval_returns: ', eval_returns)
    else:
        save_id = 0
        best_bc_score = 0    
        for step in tqdm(range(int(args.bc_steps)), desc='bc updating ......'):
            bc_loss = bc.update(replay_buffer)

            if step % int(args.log_freq) == 0:
                eval_returns =  go1_eval.play_go1([bc], torch.FloatTensor(mean).to(device), torch.FloatTensor(std).to(device))
                # current_bc_score = bc.offline_evaluate(args.env, args.seed, mean, std)
                # if current_bc_score > best_bc_score:
                #     best_bc_score = current_bc_score
                #     bc.save(best_bc_path)
                #     np.savetxt(os.path.join(path, 'best_bc.csv'), [best_bc_score], fmt='%f', delimiter=',')
                print(f"Step: {step}, Loss: {bc_loss:.4f}")
                print('eval_returns: ', eval_returns)
                logger.add_scalar('bc_loss', bc_loss, global_step=(step+1))
                # logger.add_scalar('bc_score', current_bc_score, global_step=(step+1))
            if (step+1) % int(save_freq) == 0:
                save_id += 1
                save_path = os.path.join(path, 'bc_{}.pt'.format(save_id))
                bc.save(save_path)
        bc.save(os.path.join(path, 'bc_last.pt'))
        # bc.load(best_bc_path)

    if args.is_iql:
        # Q_bc training
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
        
        if args.is_offpolicy_update:
            Q_pi.load(Q_bc_path)
        q_eval = Q_bc