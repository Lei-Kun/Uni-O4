
import random

import gym
#import neorl
from typing import Dict, Union, Tuple
from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch

import os
from models.dynamics_model import EnsembleDynamicsModel
from dynamics import EnsembleDynamics
from utils.scaler import StandardScaler
from utils.termination_fns import get_termination_fn
from utils.load_dataset import qlearning_dataset
from utils.buffer_ import ReplayBuffer
from utils.logger import Logger, make_log_dirs




def rollout(
        policy,
        dynamics,
        Q,
        init_obss: np.ndarray,
        rollout_length: int,
        args,
        mean,
        std
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        total_q = np.array([])
        rollout_transitions = defaultdict(list)
        # rollout
        observations = init_obss
        length = 0
        for _ in range(rollout_length):
            
            if not args.is_eval_state_norm:
                if args.is_state_norm:
                    s = (observations - torch.FloatTensor(mean).to(args.device)) / torch.FloatTensor(std).to(args.device)
                else:
                    s = observations
            else:
                s = observations

            actions = policy.select_action(s, is_sample=False)

            Q_value = Q(s, actions)
            next_observations, rewards, terminals, info = dynamics.step(observations.cpu().data.numpy(), actions.cpu().data.numpy())

            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())
            total_q = np.append(total_q, Q_value.cpu().data.numpy().flatten())
            nonterm_mask = (~terminals).flatten()
            length += 1
            if nonterm_mask.sum() == 0:
                print('terminal length: {}'.format(length))
                break

            observations = torch.FloatTensor(next_observations[nonterm_mask]).to(args.device)

        return total_q.mean(), rewards_arr.mean()
    
def train_dynamics(args, replay_buffer):

    dynamics_save_path = os.path.join('1saved_models_{}_{}'.format(str(args.is_eval_state_norm), str(args.data_load_path)), args.env, str(args.seed))



    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.env)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    # train
    if os.path.exists(dynamics_save_path):

        dynamics.load(dynamics_save_path)
    if not os.path.exists(dynamics_save_path) or args.finetune_qv_dynamics:
        # log
        os.makedirs(dynamics_save_path, exist_ok=True)
        log_dirs = make_log_dirs(
            args.env, args.algo_name, args.seed, vars(args),
            record_params=["penalty_coef", "rollout_length"]
        )
        # key: output file name, value: output handler type
        output_config = {
            "consoleout_backup": "stdout",
            "policy_training_progress": "csv",
            "dynamics_training_progress": "csv",
            "tb": "tensorboardX"
        }
        logger = Logger(log_dirs, output_config)
        logger.log_hyperparameters(vars(args))
        dynamics.train(
            replay_buffer.sample_all(),
            logger,
            max_epochs_since_update=args.max_epochs_since_update,
            max_epochs=args.dynamics_max_epochs
        )
        dynamics.save(dynamics_save_path)

    return dynamics

def dynamics_eval(args, policy, Q, dynamics, replay_buffer, mean = 0., std = 1.):
    s, _, _, _, _, _, _, _ = replay_buffer.sample(args.rollout_batch_size)
    #s = replay_buffer.sample_aug_state(args.rollout_batch_size)
    Q_mean, reward_mean = rollout(policy, dynamics, Q, s, args.rollout_length, args, mean, std)
    return Q_mean, reward_mean

def get_args():
    from configs import loaded_args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mobile")
    parser.add_argument("--env", type=str, default="walker2d-medium-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--is_state_norm", default=False, type=bool)
    parser.add_argument("--is_eval_state_norm", default=False, type=bool)
    known_args, _ = parser.parse_known_args()
    default_args = loaded_args[known_args.env]
    for arg_key, default_value in default_args.items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))

    return parser.parse_args()
if __name__ == "__main__":
    from buffer import OfflineReplayBuffer
    args = get_args()
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
    # device
    args.device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"


    # offline dataset to replay buffer
    dataset = env.get_dataset()
    replay_buffer = OfflineReplayBuffer(args.device, state_dim, action_dim, len(dataset['actions']) - 1, percentage=1)
    replay_buffer.load_dataset(dataset=dataset)
    replay_buffer.compute_return(args.gamma)

    if args.is_state_norm:
        mean, std = replay_buffer.normalize_state()
    else:
        mean, std = 0., 1.
    replay_buffer.augmentaion()

    train_dynamics(args, env, replay_buffer)