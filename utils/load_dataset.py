import numpy as np
import torch
import collections


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0
            if not has_next_obs:
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def load_neorl_dataset(env, data_type, traj_num=1000):
    train_data, _ = env.get_dataset(data_type=data_type, train_num=traj_num, need_val=False)
    dataset = {}
    dataset["observations"] = train_data["obs"]
    dataset["actions"] = train_data["action"]
    dataset["next_observations"] = train_data["next_obs"]
    dataset["rewards"] = train_data["reward"]
    dataset["terminals"] = train_data["done"]
    return dataset


def normalize_rewards(dataset):
    terminals_float = np.zeros_like(dataset["rewards"])
    for i in range(len(terminals_float) - 1):
        if np.linalg.norm(dataset["observations"][i + 1] -
                            dataset["next_observations"][i]
                            ) > 1e-6 or dataset["terminals"][i] == 1.0:
            terminals_float[i] = 1
        else:
            terminals_float[i] = 0

    terminals_float[-1] = 1

    # split_into_trajectories
    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append((dataset["observations"][i], dataset["actions"][i], dataset["rewards"][i], 1.0-dataset["terminals"][i],
                        terminals_float[i], dataset["next_observations"][i]))
        if terminals_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])
    
    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    # print(compute_returns(trajs[-1]))
    # print(len(trajs[-1]))
    # print(compute_returns(trajs[0]))
    # print(len(trajs[0]))

    # normalize rewards
    dataset["rewards"] /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset["rewards"] *= len(trajs[-1])

    return dataset
