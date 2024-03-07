from copy import deepcopy
from configs.gym.default import default_args


walker2d_medium_replay_args = deepcopy(default_args)
walker2d_medium_replay_args["rollout_length"] = 1
walker2d_medium_replay_args["penalty_coef"] = 0.5