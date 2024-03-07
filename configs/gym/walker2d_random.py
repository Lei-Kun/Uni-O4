from copy import deepcopy
from configs.gym.default import default_args


walker2d_random_args = deepcopy(default_args)
walker2d_random_args["rollout_length"] = 5
walker2d_random_args["penalty_coef"] = 2.0