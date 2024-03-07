from copy import deepcopy
from configs.gym.default import default_args


walker2d_medium_expert_args = deepcopy(default_args)
walker2d_medium_expert_args["rollout_length"] = 1
walker2d_medium_expert_args["penalty_coef"] = 1.5