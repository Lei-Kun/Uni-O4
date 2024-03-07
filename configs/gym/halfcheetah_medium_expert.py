from copy import deepcopy
from configs.gym.default import default_args


halfcheetah_medium_expert_args = deepcopy(default_args)
halfcheetah_medium_expert_args["rollout_length"] = 5
halfcheetah_medium_expert_args["penalty_coef"] = 1.0
halfcheetah_medium_expert_args["real_ratio"] = 0.5