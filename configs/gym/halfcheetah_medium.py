from copy import deepcopy
from configs.gym.default import default_args


halfcheetah_medium_args = deepcopy(default_args)
halfcheetah_medium_args["rollout_length"] = 5
halfcheetah_medium_args["penalty_coef"] = 0.5