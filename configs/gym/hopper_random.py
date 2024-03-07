from copy import deepcopy
from configs.gym.default import default_args


hopper_random_args = deepcopy(default_args)
hopper_random_args["rollout_length"] = 5
hopper_random_args["penalty_coef"] = 5.0
hopper_random_args["deterministic_backup"] = True