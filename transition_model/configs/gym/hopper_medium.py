from copy import deepcopy
from transition_model.configs.gym.default import default_args


hopper_medium_args = deepcopy(default_args)
hopper_medium_args["rollout_length"] = 5
hopper_medium_args["penalty_coef"] = 1.5
hopper_medium_args["auto_alpha"] = False