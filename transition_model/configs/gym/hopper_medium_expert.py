from copy import deepcopy
from transition_model.configs.gym.default import default_args


hopper_medium_expert_args = deepcopy(default_args)
hopper_medium_expert_args["rollout_length"] = 5
hopper_medium_expert_args["penalty_coef"] = 1.5
hopper_medium_expert_args["deterministic_backup"] = True