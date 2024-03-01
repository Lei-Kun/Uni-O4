from copy import deepcopy
from transition_model.configs.neorl.default import default_args


walker2d_v3_medium_args = deepcopy(default_args)
walker2d_v3_medium_args["rollout_length"] = 1
walker2d_v3_medium_args["penalty_coef"] = 2.5