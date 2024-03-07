from copy import deepcopy
from configs.neorl.default import default_args


walker2d_v3_high_args = deepcopy(default_args)
walker2d_v3_high_args["rollout_length"] = 1
walker2d_v3_high_args["penalty_coef"] = 2.5
walker2d_v3_high_args["real_ratio"] = 0.5