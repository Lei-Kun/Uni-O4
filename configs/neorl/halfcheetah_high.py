from copy import deepcopy
from configs.neorl.default import default_args


halfcheetah_v3_high_args = deepcopy(default_args)
halfcheetah_v3_high_args["rollout_length"] = 5
halfcheetah_v3_high_args["penalty_coef"] = 1.5