from copy import deepcopy
from configs.neorl.default import default_args


halfcheetah_v3_medium_args = deepcopy(default_args)
halfcheetah_v3_medium_args["rollout_length"] = 5
halfcheetah_v3_medium_args["penalty_coef"] = 0.5