from copy import deepcopy
from configs.neorl.default import default_args


hopper_v3_high_args = deepcopy(default_args)
hopper_v3_high_args["rollout_length"] = 5
hopper_v3_high_args["penalty_coef"] = 2.5
hopper_v3_high_args["real-ratio"] = 0.5
hopper_v3_high_args["auto-alpha"] = False