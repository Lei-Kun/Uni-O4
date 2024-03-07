from copy import deepcopy
from configs.neorl.default import default_args


hopper_v3_medium_args = deepcopy(default_args)
hopper_v3_medium_args["rollout_length"] = 5
hopper_v3_medium_args["penalty_coef"] = 1.5
hopper_v3_medium_args["auto-alpha"] = False