from copy import deepcopy
from configs.adroit.default import default_args


pen_cloned_args = deepcopy(default_args)
pen_cloned_args["rollout_length"] = 1
pen_cloned_args["penalty_coef"] = 0.5
pen_cloned_args["real_ratio"] = 0.5