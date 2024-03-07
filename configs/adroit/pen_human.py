from copy import deepcopy
from configs.adroit.default import default_args


pen_human_args = deepcopy(default_args)
pen_human_args["rollout_length"] = 1
pen_human_args["penalty_coef"] = 1.0
pen_human_args["real_ratio"] = 0.8
pen_human_args["max_q_backup"] = False
pen_human_args["max_epochs_since_update"] = 20