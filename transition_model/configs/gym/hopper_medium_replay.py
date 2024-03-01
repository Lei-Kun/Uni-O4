from copy import deepcopy
from transition_model.configs.gym.default import default_args


hopper_medium_replay_args = deepcopy(default_args)
hopper_medium_replay_args["rollout_length"] = 5
hopper_medium_replay_args["penalty_coef"] = 0.1
hopper_medium_replay_args["auto_alpha"] = False