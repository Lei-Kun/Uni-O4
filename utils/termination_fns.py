import numpy as np


def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
    done = ~not_done
    done = done[:, None]
    return done

def termination_fn_neorl_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False] * obs.shape[0])
    done = done[:, None]
    return done

def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_neorl_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:, 1:2]
    angle = next_obs[:, 2:3]
    state = next_obs[:, 3:]

    min_state, max_state = (-100.0, 100.0)
    min_z, max_z = (0.7, float('inf'))
    min_angle, max_angle = (-0.2, 0.2)

    healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1, keepdims=True)
    healthy_z = np.logical_and(min_z < z, z < max_z)
    healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)

    is_healthy = np.logical_and(np.logical_and(healthy_state, healthy_z), healthy_angle)

    done = np.logical_not(is_healthy).reshape(-1, 1)
    return done

def termination_fn_halfcheetahveljump(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) \
                * (height > 0.8) \
                * (height < 2.0) \
                * (angle > -1.0) \
                * (angle < 1.0)
    done = ~not_done
    done = done[:,None]
    return done

import numpy as np

def termination_fn_go1(obs, act, next_obs):
    dof_limit = np.array([[33.5, -0.802851455917, 0.802851455917, 50],
                        [33.5, -1.0471975512, 4.18879020479, 28],
                        [33.5, -2.69653369433, -0.916297857297, 28],
                        [33.5, -0.802851455917, 0.802851455917, 50],
                        [33.5, -1.0471975512, 4.18879020479, 28],
                        [33.5, -2.69653369433, -0.916297857297, 28],
                        [33.5, -0.802851455917, 0.802851455917, 50],
                        [33.5, -1.0471975512, 4.18879020479, 28],
                        [33.5, -2.69653369433, -0.916297857297, 28],
                        [33.5, -0.802851455917, 0.802851455917, 50],
                        [33.5, -1.0471975512, 4.18879020479, 28],
                        [33.5, -2.69653369433, -0.916297857297, 28]]).astype(np.float32)
    """ self.dof, 
        self.dof_vel,
        self.action,"""
    print(dof_limit[:,2] )
    out1 = next_obs[:,0:12] >  dof_limit[:,2]  
    out2 = next_obs[:,0:12] < dof_limit[:,1] 
    out3 = next_obs[:,12:24] > dof_limit[:,0]  
    out4 = next_obs[:,12:24] < -dof_limit[:,0]

    result = out1 | out2  
    result = result | out3
    result = result | out4
    return result
    #return next_obs[:][0:12]>dof_limit[:][2] | next_obs[:][0:12]<dof_limit[:][1] | next_obs[:][12:24]>dof_limit[:][0] | next_obs[:][12:24]<-dof_limit[:][0]


def termination_fn_neorl_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    min_z, max_z = (0.8, 2.0)
    min_angle, max_angle = (-1.0, 1.0)
    min_state, max_state = (-100.0, 100.0)
    
    z = next_obs[:, 1:2]
    angle = next_obs[:, 2:3]
    state = next_obs[:, 3:]
    
    healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1, keepdims=True)
    healthy_z = np.logical_and(min_z < z, z < max_z)
    healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)
    is_healthy = np.logical_and(np.logical_and(healthy_state, healthy_z), healthy_angle)
    done = np.logical_not(is_healthy).reshape(-1, 1)
    return done

def termination_fn_pen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    obj_pos = next_obs[:, 24:27]
    done = obj_pos[:, 2] < 0.075

    done = done[:,None]
    return done

def terminaltion_fn_door(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False] * obs.shape[0])

    done = done[:, None]
    return done

def terminaltion_fn_hammer(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False] * obs.shape[0])

    done = done[:, None]
    return done

def get_termination_fn(task):
    if 'halfcheetahvel' in task:
        return termination_fn_halfcheetahveljump
    elif 'halfcheetah' in task:
        return termination_fn_halfcheetah
    elif 'HalfCheetah-v3' in task:
        return termination_fn_neorl_halfcheetah
    elif 'hopper' in task:
        return termination_fn_hopper
    elif 'Hopper-v3' in task:
        return termination_fn_neorl_hopper
    elif 'walker2d' in task:
        return termination_fn_pen
    elif 'Walker2d-v3' in task:
        return termination_fn_neorl_walker2d
    elif 'pen' in task:
        return terminaltion_fn_door
    elif 'door' in task:
        return terminaltion_fn_door
    elif 'hammer' in task:
        return terminaltion_fn_hammer
    else:
        raise np.zeros