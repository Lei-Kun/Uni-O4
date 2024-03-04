import numpy as np

class Normalization1(object):
    def __init__(self, shape, mean = None, std = None, epsilon=1e-4):
        if mean is not None:
            self.mean = mean
            self.var = std ** 2
        else:
            self.mean = np.zeros(shape, 'float64')
            self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.std = self.var ** 0.5
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    def __call__(self, x):

        x = (x - self.mean) / (self.std)# + 1e-8)
        return x


    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, mean = None, std = None):  # shape:the dimension of input data
        if mean is not None:
            print('-------------------load mean and std of state from offline replay buffer-----------------------')
            self.mean = mean
            self.std = std
            self.S = std**2
        else:        
            self.mean = np.zeros(shape)
            self.S = np.ones(shape)
            self.std = np.sqrt(self.S)
        self.n = 0
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape, mean=None, std=None):
        self.running_ms = RunningMeanStd(shape=shape, mean=mean, std=std)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std)# + 1e-8)

        return x

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
