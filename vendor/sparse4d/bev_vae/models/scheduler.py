import numpy as np


class LambdaWarmUpCosineScheduler:
    """ note: use with a base_lr of 1.0 """
    def __init__(self, warm_up, lr_start, lr_max, lr_min, max_decay):
        self.warm_up = warm_up
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.max_decay = max_decay

    def schedule(self, n):
        if n < self.warm_up:
            lr = (self.lr_max - self.lr_start) / self.warm_up * n + self.lr_start
        else:
            t = (n - self.warm_up) / (self.max_decay - self.warm_up)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(min(t, 1.0) * np.pi))
        return lr

    def __call__(self, n):
        return self.schedule(n)