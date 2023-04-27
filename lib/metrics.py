from torch import nn

class MeanMetric(nn.Module):
    def __init__(self):
        super().__init__()

        self._mean = 0
        self._count = 0

    def reset(self):
        self._mean *= 0
        self._count *= 0

    def update(self, val, n=1):
        self._mean = (self._mean * self._count + val * n) / (self._count + n)
        self._count += n

    def value(self):
        return self._mean