import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        mask = x > 0
        """
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        """
        ReLU derivative:
            1 if x > 0
            0 otherwise

        masked gradient
        """
        dx = dout * self.mask
        return dx