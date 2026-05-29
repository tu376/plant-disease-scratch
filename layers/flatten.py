import numpy as np


class Flatten:
    def __init__(self):

        # save original shape
        self.input_shape = None

    def forward(self, x):
        """
        Flatten forward pass

        Input:
            x shape:
                (B, C, H, W)

        Output:
            (B, C*H*W)
        """

        self.input_shape = x.shape

        batch_size = x.shape[0]

        return x.reshape(batch_size, -1)

    def backward(self, grad_output):
        """
        Restore original tensor shape

        Input:
            grad_output:
                (B, C*H*W)

        Output:
            (B, C, H, W)
        """

        return grad_output.reshape(
            self.input_shape
        )