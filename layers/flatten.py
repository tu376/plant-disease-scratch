import cupy as cp
class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape

        batch_size = x.shape[0]

        return x.reshape(batch_size, -1)

    def backward(self, grad_output):
        return grad_output.reshape(
            self.input_shape
        )