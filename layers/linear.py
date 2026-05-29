import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros((1, out_features))

        # gradients
        self.dW = None
        self.db = None

        # cache
        self.x = None
    def parameters(self):
        """Returns the learnable weights and biases of this layer."""
        return [self.W, self.b]
    def forward(self, x):
        """
        x shape: (batch_size, in_features)
        """
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout):
        """
        dout shape: (batch_size, out_features)
        """

        batch_size = self.x.shape[0]

        # gradients
        self.dW = self.x.T @ dout / batch_size
        self.db = np.sum(dout, axis=0, keepdims=True) / batch_size

        # gradient wrt input
        dx = dout @ self.W.T

        return dx

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db