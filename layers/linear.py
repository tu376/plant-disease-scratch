import cupy as cp

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = cp.random.randn(in_features, out_features) * cp.sqrt(2 / in_features)
        self.bias = cp.zeros((1, out_features))

        # gradients
        self.dw = None
        self.d = None

        # cache
        self.x = None
    def parameters(self):
        """Returns the learnable weights and biases of this layer."""
        return [self.weight, self.bias]
    def forward(self, x):
        """
        x shape: (batch_size, in_features)
        """
        self.x = x
        out = x @ self.weight + self.bias
        return out

    def backward(self, dout):
        """
        dout shape: (batch_size, out_features)
        """

        batch_size = self.x.shape[0]

        # gradients
        self.dw = self.x.T @ dout / batch_size
        self.db = cp.sum(dout, axis=0, keepdims=True) / batch_size

        # gradient wrt input
        dx = dout @ self.weight.T

        return dx

    def step(self, lr):
        self.weight -= lr * self.dw
        self.bias -= lr * self.db