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
        self.x = x
        
        # --- LÁCH LỖI CUBLAS CHO CHIỀU FORWARD TẦNG LINEAR ---
        try:
            out = x @ self.weight
        except Exception as e:
            import numpy as np
            x_cpu = cp.asnumpy(x)
            w_cpu = cp.asnumpy(self.weight)
            out = cp.array(x_cpu @ w_cpu)
            
        if self.bias is not None:
            out += self.bias
            
        return out

    def backward(self, dout):
        batch_size = dout.shape[0]
        
        # --- LÁCH LỖI CUBLAS CHO self.dw ---
        try:
            self.dw = self.x.T @ dout / batch_size
        except Exception as e:
            import numpy as np
            x_cpu = cp.asnumpy(self.x)
            dout_cpu = cp.asnumpy(dout)
            self.dw = cp.array(x_cpu.T @ dout_cpu / batch_size)
            
        self.db = cp.sum(dout, axis=0) / batch_size
        
        # --- LÁCH LỖI CUBLAS CHO dx ---
        try:
            dx = dout @ self.weight.T
        except Exception as e:
            import numpy as np
            dout_cpu = cp.asnumpy(dout)
            w_cpu = cp.asnumpy(self.weight)
            dx = cp.array(dout_cpu @ w_cpu.T)
            
        return dx

    def step(self, lr):
        self.weight -= lr * self.dw
        self.bias -= lr * self.db