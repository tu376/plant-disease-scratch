import numpy as np

class MaxPool2D:
    """
    MaxPool2D layer with forward and backward pass.
    Stores binary mask of argmax positions for backprop.
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.mask = None

    def forward(self, input):
        """
        Forward pass using sliding window max pooling.
        Input:  (C, H, W)
        Output: (C, H_out, W_out)
        """
        self.input = input
        C, H, W = input.shape

        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        output = np.zeros((C, H_out, W_out))
        self.mask = np.zeros_like(input)

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    h_end   = h_start + self.pool_size
                    w_start = j * self.stride
                    w_end   = w_start + self.pool_size

                    patch = input[c, h_start:h_end, w_start:w_end]
                    max_val = np.max(patch)
                    output[c, i, j] = max_val

                    # Build mask: 1 at the position of the maximum value
                    max_idx = np.argmax(patch)
                    row = max_idx // self.pool_size
                    col = max_idx % self.pool_size
                    self.mask[c, h_start + row, w_start + col] = 1

        return output

    def backward(self, d_output):
        """
        Backward pass: route gradient only to the max position.
        d_output: (C, H_out, W_out)
        Returns d_input: (C, H, W)
        """
        C, H_out, W_out = d_output.shape
        d_input = np.zeros_like(self.input)

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    h_end   = h_start + self.pool_size
                    w_start = j * self.stride
                    w_end   = w_start + self.pool_size

                    patch_mask = self.mask[c, h_start:h_end, w_start:w_end]
                    d_input[c, h_start:h_end, w_start:w_end] += patch_mask * d_output[c, i, j]

        return d_input