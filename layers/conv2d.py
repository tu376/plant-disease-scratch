import numpy as np

class Conv2D:
    """
    2-D Convolution layer.
      input  : (C_in,  H_in,  W_in)
      output : (C_out, H_out, W_out)
      kernels: (C_out, C_in, K, K)
    stride = 1, configurable padding.
    Forward  uses im2col  (patch → column matrix).
    Backward uses col2im  (column matrix → patch accumulation).
    """

    def __init__(self, input_shape, kernel_size, C_out, padding=0):
        C_in, H_in, W_in = input_shape

        H_out = H_in + 2 * padding - kernel_size + 1
        W_out = W_in + 2 * padding - kernel_size + 1

        self.C_in        = C_in
        self.H_in        = H_in
        self.W_in        = W_in
        self.C_out       = C_out
        self.H_out       = H_out
        self.W_out       = W_out
        self.kernel_size = kernel_size
        self.padding     = padding

        self.output_shape = (C_out, H_out, W_out)
        self.kernel_shape = (C_out, C_in, kernel_size, kernel_size)

        # He initialisation
        fan_in = C_in * kernel_size * kernel_size
        self.kernels = np.random.randn(*self.kernel_shape) * np.sqrt(2.0 / fan_in)
        self.biases  = np.zeros(C_out)

        self.d_kernels = np.zeros_like(self.kernels)
        self.d_biases  = np.zeros_like(self.biases)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def pad_input(self, input):
        if self.padding == 0:
            return input
        return np.pad(input,
                      ((0, 0),
                       (self.padding, self.padding),
                       (self.padding, self.padding)),
                      mode='constant')

    def im2col(self, input_padded):
        """
        Rearrange every kernel-sized patch into a column.
        Returns array of shape (C_in*K*K, H_out*W_out).
        """
        K = self.kernel_size
        cols = []
        for i in range(self.H_out):
            for j in range(self.W_out):
                patch = input_padded[:, i:i+K, j:j+K]  # (C_in, K, K)
                cols.append(patch.reshape(-1))           # (C_in*K*K,)
        return np.array(cols).T  # (C_in*K*K, H_out*W_out)

    def col2im(self, d_input_col):
        """
        Accumulate column gradients back into the (padded) input tensor.
        d_input_col: (C_in*K*K, H_out*W_out)
        Returns d_input: (C_in, H_in, W_in)  (padding removed).
        """
        K         = self.kernel_size
        padded_H  = self.H_in + 2 * self.padding
        padded_W  = self.W_in + 2 * self.padding

        d_input_padded = np.zeros((self.C_in, padded_H, padded_W))

        col_idx = 0
        for i in range(self.H_out):
            for j in range(self.W_out):
                patch = d_input_col[:, col_idx].reshape(self.C_in, K, K)
                d_input_padded[:, i:i+K, j:j+K] += patch
                col_idx += 1

        if self.padding == 0:
            return d_input_padded
        return d_input_padded[:, self.padding:-self.padding,
                                 self.padding:-self.padding]

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------

    def forward(self, input):
        """
        Forward pass via im2col + matrix multiply.
        input:  (C_in, H_in, W_in)
        output: (C_out, H_out, W_out)
        """
        self.input        = input
        self.input_padded = self.pad_input(input)
        self.input_col    = self.im2col(self.input_padded)

        kernels_col = self.kernels.reshape(self.C_out, -1)        # (C_out, C_in*K*K)
        output      = kernels_col @ self.input_col + self.biases[:, None]  # (C_out, H_out*W_out)

        return output.reshape(self.C_out, self.H_out, self.W_out)

    def backward(self, d_output):
        """
        Backward pass; stores d_kernels, d_biases; returns d_input.
        d_output: (C_out, H_out, W_out)
        """
        d_output_col = d_output.reshape(self.C_out, -1)  # (C_out, H_out*W_out)

        # dL/db  — sum over spatial positions
        self.d_biases = d_output_col.sum(axis=1)          # (C_out,)

        # dL/dW  — outer product in column space
        d_kernels_col  = d_output_col @ self.input_col.T  # (C_out, C_in*K*K)
        self.d_kernels = d_kernels_col.reshape(self.kernel_shape)

        # dL/dX  — backprop through im2col
        kernels_col  = self.kernels.reshape(self.C_out, -1)
        d_input_col  = kernels_col.T @ d_output_col       # (C_in*K*K, H_out*W_out)
        d_input      = self.col2im(d_input_col)

        return d_input

    def update(self, learning_rate):
        self.kernels -= learning_rate * self.d_kernels
        self.biases  -= learning_rate * self.d_biases
