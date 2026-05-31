import cupy as cp
from cupy.lib.stride_tricks import as_strided

class MaxPool2D:

    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):

        self.x = x

        B, C, H, W = x.shape

        K = self.kernel_size
        S = self.stride

        out_h = (H - K) // S + 1
        out_w = (W - K) // S + 1

        shape = (
            B,
            C,
            out_h,
            out_w,
            K,
            K
        )

        strides = (
            x.strides[0],
            x.strides[1],
            x.strides[2] * S,
            x.strides[3] * S,
            x.strides[2],
            x.strides[3]
        )

        windows = as_strided(
            x,
            shape=shape,
            strides=strides
        )

        # save for backward
        self.windows = windows

        out = cp.max(
            windows,
            axis=(4, 5)
        )

        self.argmax = cp.argmax(
            windows.reshape(
                B, C, out_h, out_w, -1
            ),
            axis=-1
        )

        return out
    def backward(self, grad):

        B, C, H, W = self.x.shape

        K = self.kernel_size
        S = self.stride

        out_h = grad.shape[2]
        out_w = grad.shape[3]

        dx = cp.zeros_like(self.x)

        r = self.argmax // K
        c = self.argmax % K

        b_idx = cp.arange(B)[:, None, None, None]
        ch_idx = cp.arange(C)[None, :, None, None]

        i_idx = cp.arange(out_h)[None, None, :, None]
        j_idx = cp.arange(out_w)[None, None, None, :]

        h_idx = i_idx * S + r
        w_idx = j_idx * S + c

        cp.add.at(
            dx,
            (
                b_idx,
                ch_idx,
                h_idx,
                w_idx
            ),
            grad
        )

        return dx