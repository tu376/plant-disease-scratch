import numpy as cp
from cupy.lib.stride_tricks import as_strided


class Conv2D:

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True
    ):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        KH, KW = kernel_size

        scale = cp.sqrt(2.0 / (in_channels * KH * KW))

        self.weight = (
            cp.random.randn(
                out_channels,
                in_channels,
                KH,
                KW
            ) * scale
        )

        self.bias = (
            cp.zeros(out_channels)
            if bias else None
        )

        self.dw = None
        self.db = None

        self.cache = None

    # =================================================
    # im2col
    # =================================================

    def im2col(self, x):

        N, C, H, W = x.shape

        KH, KW = self.kernel_size
        S = self.stride
        P = self.padding

        x_padded = cp.pad(
            x,
            ((0, 0), (0, 0), (P, P), (P, P)),
            mode='constant'
        )

        H_p, W_p = x_padded.shape[2:]

        OH = (H_p - KH) // S + 1
        OW = (W_p - KW) // S + 1

        shape = (
            N,
            C,
            OH,
            OW,
            KH,
            KW
        )

        strides = (
            x_padded.strides[0],
            x_padded.strides[1],
            x_padded.strides[2] * S,
            x_padded.strides[3] * S,
            x_padded.strides[2],
            x_padded.strides[3]
        )

        cols = as_strided(
            x_padded,
            shape=shape,
            strides=strides
        )

        cols = cols.transpose(
            0, 2, 3, 1, 4, 5
        )

        cols = cols.reshape(
            N * OH * OW,
            -1
        )

        return cols, x_padded.shape

    # =================================================
    # col2im
    # =================================================

    def col2im(self, cols, x_shape):

        N, C, H_p, W_p = x_shape

        KH, KW = self.kernel_size
        S = self.stride
        P = self.padding

        OH = (H_p - KH) // S + 1
        OW = (W_p - KW) // S + 1

        cols = cols.reshape(
            N,
            OH,
            OW,
            C,
            KH,
            KW
        )

        cols = cols.transpose(
            0, 3, 1, 2, 4, 5
        )

        x_padded = cp.zeros(
            (N, C, H_p, W_p),
            dtype=cols.dtype
        )

        for y in range(KH):

            y_max = y + S * OH

            for x in range(KW):

                x_max = x + S * OW

                x_padded[
                    :,
                    :,
                    y:y_max:S,
                    x:x_max:S
                ] += cols[:, :, :, :, y, x]

        if P == 0:
            return x_padded

        return x_padded[
            :,
            :,
            P:-P,
            P:-P
        ]

    # =================================================
    # forward
    # =================================================

    def forward(self, x):

        N, C, H, W = x.shape

        KH, KW = self.kernel_size
        S = self.stride
        P = self.padding

        OH = (H + 2 * P - KH) // S + 1
        OW = (W + 2 * P - KW) // S + 1

        cols, padded_shape = self.im2col(x)

        w_col = self.weight.reshape(
            self.out_channels,
            -1
        )

        out = cols @ w_col.T

        if self.bias is not None:
            out += self.bias

        out = out.reshape(
            N,
            OH,
            OW,
            self.out_channels
        )

        out = out.transpose(
            0,
            3,
            1,
            2
        )

        self.cache = (
            x,
            cols,
            padded_shape,
            w_col
        )

        return out

    # =================================================
    # backward
    # =================================================

    def backward(self, dout):

        x, cols, padded_shape, w_col = self.cache

        N, OC, OH, OW = dout.shape

        dout_reshaped = (
            dout
            .transpose(0, 2, 3, 1)
            .reshape(-1, OC)
        )

        # ----------------------------
        # db
        # ----------------------------

        if self.bias is not None:

            self.db = cp.sum(
                dout_reshaped,
                axis=0
            )

        # ----------------------------
        # dw
        # ----------------------------

        dw = dout_reshaped.T @ cols

        self.dw = dw.reshape(
            self.weight.shape
        )

        # ----------------------------
        # dx
        # ----------------------------

        dcols = dout_reshaped @ w_col

        dx = self.col2im(
            dcols,
            padded_shape
        )

        return dx
def eval_numerical_gradient(f, x, eps=1e-5):

    grad = cp.zeros_like(x)

    it = cp.ndindex(*x.shape)

    for idx in it:

        old = x[idx]

        x[idx] = old + eps
        fx1 = f(x)

        x[idx] = old - eps
        fx2 = f(x)

        x[idx] = old

        grad[idx] = (fx1 - fx2) / (2 * eps)

    return grad
def rel_error(x, y):

    return cp.max(
        cp.abs(x - y) /
        cp.maximum(
            1e-8,
            cp.abs(x) + cp.abs(y)
        )
    )
cp.random.seed(0)

# small test
N = 2
C = 3