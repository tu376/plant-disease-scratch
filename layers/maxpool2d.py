import numpy as np


class MaxPool2D:

    def __init__(
        self,
        kernel_size=2,
        stride=2
    ):

        self.kernel_size = kernel_size
        self.stride = stride

    # =========================================
    # Forward
    # =========================================

    def forward(self, x):
        """
        x shape:
            (B, C, H, W)
        """

        self.x = x

        B, C, H, W = x.shape

        K = self.kernel_size
        S = self.stride

        out_h = (H - K) // S + 1
        out_w = (W - K) // S + 1

        out = np.zeros(
            (B, C, out_h, out_w)
        )

        # store max positions
        self.max_indices = {}

        for b in range(B):

            for c in range(C):

                for i in range(out_h):

                    for j in range(out_w):

                        h_start = i * S
                        h_end = h_start + K

                        w_start = j * S
                        w_end = w_start + K

                        window = x[
                            b,
                            c,
                            h_start:h_end,
                            w_start:w_end
                        ]

                        max_val = np.max(window)

                        out[b, c, i, j] = max_val

                        # ---------------------------------
                        # store max index for backward
                        # ---------------------------------

                        max_pos = np.unravel_index(
                            np.argmax(window),
                            window.shape
                        )

                        self.max_indices[
                            (b, c, i, j)
                        ] = (
                            h_start + max_pos[0],
                            w_start + max_pos[1]
                        )

        return out

    # =========================================
    # Backward
    # =========================================

    def backward(self, grad):
        """
        grad shape:
            (B, C, out_h, out_w)
        """

        B, C, H, W = self.x.shape

        dx = np.zeros_like(self.x)

        out_h = grad.shape[2]
        out_w = grad.shape[3]

        for b in range(B):

            for c in range(C):

                for i in range(out_h):

                    for j in range(out_w):

                        h_idx, w_idx = self.max_indices[
                            (b, c, i, j)
                        ]

                        dx[
                            b,
                            c,
                            h_idx,
                            w_idx
                        ] += grad[
                            b,
                            c,
                            i,
                            j
                        ]

        return dx

    # =========================================
    # Parameters
    # =========================================

    def parameters(self):

        return []
