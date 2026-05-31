import cupy as cp

class SGD:
    def __init__(self, params, lr=0.01):
        """
        params:
            list of layer parameters

        lr:
            learning rate
        """
        self.params = params
        self.lr = lr

    def step(self):
        """
        Update parameters
        """

        for param in self.params:

            # weight update
            if hasattr(param, "W"):
                param.W -= self.lr * param.dW

            # bias update
            if hasattr(param, "b"):
                param.b -= self.lr * param.db

    def zero_grad(self):
        """
        Reset gradients
        """

        for param in self.params:

            if hasattr(param, "dW"):
                param.dW.fill(0)

            if hasattr(param, "db"):
                param.db.fill(0)
class SGDMomentum:
    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0.9
    ):

        self.params = params
        self.lr = lr
        self.momentum = momentum

        self.vW = {}
        self.vb = {}

        for i, param in enumerate(params):

            if hasattr(param, "W"):
                self.vW[i] = cp.zeros_like(param.W)

            if hasattr(param, "b"):
                self.vb[i] = cp.zeros_like(param.b)

    def step(self):

        for i, param in enumerate(self.params):

            if hasattr(param, "W"):

                self.vW[i] = (
                    self.momentum * self.vW[i]
                    - self.lr * param.dW
                )

                param.W += self.vW[i]

            if hasattr(param, "b"):

                self.vb[i] = (
                    self.momentum * self.vb[i]
                    - self.lr * param.db
                )

                param.b += self.vb[i]

    def zero_grad(self):

        for param in self.params:

            if hasattr(param, "dW"):
                param.dW.fill(0)

            if hasattr(param, "db"):
                param.db.fill(0)
class Adam:

    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):

        self.params = params
        self.lr = lr

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        self.mw = {}
        self.vw = {}

        self.mb = {}
        self.vb = {}

        for i, param in enumerate(params):

            if hasattr(param, "weight"):

                self.mw[i] = cp.zeros_like(param.weight)
                self.vw[i] = cp.zeros_like(param.weight)

            if hasattr(param, "bias") and param.bias is not None:

                self.mb[i] = cp.zeros_like(param.bias)
                self.vb[i] = cp.zeros_like(param.bias)

    def step(self):

        self.t += 1

        for i, param in enumerate(self.params):

            # -------------------------
            # weight
            # -------------------------

            if (
                hasattr(param, "weight")
                and hasattr(param, "dw")
                and param.dw is not None
            ):

                self.mw[i] = (
                    self.beta1 * self.mw[i]
                    + (1 - self.beta1) * param.dw
                )

                self.vw[i] = (
                    self.beta2 * self.vw[i]
                    + (1 - self.beta2) * (param.dw ** 2)
                )

                m_hat = self.mw[i] / (
                    1 - self.beta1 ** self.t
                )

                v_hat = self.vw[i] / (
                    1 - self.beta2 ** self.t
                )

                param.weight -= (
                    self.lr
                    * m_hat
                    / (cp.sqrt(v_hat) + self.eps)
                )

            # -------------------------
            # bias
            # -------------------------

            if (
                hasattr(param, "bias")
                and hasattr(param, "db")
                and param.db is not None
            ):

                self.mb[i] = (
                    self.beta1 * self.mb[i]
                    + (1 - self.beta1) * param.db
                )

                self.vb[i] = (
                    self.beta2 * self.vb[i]
                    + (1 - self.beta2) * (param.db ** 2)
                )

                m_hat = self.mb[i] / (
                    1 - self.beta1 ** self.t
                )

                v_hat = self.vb[i] / (
                    1 - self.beta2 ** self.t
                )

                param.bias -= (
                    self.lr
                    * m_hat
                    / (cp.sqrt(v_hat) + self.eps)
                )

    def zero_grad(self):

        for param in self.params:

            if hasattr(param, "dw") and param.dw is not None:
                param.dw.fill(0)

            if hasattr(param, "db") and param.db is not None:
                param.db.fill(0)