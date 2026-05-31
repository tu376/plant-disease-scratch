<<<<<<< Updated upstream
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
=======
"""
optimizer.py
SGD with Momentum và Adam – tự viết hoàn toàn, không dùng torch.optim.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from backend import cp


class SGD:
    """
    Stochastic Gradient Descent with Momentum.

    Công thức update:
        v  = momentum * v - lr * grad
        W += v
    """

    def __init__(self, lr=0.01, momentum=0.9, weight_decay=1e-4):
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self._velocity    = {}   # key: (layer_idx, param_name) -> velocity

    def step(self, params_and_grads):
        """
        params_and_grads: list của (layer_idx, param_name, layer, param, grad)
        """
        for (layer_idx, name, layer, param, grad) in params_and_grads:
            key = (layer_idx, name)

            if key not in self._velocity:
                self._velocity[key] = cp.zeros_like(param)

            v = self._velocity[key]

            # L2 regularization
            effective_grad = grad + self.weight_decay * param

            # Momentum update
            v = self.momentum * v - self.lr * effective_grad
            self._velocity[key] = v

            # In-place update param trong layer
            params = layer.get_params()
            params[name] = params[name] + v
            layer.set_params(params)

    def set_lr(self, new_lr):
        self.lr = new_lr


class Adam:
    """
    Adam optimizer (Adaptive Moment Estimation).

    Công thức:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        W -= lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self._m   = {}   # first moment
        self._v   = {}   # second moment
        self._t   = {}   # timestep per param

    def step(self, params_and_grads):
        for (layer_idx, name, layer, param, grad) in params_and_grads:
            key = (layer_idx, name)

            if key not in self._m:
                self._m[key] = cp.zeros_like(param)
                self._v[key] = cp.zeros_like(param)
                self._t[key] = 0

            self._t[key] += 1
            t = self._t[key]

            # L2 regularization
            effective_grad = grad + self.weight_decay * param

            # Moment updates
            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * effective_grad
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * (effective_grad ** 2)

            # Bias correction
            m_hat = self._m[key] / (1 - self.beta1 ** t)
            v_hat = self._v[key] / (1 - self.beta2 ** t)

            # Update
            params = layer.get_params()
            params[name] = params[name] - self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
            layer.set_params(params)

    def set_lr(self, new_lr):
        self.lr = new_lr
>>>>>>> Stashed changes
