<<<<<<< Updated upstream
import cupy as cp

# =====================================================
# 1. Cross Entropy Loss
# =====================================================

class CrossEntropyLoss:

    def __init__(self):

        self.cache = None

    def softmax(self, logits):

        logits = logits - cp.max(
            logits,
            axis=1,
            keepdims=True
        )

        exp_logits = cp.exp(logits)

        probs = exp_logits / cp.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )

        return probs

    def forward(self, logits, targets):

        """
        logits:
            (N, C)

        targets:
            (N,)
        """

        N = logits.shape[0]

        probs = self.softmax(logits)

        eps = 1e-12

        correct_probs = probs[
            cp.arange(N),
            targets
        ]

        loss = -cp.log(correct_probs + eps)

        loss = cp.mean(loss)

        self.cache = (
            probs,
            targets,
            N
        )

        return loss

    def backward(self):

        probs, targets, N = self.cache

        dlogits = probs.copy()

        dlogits[
            cp.arange(N),
            targets
        ] -= 1

        dlogits /= N

        return dlogits
# =====================================================
# 2. Focal Loss
# =====================================================

class FocalLoss:
    def __init__(
        self,
        gamma=2.0,
        alpha=1.0,
        eps=1e-12
    ):

        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

        self.probs = None
        self.targets = None
        self.batch_size = None

    def softmax(self, x):

        x = x - cp.max(
            x,
            axis=1,
            keepdims=True
        )

        exp_x = cp.exp(x)

        return exp_x / cp.sum(
            exp_x,
            axis=1,
            keepdims=True
        )

    def forward(
        self,
        logits,
        targets
    ):
        """
        logits:
            (B, num_classes)

        targets:
            (B,)
        """

        self.batch_size = logits.shape[0]

        self.targets = targets

        # -------------------------
        # softmax probabilities
        # -------------------------

        probs = self.softmax(logits)

        self.probs = probs

        # -------------------------
        # true class probabilities
        # -------------------------

        pt = probs[
            cp.arange(self.batch_size),
            targets
        ]

        pt = cp.clip(
            pt,
            self.eps,
            1.0
        )

        # -------------------------
        # focal loss
        # -------------------------

        loss = (
            -self.alpha
            * ((1 - pt) ** self.gamma)
            * cp.log(pt)
        )

        return cp.mean(loss)

    def backward(self):
        """
        Approximate focal gradient

        shape:
            (B, num_classes)
        """

        grad = self.probs.copy()

        grad[
            cp.arange(self.batch_size),
            self.targets
        ] -= 1

        # -------------------------
        # focal scaling
        # -------------------------

        pt = self.probs[
            cp.arange(self.batch_size),
            self.targets
        ]
        focal_weight = (
            self.alpha
            * ((1 - pt) ** self.gamma)
        )
        grad *= focal_weight[:, None]
        grad /= self.batch_size
        return grad

=======
"""
TẠM THỜI
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend import cp


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self._probs = None
        self._y     = None
        self._N     = None

    def forward(self, logits, y_true):
        N = logits.shape[0]
        self._N = N
        self._y = y_true

        shifted = logits - cp.max(logits, axis=1, keepdims=True)
        exp_val = cp.exp(shifted)
        probs   = exp_val / cp.sum(exp_val, axis=1, keepdims=True)
        self._probs = probs

        correct_log_probs = -cp.log(probs[cp.arange(N), y_true] + 1e-9)
        loss = float(cp.mean(correct_log_probs))
        return loss

    def backward(self):
        grad = self._probs.copy()
        grad[cp.arange(self._N), self._y] -= 1.0
        grad /= self._N
        return grad

    def get_probs(self):
        return self._probs
>>>>>>> Stashed changes
