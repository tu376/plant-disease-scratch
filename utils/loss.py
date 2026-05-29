import numpy as np


# =====================================================
# 1. Cross Entropy Loss
# =====================================================

class CrossEntropyLoss:

    def __init__(self):

        self.cache = None

    def softmax(self, logits):

        logits = logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )

        exp_logits = np.exp(logits)

        probs = exp_logits / np.sum(
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
            np.arange(N),
            targets
        ]

        loss = -np.log(correct_probs + eps)

        loss = np.mean(loss)

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
            np.arange(N),
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

        x = x - np.max(
            x,
            axis=1,
            keepdims=True
        )

        exp_x = np.exp(x)

        return exp_x / np.sum(
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
            np.arange(self.batch_size),
            targets
        ]

        pt = np.clip(
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
            * np.log(pt)
        )

        return np.mean(loss)

    def backward(self):
        """
        Approximate focal gradient

        shape:
            (B, num_classes)
        """

        grad = self.probs.copy()

        grad[
            np.arange(self.batch_size),
            self.targets
        ] -= 1

        # -------------------------
        # focal scaling
        # -------------------------

        pt = self.probs[
            np.arange(self.batch_size),
            self.targets
        ]
        focal_weight = (
            self.alpha
            * ((1 - pt) ** self.gamma)
        )
        grad *= focal_weight[:, None]
        grad /= self.batch_size
        return grad

