```python
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
# 2. Mean Squared Error Loss
# =====================================================

class MSELoss:

    def __init__(self):

        self.cache = None

    def forward(self, predictions, targets):

        loss = np.mean(
            (predictions - targets) ** 2
        )

        self.cache = (
            predictions,
            targets
        )

        return loss

    def backward(self):

        predictions, targets = self.cache

        N = predictions.size

        dpred = (
            2.0 * (predictions - targets)
        ) / N

        return dpred


# =====================================================
# 3. Binary Cross Entropy Loss
# =====================================================

class BCELoss:

    def __init__(self):

        self.cache = None

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def forward(self, logits, targets):

        """
        logits:
            (N,)

        targets:
            (N,)
        """

        probs = self.sigmoid(logits)

        eps = 1e-12

        loss = -(
            targets * np.log(probs + eps)
            +
            (1 - targets)
            * np.log(1 - probs + eps)
        )

        loss = np.mean(loss)

        self.cache = (
            probs,
            targets
        )

        return loss

    def backward(self):

        probs, targets = self.cache

        N = probs.shape[0]

        dlogits = (
            probs - targets
        ) / N

        return dlogits


# =====================================================
# 4. Focal Loss
# =====================================================

class FocalLoss:

    def __init__(
        self,
        gamma=2.0,
        alpha=1.0
    ):

        self.gamma = gamma
        self.alpha = alpha

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

        pt = probs[
            np.arange(N),
            targets
        ]

        focal_weight = (
            self.alpha
            *
            (1 - pt) ** self.gamma
        )

        loss = (
            -focal_weight
            *
            np.log(pt + eps)
        )

        loss = np.mean(loss)

        self.cache = (
            probs,
            targets,
            pt,
            N
        )

        return loss

    def backward(self):

        probs, targets, pt, N = self.cache

        dlogits = probs.copy()

        dlogits[
            np.arange(N),
            targets
        ] -= 1

        focal_weight = (
            self.alpha
            *
            (1 - pt) ** self.gamma
        )

        dlogits *= focal_weight.reshape(-1, 1)

        dlogits /= N

        return dlogits


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    np.random.seed(0)

    # -------------------------------------------------
    # classification
    # -------------------------------------------------

    logits = np.random.randn(4, 10)

    targets = np.array([1, 3, 5, 2])

    # Cross Entropy
    ce = CrossEntropyLoss()

    ce_loss = ce.forward(
        logits,
        targets
    )

    ce_grad = ce.backward()

    print("CrossEntropy Loss")
    print("loss:", ce_loss)
    print("grad shape:", ce_grad.shape)

    # Focal Loss
    focal = FocalLoss(
        gamma=2.0,
        alpha=1.0
    )

    focal_loss = focal.forward(
        logits,
        targets
    )

    focal_grad = focal.backward()

    print("\nFocal Loss")
    print("loss:", focal_loss)
    print("grad shape:", focal_grad.shape)

    # -------------------------------------------------
    # regression
    # -------------------------------------------------

    pred = np.random.randn(5, 2)

    target = np.random.randn(5, 2)

    mse = MSELoss()

    mse_loss = mse.forward(
        pred,
        target
    )

    mse_grad = mse.backward()

    print("\nMSE Loss")
    print("loss:", mse_loss)
    print("grad shape:", mse_grad.shape)

    # -------------------------------------------------
    # binary classification
    # -------------------------------------------------

    binary_logits = np.random.randn(6)

    binary_targets = np.array(
        [1, 0, 1, 0, 1, 1]
    )

    bce = BCELoss()

    bce_loss = bce.forward(
        binary_logits,
        binary_targets
    )

    bce_grad = bce.backward()

    print("\nBCE Loss")
    print("loss:", bce_loss)
    print("grad shape:", bce_grad.shape)
```
