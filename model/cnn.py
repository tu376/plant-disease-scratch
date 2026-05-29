import numpy as np

from layers.conv2d_cpu import Conv2D
from layers.maxpool2d import MaxPool2D
from layers.linear import Linear
from layers.activation import ReLU


class CNN:

    def __init__(
        self,
        num_classes=4
    ):

        # =====================================
        # Feature Extractor
        # =====================================

        self.conv1 = Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu1 = ReLU()

        self.pool1 = MaxPool2D(
            kernel_size=2,
            stride=2
        )

        self.conv2 = Conv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu2 = ReLU()

        self.pool2 = MaxPool2D(
            kernel_size=2,
            stride=2
        )

        # =====================================
        # Classifier
        # =====================================

        # Example input:
        # 64x64 image
        #
        # after pool1:
        # 32x32
        #
        # after pool2:
        # 16x16
        #
        # channels = 32

        self.fc1 = Linear(
            32 * 16 * 16,
            128
        )

        self.relu3 = ReLU()

        self.fc2 = Linear(
            128,
            num_classes
        )

    # =========================================
    # Forward
    # =========================================

    def forward(self, x):

        # -------------------------------------
        # Conv Block 1
        # -------------------------------------

        x = self.conv1.forward(x)

        x = self.relu1.forward(x)

        x = self.pool1.forward(x)

        # -------------------------------------
        # Conv Block 2
        # -------------------------------------

        x = self.conv2.forward(x)

        x = self.relu2.forward(x)

        x = self.pool2.forward(x)

        # -------------------------------------
        # Flatten
        # -------------------------------------

        self.feature_shape = x.shape

        B = x.shape[0]

        x = x.reshape(B, -1)

        # -------------------------------------
        # FC Block
        # -------------------------------------

        x = self.fc1.forward(x)

        x = self.relu3.forward(x)

        logits = self.fc2.forward(x)

        return logits

    # =========================================
    # Backward
    # =========================================

    def backward(self, grad):

        # -------------------------------------
        # FC
        # -------------------------------------

        grad = self.fc2.backward(grad)

        grad = self.relu3.backward(grad)

        grad = self.fc1.backward(grad)

        # -------------------------------------
        # Unflatten
        # -------------------------------------

        grad = grad.reshape(self.feature_shape)

        # -------------------------------------
        # Conv Block 2
        # -------------------------------------

        grad = self.pool2.backward(grad)

        grad = self.relu2.backward(grad)

        grad = self.conv2.backward(grad)

        # -------------------------------------
        # Conv Block 1
        # -------------------------------------

        grad = self.pool1.backward(grad)

        grad = self.relu1.backward(grad)

        grad = self.conv1.backward(grad)

        return grad

    # =========================================
    # Parameters
    # =========================================

    def parameters(self):

        params = []

        layers = [
            self.conv1,
            self.conv2,
            self.fc1,
            self.fc2
        ]

        for layer in layers:

            params.extend(
                layer.parameters()
            )

        return params
