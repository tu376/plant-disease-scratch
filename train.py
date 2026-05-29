import argparse
import numpy as np

from model.cnn import CNN

from utils.loss import (
    CrossEntropyLoss,
    FocalLoss
)

from utils.optimizer import Adam

from evaluate import evaluate_classification


# =====================================================
# Argument Parser
# =====================================================

parser = argparse.ArgumentParser()

# -----------------------------------------
# training hyperparameters
# -----------------------------------------

parser.add_argument(
    "--epochs",
    type=int,
    default=10
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-3
)

# -----------------------------------------
# loss selection
# -----------------------------------------

parser.add_argument(
    "--loss",
    type=str,
    default="crossentropy",
    choices=[
        "crossentropy",
        "focal"
    ]
)

parser.add_argument(
    "--gamma",
    type=float,
    default=2.0
)

parser.add_argument(
    "--alpha",
    type=float,
    default=1.0
)

args = parser.parse_args()


# =====================================================
# Create Loss Function
# =====================================================

if args.loss == "crossentropy":

    criterion = CrossEntropyLoss()

else:

    criterion = FocalLoss(
        gamma=args.gamma,
        alpha=args.alpha
    )

print(f"Using loss: {args.loss}")


# =====================================================
# Mini Batch Generator
# =====================================================

def create_batches(
    X,
    y,
    batch_size
):

    num_samples = X.shape[0]

    indices = np.arange(num_samples)

    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    for i in range(
        0,
        num_samples,
        batch_size
    ):

        yield (
            X[i:i + batch_size],
            y[i:i + batch_size]
        )


# =====================================================
# Train One Epoch
# =====================================================

def train_one_epoch(
    model,
    optimizer,
    criterion,
    X_train,
    y_train,
    batch_size
):

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in create_batches(
        X_train,
        y_train,
        batch_size
    ):

        # ---------------------------------
        # forward
        # ---------------------------------

        logits = model.forward(X_batch)

        # ---------------------------------
        # loss
        # ---------------------------------

        loss = criterion.forward(
            logits,
            y_batch
        )

        total_loss += (
            loss * len(X_batch)
        )

        # ---------------------------------
        # accuracy
        # ---------------------------------

        preds = np.argmax(
            logits,
            axis=1
        )

        total_correct += np.sum(
            preds == y_batch
        )

        total_samples += len(X_batch)

        # ---------------------------------
        # backward
        # ---------------------------------

        grad = criterion.backward()

        model.backward(grad)

        # ---------------------------------
        # update
        # ---------------------------------

        optimizer.step()

        optimizer.zero_grad()

    avg_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return avg_loss, accuracy


# =====================================================
# Full Training Loop
# =====================================================

def train(
    model,
    optimizer,
    criterion,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=10,
    batch_size=32
):

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            optimizer,
            criterion,
            X_train,
            y_train,
            batch_size
        )

        # ---------------------------------
        # validation
        # ---------------------------------

        if X_val is not None:

            val_loss, val_acc = (
                evaluate_classification(
                    model,
                    X_val,
                    y_val,
                    criterion,
                    batch_size
                )
            )

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.4f} "
                f"| Val Loss: {val_loss:.4f} "
                f"| Val Acc: {val_acc:.4f}"
            )

        else:

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc:.4f}"
            )



