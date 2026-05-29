import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Classification accuracy

    Input:
        y_true: (N,)
        y_pred: (N,)

    Output:
        accuracy: float
    """

    correct = np.sum(y_true == y_pred)

    total = len(y_true)

    return correct / total


def predict(logits):
    """
    Convert logits to predicted class

    Input:
        logits:
            shape (B, num_classes)

    Output:
        predicted labels:
            shape (B,)
    """

    return np.argmax(logits, axis=1)


def evaluate_classification(
    model,
    X,
    y,
    criterion=None,
    batch_size=32
):
    """
    Evaluate classification model

    Returns:
        average loss
        accuracy
    """

    num_samples = X.shape[0]

    total_loss = 0
    total_correct = 0

    for i in range(0, num_samples, batch_size):

        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        # forward
        logits = model.forward(X_batch)

        # predictions
        preds = predict(logits)

        # accuracy
        total_correct += np.sum(
            preds == y_batch
        )

        # loss
        if criterion is not None:

            loss = criterion.forward(
                logits,
                y_batch
            )

            total_loss += (
                loss * len(X_batch)
            )

    accuracy = total_correct / num_samples

    if criterion is not None:
        avg_loss = total_loss / num_samples
    else:
        avg_loss = None

    return avg_loss, accuracy


def evaluate_regression(
    model,
    X,
    y,
    criterion,
    batch_size=32
):
    """
    Evaluate regression model
    """

    num_samples = X.shape[0]

    total_loss = 0

    for i in range(0, num_samples, batch_size):

        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        pred = model.forward(X_batch)

        loss = criterion.forward(
            pred,
            y_batch
        )

        total_loss += (
            loss * len(X_batch)
        )

    avg_loss = total_loss / num_samples

    return avg_loss


def confusion_matrix(
    y_true,
    y_pred,
    num_classes
):
    """
    Build confusion matrix

    Output:
        shape (num_classes, num_classes)

    rows:
        true labels

    cols:
        predicted labels
    """

    cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int32
    )

    for t, p in zip(y_true, y_pred):

        cm[t, p] += 1

    return cm