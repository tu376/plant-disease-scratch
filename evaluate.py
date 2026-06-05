import cupy as cp
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

    correct = cp.sum(y_true == y_pred)

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
    # Chuyển sang CPU để tính toán argmax bằng NumPy nhằm né lỗi biên dịch của CuPy
    import numpy as np
    preds_cpu = np.argmax(cp.asnumpy(logits), axis=1)
    return cp.array(preds_cpu) # Đẩy ngược kết quả lại thành mảng CuPy


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

        # --- SỬA LÁCH LỖI CP.SUM ---
        # Chuyển cả preds và y_batch sang NumPy để tính tổng số lượng đúng
        preds_cpu = cp.asnumpy(preds)
        y_batch_cpu = cp.asnumpy(y_batch)
        total_correct += int(np.sum(preds_cpu == y_batch_cpu))

        # loss
        if criterion is not None:

            loss = criterion.forward(
                logits,
                y_batch
            )

            # Ép kiểu loss về float nguyên bản để tránh tích tụ mảng CuPy gây tràn bộ nhớ
            total_loss += float(loss.item()) * len(X_batch)

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

        total_loss += float(loss.item()) * len(X_batch)

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
    # Vì Confusion Matrix thường dùng ở cuối quá trình test trên CPU để vẽ biểu đồ (seaborn/matplotlib)
    # Nên chúng ta chuyển hẳn sang NumPy để chạy vòng lặp zip() nhanh hơn rất nhiều so với lặp trên GPU
    y_true_cpu = cp.asnumpy(y_true)
    y_pred_cpu = cp.asnumpy(y_pred)

    cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int32
    )

    for t, p in zip(y_true_cpu, y_pred_cpu):
        cm[t, p] += 1

    return cp.array(cm) # Chuyển lại về mảng CuPy nếu code bên ngoài yêu cầu