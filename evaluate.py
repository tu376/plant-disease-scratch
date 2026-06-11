import cupy as cp
def evaluate(model, val_loader, criterion):

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in val_loader:

        X_batch = cp.asarray(images)
        y_batch = cp.asarray(labels)

        logits = model.forward(X_batch)

        loss = criterion.forward(
            logits,
            y_batch
        )

        preds = cp.argmax(
            logits,
            axis=1
        )

        total_correct += int(
            cp.sum(preds == y_batch).item()
        )

        total_samples += X_batch.shape[0]

        total_loss += (
            float(loss.item()) * len(X_batch)
        )

    val_loss = total_loss / total_samples
    val_acc = total_correct / total_samples

    return val_loss, val_acc