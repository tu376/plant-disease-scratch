import cupy as cp
def evaluate(model, val_loader, criterion):

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in val_loader:

        X_batch = cp.asarray(images)
        y_batch = cp.asarray(labels)

        logits = model.forward(X_batch)
        loss = criterion.forward(logits, y_batch)

        preds = cp.asnumpy(logits).argmax(axis=1)
        y_np = cp.asnumpy(y_batch)

        total_correct += int((preds == y_np).sum())
        total_samples += X_batch.shape[0]
        total_loss += float(loss.item()) * X_batch.shape[0]

    val_loss = total_loss / total_samples
    val_acc = total_correct / total_samples

    return val_loss, val_acc