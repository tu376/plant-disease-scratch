import os
os.environ["CUPY_ACCELERATORS"] = ""
import argparse
import cupy as cp
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from evaluate import evaluate
from model.cnn import CNN
from utils.optimizer import Adam, SGDMomentum
from utils.loss import CrossEntropyLoss, FocalLoss
from utils.data_loader import create_dataloaders, dataloader_to_cupy

'''
TRAIN_DIR = r"E:\GitHub\plant-disease-scratch\data\train"
VAL_DIR = r"E:\GitHub\plant-disease-scratch\data\val"
TEST_DIR = r"E:\GitHub\plant-disease-scratch\data\test"
'''
# Vinh path
TRAIN_DIR = r"C:\Users\NCPC\Pictures\plant-disease-scratch\data\train"
VAL_DIR = r"C:\Users\NCPC\Pictures\plant-disease-scratch\data\val"
TEST_DIR = r"C:\Users\NCPC\Pictures\plant-disease-scratch\data\test"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--loss", type=str, default="crossentropy", choices=["crossentropy", "focal"])
parser.add_argument("--gamma", type=float, default=2.0)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgdmomentum"])
args = parser.parse_args()

print("Using device: CUDA (CuPy)")

train_loader, val_loader, test_loader, classes = create_dataloaders(
    train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR,
    batch_size=args.batch_size, image_size=64
)
num_classes = len(classes)
print("Classes:", classes)

def get_all_features(model, dataloader):
    all_features = []
    all_labels = []
    for X_batch, y_batch in dataloader:
        X_batch = cp.asarray(X_batch)
        y_batch = cp.asarray(y_batch)
        
        features = model.extract_features(X_batch)
        all_features.append(cp.asnumpy(features))
        all_labels.append(cp.asnumpy(y_batch))
        
    return np.vstack(all_features), np.hstack(all_labels)

cnn = CNN(num_classes=num_classes)

if args.loss == "crossentropy":
    criterion = CrossEntropyLoss()
else:
    criterion = FocalLoss(gamma=args.gamma, alpha=args.alpha)

if args.optimizer == "adam":
    optimizer = Adam(cnn.parameters(), lr=args.lr)
else:
    optimizer = SGDMomentum(cnn.parameters(), lr=args.lr)

X_val, y_val = dataloader_to_cupy(val_loader)
X_test, y_test = dataloader_to_cupy(test_loader)

best_val_acc = 0.0

for epoch in range(args.epochs):
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in train_loader:
        X_batch = cp.asarray(images)
        y_batch = cp.asarray(labels)

        logits = cnn.forward(X_batch)
        loss = criterion.forward(logits, y_batch)

        preds = cp.asnumpy(logits).argmax(axis=1)
        y_np = cp.asnumpy(y_batch)

        total_correct += int((preds == y_np).sum())
        total_samples += X_batch.shape[0]
        total_loss += float(loss.item()) * X_batch.shape[0]

        grad = criterion.backward()
        cnn.backward(grad)
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    val_loss, val_acc = evaluate(cnn, val_loader, criterion)

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        cp.savez(
            "best_cnn_weights.npz",
            conv1_w=cnn.conv1.weight,
            conv1_b=cnn.conv1.bias,
            conv2_w=cnn.conv2.weight,
            conv2_b=cnn.conv2.bias,
            fc1_w=cnn.fc1.weight,
            fc1_b=cnn.fc1.bias,
            fc2_w=cnn.fc2.weight,
            fc2_b=cnn.fc2.bias
        )

    print(
        f"Epoch [{epoch+1}/{args.epochs}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_acc:.4f}"
    )
print("\nEvaluating BEST model on Test Set...")
weights = cp.load("best_cnn_weights.npz")
cnn.conv1.weight[:], cnn.conv1.bias[:] = weights["conv1_w"], weights["conv1_b"]
cnn.conv2.weight[:], cnn.conv2.bias[:] = weights["conv2_w"], weights["conv2_b"]
cnn.fc1.weight[:], cnn.fc1.bias[:] = weights["fc1_w"], weights["fc1_b"]
cnn.fc2.weight[:], cnn.fc2.bias[:] = weights["fc2_w"], weights["fc2_b"]

all_preds, all_targets = [], []
for images, labels in test_loader:
    X_batch, y_batch = cp.asarray(images), cp.asarray(labels)
    logits = cnn.forward(X_batch)
    preds = cp.argmax(logits, axis=1)
    
    all_preds.extend(cp.asnumpy(preds))
    all_targets.extend(cp.asnumpy(y_batch))

all_preds, all_targets = np.array(all_preds), np.array(all_targets)

print("\n" + "=" * 60 + "\nCLASSIFICATION REPORT\n" + "=" * 60)
print(classification_report(all_targets, all_preds, target_names=classes, digits=4))

print("\n" + "=" * 60 + "\nCONFUSION MATRIX\n" + "=" * 60)
print(confusion_matrix(all_targets, all_preds))

print("\n" + "=" * 60 + "\nExtracting features for classical ML models...")
X_train, y_train = get_all_features(cnn, train_loader)
X_val, y_val = get_all_features(cnn, val_loader)
X_test, y_test = get_all_features(cnn, test_loader)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

os.makedirs("features", exist_ok=True)

np.save("features/classes.npy", np.array(classes))
np.save("features/X_train.npy", X_train)
np.save("features/y_train.npy", y_train)
np.save("features/X_val.npy", X_val)
np.save("features/y_val.npy", y_val)
np.save("features/X_test.npy", X_test)
np.save("features/y_test.npy", y_test)

print("Features saved successfully.")