import argparse
import cupy as cp
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import evaluate
from model.cnn import CNN
from model.xgboost import XGBoostModel
from model.svm import SVMModel
from model.random_forest import RandomForestModel
from model.svm import SVMModel
from model.logistic_regression import LogisticRegressionModel
from model.knn import KNeighborsClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt

from utils.optimizer import (
    Adam,
    SGDMomentum
)

from utils.loss import (
    CrossEntropyLoss,
    FocalLoss
)

from utils.data_loader import (
    create_dataloaders,
    dataloader_to_cupy
)

# =====================================================
# Paths
# =====================================================

TRAIN_DIR = r"E:\GitHub\plant-disease-scratch\data\train"
VAL_DIR = r"E:\GitHub\plant-disease-scratch\data\val"
TEST_DIR = r"E:\GitHub\plant-disease-scratch\data\test"

# =====================================================
# Args
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="cnn",
    choices=["cnn", "xgboost", "random_forest", "logistic_regression", 'knn','svm']
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=4
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-3
)

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

parser.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    choices=[
        "adam",
        "sgdmomentum"
    ]
)

args = parser.parse_args()

# =====================================================
# Data
# =====================================================

print("Using device: CUDA (CuPy)")

train_loader, val_loader, test_loader, classes = create_dataloaders(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    test_dir=TEST_DIR,
    batch_size=args.batch_size,
    image_size=64
)

num_classes = len(classes)

print("Classes:", classes)

# =====================================================
# Helper
# =====================================================

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

# =====================================================
# CNN
# =====================================================

cnn = CNN(
    num_classes=num_classes
)

# -----------------------------
# Loss
# -----------------------------

if args.loss == "crossentropy":

    criterion = CrossEntropyLoss()

else:

    criterion = FocalLoss(
        gamma=args.gamma,
        alpha=args.alpha
    )

# -----------------------------
# Optimizer
# -----------------------------

if args.optimizer == "adam":

    optimizer = Adam(
        cnn.parameters(),
        lr=args.lr
    )

else:

    optimizer = SGDMomentum(
        cnn.parameters(),
        lr=args.lr
    )

# -----------------------------
# Validation set
# -----------------------------

X_val, y_val = dataloader_to_cupy(
    val_loader
)

X_test, y_test = dataloader_to_cupy(
    test_loader
)

# -----------------------------
# Training loop
# -----------------------------
best_val_acc = 0.0
for epoch in range(args.epochs):

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:

        X_batch = cp.asarray(images)
        y_batch = cp.asarray(labels)

        # forward

        logits = cnn.forward(
            X_batch
        )

        loss = criterion.forward(
            logits,
            y_batch
        )

        # accuracy

        preds = cp.argmax(
            logits,
            axis=1
        )

        total_correct += int(
            cp.sum(
                preds == y_batch
            ).item()
        )

        total_samples += X_batch.shape[0]

        total_loss += (
            float(loss.item())
            * X_batch.shape[0]
        )

        # backward

        grad = criterion.backward()

        cnn.backward(
            grad
        )

        # update

        optimizer.step()

        optimizer.zero_grad()

    train_loss = (
        total_loss / total_samples
    )

    train_acc = (
        total_correct / total_samples
    )

    # validation

    val_loss, val_acc = (
        evaluate(
            cnn,
            X_val,
            y_val,
            criterion
        )
    )

    val_loss = float(
        val_loss.item()
    )

    val_acc = float(
        val_acc.item()
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Save the best model weights
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
weights = cp.load("best_cnn_weights.npz")

cnn.conv1.weight[:] = weights["conv1_w"]
cnn.conv1.bias[:]   = weights["conv1_b"]

cnn.conv2.weight[:] = weights["conv2_w"]
cnn.conv2.bias[:]   = weights["conv2_b"]

cnn.fc1.weight[:] = weights["fc1_w"]
cnn.fc1.bias[:]   = weights["fc1_b"]

cnn.fc2.weight[:] = weights["fc2_w"]
cnn.fc2.bias[:]   = weights["fc2_b"]
print("\nEvaluating BEST model on Test Set...")

all_preds = []
all_targets = []

for images, labels in test_loader:

    X_batch = cp.asarray(images)
    y_batch = cp.asarray(labels)

    logits = cnn.forward(X_batch)

    preds = cp.argmax(logits, axis=1)

    all_preds.extend(cp.asnumpy(preds))
    all_targets.extend(cp.asnumpy(y_batch))

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_targets,
        all_preds,
        target_names=classes,
        digits=4
    )
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    all_targets,
    all_preds
)

print(cm)
print("preparing features for classical ML models...")
print("\n" + "=" * 60)
print(
    "Extracting features..."
)

X_train, y_train = get_all_features(
    cnn,
    train_loader
)

X_val, y_val = get_all_features(
    cnn,
    val_loader
)

X_test, y_test = get_all_features(
    cnn,
    test_loader
)

print(
    "Train:",
    X_train.shape,
    y_train.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)

if args.model == "xgboost":
# -----------------------------
# XGBoost
# -----------------------------

    xgb = XGBoostModel(
        num_classes=num_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )

    print(
        "Training XGBoost..."
    )

    xgb.fit(
        X_train,
        y_train
    )

    pred = xgb.predict(
        X_test
    )

    acc = (
        pred == y_test
    ).mean()

    print(
        f"Test Accuracy: {acc:.4f}"
    )
    plot_importance(
        xgb.model,
        importance_type='gain',
        max_num_features=20
    )
    f1 = f1_score(
        y_test,
        pred,
        average="weighted"
    )

    print(f"F1-score: {f1:.4f}")
    print(
        classification_report(
            y_test,
            pred,
            target_names=classes,
            digits=4
        )
    )

    plt.title("Feature Importance")
    plt.show()
elif args.model == "svm":
    C_values = [0.01, 0.1, 1, 10, 100]
    accuracies = []
    f1 = []
    for c_value in C_values:
        svm = SVMModel(
            C=c_value,
            kernel="rbf",
            gamma="scale"
        )

        print("Training SVM...")

        svm.fit(
            X_train,
            y_train
        )

        print("Predicting on Test Set...")
        y_pred = svm.predict(
            X_test
        )
    # 1. Overall Accuracy
        acc = accuracy_score(
            y_test,
            y_pred
        )
        accuracies.append(acc)
        f1.append(f1_score(y_test, y_pred, average='weighted'))
    print(f"Best test Accuracy: {max(accuracies):.4f} at C={C_values[accuracies.index(max(accuracies))]}")
    print(f"Best F1_score: {max(f1):.4f} at C={C_values[f1.index(max(f1))]}")
    plt.figure(figsize=(8,5))
    plt.plot(C_values, accuracies, marker='o')
    plt.xscale('log')

    plt.xlabel('C')
    plt.ylabel('Accuracy (%)')
    plt.title('SVM Accuracy vs C')
    plt.grid(True)

    plt.show()

elif args.model == "random_forest":
    n_trees = [10, 50, 100, 200, 300]
    accuracies = []
    f1 = []
    for n in n_trees:
        rf = RandomForestModel(
            n_estimators=n,
            random_state=42
        )

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        accuracies.append(
            accuracy_score(y_test, y_pred)
        )
        f1.append(f1_score(y_test, y_pred, average='weighted'))
    print(f"Best test Accuracy: {max(accuracies):.4f}% at n_estimators={n_trees[accuracies.index(max(accuracies))]}")
    print(f"Best F1_score: {max(f1):.4f} at n_estimators={n_trees[f1.index(max(f1))]}")
    plt.plot(n_trees, accuracies, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy vs Number of Trees')
    plt.grid(True)
    plt.show()
elif args.model == "knn":
    n_neighbors = [1, 3, 5, 7, 9]
    accuracies = []
    f1 = []
    for k in n_neighbors:
        knn = KNeighborsClassifier(
            n_neighbors=k
        )

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracies.append(
            accuracy_score(y_test, y_pred) 
        )
        f1.append(
            f1_score(y_test, y_pred, average='weighted') 
        )
    print(f"Best test Accuracy: {max(accuracies):.4f} at k={n_neighbors[accuracies.index(max(accuracies))]}")
    print(f"F1_score: max={max(f1):.4f} at k={n_neighbors[f1.index(max(f1))]}")
    plt.plot(n_neighbors, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy (%)')
    plt.title('KNN Accuracy vs Number of Neighbors')
    plt.grid(True)
    plt.show()
elif args.model == "logistic_regression":
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    accuracies = []
    f1 = []

    for C in C_values:
        model = LogisticRegressionModel(
            C=C,
            max_iter=5000
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracies.append(
            accuracy_score(y_test, pred) 
        )
        f1.append(
            f1_score(y_test, pred, average='weighted')
        )
    print(f"F1_score: max={max(f1):.4f} at C={C_values[f1.index(max(f1))]}")
    print(f"Best test Accuracy: {max(accuracies):.4f} at C={C_values[accuracies.index(max(accuracies))]}")
