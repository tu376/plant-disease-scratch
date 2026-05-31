import argparse
import cupy as cp

from evaluate import (
    accuracy_score,
    evaluate_classification
)

from model.cnn import CNN
from model.xgboost import XGBoostModel
from model.svm import SVMModel

from utils.optimizer import (
    Adam,
    SGD,
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

# =====================================================
# Args
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="cnn",
    choices=["cnn", "xgboost"]
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
        "sgd",
        "sgdmomentum"
    ]
)

args = parser.parse_args()

# =====================================================
# Data
# =====================================================

print("Using device: CUDA (CuPy)")

train_loader, val_loader, classes = create_dataloaders(
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    batch_size=args.batch_size,
    image_size=64
)

num_classes = len(classes)

print("Classes:", classes)

# =====================================================
# Helper
# =====================================================

def extract_features(cnn, loader):

    X, y = dataloader_to_cupy(loader)

    features = cnn.extract_features(X)

    return features, y

# =====================================================
# CNN
# =====================================================

if args.model == "cnn":

    model = CNN(
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
            model.parameters(),
            lr=args.lr
        )

    elif args.optimizer == "sgd":

        optimizer = SGD(
            model.parameters(),
            lr=args.lr
        )

    else:

        optimizer = SGDMomentum(
            model.parameters(),
            lr=args.lr
        )

    # -----------------------------
    # Validation set
    # -----------------------------

    X_val, y_val = dataloader_to_cupy(
        val_loader
    )

    # -----------------------------
    # Training loop
    # -----------------------------

    for epoch in range(args.epochs):

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:

            X_batch = cp.asarray(images)
            y_batch = cp.asarray(labels)

            # forward

            logits = model.forward(
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

            model.backward(
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
            evaluate_classification(
                model,
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

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )
        cp.savez(
            "cnn_weights.npz",

            conv1_w=model.conv1.weight,
            conv1_b=model.conv1.bias,

            conv2_w=model.conv2.weight,
            conv2_b=model.conv2.bias,

            fc1_w=model.fc1.weight,
            fc1_b=model.fc1.bias,

            fc2_w=model.fc2.weight,
            fc2_b=model.fc2.bias,
        )

# =====================================================
# XGBoost
# =====================================================

else:

    cnn = CNN(
        num_classes=num_classes
    )

    weights = cp.load(
            "cnn_weights.npz"
    )
    cnn.conv1.weight[:] = weights["conv1_w"]
    cnn.conv1.bias[:]   = weights["conv1_b"]

    cnn.conv2.weight[:] = weights["conv2_w"]
    cnn.conv2.bias[:]   = weights["conv2_b"]

    cnn.fc1.weight[:] = weights["fc1_w"]
    cnn.fc1.bias[:]   = weights["fc1_b"]

    cnn.fc2.weight[:] = weights["fc2_w"]
    cnn.fc2.bias[:]   = weights["fc2_b"]
    print(
        "Extracting train features..."
    )

    X_train, y_train = extract_features(
        cnn,
        train_loader
    )

    print(
        "Extracting validation features..."
    )

    X_val, y_val = extract_features(
        cnn,
        val_loader
    )

    # -----------------------------
    # CuPy -> NumPy
    # -----------------------------

    X_train = cp.asnumpy(
        X_train
    )

    y_train = cp.asnumpy(
        y_train
    )

    X_val = cp.asnumpy(
        X_val
    )

    y_val = cp.asnumpy(
        y_val
    )

    print(
        "Train:",
        X_train.shape,
        y_train.shape
    )

    print(
        "Val:",
        X_val.shape,
        y_val.shape
    )
    if args.model == "svm":

        svm = SVMModel(
            num_classes=num_classes,
            C=1.0,
            kernel="rbf",
            gamma="scale"
        )

        print(
            "Training SVM..."
        )

        svm.fit(
            X_train,
            y_train
        )

        pred = svm.predict(
            X_val
        )

        acc = accuracy_score(
            y_val,
            pred
        )

        print(
            f"Validation Accuracy: {acc:.4f}"
        )
    elif args.model == "xgboost":
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
            X_val
        )

        acc = (
            pred == y_val
        ).mean()

        print(
            f"Validation Accuracy: {acc:.4f}"
        )