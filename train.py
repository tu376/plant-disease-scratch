<<<<<<< Updated upstream
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
=======
"""
train.py
Vòng lặp huấn luyện đầy đủ: forward, loss, backward, optimizer, scheduler.
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from backend    import cp, to_numpy, to_device, BACKEND
from model      import PlantDiseaseCNN
from optimizer  import Adam, SGD
from data_loader import PlantDiseaseDataset, create_batches


# -------------------------------------------------------------------------
# Learning rate scheduler (cosine annealing, tự viết)
# -------------------------------------------------------------------------
def cosine_lr(base_lr, epoch, total_epochs):
    """Cosine annealing without restart."""
    return base_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


# -------------------------------------------------------------------------
# One epoch training
# -------------------------------------------------------------------------
def train_one_epoch(model, optimizer, dataset, batch_size, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    batch_count = 0

    for X_np, y_np in create_batches(dataset, batch_size=batch_size, shuffle=True):
        # Chuyển lên GPU nếu dùng CuPy
        X = to_device(X_np)
        y = to_device(y_np.astype(np.int32))

        # ---- Forward ----
        logits = model.forward(X)
        loss   = model.loss(logits, y)

        # ---- Backward ----
        model.backward()

        # ---- Optimizer step ----
        params_grads = model.get_all_params_and_grads()
        optimizer.step(params_grads)

        # ---- Metrics ----
        preds   = cp.argmax(logits, axis=1)
        correct = int(cp.sum(preds == y))
        total_loss    += loss * X.shape[0]
        total_correct += correct
        total_samples += X.shape[0]
        batch_count   += 1

        if batch_count % 10 == 0:
            acc = correct / X.shape[0] * 100
            print(f"  [batch {batch_count:4d}] loss={loss:.4f}  batch_acc={acc:.1f}%")

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples * 100
    return avg_loss, avg_acc


# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------
def evaluate(model, dataset, batch_size):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X_np, y_np in create_batches(dataset, batch_size=batch_size, shuffle=False):
        X = to_device(X_np)
        y = to_device(y_np.astype(np.int32))

        logits = model.forward(X)
        loss   = model.loss_fn.forward(logits, y)

        preds   = cp.argmax(logits, axis=1)
        correct = int(cp.sum(preds == y))
        total_loss    += loss * X.shape[0]
        total_correct += correct
        total_samples += X.shape[0]

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples * 100
    return avg_loss, avg_acc


# -------------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------------
def train(args):
    print(f"\n{'='*60}")
    print(f"  Plant Disease CNN – From Scratch | Backend: {BACKEND.upper()}")
    print(f"{'='*60}\n")

    # Dataset
    train_dataset = PlantDiseaseDataset(
        root=os.path.join(args.data_dir, "train"),
        img_size=args.img_size,
        augment=True
    )
    val_dataset = PlantDiseaseDataset(
        root=os.path.join(args.data_dir, "val"),
        img_size=args.img_size,
        augment=False
    )

    num_classes = train_dataset.num_classes
    print(f"\n[config] Số lớp: {num_classes} | Img size: {args.img_size}x{args.img_size}")
    print(f"[config] Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}\n")

    # Model
    model = PlantDiseaseCNN(num_classes=num_classes, img_size=args.img_size)

    # Load checkpoint nếu có
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pkl")
    if args.resume and os.path.exists(ckpt_path):
        model.load(ckpt_path)
        print("[train] Resume từ checkpoint\n")

    # Optimizer
    if args.optimizer == "adam":
        optimizer = Adam(lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Lịch sử để vẽ đồ thị
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Cosine LR
        current_lr = cosine_lr(args.lr, epoch - 1, args.epochs)
        optimizer.set_lr(current_lr)

        t0 = time.time()
        print(f"\n{'─'*50}")
        print(f" Epoch {epoch}/{args.epochs}  |  LR = {current_lr:.6f}")
        print(f"{'─'*50}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_dataset, args.batch_size, epoch
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_dataset, args.batch_size)

        elapsed = time.time() - t0
        print(f"\n  Train: loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val  : loss={val_loss:.4f}  acc={val_acc:.2f}%  [{elapsed:.1f}s]")

        # Lưu lịch sử
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Lưu best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(ckpt_path)
            print(f"  ✅ Best model saved! (val_acc={best_val_acc:.2f}%)")

        # Lưu checkpoint định kỳ
        if epoch % args.save_every == 0:
            ep_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pkl")
            model.save(ep_path)

    print(f"\n{'='*60}")
    print(f"  Hoàn tất! Best val acc = {best_val_acc:.2f}%")
    print(f"{'='*60}\n")

    # Vẽ đồ thị (tùy chọn)
    _plot_history(history, args.checkpoint_dir)
    return model, history


def _plot_history(history, save_dir):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
        axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato")
        axes[0].set_title("Loss theo Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="steelblue")
        axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="tomato")
        axes[1].set_title("Accuracy theo Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(save_dir, "training_curve.png")
        plt.savefig(out_path, dpi=120)
        print(f"[plot] Đã lưu training curve -> {out_path}")
        plt.close()
    except Exception as e:
        print(f"[plot] Không vẽ được đồ thị: {e}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN from scratch cho Plant Disease")
    parser.add_argument("--data_dir",      type=str, default="data",        help="Thư mục data (chứa train/ và val/)")
    parser.add_argument("--checkpoint_dir",type=str, default="checkpoints", help="Thư mục lưu checkpoint")
    parser.add_argument("--img_size",      type=int, default=64,            help="Kích thước ảnh (64 hoặc 128)")
    parser.add_argument("--epochs",        type=int, default=30,            help="Số epoch")
    parser.add_argument("--batch_size",    type=int, default=32,            help="Batch size")
    parser.add_argument("--lr",            type=float, default=1e-3,        help="Learning rate ban đầu")
    parser.add_argument("--weight_decay",  type=float, default=1e-4,        help="L2 regularization")
    parser.add_argument("--optimizer",     type=str, default="adam",        choices=["adam", "sgd"])
    parser.add_argument("--save_every",    type=int, default=5,             help="Lưu checkpoint mỗi N epoch")
    parser.add_argument("--resume",        action="store_true",             help="Tiếp tục từ checkpoint")
    args = parser.parse_args()

    train(args)
>>>>>>> Stashed changes
