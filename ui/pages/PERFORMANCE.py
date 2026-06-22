"""
PERFORMANCE page – Model evaluation: metrics, confusion matrix,
loss/accuracy curves, per-class analysis.
"""
import os
import sys
import streamlit as st
import numpy as np

# ── Path setup ────────────────────────────────────────────────
# app.py already adds project root and ui/ to sys.path.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from components.metrics_card import render_metrics_row
from components.matrix_view  import render_confusion_matrix
from components.charts        import (
    render_loss_curve,
    render_accuracy_curve,
    render_per_class_accuracy,
    render_class_distribution,
    render_precision_recall_bar,
)

# ── Default class list ────────────────────────────────────────
DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
]


# ── Helpers ───────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def detect_classes(data_dir: str):
    if os.path.isdir(data_dir):
        classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        if classes:
            return classes
    return DEFAULT_CLASSES


def compute_per_class_metrics(cm: np.ndarray):
    """Compute precision, recall, F1 per class from confusion matrix."""
    n = cm.shape[0]
    precisions, recalls, f1s = [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return precisions, recalls, f1s


@st.cache_resource(show_spinner="⏳ Đang tải mô hình và đánh giá...")
def run_full_evaluation(weights_path: str, data_dir: str, batch_size: int = 32):
    """
    Load CNN, run inference on the val set, return full metrics dict.
    Returns a dict or raises an exception string.
    """
    try:
        import cupy as cp
        from model.cnn import CNN
        from utils.data_loader import create_dataloaders
        from evaluate import predict, confusion_matrix as build_cm

        # Detect classes
        class_names = detect_classes(data_dir)
        num_classes  = len(class_names)

        # Load model
        model = CNN(num_classes=num_classes)
        weights = cp.load(weights_path)
        model.conv1.weight[:] = weights["conv1_w"]
        model.conv1.bias[:]   = weights["conv1_b"]
        model.conv2.weight[:] = weights["conv2_w"]
        model.conv2.bias[:]   = weights["conv2_b"]
        model.fc1.weight[:]   = weights["fc1_w"]
        model.fc1.bias[:]     = weights["fc1_b"]
        model.fc2.weight[:]   = weights["fc2_w"]
        model.fc2.bias[:]     = weights["fc2_b"]

        # Data
        train_dir = os.path.join(data_dir.rstrip("/\\").rstrip("train"), "train")
        val_dir   = os.path.join(data_dir.rstrip("/\\").rstrip("train"), "val")
        if not os.path.isdir(val_dir):
            return {"error": f"Không tìm thấy thư mục val: {val_dir}"}

        _, val_loader, _ = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            image_size=64,
        )

        # Inference
        all_preds  = []
        all_labels = []
        all_logits = []

        for X_batch, y_batch in val_loader:
            X_gpu = cp.asarray(X_batch)
            logits = model.forward(X_gpu)
            preds  = predict(logits)
            all_preds.append(cp.asnumpy(preds))
            all_labels.append(cp.asnumpy(y_batch))
            all_logits.append(cp.asnumpy(logits))

        y_pred_np = np.concatenate(all_preds)
        y_true_np = np.concatenate(all_labels)
        logits_np = np.concatenate(all_logits, axis=0)

        # Metrics
        accuracy = float(np.mean(y_pred_np == y_true_np))
        probs_np = softmax(logits_np)

        cm_gpu = build_cm(
            cp.array(y_true_np),
            cp.array(y_pred_np),
            num_classes,
        )
        cm = cp.asnumpy(cm_gpu)

        precisions, recalls, f1s = compute_per_class_metrics(cm)
        macro_f1 = float(np.mean(f1s))

        # Per-class accuracy
        class_accs = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0 for i in range(num_classes)]

        return {
            "accuracy":    accuracy,
            "macro_f1":    macro_f1,
            "cm":          cm,
            "class_names": class_names,
            "num_classes": num_classes,
            "precisions":  precisions,
            "recalls":     recalls,
            "f1s":         f1s,
            "class_accs":  class_accs,
            "n_samples":   len(y_true_np),
            "error":       None,
        }
    except Exception as e:
        import traceback
        return {"error": str(e) + "\n" + traceback.format_exc()}


def _generate_demo_data(class_names):
    """Return plausible fake evaluation data for demo mode."""
    rng = np.random.default_rng(7)
    n   = len(class_names)

    # Confusion matrix with high diagonal
    cm = rng.integers(0, 15, (n, n))
    for i in range(n):
        cm[i, i] = rng.integers(70, 120)

    accuracy   = float(np.trace(cm) / cm.sum())
    precisions, recalls, f1s = compute_per_class_metrics(cm)
    macro_f1   = float(np.mean(f1s))
    class_accs = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0 for i in range(n)]

    # Fake training curves (30 epochs)
    train_losses = [0.9 * np.exp(-0.12 * e) + 0.1 + rng.normal(0, 0.02) for e in range(30)]
    val_losses   = [0.95 * np.exp(-0.1  * e) + 0.15 + rng.normal(0, 0.03) for e in range(30)]
    train_accs   = [min(0.98, 1 - 0.85 * np.exp(-0.15 * e) + rng.normal(0, 0.01)) for e in range(30)]
    val_accs     = [min(0.97, 1 - 0.9  * np.exp(-0.12 * e) + rng.normal(0, 0.015)) for e in range(30)]

    # Clamp
    train_losses = [max(0.05, x) for x in train_losses]
    val_losses   = [max(0.08, x) for x in val_losses]
    train_accs   = [max(0.05, min(0.99, x)) for x in train_accs]
    val_accs     = [max(0.05, min(0.99, x)) for x in val_accs]

    class_counts = [int(rng.integers(80, 200)) for _ in range(n)]

    return {
        "accuracy":     accuracy,
        "macro_f1":     macro_f1,
        "cm":           cm,
        "class_names":  class_names,
        "num_classes":  n,
        "precisions":   precisions,
        "recalls":      recalls,
        "f1s":          f1s,
        "class_accs":   class_accs,
        "n_samples":    cm.sum(),
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "train_accs":   train_accs,
        "val_accs":     val_accs,
        "class_counts": class_counts,
        "error":        None,
    }


# ── Main render ───────────────────────────────────────────────

def render():
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin-bottom:0.3rem;">📊 Model Performance Evaluation</h1>
            <p style="color:#9ca3af; font-size:0.95rem;">
                Confusion matrix, loss & accuracy curves, per-class metrics and model comparison.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='color:#34d399; font-weight:600; font-size:0.85rem;"
            "text-transform:uppercase; letter-spacing:0.06em; margin:1rem 0 0.5rem;'>"
            "⚙️ Configuration</div>",
            unsafe_allow_html=True,
        )

        weights_path = st.text_input(
            "Weights path (.npz)",
            value=os.path.join(ROOT, "cnn_weights.npz"),
        )
        data_dir = st.text_input(
            "Dataset root directory",
            value=os.path.join(ROOT, "data"),
            help="Directory containing train/ and val/ folders",
        )
        batch_size = st.slider("Batch size for evaluation", 8, 64, 32, step=8)
        normalize_cm = st.checkbox("Normalize confusion matrix", value=False)

        run_eval = st.button("▶ Run Evaluation", type="primary", use_container_width=True)
        use_demo = st.checkbox("🎭 Use Demo Data", value=True,
                               help="Display fake data when real weights are not available")

    # ── Load / demo data ─────────────────────────────────────
    class_names = detect_classes(os.path.join(data_dir, "train"))
    data_key    = "perf_data"

    if run_eval and not use_demo:
        if not os.path.isfile(weights_path):
            st.error(f"❌ Cannot find weights: `{weights_path}`")
            return
        with st.spinner("⏳ Running evaluation..."):
            result = run_full_evaluation(weights_path, os.path.join(data_dir, "train"), batch_size)
        st.session_state[data_key] = result
    elif use_demo and data_key not in st.session_state:
        st.session_state[data_key] = _generate_demo_data(class_names)

    data = st.session_state.get(data_key)

    if data is None:
        st.info("👆 Click **▶ Run Evaluation** or check **🎭 Use Demo Data** to view results.")
        return

    if data.get("error"):
        st.error(f"❌ Error:\n```\n{data['error']}\n```")
        return

    if use_demo:
        st.info("🎭 **Demo Mode** – Fake data. Run `python train_cnn.py` then click *▶ Run* to see real results.")

    # ── Overview metrics ─────────────────────────────────────
    st.markdown("### 📈 Overview")
    render_metrics_row({
        "Accuracy":   (f"{data['accuracy']*100:.2f}%", None),
        "Macro F1":   (f"{data['macro_f1']:.4f}", None),
        "Number of classes": (str(data["num_classes"]), None),
        "Validation samples":(str(data["n_samples"]), None),
    })

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab_cm, tab_curves, tab_class, tab_compare = st.tabs([
        "🟩 Confusion Matrix",
        "📉 Loss & Accuracy Curves",
        "📊 Per-Class Metrics",
        "🏆 Model Comparison",
    ])

    # ── Tab 1: Confusion Matrix ───────────────────────────────
    with tab_cm:
        st.markdown("#### Confusion Matrix")
        st.caption("Row = true label | Column = predicted label")

        col_cm, col_info = st.columns([3, 1])
        with col_cm:
            render_confusion_matrix(
                cm=data["cm"],
                class_names=data["class_names"],
                title="Confusion Matrix – CNN (val set)",
                figsize=(10, 8),
                normalize=normalize_cm,
            )
        with col_info:
            st.markdown(
                """
                <div class="plant-card" style="padding:1rem;">
                    <div style="font-size:0.85rem; color:#34d399; font-weight:600; margin-bottom:0.7rem;">
                        📌 How to Read
                    </div>
                    <ul style="font-size:0.78rem; color:#9ca3af; line-height:1.8; padding-left:1.1rem; margin:0;">
                        <li><b style="color:#f0fdf4;">Diagonal entries</b> = correct predictions</li>
                        <li><b style="color:#f0fdf4;">Off-diagonal entries</b> = classification errors</li>
                        <li>Darker blue = higher quantity</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Per-class accuracy mini table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.8rem; color:#34d399; font-weight:600; margin-bottom:0.5rem;'>"
                "Class Accuracy</div>",
                unsafe_allow_html=True,
            )
            for cls, acc in zip(data["class_names"], data["class_accs"]):
                short = cls.split("___")[-1].replace("_", " ")[:18]
                color = "#10b981" if acc >= 0.8 else "#f59e0b" if acc >= 0.6 else "#ef4444"
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-size:0.75rem; margin-bottom:0.25rem;'>"
                    f"<span style='color:#9ca3af;'>{short}</span>"
                    f"<span style='color:{color}; font-weight:600;'>{acc*100:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Curves ─────────────────────────────────────────
    with tab_curves:
        if "train_losses" not in data:
            st.info(
                "📌 Loss & Accuracy Curves require training history data.\n\n"
                "Turn on **Demo Mode** to see an example, or save the training history when training "
                "and upload it here."
            )
            # Allow manual JSON upload
            uploaded_history = st.file_uploader(
                "Upload training_history.json (optional)",
                type=["json"],
                key="history_upload",
            )
            if uploaded_history:
                import json
                history = json.load(uploaded_history)
                st.session_state[data_key]["train_losses"] = history.get("train_losses", [])
                st.session_state[data_key]["val_losses"]   = history.get("val_losses",   [])
                st.session_state[data_key]["train_accs"]   = history.get("train_accs",   [])
                st.session_state[data_key]["val_accs"]     = history.get("val_accs",     [])
                st.rerun()
        else:
            col_l, col_a = st.columns(2)
            with col_l:
                render_loss_curve(data["train_losses"], data["val_losses"])
            with col_a:
                render_accuracy_curve(data["train_accs"], data["val_accs"])

            # Summary stats
            if data["train_losses"]:
                best_epoch = int(np.argmin(data["val_losses"]))
                st.markdown(
                    f"""
                    <div class="plant-card" style="padding:1rem 1.5rem; margin-top:0.5rem;">
                        <div style="display:flex; gap:3rem; flex-wrap:wrap;">
                            <div>
                                <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase;">
                                    Best Val Loss</div>
                                <div style="font-size:1.4rem; font-weight:700; color:#f59e0b;">
                                    {min(data['val_losses']):.4f}</div>
                                <div style="font-size:0.75rem; color:#4b5563;">Epoch {best_epoch+1}</div>
                            </div>
                            <div>
                                <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase;">
                                    Best Val Acc</div>
                                <div style="font-size:1.4rem; font-weight:700; color:#34d399;">
                                    {max(data['val_accs'])*100:.2f}%</div>
                                <div style="font-size:0.75rem; color:#4b5563;">
                                    Epoch {int(np.argmax(data['val_accs']))+1}</div>
                            </div>
                            <div>
                                <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase;">
                                    Final Train Loss</div>
                                <div style="font-size:1.4rem; font-weight:700; color:#10b981;">
                                    {data['train_losses'][-1]:.4f}</div>
                            </div>
                            <div>
                                <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase;">
                                    Total Epochs</div>
                                <div style="font-size:1.4rem; font-weight:700; color:#a78bfa;">
                                    {len(data['train_losses'])}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Tab 3: Per-class ─────────────────────────────────────
    with tab_class:
        st.markdown("#### Per-Class Accuracy")
        render_per_class_accuracy(
            data["class_names"],
            data["class_accs"],
            title="Per-Class Accuracy",
        )

        st.markdown("#### Precision / Recall / F1 per Class")
        render_precision_recall_bar(
            data["class_names"],
            data["precisions"],
            data["recalls"],
            data["f1s"],
        )

        if "class_counts" in data:
            st.markdown("#### Class Distribution in Validation Set")
            render_class_distribution(data["class_names"], data["class_counts"])

        # Table
        st.markdown("#### 📋 Detailed Table")
        import pandas as pd
        df = pd.DataFrame({
            "Class": [c.split("___")[-1].replace("_", " ") for c in data["class_names"]],
            "Accuracy (%)":  [f"{a*100:.1f}" for a in data["class_accs"]],
            "Precision (%)": [f"{p*100:.1f}" for p in data["precisions"]],
            "Recall (%)":    [f"{r*100:.1f}" for r in data["recalls"]],
            "F1 (%)":        [f"{f*100:.1f}" for f in data["f1s"]],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Tab 4: Model comparison ───────────────────────────────
    with tab_compare:
        st.markdown("#### 🏆 Model Comparison")
        st.caption(
            "This result is for illustration purposes only. "
            "Run the baseline models (XGBoost, SVM, RF) and update the table manually."
        )

        compare_data = {
            "Model": ["CNN (from scratch)", "XGBoost + CNN features", "Random Forest + CNN", "SVM (RBF) + CNN"],
            "Val Accuracy": ["~92.4%", "~89.1%", "~86.7%", "~85.3%"],
            "Macro F1":    ["~0.921", "~0.888", "~0.862", "~0.849"],
            "Training Time": ["~45 min/epoch (GPU)", "~5 min", "~3 min", "~8 min"],
            "Inference":   ["Fast (GPU)", "Fast", "Fast", "Medium"],
        }

        import pandas as pd
        df_cmp = pd.DataFrame(compare_data)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        # Bar chart comparison
        st.markdown("#### Accuracy Comparison")
        models_names = ["CNN\n(scratch)", "XGBoost\n+CNN", "Random\nForest", "SVM\n(RBF)"]
        accs_pct = [92.4, 89.1, 86.7, 85.3]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0a0f0d")
        ax.set_facecolor("#111a15")
        colors = ["#10b981", "#f59e0b", "#60a5fa", "#a78bfa"]
        bars = ax.bar(models_names, accs_pct, color=colors, width=0.5,
                      edgecolor="#064e3b", linewidth=0.5)
        ax.set_ylim(75, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        for bar, val in zip(bars, accs_pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val}%", ha="center", va="bottom", fontsize=10,
                    color="#f0fdf4", fontweight="bold")
        ax.tick_params(colors="#9ca3af")
        for spine in ax.spines.values():
            spine.set_edgecolor("#064e3b")
        ax.set_title("Validation Accuracy – Model Comparison",
                     color="#34d399", fontweight="bold", pad=10)
        ax.grid(color="#1f2d27", linestyle="--", linewidth=0.6, alpha=0.5, axis="y")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Key takeaways
        st.markdown(
            """
            <div class="plant-card" style="margin-top:1rem;">
                <div style="font-size:0.9rem; font-weight:600; color:#34d399; margin-bottom:0.7rem;">
                    🔑 Key Takeaways
                </div>
                <ul style="font-size:0.82rem; color:#9ca3af; line-height:1.8; padding-left:1.2rem; margin:0;">
                    <li><b style="color:#f0fdf4;">CNN from scratch</b> achieves the highest accuracy due to end-to-end feature learning.</li>
                    <li>XGBoost and RF use <b style="color:#f0fdf4;">CNN features</b> (128-dim vector) instead of raw pixels,
                        leading to significant improvements over using raw features.</li>
                    <li>SVM with RBF kernel is suitable for small datasets, but underperforms CNN on larger datasets.</li>
                    <li>All models benefit from <b style="color:#f0fdf4;">CNN feature extraction</b>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
