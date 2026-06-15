"""
Charts component – loss curves, accuracy curves, per-class bar charts.
"""
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Shared style helpers ──────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#111a15")
    ax.tick_params(colors="#9ca3af")
    ax.xaxis.label.set_color("#9ca3af")
    ax.yaxis.label.set_color("#9ca3af")
    for spine in ax.spines.values():
        spine.set_edgecolor("#064e3b")
    if title:
        ax.set_title(title, color="#34d399", fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color="#9ca3af")
    if ylabel:
        ax.set_ylabel(ylabel, color="#9ca3af")
    ax.grid(color="#1f2d27", linestyle="--", linewidth=0.6, alpha=0.7)


def _dark_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0a0f0d")
    return fig, ax


# ── Public chart functions ────────────────────────────────────

def render_loss_curve(
    train_losses: list,
    val_losses: list,
    title: str = "Training & Validation Loss",
):
    """Line chart of train vs. val loss over epochs."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = _dark_fig(figsize=(9, 4))
    ax.plot(epochs, train_losses, color="#10b981", linewidth=2, label="Train Loss", marker="o", markersize=4)
    ax.plot(epochs, val_losses,   color="#f59e0b", linewidth=2, label="Val Loss",   marker="s", markersize=4,
            linestyle="--")
    ax.fill_between(epochs, train_losses, alpha=0.08, color="#10b981")
    ax.fill_between(epochs, val_losses,   alpha=0.08, color="#f59e0b")
    ax.legend(facecolor="#111a15", edgecolor="#064e3b", labelcolor="#f0fdf4", fontsize=9)
    _style_ax(ax, title=title, xlabel="Epoch", ylabel="Loss")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_accuracy_curve(
    train_accs: list,
    val_accs: list,
    title: str = "Training & Validation Accuracy",
):
    """Line chart of train vs. val accuracy over epochs."""
    epochs = range(1, len(train_accs) + 1)
    fig, ax = _dark_fig(figsize=(9, 4))
    ax.plot(epochs, [a * 100 for a in train_accs], color="#34d399", linewidth=2,
            label="Train Acc", marker="o", markersize=4)
    ax.plot(epochs, [a * 100 for a in val_accs],   color="#a78bfa", linewidth=2,
            label="Val Acc",   marker="s", markersize=4, linestyle="--")
    ax.fill_between(epochs, [a * 100 for a in train_accs], alpha=0.08, color="#34d399")
    ax.fill_between(epochs, [a * 100 for a in val_accs],   alpha=0.08, color="#a78bfa")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(facecolor="#111a15", edgecolor="#064e3b", labelcolor="#f0fdf4", fontsize=9)
    _style_ax(ax, title=title, xlabel="Epoch", ylabel="Accuracy (%)")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_per_class_accuracy(
    class_names: list,
    class_accs: list,
    title: str = "Per-Class Accuracy",
):
    """Horizontal bar chart of per-class accuracy."""
    short_names = [n.split("___")[-1].replace("_", " ")[:22] for n in class_names]
    accs_pct = [a * 100 for a in class_accs]
    colors = ["#10b981" if a >= 80 else "#f59e0b" if a >= 60 else "#ef4444" for a in accs_pct]

    fig, ax = _dark_fig(figsize=(9, max(4, len(class_names) * 0.45)))
    bars = ax.barh(short_names, accs_pct, color=colors, height=0.6, edgecolor="#064e3b", linewidth=0.5)
    ax.set_xlim(0, 110)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    for bar, val in zip(bars, accs_pct):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#f0fdf4", fontsize=8)
    ax.invert_yaxis()
    _style_ax(ax, title=title, xlabel="Accuracy (%)")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_class_distribution(
    class_names: list,
    counts: list,
    title: str = "Class Distribution",
):
    """Vertical bar chart of sample counts per class."""
    short_names = [n.split("___")[-1].replace("_", " ")[:16] for n in class_names]
    fig, ax = _dark_fig(figsize=(10, 4))
    bars = ax.bar(
        short_names, counts,
        color="#10b981", edgecolor="#064e3b", linewidth=0.5,
        width=0.65,
    )
    # Gradient-ish coloring by count
    max_c = max(counts) if counts else 1
    for bar, c in zip(bars, counts):
        ratio = c / max_c
        bar.set_alpha(0.4 + 0.6 * ratio)
    ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8, color="#9ca3af")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=7.5, color="#9ca3af")
    _style_ax(ax, title=title, xlabel="Class", ylabel="Samples")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_precision_recall_bar(
    class_names: list,
    precisions: list,
    recalls: list,
    f1s: list,
    title: str = "Precision / Recall / F1 per Class",
):
    """Grouped bar chart."""
    short_names = [n.split("___")[-1].replace("_", " ")[:14] for n in class_names]
    x = np.arange(len(short_names))
    w = 0.25

    fig, ax = _dark_fig(figsize=(12, 5))
    ax.bar(x - w, [p * 100 for p in precisions], w, label="Precision", color="#34d399",
           edgecolor="#064e3b", linewidth=0.4)
    ax.bar(x,     [r * 100 for r in recalls],    w, label="Recall",    color="#60a5fa",
           edgecolor="#1e3a5f", linewidth=0.4)
    ax.bar(x + w, [f * 100 for f in f1s],        w, label="F1",        color="#a78bfa",
           edgecolor="#3b1e5f", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8, color="#9ca3af")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(facecolor="#111a15", edgecolor="#064e3b", labelcolor="#f0fdf4", fontsize=9)
    _style_ax(ax, title=title, ylabel="%")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
