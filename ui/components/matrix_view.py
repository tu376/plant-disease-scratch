"""
Confusion matrix visualization component.
"""
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def render_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    normalize: bool = False,
):
    """
    Render a styled confusion matrix heatmap.

    Parameters
    ----------
    cm          : (N, N) int or float array
    class_names : list of class label strings (length N)
    title       : chart title
    figsize     : matplotlib figure size
    normalize   : if True, normalize rows to show recall per class
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_display = cm.astype(int)
        fmt = "d"

    n = len(class_names)

    # Shorten class names for readability
    short_names = []
    for name in class_names:
        parts = name.replace("(", "").replace(")", "").split("___")
        if len(parts) == 2:
            short_names.append(f"{parts[0][:4]}./{parts[1][:10]}")
        else:
            short_names.append(name[:14])

    # Dark green colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "plant_green",
        ["#0a0f0d", "#064e3b", "#10b981", "#6ee7b7"],
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0a0f0d")
    ax.set_facecolor("#111a15")

    im = ax.imshow(cm_display, aspect="auto", cmap=cmap)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # Ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8, color="#9ca3af")
    ax.set_yticklabels(short_names, fontsize=8, color="#9ca3af")

    # Cell annotations
    thresh = cm_display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            txt = f"{val:{fmt}}" if not normalize else f"{val:.2f}"
            color = "#0a0f0d" if val > thresh else "#6ee7b7"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=max(5, 9 - n // 4), color=color, fontweight="bold")

    ax.set_xlabel("Predicted Label", color="#9ca3af", labelpad=10)
    ax.set_ylabel("True Label", color="#9ca3af", labelpad=10)
    ax.set_title(title, color="#34d399", fontsize=14, fontweight="bold", pad=14)

    for spine in ax.spines.values():
        spine.set_edgecolor("#064e3b")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
