"""
Prediction result card component.
"""
import streamlit as st
import numpy as np


# ── Disease metadata ──────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab":                  ("🍎", "Apple – Scab",          "#ef4444"),
    "Apple___Black_rot":                   ("🍎", "Apple – Black Rot",           "#dc2626"),
    "Apple___Cedar_apple_rust":            ("🍎", "Apple – Cedar Apple Rust",      "#f97316"),
    "Apple___healthy":                     ("🍎", "Apple – Healthy",          "#10b981"),
    "Blueberry___healthy":                 ("🫐", "Blueberry – Healthy",   "#10b981"),
    "Cherry_(including_sour)___Powdery_mildew": ("🍒","Cherry – Powdery Mildew","#a855f7"),
    "Cherry_(including_sour)___healthy":   ("🍒", "Cherry – Healthy",    "#10b981"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("🌽","Corn – Gray Leaf Spot","#6b7280"),
    "Corn_(maize)___Common_rust_":         ("🌽", "Corn – Common Rust","#f59e0b"),
    "Corn_(maize)___Northern_Leaf_Blight": ("🌽", "Corn – Northern Leaf Blight",  "#ef4444"),
    "Corn_(maize)___healthy":              ("🌽", "Corn – Healthy",          "#10b981"),
    "Grape___Black_rot":                   ("🍇", "Grape – Black Rot",           "#7c3aed"),
    "Grape___Esca_(Black_Measles)":        ("🍇", "Grape – Esca",               "#9333ea"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("🍇","Grape – Leaf Blight","#c026d3"),
    "Grape___healthy":                     ("🍇", "Grape – Healthy",          "#10b981"),
    "Orange___Haunglongbing_(Citrus_greening)": ("🍊", "Orange – Huanglongbing", "#f59e0b"),
    "Peach___Bacterial_spot":              ("🍑", "Peach – Bacterial Spot",   "#dc2626"),
    "Peach___healthy":                     ("🍑", "Peach – Healthy",          "#10b981"),
    "Pepper,_bell___Bacterial_spot":       ("🌶️", "Pepper – Bacterial Spot",  "#dc2626"),
    "Pepper,_bell___healthy":              ("🌶️", "Pepper – Healthy",         "#10b981"),
}

def _get_info(class_name: str):
    for key, val in DISEASE_INFO.items():
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return val
    return ("🌿", class_name.replace("_", " "), "#6b7280")


def render_prediction_card(
    class_name: str,
    confidence: float,
    all_probs: np.ndarray,
    class_names: list,
    top_k: int = 5,
):
    """
    Renders a full prediction result card.

    Parameters
    ----------
    class_name   : predicted class label
    confidence   : confidence of the top prediction (0-1)
    all_probs    : softmax probability array (num_classes,)
    class_names  : ordered list of class names
    top_k        : number of top predictions to show
    """
    icon, display_name, color = _get_info(class_name)
    is_healthy = "healthy" in class_name.lower()
    status_icon = "✅" if is_healthy else "⚠️"
    status_text = "Healthy" if is_healthy else "Disease Detected"
    status_color = "#10b981" if is_healthy else "#f59e0b"

    # ── Top result ────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="plant-card" style="margin-bottom:1.2rem;">
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.2rem;">
                <div style="font-size:3.5rem; line-height:1;">{icon}</div>
                <div>
                    <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase;
                                letter-spacing:0.08em; margin-bottom:0.2rem;">
                        {status_icon} {status_text}
                    </div>
                    <div style="font-size:1.35rem; font-weight:700; color:#f0fdf4;
                                line-height:1.2;">
                        {display_name}
                    </div>
                    <div style="font-size:0.8rem; color:#9ca3af; margin-top:0.2rem;
                                font-family:'JetBrains Mono',monospace;">
                        {class_name}
                    </div>
                </div>
                <div style="margin-left:auto; text-align:right;">
                    <div style="font-size:2.2rem; font-weight:800; color:{color};">
                        {confidence*100:.1f}%
                    </div>
                    <div style="font-size:0.72rem; color:#6b7280;">confidence</div>
                </div>
            </div>
            <!-- Confidence bar -->
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence*100:.1f}%;
                     background:linear-gradient(90deg, {color}88, {color});"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Top-K predictions ──────────────────────────────────────
    st.markdown("#### 🔢 Prediction Probabilities")

    top_indices = np.argsort(all_probs)[::-1][:top_k]
    for rank, idx in enumerate(top_indices):
        cname = class_names[idx]
        prob  = float(all_probs[idx])
        _, dname, dcolor = _get_info(cname)
        bar_w = max(prob * 100, 1)
        medal = ["🥇","🥈","🥉"] [rank] if rank < 3 else f"#{rank+1}"
        st.markdown(
            f"""
            <div class="conf-bar-wrap">
                <div class="conf-label">
                    <span>{medal} {dname}</span>
                    <span style="color:{dcolor}; font-weight:600;">{prob*100:.2f}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{bar_w:.1f}%;
                         background:linear-gradient(90deg,{dcolor}66,{dcolor});"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
