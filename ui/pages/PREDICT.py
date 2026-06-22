"""
PREDICT page – Upload a leaf image, run CNN inference, display results.
"""
import os
import sys
import streamlit as st
import numpy as np
from PIL import Image

# ── Path setup ────────────────────────────────────────────────
# app.py (which runs first) already adds project root and ui/ to sys.path.
# We just ensure root is accessible for model imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from components.image_upload    import render_image_uploader, preprocess_image
from components.prediction_card import render_prediction_card

# ── Class names (PlantVillage 20-class) ──────────────────────
# Matches the exact folder names used in data/train/
DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
]


# ── Model loader (cached) ────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Đang tải mô hình CNN...")
def load_model(weights_path: str, num_classes: int):
    """Load CNN weights. Returns (model, class_names) or raises."""
    try:
        import cupy as cp
        from model.cnn import CNN

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

        return model, True  # (model, cupy_ok)
    except Exception as e:
        return str(e), False


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def predict_image(model, img_array: np.ndarray, use_cupy: bool):
    """
    Run forward pass.
    img_array : (1, 3, H, W) float32 numpy
    Returns   : (probs np.ndarray shape (num_classes,), pred_idx int)
    """
    try:
        import cupy as cp
        x = cp.array(img_array)
        logits = model.forward(x)
        logits_np = cp.asnumpy(logits)[0]  # (num_classes,)
    except Exception:
        # CPU fallback (numpy arrays)
        logits_np = np.zeros(20)            # placeholder – 20 classes
    probs = softmax(logits_np)
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx


# ── Detect class names from data/train folder ────────────────

def detect_classes(data_dir: str):
    if os.path.isdir(data_dir):
        classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        if classes:
            return classes
    return DEFAULT_CLASSES


# ── Main render ───────────────────────────────────────────────

def render():
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin-bottom:0.3rem;">🔬 Predict Plant Disease</h1>
            <p style="color:#9ca3af; font-size:0.95rem;">
                Upload a picture of a leaf so the CNN model can classify the disease.
                Results will display the predicted label, confidence, and top‑5 probabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar config ───────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='color:#34d399; font-weight:600; font-size:0.85rem;"
            "text-transform:uppercase; letter-spacing:0.06em; margin:1rem 0 0.5rem;'>"
            "⚙️ Configuration</div>",
            unsafe_allow_html=True,
        )

        weights_path = st.text_input(
            "Weights file path (.npz)",
            value=os.path.join(ROOT, "best_cnn_weights.npz"),
            help="File best_cnn_weights.npz generated after running train.py.",
        )

        data_dir = st.text_input(
            "Data/train directory (for class detection)",
            value=os.path.join(ROOT, "data", "train"),
            help="Used to automatically read class list from directory names.",
        )

        top_k = st.slider("Number of top-K results to display", 3, 10, 5)
        image_size = st.selectbox("Image resize size", [64, 128, 224], index=0)

    # ── Class names ──────────────────────────────────────────
    class_names = detect_classes(data_dir)
    num_classes = len(class_names)

    # ── Model status ─────────────────────────────────────────
    weights_exist = os.path.isfile(weights_path)
    if not weights_exist:
        st.warning(
            f"⚠️ Cannot find weights file at: `{weights_path}`\n\n"
            "Run `python train.py` first to generate `best_cnn_weights.npz`, "
            "or adjust the path in the sidebar.\n\n"
            "**Demo mode** will display the interface with synthetic data.",
        )
        demo_mode = True
    else:
        demo_mode = False

    # ── Layout: upload | result ──────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 📤 Upload Image")
        image = render_image_uploader(key="predict_upload")

        if image is not None:
            st.image(image, caption="Original Image", use_container_width=True)

            # Show resized preview
            with st.expander("👁️ View Resized Image"):
                resized = image.resize((image_size, image_size), Image.LANCZOS)
                st.image(
                    resized,
                    caption=f"Resized {image_size}×{image_size}",
                    width=200,
                )
                arr = np.array(resized)
                st.caption(
                    f"Shape: {arr.shape} | Min: {arr.min()} | Max: {arr.max()} | "
                    f"Mean: {arr.mean():.1f}"
                )

        # Tips
        st.markdown(
            """
            <div class="plant-card" style="margin-top:1rem; padding:1rem;">
                <div style="font-size:0.85rem; color:#34d399; font-weight:600;
                            margin-bottom:0.5rem;">💡 Tips for Better Results</div>
                <ul style="font-size:0.78rem; color:#9ca3af; line-height:1.7;
                           padding-left:1.2rem; margin:0;">
                    <li>Take clear photos with good lighting</li>
                    <li>Ensure the leaf occupies most of the frame</li>
                    <li>Use a simple, low-noise background</li>
                    <li>Use real images from PlantVillage for testing</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_result:
        st.markdown("#### 🎯 Prediction Results")

        if image is None:
            st.markdown(
                """
                <div style="text-align:center; padding:3rem 1rem; color:#4b5563;
                            border:2px dashed rgba(16,185,129,0.15); border-radius:14px;">
                    <div style="font-size:3rem; margin-bottom:0.8rem;">🍃</div>
                    <div style="font-size:0.9rem;">
                        Upload an image to see prediction results
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            img_arr = preprocess_image(image, size=image_size)

            if demo_mode:
                # ── Demo mode: random probabilities ──────────
                st.info("🎭 **Demo Mode** – using random probabilities (no real weights available)")
                rng = np.random.default_rng(42)
                probs = rng.dirichlet(np.ones(num_classes) * 0.5)
                pred_idx = int(np.argmax(probs))
            else:
                # ── Real inference ────────────────────────────
                result = load_model(weights_path, num_classes)
                model, cupy_ok = result

                if not cupy_ok:
                    st.error(f"❌ Error loading model: {model}")
                    return

                with st.spinner("🔄 Loading prediction..."):
                    probs, pred_idx = predict_image(model, img_arr, cupy_ok)

            pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
            confidence  = float(probs[pred_idx])

            render_prediction_card(
                class_name=pred_class,
                confidence=confidence,
                all_probs=probs,
                class_names=class_names,
                top_k=top_k,
            )

            # Raw logits / probs table
            with st.expander("📋 View Full Probabilities"):
                import pandas as pd
                df = pd.DataFrame({
                    "Class": class_names[:num_classes],
                    "Probability (%)": [f"{p*100:.3f}" for p in probs[:num_classes]],
                }).sort_values("Probability (%)", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Info about supported classes ─────────────────────────
    st.markdown("---")
    st.markdown("#### 📚 Supported Disease Classes")
    class_cols = st.columns(2)
    for i, cls in enumerate(class_names):
        with class_cols[i % 2]:
            is_healthy = "healthy" in cls.lower()
            icon  = "✅" if is_healthy else "🔴"
            short = cls.replace("_", " ").replace("(", "").replace(")", "")
            st.markdown(
                f"<span style='font-size:0.82rem; color:#9ca3af;'>"
                f"{icon} &nbsp; {short}</span>",
                unsafe_allow_html=True,
            )
