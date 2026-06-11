"""
Image uploader component.
"""
import streamlit as st
from PIL import Image
import numpy as np


def render_image_uploader(key: str = "leaf_upload"):
    """
    Renders a styled image upload zone.

    Returns
    -------
    PIL.Image or None
    """
    st.markdown(
        """
        <div style="text-align:center; padding:0.5rem 0 1rem;">
            <div style="font-size:2.5rem;">🍃</div>
            <div style="font-size:1rem; font-weight:600; color:#34d399; margin-bottom:0.3rem;">
                Upload Leaf Image
            </div>
            <div style="font-size:0.8rem; color:#6b7280;">
                Supports JPG · JPEG · PNG · Drag & drop or browse
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        label="Choose an image file",
        type=["jpg", "jpeg", "png"],
        key=key,
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        return image

    return None


def preprocess_image(image: Image.Image, size: int = 64) -> np.ndarray:
    """
    Resize + normalize PIL image → (1, 3, H, W) numpy float32 array
    ready to be passed to CuPy model.
    """
    img = image.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0          # (H, W, 3)
    arr = arr.transpose(2, 0, 1)                            # (3, H, W)
    arr = arr[np.newaxis, ...]                              # (1, 3, H, W)
    return arr
