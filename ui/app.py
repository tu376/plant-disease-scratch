"""
app.py – Entry point for the Plant Disease Classifier Streamlit app.

Run with:
    streamlit run ui/app.py
"""
import os
import sys
import streamlit as st

# ── Ensure project root is on sys.path ───────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Also ensure ui/ itself is importable ─────────────────────
UI_DIR = os.path.dirname(os.path.abspath(__file__))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

# ── Page config (MUST be first Streamlit call) ────────────────
st.set_page_config(
    page_title="🌱 PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help":     None,
        "Report a bug": None,
        "About":        "Plant Disease Classifier – CNN from Scratch · CuPy",
    },
)

# ── Inject CSS ────────────────────────────────────────────────
css_path = os.path.join(UI_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
from components.sidebar import render_sidebar
render_sidebar()

# ── Route to current page ─────────────────────────────────────
page = st.session_state.get("page", "home")

if page == "home":
    from pages.HOME import render
    render()
elif page == "predict":
    from pages.PREDICT import render
    render()
elif page == "performance":
    from pages.PERFORMANCE import render
    render()
else:
    st.error(f"Page '{page}' not found.")