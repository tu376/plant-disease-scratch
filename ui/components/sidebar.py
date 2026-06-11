"""
Sidebar navigation component.
"""
import streamlit as st


def render_sidebar():
    with st.sidebar:
        # Logo / brand
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 1.5rem;">
                <div style="font-size:3rem; margin-bottom:0.4rem;">🌿</div>
                <div style="font-size:1.1rem; font-weight:700; color:#34d399; letter-spacing:-0.01em;">
                    PlantGuard AI
                </div>
                <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase;
                            letter-spacing:0.08em; margin-top:0.2rem;">
                    CNN from Scratch · CuPy
                </div>
            </div>
            <hr style="border-color:rgba(16,185,129,0.15); margin:0 0 1.2rem;">
            """,
            unsafe_allow_html=True,
        )

        # Navigation
        nav_items = [
            ("🏠", "Trang chủ",      "home"),
            ("🔬", "Dự đoán",        "predict"),
            ("📊", "Hiệu suất",      "performance"),
        ]

        current = st.session_state.get("page", "home")

        for icon, label, key in nav_items:
            is_active = current == key
            btn_style = (
                "background:rgba(16,185,129,0.18); color:#34d399; "
                "border:1px solid rgba(16,185,129,0.4);"
            ) if is_active else (
                "background:transparent; color:#9ca3af; border:1px solid transparent;"
            )
            clicked = st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
            )
            if clicked:
                st.session_state["page"] = key
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <hr style="border-color:rgba(16,185,129,0.12); margin-bottom:1rem;">
            <div style="font-size:0.7rem; color:#4b5563; text-align:center; line-height:1.6;">
                🎓 Đồ án môn học · 2025<br>
                Nhóm 08 · Python &amp; CuPy
            </div>
            """,
            unsafe_allow_html=True,
        )
