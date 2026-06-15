"""
Metrics card component.
"""
import streamlit as st


def render_metrics_row(metrics: dict):
    """
    Render a horizontal row of metric tiles.

    Parameters
    ----------
    metrics : dict
        { "label": (value_str, delta_str_or_None), ... }

    Example
    -------
    render_metrics_row({
        "Accuracy":  ("92.4%",  "+1.2%"),
        "Val Loss":  ("0.231",  None),
        "F1 Score":  ("0.918",  None),
        "Classes":   ("10",     None),
    })
    """
    cols = st.columns(len(metrics))
    for col, (label, payload) in zip(cols, metrics.items()):
        value = payload[0] if isinstance(payload, (tuple, list)) else payload
        delta = payload[1] if isinstance(payload, (tuple, list)) and len(payload) > 1 else None
        with col:
            if delta:
                st.metric(label=label, value=value, delta=delta)
            else:
                st.metric(label=label, value=value)


def render_single_metric(label: str, value: str, color: str = "#34d399", icon: str = ""):
    """Render a standalone decorative metric tile."""
    st.markdown(
        f"""
        <div class="metric-tile">
            <div style="font-size:1.5rem; margin-bottom:0.3rem;">{icon}</div>
            <div class="value" style="color:{color};">{value}</div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
