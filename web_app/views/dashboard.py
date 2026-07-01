"""Dashboard page — shows summary stats and recent activity."""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from database import get_dashboard_stats, get_recent_predictions


def render():
    st.markdown('<p class="main-header">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">System overview and recent activity</p>', unsafe_allow_html=True)
    st.markdown("")

    stats = get_dashboard_stats()

    # Metric cards row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-value">{stats["total_patients"]}</div>
            <div class="metric-label">👤 Total Patients</div></div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-value">{stats["total_predictions"]}</div>
            <div class="metric-label">🔬 Total Predictions</div></div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-value">{stats["positive_predictions"]}</div>
            <div class="metric-label">⚠️ Disease Detected</div></div>''', unsafe_allow_html=True)
    with c4:
        safe = stats["total_predictions"] - stats["positive_predictions"]
        st.markdown(f'''<div class="metric-card">
            <div class="metric-value">{safe}</div>
            <div class="metric-label">✅ Healthy Results</div></div>''', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predictions by Disease Type")
        if stats["by_disease"]:
            disease_map = {"heart": "🫀 Heart", "diabetes": "🩸 Diabetes", "kidney": "🫘 Kidney"}
            labels = [disease_map.get(k, k) for k in stats["by_disease"].keys()]
            values = list(stats["by_disease"].values())
            colors = ["#0ea5e9", "#6366f1", "#8b5cf6"]
            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors[:len(labels)], marker_line_width=0)])
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", height=350,
                xaxis=dict(showgrid=False, linecolor="rgba(255,255,255,0.05)"), 
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.05)"),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No predictions yet. Go to **Disease Prediction** to get started!")

    with col_right:
        st.subheader("Risk Level Distribution")
        if stats["by_risk"]:
            color_map = {"Low": "#10b981", "Moderate": "#f59e0b", "High": "#f43f5e"}
            labels = list(stats["by_risk"].keys())
            values = list(stats["by_risk"].values())
            colors = [color_map.get(k, "#64748b") for k in labels]
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6,
                                         marker=dict(colors=colors, line=dict(color="#131b2e", width=2)))])
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", height=350, showlegend=True,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No risk data available yet.")

    # Recent predictions table
    st.markdown("")
    st.subheader("🕐 Recent Predictions")
    recent = get_recent_predictions(10)
    if recent:
        df = pd.DataFrame(recent)
        display_df = df[["patient_name", "disease_type", "label", "probability", "risk_level", "created_at"]].copy()
        display_df.columns = ["Patient", "Disease", "Result", "Confidence", "Risk", "Date"]
        display_df["Confidence"] = (display_df["Confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions recorded yet. Start by registering a patient and running a prediction!")
