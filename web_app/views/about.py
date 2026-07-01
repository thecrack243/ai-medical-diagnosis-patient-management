"""About page — project info, team, and model performance."""
import streamlit as st
import pandas as pd
import os


def render():
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI Medical Diagnosis & Patient Management System</p>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown("""
    ### 🏥 Project Description
    
    This is an **AI-powered Medical Diagnosis System** that uses machine learning models to predict 
    the likelihood of three diseases: **Heart Disease**, **Diabetes**, and **Chronic Kidney Disease**.
    
    The system combines trained ML models with a user-friendly web interface for patient management 
    and health risk assessment. It is designed as a decision-support tool for healthcare professionals.
    
    > ⚠️ **Disclaimer:** This system is for educational purposes only and should not be used as a 
    substitute for professional medical advice, diagnosis, or treatment.
    """)

    st.markdown("---")

    # Team section
    st.subheader("👥 Team Members")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''<div class="metric-card">
            <div style="font-size:2.5rem">🤖</div>
            <div style="font-size:1.2rem;font-weight:600;color:#0ea5e9;margin:8px 0">Emmanuel</div>
            <div class="metric-label">Machine Learning Engineer</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:8px">
                Data collection & preprocessing<br>
                Model training (Logistic Regression, Random Forest, MLP)<br>
                Model evaluation & selection<br>
                Export trained models (.pkl)
            </div>
        </div>''', unsafe_allow_html=True)
    with col2:
        st.markdown('''<div class="metric-card">
            <div style="font-size:2.5rem">🌐</div>
            <div style="font-size:1.2rem;font-weight:600;color:#8b5cf6;margin:8px 0">Ibrahim</div>
            <div class="metric-label">Web Application Developer</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:8px">
                Streamlit web application<br>
                Patient registration & database (SQLite)<br>
                ML model integration & prediction UI<br>
                Dashboard & patient records management
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    # Model performance
    st.subheader("📈 Model Performance")
    st.markdown("Results from Emmanuel's model training and evaluation:")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ml', 'results')

    for disease, file_name, color in [
        ("🫀 Heart Disease", "heart_results.csv", "#ef4444"),
        ("🩸 Diabetes", "diabetes_results.csv", "#f59e0b"),
        ("🫘 Kidney Disease", "kidney_results.csv", "#6366f1")
    ]:
        csv_path = os.path.join(results_dir, file_name)
        if os.path.exists(csv_path):
            st.markdown(f"**{disease}**")
            df = pd.read_csv(csv_path)
            # Format percentage columns
            for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
                if col in df.columns:
                    df[col] = (df[col] * 100).round(1).astype(str) + "%"
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("")

    st.markdown("---")

    # Technologies
    st.subheader("🛠️ Technologies Used")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        st.markdown('''<div style="background: rgba(19, 27, 46, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; min-height: 180px;">
            <strong style="color: #0ea5e9; font-size: 1.1rem; display: block; margin-bottom: 10px;">🤖 Machine Learning</strong>
            <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                • Python & Scikit-learn<br>
                • PyTorch Neural Nets<br>
                • Pandas & NumPy Preprocessing<br>
                • Joblib Model Serialization
            </div>
        </div>''', unsafe_allow_html=True)
    with tech_col2:
        st.markdown('''<div style="background: rgba(19, 27, 46, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; min-height: 180px;">
            <strong style="color: #8b5cf6; font-size: 1.1rem; display: block; margin-bottom: 10px;">🌐 Web Application</strong>
            <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                • Streamlit Interactive WebApp<br>
                • Custom Glassmorphic CSS Styling<br>
                • Plotly Analytics & Reporting<br>
                • SQLite Relational Database
            </div>
        </div>''', unsafe_allow_html=True)
    with tech_col3:
        st.markdown('''<div style="background: rgba(19, 27, 46, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; min-height: 180px;">
            <strong style="color: #10b981; font-size: 1.1rem; display: block; margin-bottom: 10px;">📊 Clinical Datasets</strong>
            <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                • Cleveland UCI Heart Disease<br>
                • Pima Indians Diabetes Dataset<br>
                • UCI Chronic Kidney Disease<br>
                • Synthetic Patient Clinical Logs
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#64748b">Python Final Project — 2026</p>',
        unsafe_allow_html=True
    )
