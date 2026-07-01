"""Patient Records page — view, search, and manage patients."""
import streamlit as st
import pandas as pd
import json
from database import get_all_patients, search_patients, get_patient, get_patient_predictions, delete_patient


def render():
    st.markdown('<p class="main-header">📊 Patient Records</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View and manage patient data and prediction history</p>', unsafe_allow_html=True)
    st.markdown("")

    # Search bar
    search_query = st.text_input("🔍 Search patients by name or phone", placeholder="Type to search...")

    if search_query:
        patients = search_patients(search_query)
    else:
        patients = get_all_patients()

    if not patients:
        if search_query:
            st.warning(f"No patients found matching '{search_query}'")
        else:
            st.info("No patients registered yet. Go to **Patient Registration** to add patients.")
        return

    st.markdown(f"**{len(patients)} patient(s) found**")
    st.markdown("")

    # Patient list
    patient_names = {f"{p['name']} (ID: {p['id']}, Age: {p['age']})": p["id"] for p in patients}
    selected = st.selectbox("Select a patient to view details", list(patient_names.keys()))
    patient_id = patient_names[selected]

    patient = get_patient(patient_id)
    if not patient:
        st.error("Patient not found.")
        return

    st.markdown("---")

    # Patient info card
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{'👨' if patient['gender'] == 'Male' else '👩'} {patient['name']}")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.markdown(f"**Age:** {patient['age']}")
            st.markdown(f"**Gender:** {patient['gender']}")
        with info_col2:
            st.markdown(f"**Phone:** {patient['phone'] or 'N/A'}")
            st.markdown(f"**Email:** {patient['email'] or 'N/A'}")
        with info_col3:
            st.markdown(f"**Address:** {patient['address'] or 'N/A'}")
            st.markdown(f"**Registered:** {patient['created_at']}")

    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("🗑️ Delete Patient", use_container_width=True, type="secondary"):
            st.session_state["confirm_delete"] = patient_id

        if st.session_state.get("confirm_delete") == patient_id:
            st.warning(f"Are you sure you want to delete **{patient['name']}**?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, Delete", use_container_width=True):
                    delete_patient(patient_id)
                    st.session_state.pop("confirm_delete", None)
                    st.success("Patient deleted.")
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()

    # Prediction history
    st.markdown("---")
    st.subheader("🔬 Prediction History")

    predictions = get_patient_predictions(patient_id)
    if predictions:
        for pred in predictions:
            disease_icons = {"heart": "🫀", "diabetes": "🩸", "kidney": "🫘"}
            risk_colors = {"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"}
            icon = disease_icons.get(pred["disease_type"], "🔬")
            color = risk_colors.get(pred["risk_level"], "#64748b")
            prob_pct = round(pred["probability"] * 100, 1)

            st.markdown(f'''<div class="patient-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.3rem;">{icon}</span>
                    <strong style="font-size:1.05rem; color:#f1f5f9;">{pred["disease_type"].title()} Diagnosis</strong>
                </div>
                <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
                    <span style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); color:#e2e8f0; font-size:0.78rem; font-weight:600; padding:3px 8px; border-radius:8px;">{pred["label"]}</span>
                    <span style="background:rgba(14,165,233,0.1); border:1px solid rgba(14,165,233,0.2); color:#0ea5e9; font-size:0.78rem; font-weight:600; padding:3px 8px; border-radius:8px;">{prob_pct}% Conf.</span>
                    <span style="background:{color}15; border:1px solid {color}35; color:{color}; font-size:0.78rem; font-weight:700; padding:3px 8px; border-radius:8px;">{pred["risk_level"]} Risk</span>
                </div>
                <div style="color:#64748b; font-size:0.75rem; font-weight:500;">
                    {pred["created_at"]}
                </div>
            </div>''', unsafe_allow_html=True)

        # Also show as downloadable table
        st.markdown("")
        df = pd.DataFrame(predictions)
        display_df = df[["disease_type", "label", "probability", "risk_level", "created_at"]].copy()
        display_df.columns = ["Disease", "Result", "Confidence", "Risk Level", "Date"]
        display_df["Confidence"] = (display_df["Confidence"] * 100).round(1).astype(str) + "%"
        
        csv = display_df.to_csv(index=False)
        st.download_button("📥 Download History as CSV", csv, 
                          f"patient_{patient_id}_history.csv", "text/csv",
                          use_container_width=True)
    else:
        st.info("No predictions yet for this patient. Go to **Disease Prediction** to run one.")
