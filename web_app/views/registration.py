"""Patient Registration page."""
import streamlit as st
from database import add_patient, get_all_patients


def render():
    st.markdown('<p class="main-header">📋 Patient Registration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Register a new patient into the system</p>', unsafe_allow_html=True)
    st.markdown("")

    with st.form("registration_form", clear_on_submit=True):
        st.subheader("New Patient Information")

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *", placeholder="e.g. John Doe")
            age = st.number_input("Age *", min_value=1, max_value=120, value=30, step=1)
            gender = st.selectbox("Gender *", ["Male", "Female"])
        with col2:
            phone = st.text_input("Phone Number", placeholder="e.g. +1234567890")
            email = st.text_input("Email", placeholder="e.g. john@email.com")
            address = st.text_area("Address", placeholder="Patient address", height=80)

        st.markdown("")
        submitted = st.form_submit_button("✅ Register Patient", use_container_width=True)

        if submitted:
            if not name.strip():
                st.error("❌ Patient name is required!")
            elif age < 1 or age > 120:
                st.error("❌ Please enter a valid age (1-120).")
            else:
                patient_id = add_patient(name.strip(), age, gender, phone.strip(), email.strip(), address.strip())
                st.success(f"✅ Patient **{name}** registered successfully! (ID: {patient_id})")
                st.balloons()

    # Show recently registered patients
    st.markdown("")
    st.markdown("---")
    st.subheader("📋 Registered Patients")

    patients = get_all_patients()
    if patients:
        for p in patients[:10]:
            gender_icon = "👨" if p["gender"] == "Male" else "👩"
            st.markdown(f'''<div class="patient-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.3rem;">{gender_icon}</span>
                    <strong style="font-size:1.05rem; color:#f1f5f9;">{p["name"]}</strong>
                </div>
                <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
                    <span style="background:rgba(14,165,233,0.1); border:1px solid rgba(14,165,233,0.2); color:#0ea5e9; font-size:0.78rem; font-weight:600; padding:3px 8px; border-radius:8px;">Age: {p["age"]}</span>
                    <span style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.2); color:#a5b4fc; font-size:0.78rem; font-weight:600; padding:3px 8px; border-radius:8px;">{p["gender"]}</span>
                    <span style="color:#94a3b8; font-size:0.82rem; font-weight:500;">📞 {p["phone"] or "N/A"}</span>
                </div>
                <div style="color:#64748b; font-size:0.75rem; font-weight:500;">
                    ID: {p["id"]} • {p["created_at"]}
                </div>
            </div>''', unsafe_allow_html=True)
        if len(patients) > 10:
            st.caption(f"Showing 10 of {len(patients)} patients. View all in Patient Records.")
    else:
        st.info("No patients registered yet. Fill out the form above to get started!")
