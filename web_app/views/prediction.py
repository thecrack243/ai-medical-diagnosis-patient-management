"""Disease Prediction page — connects ML models to the UI."""
import streamlit as st
import json
from database import get_all_patients, add_prediction
from ml.predictor import (
    predict_heart, predict_diabetes, predict_kidney,
    get_heart_feature_info, get_diabetes_feature_info, get_kidney_feature_info
)

DISEASE_CONFIG = {
    "🫀 Heart Disease": {
        "key": "heart", "predict_fn": predict_heart,
        "features_fn": get_heart_feature_info,
        "icon": "🫀", "color": "#ef4444",
        "recommendations": {
            1: [
                "Consult a cardiologist as soon as possible",
                "Monitor blood pressure and cholesterol regularly",
                "Maintain a heart-healthy diet (low sodium, low saturated fat)",
                "Exercise regularly (at least 150 min/week of moderate activity)",
                "Avoid smoking and limit alcohol consumption"
            ],
            0: [
                "Continue maintaining a healthy lifestyle",
                "Regular cardiovascular check-ups recommended",
                "Stay active with regular physical exercise",
                "Monitor blood pressure periodically"
            ]
        }
    },
    "🩸 Diabetes": {
        "key": "diabetes", "predict_fn": predict_diabetes,
        "features_fn": get_diabetes_feature_info,
        "icon": "🩸", "color": "#f59e0b",
        "recommendations": {
            1: [
                "Consult an endocrinologist for proper diagnosis",
                "Monitor blood glucose levels regularly",
                "Follow a balanced diet with controlled carbohydrate intake",
                "Maintain a healthy weight through diet and exercise",
                "Regular HbA1c testing every 3-6 months"
            ],
            0: [
                "Maintain a balanced diet and healthy weight",
                "Regular blood glucose screening recommended",
                "Stay physically active",
                "Limit sugar and processed food intake"
            ]
        }
    },
    "🫘 Kidney Disease": {
        "key": "kidney", "predict_fn": predict_kidney,
        "features_fn": get_kidney_feature_info,
        "icon": "🫘", "color": "#6366f1",
        "recommendations": {
            1: [
                "Consult a nephrologist immediately",
                "Monitor kidney function tests (creatinine, BUN) regularly",
                "Control blood pressure and blood sugar levels",
                "Follow a kidney-friendly diet (low protein, low sodium, low potassium)",
                "Stay well hydrated and avoid nephrotoxic medications"
            ],
            0: [
                "Continue maintaining good hydration habits",
                "Regular kidney function screening recommended",
                "Maintain healthy blood pressure levels",
                "Avoid excessive use of painkillers (NSAIDs)"
            ]
        }
    }
}


def build_input_form(features_info):
    """Dynamically build input form from feature info dict with ranges, types, and labels."""
    data = {}
    cols = st.columns(2)
    for i, (key, info) in enumerate(features_info.items()):
        with cols[i % 2]:
            if info["type"] == "number":
                # Determine if we should display as integer based on the step value
                is_int = float(info["step"]).is_integer()
                
                val_min = int(info["min"]) if is_int else float(info["min"])
                val_max = int(info["max"]) if is_int else float(info["max"])
                val_step = int(info["step"]) if is_int else float(info["step"])
                
                # Fetch default or fall back to min
                default_val = info.get("default", info["min"])
                val_default = int(default_val) if is_int else float(default_val)
                
                # Add range indicator directly to the label
                label_with_range = f"{info['label']} [{val_min} - {val_max}]"
                
                data[key] = st.number_input(
                    label_with_range, 
                    min_value=val_min,
                    max_value=val_max, 
                    step=val_step,
                    value=val_default, 
                    key=f"pred_{key}"
                )
            elif info["type"] == "select":
                options = info["options"]
                labels = info.get("labels", {})
                
                # Find default index
                default_idx = 0
                if "default" in info and info["default"] in options:
                    default_idx = options.index(info["default"])
                
                data[key] = st.selectbox(
                    info["label"], 
                    options, 
                    index=default_idx,
                    format_func=lambda x: labels.get(x, str(x)),
                    key=f"pred_{key}"
                )
    return data


def display_result(result, config):
    """Display prediction result with styling."""
    risk = result["risk_level"]
    risk_class = f"risk-{risk.lower()}"
    prob_pct = round(result["probability"] * 100, 1)

    st.markdown("")
    st.markdown(f'''<div class="{risk_class}">
        <span style="font-size:1.5rem; margin-right:8px;">{config["icon"]}</span> 
        {result["label"]} &nbsp;•&nbsp; Risk Level: <strong>{risk}</strong> ({prob_pct}%)
    </div>''', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diagnosis", result["label"])
    with col2:
        st.metric("Confidence", f"{prob_pct}%")
    with col3:
        st.metric("Risk Level", risk)

    st.progress(result["probability"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Recommendations
    st.markdown("")
    st.subheader("💡 Clinical Recommendations")
    recs = config["recommendations"].get(result["prediction"], [])
    for rec in recs:
        st.markdown(f'''<div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 12px; padding: 12px 18px; margin-bottom: 8px; display: flex; align-items: center; gap: 12px;">
            <span style="color:#0ea5e9; font-size:1.2rem;">✨</span>
            <span style="color:#cbd5e1; font-size:0.95rem; font-weight:500;">{rec}</span>
        </div>''', unsafe_allow_html=True)


def render():
    st.markdown('<p class="main-header">🔬 Disease Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered diagnosis using trained machine learning models</p>', unsafe_allow_html=True)
    st.markdown("")

    # Step 1: Select patient
    patients = get_all_patients()
    if not patients:
        st.warning("⚠️ No patients registered yet. Please go to **Patient Registration** first.")
        return

    patient_options = {f"{p['name']} (ID: {p['id']})": p["id"] for p in patients}
    selected = st.selectbox("👤 Select Patient", list(patient_options.keys()))
    patient_id = patient_options[selected]

    st.markdown("")

    # Step 2: Select disease
    disease = st.selectbox("🦠 Select Disease to Predict", list(DISEASE_CONFIG.keys()))
    config = DISEASE_CONFIG[disease]

    st.markdown("---")

    # Step 3: Input form
    st.subheader(f"{config['icon']} Enter Medical Data for {disease}")
    features_info = config["features_fn"]()
    input_data = build_input_form(features_info)

    st.markdown("")

    # Step 4: Predict
    if st.button("🚀 Run Prediction", use_container_width=True):
        with st.spinner("Analyzing medical data..."):
            try:
                result = config["predict_fn"](input_data)

                # Save to database
                add_prediction(
                    patient_id=patient_id,
                    disease_type=config["key"],
                    input_data=input_data,
                    prediction=result["prediction"],
                    probability=result["probability"],
                    risk_level=result["risk_level"],
                    label=result["label"]
                )

                # Display results
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                display_result(result, config)
                st.success("✅ Result saved to patient records.")

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Make sure the ML models are properly trained. Check the `ml/models/` directory.")
