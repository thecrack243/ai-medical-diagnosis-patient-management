"""
AI Medical Diagnosis & Patient Management System
Main Streamlit Application - by Ibrahim
"""
import streamlit as st
import sys, os

# Add parent directory so we can import ml.predictor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from database import init_db
from views import dashboard, registration, prediction, records, about

# Initialize database
init_db()

# Page config
st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Clean UI Overrides */
    header { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem; 
        font-weight: 800; 
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .sub-header { 
        color: #64748b; 
        font-size: 1.05rem; 
        margin-top: -8px; 
        margin-bottom: 25px; 
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(19, 27, 46, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        text-align: center !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .metric-card:hover {
        transform: translateY(-5px) !important;
        border-color: rgba(14, 165, 233, 0.4) !important;
        box-shadow: 0 12px 40px rgba(14, 165, 233, 0.15) !important;
    }
    .metric-value { 
        font-size: 2.6rem; 
        font-weight: 800; 
        color: #0ea5e9; 
    }
    .metric-label { 
        font-size: 0.95rem; 
        color: #94a3b8; 
        margin-top: 6px; 
        font-weight: 500;
    }
    
    /* Risk Alert Containers */
    .risk-high {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(225, 29, 72, 0.05)) !important;
        border-left: 5px solid #f43f5e !important;
        border-top: 1px solid rgba(244, 63, 94, 0.1) !important;
        border-right: 1px solid rgba(244, 63, 94, 0.1) !important;
        border-bottom: 1px solid rgba(244, 63, 94, 0.1) !important;
        color: #fda4af !important;
        padding: 20px !important;
        border-radius: 14px !important;
        font-size: 1.25rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(244, 63, 94, 0.15) !important;
    }
    .risk-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.05)) !important;
        border-left: 5px solid #f59e0b !important;
        border-top: 1px solid rgba(245, 158, 11, 0.1) !important;
        border-right: 1px solid rgba(245, 158, 11, 0.1) !important;
        border-bottom: 1px solid rgba(245, 158, 11, 0.1) !important;
        color: #fde68a !important;
        padding: 20px !important;
        border-radius: 14px !important;
        font-size: 1.25rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15) !important;
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05)) !important;
        border-left: 5px solid #10b981 !important;
        border-top: 1px solid rgba(16, 185, 129, 0.1) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.1) !important;
        border-bottom: 1px solid rgba(16, 185, 129, 0.1) !important;
        color: #a7f3d0 !important;
        padding: 20px !important;
        border-radius: 14px !important;
        font-size: 1.25rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15) !important;
    }
    
    /* Patient Profile Card */
    .patient-card {
        background: rgba(19, 27, 46, 0.45) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    .patient-card:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.12) !important;
        transform: translateY(-2px);
    }
    
    /* Result Diagnostic Box */
    .result-box {
        background: rgba(19, 27, 46, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Form Customization */
    div[data-testid="stForm"] {
        background: rgba(19, 27, 46, 0.4) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Rounded inputs, selectboxes, and textareas */
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {
        border-radius: 12px !important;
        background-color: rgba(11, 15, 25, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.3s ease !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 10px rgba(14, 165, 233, 0.15) !important;
    }
    
    /* Buttons Customization */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Secondary and download button custom styles */
    .stButton > button[type="secondary"], .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button[type="secondary"]:hover, .stDownloadButton > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar Overrides & Navigation Pills */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #131b2e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .sidebar-title {
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem; 
        font-weight: 800; 
        text-align: center;
        padding: 15px 0; 
        margin-bottom: 20px;
        letter-spacing: -0.5px;
    }
    
    /* Convert radio selections into beautiful button list items */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] { 
        display: none !important; 
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-checked="true"] { 
        display: none !important; 
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: rgba(14, 165, 233, 0.08) !important;
        border-color: rgba(14, 165, 233, 0.25) !important;
        transform: translateX(4px);
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(99, 102, 241, 0.15)) !important;
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 15px rgba(14, 165, 233, 0.2) !important;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #0ea5e9 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown('<p class="sidebar-title">🏥 MedDiagnosis AI</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📋 Patient Registration", "🔬 Disease Prediction", 
         "📊 Patient Records", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#64748b;font-size:0.75rem;">'
        '© 2026 Medical AI System<br>Final Project — Python</p>',
        unsafe_allow_html=True
    )

# Route to pages
if page == "🏠 Dashboard":
    dashboard.render()
elif page == "📋 Patient Registration":
    registration.render()
elif page == "🔬 Disease Prediction":
    prediction.render()
elif page == "📊 Patient Records":
    records.render()
elif page == "ℹ️ About":
    about.render()
