import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# MODEL LOADING
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def safe_load(path):
    """Safely load pickle/joblib files with fallback methods."""
    # Try joblib first (more compatible)
    try:
        import joblib
        return joblib.load(path)
    except Exception as e1:
        # Fallback to pickle with different protocols
        try:
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            # Try with encoding for Python 2/3 compatibility
            try:
                import pickle
                with open(path, 'rb') as f:
                    return pickle.load(f, encoding='latin1')
            except Exception as e3:
                raise RuntimeError(
                    f"Could not load {path}.\n"
                    f"joblib error: {e1}\n"
                    f"pickle error: {e2}\n"
                    f"pickle latin1 error: {e3}\n\n"
                    f"This usually means the file was saved with a different Python version.\n"
                    f"Solution: Re-run the training scripts to regenerate the .pkl files."
                )

# --- Heart Disease Model ---
_heart_scaler = None
_heart_model = None
_heart_model_type = None

def _load_heart_model():
    global _heart_scaler, _heart_model, _heart_model_type
    if _heart_model is not None:
        return
    
    scaler_path = os.path.join(MODELS_DIR, 'heart_scaler.pkl')
    model_pkl = os.path.join(MODELS_DIR, 'heart_model.pkl')
    model_pth = os.path.join(MODELS_DIR, 'heart_model.pth')
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Heart scaler not found at: {scaler_path}\n"
            f"Please run notebooks/heart_training.py first."
        )
    
    _heart_scaler = safe_load(scaler_path)
    
    if os.path.exists(model_pkl):
        _heart_model = safe_load(model_pkl)
        _heart_model_type = 'sklearn'
    elif os.path.exists(model_pth):
        import torch
        import torch.nn as nn
        checkpoint = torch.load(model_pth, map_location='cpu', weights_only=False)
        class MLP(nn.Module):
            def __init__(self, input_dim):
                super(MLP, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, 1), nn.Sigmoid()
                )
            def forward(self, x):
                return self.net(x)
        _heart_model = MLP(checkpoint['input_dim'])
        _heart_model.load_state_dict(checkpoint['state_dict'])
        _heart_model.eval()
        _heart_model_type = 'pytorch'
    else:
        raise FileNotFoundError(
            f"Heart model not found. Expected:\n"
            f"  - {model_pkl} (sklearn model)\n"
            f"  - {model_pth} (PyTorch model)\n"
            f"Please run notebooks/heart_training.py first."
        )

# --- Diabetes Model ---
_diabetes_scaler = None
_diabetes_model = None
_diabetes_model_type = None

def _load_diabetes_model():
    global _diabetes_scaler, _diabetes_model, _diabetes_model_type
    if _diabetes_model is not None:
        return
    
    scaler_path = os.path.join(MODELS_DIR, 'diabetes_scaler.pkl')
    model_pkl = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
    model_pth = os.path.join(MODELS_DIR, 'diabetes_model.pth')
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Diabetes scaler not found at: {scaler_path}\n"
            f"Please run notebooks/diabetes_training.py first."
        )
    
    _diabetes_scaler = safe_load(scaler_path)
    
    if os.path.exists(model_pkl):
        _diabetes_model = safe_load(model_pkl)
        _diabetes_model_type = 'sklearn'
    elif os.path.exists(model_pth):
        import torch
        import torch.nn as nn
        checkpoint = torch.load(model_pth, map_location='cpu', weights_only=False)
        class MLP(nn.Module):
            def __init__(self, input_dim):
                super(MLP, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, 1), nn.Sigmoid()
                )
            def forward(self, x):
                return self.net(x)
        _diabetes_model = MLP(checkpoint['input_dim'])
        _diabetes_model.load_state_dict(checkpoint['state_dict'])
        _diabetes_model.eval()
        _diabetes_model_type = 'pytorch'
    else:
        raise FileNotFoundError(
            f"Diabetes model not found. Please run notebooks/diabetes_training.py first."
        )

# --- Kidney Model ---
_kidney_scaler = None
_kidney_model = None
_kidney_model_type = None
_kidney_feature_names = None

def _load_kidney_model():
    global _kidney_scaler, _kidney_model, _kidney_model_type, _kidney_feature_names
    if _kidney_model is not None:
        return
    
    scaler_path = os.path.join(MODELS_DIR, 'kidney_scaler.pkl')
    model_pkl = os.path.join(MODELS_DIR, 'kidney_model.pkl')
    model_pth = os.path.join(MODELS_DIR, 'kidney_model.pth')
    feat_path = os.path.join(MODELS_DIR, 'kidney_feature_names.pkl')
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Kidney scaler not found at: {scaler_path}\n"
            f"Please run notebooks/kidney_training.py first."
        )
    
    _kidney_scaler = safe_load(scaler_path)
    if os.path.exists(feat_path):
        _kidney_feature_names = safe_load(feat_path)
    
    if os.path.exists(model_pkl):
        _kidney_model = safe_load(model_pkl)
        _kidney_model_type = 'sklearn'
    elif os.path.exists(model_pth):
        import torch
        import torch.nn as nn
        checkpoint = torch.load(model_pth, map_location='cpu', weights_only=False)
        class MLP(nn.Module):
            def __init__(self, input_dim):
                super(MLP, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, 1), nn.Sigmoid()
                )
            def forward(self, x):
                return self.net(x)
        _kidney_model = MLP(checkpoint['input_dim'])
        _kidney_model.load_state_dict(checkpoint['state_dict'])
        _kidney_model.eval()
        _kidney_model_type = 'pytorch'
    else:
        raise FileNotFoundError(
            f"Kidney model not found. Please run notebooks/kidney_training.py first."
        )


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_heart(data):
    _load_heart_model()
    
    feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    values = [float(data.get(f, 0)) for f in feature_order]
    X = np.array(values).reshape(1, -1)
    X_scaled = _heart_scaler.transform(X)
    
    if _heart_model_type == 'sklearn':
        prob = _heart_model.predict_proba(X_scaled)[0, 1]
        pred = int(_heart_model.predict(X_scaled)[0])
    else:
        import torch
        with torch.no_grad():
            prob = _heart_model(torch.FloatTensor(X_scaled)).item()
            pred = 1 if prob > 0.5 else 0
    
    risk = 'Low' if prob < 0.3 else 'Moderate' if prob < 0.7 else 'High'
    
    return {
        'prediction': pred,
        'probability': round(float(prob), 4),
        'label': 'Heart Disease' if pred == 1 else 'Healthy',
        'risk_level': risk
    }


def predict_diabetes(data):
    _load_diabetes_model()
    
    feature_order = ['Pregnancies', 'Glucose', 'BloodPressure',
                     'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age']
    
    values = [float(data.get(f, 0)) for f in feature_order]
    X = np.array(values).reshape(1, -1)
    X_scaled = _diabetes_scaler.transform(X)
    
    if _diabetes_model_type == 'sklearn':
        prob = _diabetes_model.predict_proba(X_scaled)[0, 1]
        pred = int(_diabetes_model.predict(X_scaled)[0])
    else:
        import torch
        with torch.no_grad():
            prob = _diabetes_model(torch.FloatTensor(X_scaled)).item()
            pred = 1 if prob > 0.5 else 0
    
    risk = 'Low' if prob < 0.3 else 'Moderate' if prob < 0.7 else 'High'
    
    return {
        'prediction': pred,
        'probability': round(float(prob), 4),
        'label': 'Diabetes' if pred == 1 else 'Non-Diabetic',
        'risk_level': risk
    }


def predict_kidney(data):
    _load_kidney_model()
    
    if _kidney_feature_names:
        feature_order = _kidney_feature_names
    else:
        feature_order = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc',
                         'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                         'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet',
                         'pe', 'ane']
    
    values = []
    for f in feature_order:
        val = data.get(f, data.get(f.lower(), data.get(f.upper(), 0)))
        values.append(float(val) if val is not None else 0)
    
    X = np.array(values).reshape(1, -1)
    X_scaled = _kidney_scaler.transform(X)
    
    if _kidney_model_type == 'sklearn':
        prob = _kidney_model.predict_proba(X_scaled)[0, 1]
        pred = int(_kidney_model.predict(X_scaled)[0])
    else:
        import torch
        with torch.no_grad():
            prob = _kidney_model(torch.FloatTensor(X_scaled)).item()
            pred = 1 if prob > 0.5 else 0
    
    risk = 'Low' if prob < 0.3 else 'Moderate' if prob < 0.7 else 'High'
    
    return {
        'prediction': pred,
        'probability': round(float(prob), 4),
        'label': 'Chronic Kidney Disease' if pred == 1 else 'Healthy',
        'risk_level': risk
    }


# ============================================================
# STREAMLIT INTEGRATION HELPERS
# ============================================================

def get_heart_feature_info():
    return {
        'age': {'label': 'Age', 'type': 'number', 'min': 1, 'max': 120, 'step': 1},
        'sex': {'label': 'Sex (1=Male, 0=Female)', 'type': 'select', 'options': [0, 1]},
        'cp': {'label': 'Chest Pain Type (1-4)', 'type': 'select', 'options': [1, 2, 3, 4]},
        'trestbps': {'label': 'Resting Blood Pressure (mm Hg)', 'type': 'number', 'min': 50, 'max': 250, 'step': 1},
        'chol': {'label': 'Serum Cholesterol (mg/dl)', 'type': 'number', 'min': 50, 'max': 600, 'step': 1},
        'fbs': {'label': 'Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)', 'type': 'select', 'options': [0, 1]},
        'restecg': {'label': 'Resting ECG (0-2)', 'type': 'select', 'options': [0, 1, 2]},
        'thalach': {'label': 'Max Heart Rate Achieved', 'type': 'number', 'min': 50, 'max': 250, 'step': 1},
        'exang': {'label': 'Exercise Induced Angina (1=Yes, 0=No)', 'type': 'select', 'options': [0, 1]},
        'oldpeak': {'label': 'ST Depression (oldpeak)', 'type': 'number', 'min': 0, 'max': 10, 'step': 0.1},
        'slope': {'label': 'Slope of Peak Exercise ST (1-3)', 'type': 'select', 'options': [1, 2, 3]},
        'ca': {'label': 'Number of Major Vessels (0-3)', 'type': 'select', 'options': [0, 1, 2, 3]},
        'thal': {'label': 'Thalassemia (3=Normal, 6=Fixed, 7=Reversible)', 'type': 'select', 'options': [3, 6, 7]},
    }

def get_diabetes_feature_info():
    return {
        'Pregnancies': {'label': 'Number of Pregnancies', 'type': 'number', 'min': 0, 'max': 20, 'step': 1},
        'Glucose': {'label': 'Glucose Level (mg/dl)', 'type': 'number', 'min': 0, 'max': 300, 'step': 1},
        'BloodPressure': {'label': 'Blood Pressure (mm Hg)', 'type': 'number', 'min': 0, 'max': 200, 'step': 1},
        'SkinThickness': {'label': 'Skin Thickness (mm)', 'type': 'number', 'min': 0, 'max': 100, 'step': 1},
        'Insulin': {'label': 'Insulin Level (mu U/ml)', 'type': 'number', 'min': 0, 'max': 900, 'step': 1},
        'BMI': {'label': 'BMI', 'type': 'number', 'min': 0, 'max': 70, 'step': 0.1},
        'DiabetesPedigreeFunction': {'label': 'Diabetes Pedigree Function', 'type': 'number', 'min': 0, 'max': 3, 'step': 0.01},
        'Age': {'label': 'Age', 'type': 'number', 'min': 1, 'max': 120, 'step': 1},
    }

def get_kidney_feature_info():
    return {
        'age': {'label': 'Age', 'type': 'number', 'min': 1, 'max': 120, 'step': 1},
        'bp': {'label': 'Blood Pressure', 'type': 'number', 'min': 0, 'max': 200, 'step': 1},
        'sg': {'label': 'Specific Gravity', 'type': 'number', 'min': 1.0, 'max': 1.03, 'step': 0.005},
        'al': {'label': 'Albumin', 'type': 'select', 'options': [0, 1, 2, 3, 4, 5]},
        'su': {'label': 'Sugar', 'type': 'select', 'options': [0, 1, 2, 3, 4, 5]},
        'bgr': {'label': 'Blood Glucose Random', 'type': 'number', 'min': 0, 'max': 500, 'step': 1},
        'bu': {'label': 'Blood Urea', 'type': 'number', 'min': 0, 'max': 400, 'step': 1},
        'sc': {'label': 'Serum Creatinine', 'type': 'number', 'min': 0, 'max': 80, 'step': 0.1},
        'sod': {'label': 'Sodium', 'type': 'number', 'min': 0, 'max': 200, 'step': 1},
        'pot': {'label': 'Potassium', 'type': 'number', 'min': 0, 'max': 50, 'step': 0.1},
        'hemo': {'label': 'Hemoglobin', 'type': 'number', 'min': 0, 'max': 20, 'step': 0.1},
    }


if __name__ == '__main__':
    print("Testing predictors...")
    
    heart_test = {
        'age': 63, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 2, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 3, 'ca': 0, 'thal': 6
    }
    print("\nHeart Test:", predict_heart(heart_test))
    
    diabetes_test = {
        'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72,
        'SkinThickness': 35, 'Insulin': 0, 'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627, 'Age': 50
    }
    print("Diabetes Test:", predict_diabetes(diabetes_test))
    