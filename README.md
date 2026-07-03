
# 🩺 AI Medical Diagnosis & Patient Management System

A Python final project combining machine learning disease prediction with a patient management web application. Built with Streamlit, scikit-learn, PyTorch, and SQLite.

## 🎯 Features

- **Disease Prediction** — ML models for 3 diseases:
  - ❤️ Heart Disease (UCI Cleveland, 303 samples)
  - 🩸 Diabetes (UCI Pima Indians, 768 samples)
  - 🫘 Chronic Kidney Disease (UCI/Kaggle, 400 samples)
- **Patient Management** — Register, track, and manage patient records
- **Dashboard Analytics** — Visualize predictions, risk distributions, and trends
- **Dark Theme UI** — Modern glassmorphism design with responsive layout

## 🧠 Machine Learning

| Disease | Best Model | ROC-AUC |
|---------|-----------|---------|
| Heart | Logistic Regression | 0.9513 |
| Diabetes | Logistic Regression | 0.8361 |
| Kidney | Random Forest | 0.9583 |

**Models compared:** Logistic Regression, Random Forest, PyTorch MLP

**Pipeline:** Data loading → Cleaning → EDA → Missing value imputation → Encoding → Standardization → 80/20 stratified split → Training → Evaluation → Export

## 🏗️ Architecture

```
├── ml/                          # Machine Learning module
│   ├── predictor.py             # Prediction API for Streamlit integration
│   ├── data/                    # UCI datasets
│   │   ├── diabetes.csv
│   │   ├── heart.csv
│   │   └── kidney.csv
│   ├── models/                  # Trained models + scalers
│   │   ├── heart_model.pkl
│   │   ├── heart_scaler.pkl
│   │   ├── diabetes_model.pkl
│   │   ├── diabetes_scaler.pkl
│   │   ├── kidney_model.pkl
│   │   ├── kidney_scaler.pkl
│   │   └── kidney_feature_names.pkl
│   ├── notebooks/               # Training pipelines (Jupyter notebooks)
│   │   ├── heart_training.ipynb
│   │   ├── diabetes_training.ipynb
│   │   ├── kidney_training.ipynb
│   │   └── generate_results.ipynb
│   └── results/                 # Model comparison tables
│       ├── heart_results.csv
│       ├── diabetes_results.csv
│       ├── kidney_results.csv
│       └── model_results.xlsx
│
└── web_app/                     # Streamlit web application
    ├── app.py                   # Main entry point
    ├── database.py              # SQLite patient database
    ├── patients.db              # SQLite database file
    ├── requirements.txt         # Python dependencies
    ├── .streamlit/
    │   └── config.toml          # Streamlit configuration
    └── views/                   # Page components
        ├── dashboard.py
        ├── registration.py
        ├── prediction.py
        ├── records.py
        ├── about.py
        └── __init__.py
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/yourusername/medical-diagnosis-ai.git
cd medical-diagnosis-ai

# Install dependencies
pip install -r web_app/requirements.txt

# Run training (optional — models are pre-trained)
cd ml/notebooks
# Open and run: heart_training.ipynb, diabetes_training.ipynb, kidney_training.ipynb

# Launch app
cd ../../web_app
streamlit run app.py
```

## 👥 Team

| Role | Member | Contribution |
|------|--------|-------------|
| ML Engineer | Emmanuel Ilunga | Dataset collection, model training, predictor API |
| Full-Stack Developer | Ibrahim | Streamlit UI, patient database, dashboard |

## 📚 Datasets

- [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [UCI Diabetes](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)
- [Kaggle CKD](https://www.kaggle.com/datasets/mansoordaku/ckdisease)

## 🛠️ Tech Stack

- **ML:** scikit-learn, PyTorch, pandas, numpy
- **Web:** Streamlit, SQLite
- **Viz:** Plotly, matplotlib
