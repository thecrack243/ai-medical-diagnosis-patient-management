"""
Database Manager for AI Medical Diagnosis & Patient Management System.
Uses SQLite to store patient records and prediction history.
"""

import sqlite3
import os
import json
from datetime import datetime

# Database path (same directory as this file)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patients.db')


def get_connection():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            disease_type TEXT NOT NULL,
            input_data TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            label TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()


# ============================================================
# PATIENT OPERATIONS
# ============================================================

def add_patient(name, age, gender, phone="", email="", address=""):
    """Register a new patient. Returns the new patient ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO patients (name, age, gender, phone, email, address) VALUES (?, ?, ?, ?, ?, ?)",
        (name, age, gender, phone, email, address)
    )
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id


def get_all_patients():
    """Get all patients ordered by most recent first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_patient(patient_id):
    """Get a single patient by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def search_patients(query):
    """Search patients by name or phone number."""
    conn = get_connection()
    cursor = conn.cursor()
    search = f"%{query}%"
    cursor.execute(
        "SELECT * FROM patients WHERE name LIKE ? OR phone LIKE ? ORDER BY created_at DESC",
        (search, search)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def delete_patient(patient_id):
    """Delete a patient and all their predictions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()


# ============================================================
# PREDICTION OPERATIONS
# ============================================================

def add_prediction(patient_id, disease_type, input_data, prediction, probability, risk_level, label):
    """Save a prediction result. input_data should be a dict (will be JSON-encoded)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO predictions 
           (patient_id, disease_type, input_data, prediction, probability, risk_level, label) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, disease_type, json.dumps(input_data), prediction, probability, risk_level, label)
    )
    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()
    return prediction_id


def get_patient_predictions(patient_id):
    """Get all predictions for a specific patient."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_all_predictions():
    """Get all predictions with patient names."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, pt.name as patient_name 
        FROM predictions p 
        JOIN patients pt ON p.patient_id = pt.id 
        ORDER BY p.created_at DESC
    ''')
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_recent_predictions(limit=10):
    """Get most recent predictions with patient names."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, pt.name as patient_name 
        FROM predictions p 
        JOIN patients pt ON p.patient_id = pt.id 
        ORDER BY p.created_at DESC 
        LIMIT ?
    ''', (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

def get_dashboard_stats():
    """Get summary statistics for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()

    # Total patients
    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    # Predictions by disease type
    cursor.execute("""
        SELECT disease_type, COUNT(*) as count 
        FROM predictions 
        GROUP BY disease_type
    """)
    by_disease = {row['disease_type']: row['count'] for row in cursor.fetchall()}

    # Risk level distribution
    cursor.execute("""
        SELECT risk_level, COUNT(*) as count 
        FROM predictions 
        GROUP BY risk_level
    """)
    by_risk = {row['risk_level']: row['count'] for row in cursor.fetchall()}

    # Positive predictions (disease detected)
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
    positive_predictions = cursor.fetchone()[0]

    conn.close()

    return {
        'total_patients': total_patients,
        'total_predictions': total_predictions,
        'positive_predictions': positive_predictions,
        'by_disease': by_disease,
        'by_risk': by_risk
    }
