"""
database/db_manager.py
======================
Simple MySQL helper for storing and retrieving hearing-loss records.

All public functions return None and print a warning when MySQL is
unavailable, so the rest of the system keeps working without a DB.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_CONFIG

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection():
    """Return a live MySQL connection or raise if unavailable."""
    if not MYSQL_AVAILABLE:
        raise RuntimeError("mysql-connector-python not installed.")
    return mysql.connector.connect(**DB_CONFIG)


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

def insert_patient(name: str, age: int, gender: int,
                   noise_exposure: float = 0.0, tinnitus: int = 0) -> int | None:
    """Insert a new patient and return their auto-incremented ID."""
    sql = """
        INSERT INTO patients (name, age, gender, noise_exposure, tinnitus)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (name, age, gender, noise_exposure, tinnitus))
        conn.commit()
        patient_id = cur.lastrowid
        cur.close(); conn.close()
        print(f"[db] Patient inserted  id={patient_id}")
        return patient_id
    except Exception as e:
        print(f"[db] insert_patient error: {e}")
        return None


def get_all_patients() -> list:
    """Return all patient rows as a list of dicts."""
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM patients ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close(); conn.close()
        return rows
    except Exception as e:
        print(f"[db] get_all_patients error: {e}")
        return []


# ---------------------------------------------------------------------------
# Audiometric data
# ---------------------------------------------------------------------------

def insert_audiometric(patient_id: int, data: dict) -> int | None:
    """
    Insert one audiometric record.

    data  — dict with keys matching the audiometric_data column names
            (hearing_250hz_left … hearing_8000hz_right).
    """
    cols = [
        "hearing_250hz_left",  "hearing_500hz_left",
        "hearing_1000hz_left", "hearing_2000hz_left",
        "hearing_4000hz_left", "hearing_8000hz_left",
        "hearing_250hz_right", "hearing_500hz_right",
        "hearing_1000hz_right","hearing_2000hz_right",
        "hearing_4000hz_right","hearing_8000hz_right",
    ]
    values = tuple(data.get(c) for c in cols)
    placeholders = ", ".join(["%s"] * len(cols))
    col_names    = ", ".join(cols)
    sql = f"INSERT INTO audiometric_data (patient_id, {col_names}) VALUES (%s, {placeholders})"
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (patient_id, *values))
        conn.commit()
        row_id = cur.lastrowid
        cur.close(); conn.close()
        return row_id
    except Exception as e:
        print(f"[db] insert_audiometric error: {e}")
        return None


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def insert_prediction(patient_id: int, label: str, cls: int,
                      confidence: float, source: str = "manual") -> int | None:
    """Save a model prediction result."""
    sql = """
        INSERT INTO predictions (patient_id, severity_label, severity_class, confidence, input_source)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, (patient_id, label, cls, confidence, source))
        conn.commit()
        row_id = cur.lastrowid
        cur.close(); conn.close()
        print(f"[db] Prediction stored id={row_id} → {label} ({confidence:.2%})")
        return row_id
    except Exception as e:
        print(f"[db] insert_prediction error: {e}")
        return None


def get_patient_predictions(patient_id: int) -> list:
    """Return all predictions for a given patient."""
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM predictions WHERE patient_id=%s ORDER BY predicted_at DESC",
            (patient_id,)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        return rows
    except Exception as e:
        print(f"[db] get_patient_predictions error: {e}")
        return []
