"""
Utility functions for Streamlit app
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_age(age, X_train):
    """Normalize age using training data statistics"""
    age_mean = (
        X_train[[col for col in X_train.columns if "age" in col.lower()]].mean().mean()
    )
    age_std = (
        X_train[[col for col in X_train.columns if "age" in col.lower()]].std().mean()
    )
    return (age - age_mean) / age_std


def create_patient_vector(patient_data, X_train, feature_names):
    """
    Convert patient input data to feature vector matching model expectations

    Args:
        patient_data: dict with patient information
        X_train: training data for scaling reference
        feature_names: list of feature names from model

    Returns:
        np.array of shape (1, n_features) ready for model prediction
    """
    patient_vector = np.zeros((1, len(feature_names)))

    # Map available patient features to model features
    for i, feature in enumerate(feature_names):
        feature_lower = feature.lower()

        # Age features
        if "age" in feature_lower:
            patient_vector[0, i] = normalize_age(patient_data.get("age", 65), X_train)

        # Hospital utilization
        elif "num_programs" in feature_lower:
            patient_vector[0, i] = patient_data.get("num_hospital_visits", 0)
        elif "number_outpatient" in feature_lower:
            patient_vector[0, i] = patient_data.get("num_outpatient_visits", 0)
        elif "number_emergency" in feature_lower:
            patient_vector[0, i] = patient_data.get("num_er_visits", 0)
        elif "number_inpatient" in feature_lower:
            patient_vector[0, i] = patient_data.get("num_inpatient_visits", 0)

        # Days in hospital
        elif "time_in_hospital" in feature_lower:
            patient_vector[0, i] = patient_data.get("days_in_hospital", 5)

        # Medications and procedures
        elif "num_medications" in feature_lower:
            patient_vector[0, i] = patient_data.get("num_medications", 10)
        elif "num_procedures" in feature_lower:
            patient_vector[0, i] = patient_data.get("num_procedures", 3)

        # Diagnoses
        elif "diabetes" in feature_lower:
            patient_vector[0, i] = 1 if patient_data.get("has_diabetes", False) else 0

        # Demographics (one-hot encoded)
        elif "gender" in feature_lower:
            if "male" in feature_lower:
                patient_vector[0, i] = 1 if patient_data.get("gender") == "Male" else 0
            elif "unknown" in feature_lower:
                patient_vector[0, i] = (
                    1 if patient_data.get("gender") == "Unknown" else 0
                )

        elif "race" in feature_lower:
            race_map = {
                "Caucasian": "Caucasian",
                "African American": "AfricanAmerican",
                "Asian": "Asian",
                "Hispanic": "Hispanic",
                "Other": "Other",
            }
            race_value = race_map.get(patient_data.get("race", "Other"), "Other")
            if race_value in feature_lower:
                patient_vector[0, i] = (
                    1 if feature_lower.endswith(race_value.lower()) else 0
                )

    return patient_vector


def get_risk_level(risk_percent):
    """Categorize risk level"""
    if risk_percent >= 60:
        return "HIGH", "🔴"
    elif risk_percent >= 35:
        return "MODERATE", "🟠"
    else:
        return "LOW", "🟢"


def format_shap_value(shap_val, feature_name):
    """Format SHAP value for display"""
    direction = "↑ Increases" if shap_val > 0 else "↓ Decreases"
    magnitude = abs(shap_val)

    if magnitude > 0.5:
        impact = "Strong"
    elif magnitude > 0.1:
        impact = "Moderate"
    else:
        impact = "Weak"

    return f"{direction} risk ({impact} impact)"
