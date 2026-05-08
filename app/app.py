"""
Hospital Readmission Prediction - Streamlit Deployment App
Phase 7: Interactive Risk Prediction with SHAP Explainability & Fairness Alerts

Features:
- Real-time risk score prediction (0-100%)
- Top 5 SHAP feature importance factors
- Low confidence detection & alerts
- Fairness warnings for demographic parity
- Patient demographic tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown(
    """
<style>
    .main {
        padding: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff9900;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
    .fairness-warning {
        background-color: #e8d4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #9933ff;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SECTION 1: DATA LOADING & MODEL SETUP
# ============================================================================


@st.cache_resource
def load_model_and_data():
    """Load preprocessed data, trained model, and feature information"""
    try:
        # Load preprocessed data for reference and SHAP background
        X_train = pd.read_csv("./data/preprocessed/X_train.csv")
        X_test = pd.read_csv("./data/preprocessed/X_test.csv")
        y_train = pd.read_csv("./data/preprocessed/y_train.csv").values.ravel()

        # Train logistic regression model (same as in fairness notebook)
        model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced", n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Create SHAP explainer
        explainer = shap.LinearExplainer(model, X_train)

        return {
            "model": model,
            "explainer": explainer,
            "X_train": X_train,
            "X_test": X_test,
            "feature_names": X_train.columns.tolist(),
            "n_features": X_train.shape[1],
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load model
model_data = load_model_and_data()
if model_data is None:
    st.error("⚠️ Failed to load model. Please check the data files.")
    st.stop()

model = model_data["model"]
explainer = model_data["explainer"]
X_train = model_data["X_train"]
feature_names = model_data["feature_names"]

# ============================================================================
# SECTION 2: FAIRNESS REFERENCE DATA
# ============================================================================


# Load fairness metrics for context
@st.cache_data
def load_fairness_metrics():
    """Load fairness audit metrics to inform demographic alerts"""
    try:
        df = pd.read_csv("./results/fairness_metrics_by_group.csv")

        # Extract race metrics (rows where Group Type == 'Race')
        race_rows = df[df["Group Type"] == "Race"].copy()
        race_rows.set_index("Demographic Group", inplace=True)

        # Keep only race demographic columns
        race_df = race_rows[
            ["Asian", "Caucasian", "AfricanAmerican", "Other", "Hispanic"]
        ]

        return race_df
    except Exception as e:
        print(f"Error loading fairness metrics: {e}")
        return None


fairness_df = load_fairness_metrics()

# ============================================================================
# SECTION 3: STREAMLIT UI - MAIN APP
# ============================================================================

st.markdown("# 🏥 Hospital Readmission Risk Predictor")
st.markdown("**AI-Powered 30-Day Readmission Risk Assessment with Explainability**")
st.markdown("---")

# Create two-column layout
col1, col2 = st.columns([1.5, 1], gap="large")

# ============================================================================
# LEFT COLUMN: PATIENT INPUT FORM
# ============================================================================

with col1:
    st.markdown("### 👤 Patient Information")

    # Create tabs for different input sections
    input_tabs = st.tabs(
        ["Demographics", "Clinical History", "Diagnoses", "Medications"]
    )

    # TAB 1: Demographics
    with input_tabs[0]:
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            age = st.slider("Age", min_value=0, max_value=120, value=65, step=1)
            gender = st.radio("Gender", ["Female", "Male", "Unknown"], horizontal=True)

        with col_d2:
            race = st.selectbox(
                "Race/Ethnicity",
                ["Caucasian", "African American", "Hispanic", "Asian", "Other"],
            )

        st.info(
            "⚠️ **Fairness Note:** Demographic factors affect baseline model predictions. "
            "See fairness analysis in sidebar."
        )

    # TAB 2: Clinical History
    with input_tabs[1]:
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            num_hospital_visits = st.number_input(
                "Previous Hospital Visits (this year)",
                min_value=0,
                max_value=50,
                value=2,
                step=1,
            )

            num_er_visits = st.number_input(
                "Previous ER Visits (this year)",
                min_value=0,
                max_value=50,
                value=1,
                step=1,
            )

            days_in_hospital = st.number_input(
                "Days in Hospital (this admission)",
                min_value=0,
                max_value=200,
                value=5,
                step=1,
            )

        with col_c2:
            num_medications = st.number_input(
                "Number of Medications", min_value=0, max_value=100, value=10, step=1
            )

            num_procedures = st.number_input(
                "Number of Lab/Procedures", min_value=0, max_value=100, value=3, step=1
            )

    # TAB 3: Diagnoses
    with input_tabs[2]:
        st.markdown("**Primary Diagnoses**")

        col_diag1, col_diag2 = st.columns(2)

        with col_diag1:
            has_diabetes = st.checkbox("Diabetes", value=True)
            has_hypertension = st.checkbox("Hypertension", value=False)
            has_heart_failure = st.checkbox("Heart Failure", value=False)

        with col_diag2:
            has_ischemic_heart = st.checkbox("Ischemic Heart Disease", value=False)
            has_kidney_disease = st.checkbox("Kidney Disease", value=False)
            has_pneumonia = st.checkbox("Pneumonia", value=False)

        num_diagnoses = sum(
            [
                has_diabetes,
                has_hypertension,
                has_heart_failure,
                has_ischemic_heart,
                has_kidney_disease,
                has_pneumonia,
            ]
        )

    # TAB 4: Medications
    with input_tabs[3]:
        st.markdown("**Current Medications (if any)**")

        col_med1, col_med2 = st.columns(2)

        with col_med1:
            on_metformin = st.checkbox("Metformin", value=True)
            on_insulin = st.checkbox("Insulin", value=False)
            on_glipizide = st.checkbox("Glipizide", value=False)

        with col_med2:
            on_sulfonylurea = st.checkbox("Sulfonylurea", value=False)
            on_thiazolidinedione = st.checkbox("Thiazolidinedione", value=False)
            on_rosiglitazone = st.checkbox("Rosiglitazone", value=False)

    # Build feature vector (simplified for demonstration)
    st.markdown("---")

    if st.button(
        "🔮 Predict Readmission Risk", use_container_width=True, type="primary"
    ):
        st.session_state.predict = True

    if st.button("🔄 Clear Form", use_container_width=True):
        st.session_state.predict = False
        st.rerun()

# ============================================================================
# RIGHT COLUMN: RISK PREDICTION & EXPLANATION
# ============================================================================

with col2:
    st.markdown("### 📊 Risk Assessment")

    if "predict" in st.session_state and st.session_state.predict:
        # Create patient input vector using training data mean/std for standardization

        try:
            # Initialize with training data means as baseline
            patient_vector = X_train.mean().values.copy()

            # Age normalization
            age_idx = next(
                (i for i, name in enumerate(feature_names) if "age" in name.lower()), 0
            )
            if age_idx >= 0:
                patient_vector[age_idx] = (
                    age - X_train.iloc[:, age_idx].mean()
                ) / X_train.iloc[:, age_idx].std()

            # Hospital visits
            hosp_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "num_programs" in name.lower()
                ),
                -1,
            )
            if hosp_idx >= 0:
                patient_vector[hosp_idx] = num_hospital_visits

            # Reshape for prediction
            patient_vector = patient_vector.reshape(1, -1)

            # Get prediction
            pred_proba = model.predict_proba(patient_vector)
            risk_prob = (
                pred_proba[0, 1] if pred_proba.shape[1] > 1 else pred_proba[0, 0]
            )
            risk_percent = risk_prob * 100

            # Get SHAP explanation
            shap_vals = explainer.shap_values(patient_vector)
            # For binary classification, shap_values returns shape (n_samples, n_features)
            # We need the first (and only) sample
            if isinstance(shap_vals, list):
                shap_values = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            else:
                shap_values = shap_vals[0]

            # ================================================================
            # RISK SCORE DISPLAY
            # ================================================================

            # Determine risk level
            if risk_percent >= 60:
                risk_level = "🔴 HIGH RISK"
                risk_class = "risk-high"
                color = "#ff0000"
            elif risk_percent >= 35:
                risk_level = "🟠 MODERATE RISK"
                risk_class = "risk-medium"
                color = "#ff9900"
            else:
                risk_level = "🟢 LOW RISK"
                risk_class = "risk-low"
                color = "#00cc00"

            # Display risk score
            st.markdown(
                f"""
            <div class="{risk_class}">
                <h2 style="margin: 0; text-align: center;">{risk_level}</h2>
                <h1 style="margin: 10px 0 0 0; text-align: center; font-size: 3em;">{risk_percent:.1f}%</h1>
                <p style="margin: 5px 0 0 0; text-align: center; font-size: 0.9em;">
                    30-day readmission probability
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Model confidence
            confidence = max(model.predict_proba(patient_vector)[0])
            st.metric(
                "Model Confidence",
                f"{confidence*100:.1f}%",
                delta="Normal" if confidence > 0.7 else "⚠️ Low",
            )

            if confidence < 0.7:
                st.warning(
                    "⚠️ **Low Model Confidence**: Prediction uncertainty is high. "
                    "Recommend clinical review before making decisions.",
                    icon="⚠️",
                )

            # ================================================================
            # FAIRNESS ALERT
            # ================================================================

            if fairness_df is not None and "AUC" in fairness_df.index:
                # Normalize race name for lookup
                race_lookup = race.replace(" ", "")
                if race_lookup in fairness_df.columns:
                    race_auc = fairness_df.loc["AUC", race_lookup]
                    avg_auc = fairness_df.loc["AUC"].mean()

                    if race_auc < avg_auc - 0.02:
                        st.markdown(
                            """
                        <div class="fairness-warning">
                            <strong>⚖️ Fairness Alert</strong><br>
                            Model performance for this demographic group is lower than average.
                            Recommendations should be reviewed carefully with clinician input.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            # ================================================================
            # TOP 5 SHAP FACTORS
            # ================================================================

            st.markdown("---")
            st.markdown("### 📈 Top Factors Driving Prediction")

            # Get top features by SHAP importance
            feature_importance = list(zip(feature_names, shap_values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            top_5 = feature_importance[:5]

            # Display top factors
            for rank, (feature, shap_val) in enumerate(top_5, 1):
                direction = "↑ Increases" if shap_val > 0 else "↓ Decreases"
                magnitude = (
                    "Strong"
                    if abs(shap_val) > 0.5
                    else "Moderate" if abs(shap_val) > 0.1 else "Weak"
                )
                impact_color = "#ff6b6b" if shap_val > 0 else "#51cf66"

                st.markdown(
                    f"""
                **{rank}. {feature}**  
                {direction} risk ({magnitude} impact)  
                <span style="color: {impact_color}; font-weight: bold;">SHAP: {shap_val:.3f}</span>
                """,
                    unsafe_allow_html=True,
                )

            # SHAP waterfall plot
            st.markdown("---")
            st.markdown("### 🌊 SHAP Waterfall Plot")

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create explanation object for waterfall plot
            try:
                shap_exp = shap.Explanation(
                    values=shap_values,
                    base_values=model.intercept_[0],
                    data=patient_vector[0],
                    feature_names=feature_names,
                )
                shap.plots._waterfall.waterfall_legacy(shap_exp, max_display=5)
            except Exception as e:
                st.warning(f"Could not generate waterfall plot: {str(e)}")
                # Fallback: show bar plot instead
                fig, ax = plt.subplots(figsize=(10, 6))
                top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
                ax.barh(range(len(top_indices)), shap_values[top_indices])
                ax.set_yticks(range(len(top_indices)))
                ax.set_yticklabels([feature_names[i] for i in top_indices])
                ax.set_xlabel("SHAP Value")
                ax.set_title("Top 5 SHAP Feature Contributions")
                st.pyplot(fig)

            # Note: plotting handled in try-except above

            # ================================================================
            # CLINICAL RECOMMENDATIONS
            # ================================================================

            st.markdown("---")
            st.markdown("### 💊 Clinical Recommendations")

            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.markdown("""
                **For HIGH RISK patients:**
                - Schedule early post-discharge follow-up
                - Ensure medication adherence programs
                - Consider home health services
                - Intensive care coordination
                """)

            with rec_col2:
                st.markdown("""
                **For MODERATE/LOW RISK patients:**
                - Standard discharge planning
                - Routine follow-up appointment
                - Monitor for warning signs
                - Community support as needed
                """)

            # ================================================================
            # PATIENT SUMMARY
            # ================================================================

            with st.expander("📋 Patient Summary"):
                summary_df = pd.DataFrame(
                    {
                        "Attribute": [
                            "Age",
                            "Gender",
                            "Race",
                            "Hospital Visits",
                            "ER Visits",
                            "Days Hospitalized",
                            "Medications",
                            "Procedures",
                        ],
                        "Value": [
                            f"{age} years",
                            gender,
                            race,
                            f"{num_hospital_visits}",
                            f"{num_er_visits}",
                            f"{days_in_hospital}",
                            f"{num_medications}",
                            f"{num_procedures}",
                        ],
                    }
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Please check the data format and try again.")

    else:
        st.info(
            "👈 Fill in patient information and click 'Predict Readmission Risk' to get started."
        )

# ============================================================================
# SIDEBAR: INFORMATION & FAIRNESS ANALYSIS
# ============================================================================

with st.sidebar:
    st.markdown("## ℹ️ About This Model")

    st.markdown("""
    **Hospital Readmission Prediction Model**
    
    - **Dataset:** Diabetes 130-US Hospitals (1999-2008)
    - **Model:** Logistic Regression with class balancing
    - **Validation:** Test AUC = 0.637
    - **Primary Use:** Clinical decision support for discharge planning
    
    ⚠️ **Disclaimer:** This is a clinical decision support tool, not a replacement 
    for clinical judgment.
    """)

    st.markdown("---")

    st.markdown("## 📊 Model Fairness Analysis")

    if fairness_df is not None:
        st.markdown("**Performance by Race:**")
        # Select key metrics rows for display
        metrics_to_show = ["AUC", "Sensitivity (TPR)", "False Positive Rate"]
        race_performance = fairness_df.loc[
            fairness_df.index.isin(metrics_to_show)
        ].copy()
        st.dataframe(race_performance.round(4), use_container_width=True)

        st.markdown("""
        **Key Finding:** African American patients show slightly lower AUC 
        (0.61 vs 0.63+ for other groups). Model predictions should be interpreted 
        with awareness of these disparities.
        """)

    st.markdown("---")

    st.markdown("## 🔧 Technical Details")

    st.markdown(f"""
    - **Input Features:** {model_data['n_features']}
    - **Model Type:** Logistic Regression
    - **Explainability:** SHAP (SHapley Additive exPlanations)
    - **Feature Scaling:** Standardized
    """)

    st.markdown("---")

    st.markdown("## 📚 Documentation")

    st.markdown("""
    - [Phase 4: Explainability Analysis](../notebooks/04_explainability.ipynb)
    - [Phase 5: Fairness Audit](../notebooks/05_fairness.ipynb)
    - [GitHub Repository](https://github.com)
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 0.85em;">
    <p>Hospital Readmission Prediction System | Developed with Fairness & Explainability</p>
    <p>⚠️ For clinical use only. Always validate with domain experts.</p>
</div>
""",
    unsafe_allow_html=True,
)
