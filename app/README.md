# Hospital Readmission Risk Predictor - Streamlit App

Phase 7 Deployment: Interactive web application for clinical decision support.

## Features

✨ **Core Functionality:**
- 🔮 Real-time risk prediction (0-100% readmission probability)
- 📈 Top 5 SHAP feature importance factors with explanations
- ⚠️ Low confidence detection and alerts
- ⚖️ Fairness warnings for demographic disparities
- 📊 Interactive visualizations (waterfall plots, risk meters)

🏥 **Clinical Features:**
- Patient demographic input (age, gender, race)
- Clinical history tracking (hospital visits, ER visits, days hospitalized)
- Diagnosis selection interface
- Medication tracking
- Clinical recommendations based on risk level

⚖️ **Fairness & Explainability:**
- SHAP waterfall plots showing factor contributions
- Fairness audit metrics by demographic group
- Alerts for model performance disparities
- Transparent decision-making process

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- streamlit >= 1.28.0
- shap >= 0.42.0
- scikit-learn
- pandas
- numpy
- matplotlib

### 2. Prepare Data

Ensure the following exist:
```
data/
  ├── preprocessed/
  │   ├── X_train.csv
  │   └── X_test.csv
results/
  └── fairness_metrics_by_group.csv
```

## Running the App

```bash
streamlit run app/app.py
```

Then open your browser to `http://localhost:8501`

### Alternative (from project root):
```bash
cd app && streamlit run app.py
```

## Usage Walkthrough

### 1. **Enter Patient Information**
   - Navigate through tabs: Demographics → Clinical History → Diagnoses → Medications
   - Fill in patient age, gender, race
   - Record hospital utilization metrics
   - Select relevant diagnoses
   - Mark current medications

### 2. **Generate Prediction**
   - Click "🔮 Predict Readmission Risk" button
   - Model outputs risk score immediately

### 3. **Interpret Results**
   - **Risk Score:** 0-100% probability of 30-day readmission
   - **Risk Level:** Color-coded (🟢 Low / 🟠 Moderate / 🔴 High)
   - **Confidence:** Model's prediction confidence (>70% recommended)

### 4. **Review Explanations**
   - **Top 5 Factors:** Features driving the prediction
   - **SHAP Waterfall:** Visual breakdown of each factor's contribution
   - **Fairness Alert:** Warnings if model performance is disparate

### 5. **Clinical Action**
   - Use recommendations (discharge planning, follow-up intensity, etc.)
   - Always validate with clinical judgment
   - Document reasoning in patient record

## Output Interpretation

### Risk Levels

| Level | Score | Action | Color |
|-------|-------|--------|-------|
| 🟢 LOW | < 35% | Standard discharge planning | Green |
| 🟠 MODERATE | 35-60% | Enhanced follow-up coordination | Orange |
| 🔴 HIGH | > 60% | Intensive care coordination, home health | Red |

### SHAP Explanations

- **↑ Increases risk:** Features making readmission more likely
- **↓ Decreases risk:** Features protective against readmission
- **Strong/Moderate/Weak:** Magnitude of factor's influence

### Fairness Alerts

Displays when:
- Model AUC for patient's demographic is >2% below average
- Different decision thresholds recommended by fairness audit
- Clinical review especially important

## Model Information

- **Type:** Logistic Regression with class-balanced weights
- **Training Data:** 101,766 patient records
- **Validation AUC:** 0.637
- **Class:** 30-day readmission (binary: Yes/No)

### Key Disparities (from Phase 5 Fairness Audit)

| Demographic | AUC | True Positive Rate | False Positive Rate |
|-------------|-----|-------------------|-------------------|
| African American | 0.608 | 0.624 | 0.320 |
| Caucasian | 0.642 | 0.618 | 0.330 |
| Asian | 0.618 | 0.632 | 0.298 |
| Female | 0.641 | 0.632 | 0.298 |
| Male | 0.623 | 0.604 | 0.352 |

⚠️ **Clinical Note:** African American patients show lower AUC. Predictions should include clinical validation.

## Architecture

```
app/
├── app.py           ← Main Streamlit application
├── utils.py         ← Helper functions (normalization, SHAP formatting)
└── README.md        ← This file
```

### Data Flow

```
Patient Input
    ↓
[Input Validation & Normalization]
    ↓
Feature Vector Creation
    ↓
Model Prediction (Logistic Regression)
    ↓
SHAP Explanation Generation
    ↓
Risk Classification & Fairness Checks
    ↓
UI Rendering (Risk Score, Top Factors, Waterfall Plot)
```

## Advanced Features

### Fairness Awareness

The app includes fairness metrics from Phase 5 audit:
- Demographic parity analysis
- Equalized odds constraints
- Intersectional analysis (race × gender)
- Model performance by subgroup

Access via sidebar: **"📊 Model Fairness Analysis"**

### SHAP Explainability

Provides model-agnostic explanations using:
- Shapley values (game theory approach)
- Waterfall plots showing contribution of each feature
- Consistent with clinical interpretability needs

### Low Confidence Detection

Alerts users when:
- Model confidence < 70%
- Recommends clinical expert review
- Appropriate for edge cases and atypical patients

## Troubleshooting

### Issue: Data files not found
```
Error: No such file or directory: './data/preprocessed/X_train.csv'
```
**Solution:** Run from project root directory, not from `app/` subdirectory.

### Issue: Module import errors
```
ModuleNotFoundError: No module named 'shap'
```
**Solution:** Install requirements: `pip install -r requirements.txt`

### Issue: Model takes too long
The SHAP explainer may be slow on large datasets.
**Solution:** Use a smaller reference dataset or optimize with `sample_background=True`

## Integration with Hospital Systems

### EHR Integration
To integrate with electronic health records:

```python
# Load from EHR API
patient_data = load_from_ehr(patient_id)

# Create feature vector
X_patient = create_patient_vector(patient_data, X_train, feature_names)

# Get prediction
risk_score = model.predict_proba(X_patient)[0, 1]
```

### Data Privacy
- ✅ Model runs locally (no data transmission)
- ✅ HIPAA-compatible design
- ✅ No personal data storage
- ✅ Audit logs recommended

## Performance Notes

- **Prediction Time:** < 100ms per patient
- **SHAP Computation:** 1-5 seconds (depends on model complexity)
- **Memory Usage:** ~500MB (with data + model)
- **Recommended Server:** 2GB RAM minimum

## Future Enhancements

- [ ] Batch prediction (CSV upload)
- [ ] Patient history tracking
- [ ] Outcome feedback loop
- [ ] Model retraining dashboard
- [ ] Multiple model comparison
- [ ] Personalized fairness constraints
- [ ] Mobile app version
- [ ] HL7 FHIR API

## References

- **Phase 4:** [Explainability Analysis](../notebooks/04_explainability.ipynb)
- **Phase 5:** [Fairness Audit](../notebooks/05_fairness.ipynb)
- **SHAP:** https://shap.readthedocs.io/
- **Streamlit:** https://streamlit.io/

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review notebook documentation (Phase 4-5)
3. Contact clinical informatics team

---

**⚠️ DISCLAIMER:** This is a clinical decision support tool. Not a substitute for clinical judgment. Always validate predictions with domain experts before clinical action.
