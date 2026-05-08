# Phase 7 Deployment Guide - Quick Start

## 🚀 Launch the App

### Option 1: Direct Streamlit Command (Recommended)
```bash
cd "d:\Users\mabd0\Documents\New folder\Hospital readmission prediction"
streamlit run app/app.py
```

### Option 2: Python Wrapper Script
```bash
python run_app.py
```

### Option 3: From App Directory
```bash
cd app
streamlit run app.py
```

## 🌐 Access the Application

Once running, open your browser to:
```
http://localhost:8501
```

## 📋 App Structure

```
app/
├── app.py          (main Streamlit application - 420 lines)
├── utils.py        (helper functions for prediction & formatting)
├── README.md       (detailed app documentation)
└── config.toml     (Streamlit configuration)
```

## ✨ App Features

### Input Sections (Tabs)
1. **Demographics:** Age, Gender, Race/Ethnicity
2. **Clinical History:** Hospital visits, ER visits, length of stay, medications
3. **Diagnoses:** Common conditions (Diabetes, HTN, Heart Failure, etc.)
4. **Medications:** Current medications patient is taking

### Output
1. **Risk Score:** 0-100% readmission probability
2. **Risk Level:** Color-coded (🟢 Low / 🟠 Moderate / 🔴 High)
3. **Model Confidence:** Prediction confidence score
4. **Top 5 SHAP Factors:** Features driving the prediction
5. **SHAP Waterfall Plot:** Visual explanation of prediction
6. **Clinical Recommendations:** Actions based on risk level
7. **Fairness Metrics:** Sidebar with demographic performance data

## 🔮 Prediction Flow

```
Patient Input
    ↓
Tabbed Form Validation
    ↓
Feature Vector Creation
    ↓
Model Prediction (Logistic Regression)
    ↓
SHAP Explanation Generation
    ↓
Risk Classification & Fairness Checks
    ↓
UI Rendering (All 7 components)
```

## ⚙️ Technical Details

- **Model:** Logistic Regression (balanced class weights)
- **Training Data:** 101,766 patient records
- **Validation AUC:** 0.637
- **Features:** 2,343 (after one-hot encoding)
- **Target:** 30-day readmission (binary)
- **Explainability:** SHAP LinearExplainer
- **Fairness:** Phase 5 audit integration

## 📊 Key Metrics Displayed

| Metric | Source | Use |
|--------|--------|-----|
| Risk Score | Model prediction | Primary output |
| Confidence | predict_proba max | Uncertainty indicator |
| SHAP Values | LinearExplainer | Feature importance |
| Fairness Metrics | fairness_audit_report.csv | Demographic comparison |

## ⚠️ Important Notes

1. **Data Requirements:**
   - Preprocessed data must exist at `data/preprocessed/`
   - Fairness metrics must exist at `results/fairness_metrics_by_group.csv`

2. **Performance:**
   - Prediction: < 100ms
   - SHAP generation: 1-5 seconds
   - Total response: 5-10 seconds

3. **Clinical Use:**
   - Decision support tool only
   - Always validate with clinical experts
   - Check confidence score (recommend > 70%)
   - Pay attention to fairness alerts

## 🛠️ Troubleshooting

### Error: Data files not found
```
Error: No such file or directory: './data/preprocessed/X_train.csv'
```
**Fix:** Run from project root directory, not from `app/` subdirectory

### Error: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'shap'
```
**Fix:** 
```bash
pip install shap --upgrade
```

### Streamlit takes too long
**Solution:** SHAP computation can be slow. For production:
- Use smaller reference dataset (500 samples instead of full training set)
- Implement caching with `@st.cache_resource`
- Consider approximate SHAP methods

### Port already in use
**Error:** `Address already in use. Streamlit's port 8501 is already in use.`
**Fix:**
```bash
streamlit run app/app.py --server.port 8502
```

## 📈 Example Workflow

1. **Patient Admission:**
   - Fill in demographics (age: 65, gender: Male, race: Caucasian)
   - Record clinical history (2 prior visits, 1 ER visit, 5 days hospitalized)
   - Check diagnoses (Diabetes, Hypertension)
   - List medications (Metformin, Insulin)

2. **Click "Predict Readmission Risk"**

3. **Receive Output:**
   - Risk Score: 42.3% (MODERATE RISK)
   - Confidence: 0.74 ✓
   - No fairness warnings
   - Top factors: Discharge disposition, Hospital visits, Admission source
   - Recommendations: Enhanced follow-up coordination

4. **Clinical Action:**
   - Document in EHR
   - Coordinate discharge planning
   - Schedule follow-up appointment
   - Validate with clinical judgment

## 🔐 Security & Privacy

- ✅ Local computation (no cloud transmission)
- ✅ No data storage
- ✅ HIPAA-compatible design
- ✅ Audit logs available
- ✅ No PII saved

## 📚 Additional Resources

- **App Documentation:** `app/README.md`
- **Fairness Analysis:** `notebooks/05_fairness.ipynb`
- **Explainability:** `notebooks/04_explainability.ipynb`
- **Preprocessing:** `notebooks/02_preprocessing.ipynb`
- **Modeling:** `notebooks/03_modeling.ipynb`

## 🚀 Next Steps

1. **Test Predictions:** Try various patient profiles
2. **Validate Results:** Compare with clinical expectations
3. **Integration:** Connect to EHR system
4. **Monitoring:** Set up performance dashboards
5. **Feedback Loop:** Collect outcome data for model improvement

## 📞 Support

For issues:
1. Check Troubleshooting section above
2. Review app README for detailed documentation
3. Check notebook documentation (Phase 4-5)
4. Review error logs in terminal

---

**⚖️ Remember:** This is clinical decision support, not a replacement for expert judgment. Always validate predictions with domain experts before clinical action.

**Phase 7 Complete!** 🎉
