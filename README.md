# Hospital Readmission Prediction Project

Predicting 30-day hospital readmission risk using the Diabetes 130-US Hospitals dataset with focus on fairness and explainability.

## Project Structure

```
project/
├── diabetes+130-us.../
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_fairness.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── explain.py
├── app/
│   └── app.py        ← Streamlit app
├── requirements.txt
└── README.md
```

## Dataset

**Source:** Diabetes 130-US Hospitals for Years 1999-2008

- **Target:** Readmission status (3 classes: <30 days, >30 days, NO)
- **Binary Target:** Readmitted within 30 days (1) vs. others (0)
- **Class Distribution:** ~11% readmitted <30 days (imbalanced)

## Key Challenges

1. **Class Imbalance:** ~11% minority class → SMOTE/class weights required
2. **Missing Values:** weight (~97%), payer_code, medical_specialty
3. **Fairness:** Investigate readmission disparities by race and gender
4. **Model Explainability:** SHAP values for clinical interpretability

## Phase 1: EDA & Analysis (01_EDA.ipynb)

### Key Findings:
- **Dataset Size:** 101,766 patient records from 130 US hospitals (1999-2008)
- **Target Distribution:** 
  - No readmission: 76.9%
  - Readmitted >30 days: 12.0%
  - Readmitted <30 days: 11.1% (minority class)
- **Class Imbalance Ratio:** ~9:1 (non-readmitted to readmitted)
- **Missing Values:** weight (96.6%), payer_code (49.0%), medical_specialty (3.3%)
- **Patient Demographics:**
  - Age range: Pediatric to 90+ years
  - Race distribution: ~75% Caucasian, ~19% African American, ~6% Other
  - Gender: ~47% Female, ~53% Male
- **Key Correlations with Readmission:**
  - Prior hospitalizations (weak positive)
  - Medication changes (weak positive)
  - Discharge type (conditional)
  - Length of stay (moderate positive)

### EDA Insights:
- Readmission risk increases with number of prior visits
- Patients with multiple medication changes show higher readmission rates
- Age group 70-80 years shows elevated readmission rates
- Potential demographic disparities by race/gender detected

---

## Phase 2: Preprocessing & Feature Engineering (02_preprocessing.ipynb)

### Data Processing Steps:
1. **Dropped Features:** weight (~97% missing), payer_code (49% missing)
2. **Imputation:**
   - Medical specialty: Filled with 'Unknown' (3% missing)
   - Admission type/source: Mode imputation
3. **Encoding:**
   - Categorical features: One-hot encoding
   - Target variable: Binary (1=readmitted <30 days, 0=not readmitted)
4. **Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling)

### Feature Engineering Results:
- **Initial Features:** 46 raw features
- **Final Features After One-Hot Encoding:** 2,343 features
- **Data Split:** 70% train, 15% validation, 15% test
  - Training samples: 91,314 (after SMOTE)
  - Validation samples: 10,728
  - Test samples: 10,728

### Preprocessing Insights:
- SMOTE balanced training data: 90,370 non-readmitted vs. 90,944 readmitted (1:1 ratio)
- One-hot encoding expanded feature space significantly due to high cardinality categorical variables
- Scaler saved for consistent feature normalization during inference
- 70-15-15 split ensures robust validation and testing

---

## Phase 3: Model Development & Training (03_modeling.ipynb)

### Models Trained & Performance:

| Model | AUC-ROC | F1-Score | Avg Precision | Training Time |
|-------|---------|----------|--------------|---------------|
| **Logistic Regression** (Baseline) | 0.6270 | 0.2167 | 0.1443 | ~1s |
| **Random Forest** (Ensemble) | 0.6319 | 0.0358 | 0.1502 | ~5s |
| **XGBoost** (Best) | **0.6721** ⭐ | 0.0247 | **0.1807** ⭐ | ~10s |
| **Neural Network (PyTorch + GPU)** | 0.6014 | 0.1773 | 0.1283 | ~2s (GPU) |

### Best Model: XGBoost
- **AUC-ROC:** 0.6721 (best discrimination ability)
- **Average Precision:** 0.1807 (best at identifying readmission risk)
- **Advantages:** Handles feature interactions, interpretable, CPU-optimized
- **Trade-offs:** Lower F1-score due to class imbalance

### Model Insights:
1. **Class Imbalance Challenge:** All models show low F1-scores due to ~10:1 class imbalance even with SMOTE
2. **GPU Acceleration:** Neural Network successfully trained on NVIDIA RTX A5000 (17.18 GB VRAM) with CUDA 12.8
   - Training completed with early stopping at epoch 11
   - GPU acceleration provides 10-15x speedup vs. CPU
3. **Ensemble vs. Linear:** XGBoost (ensemble) outperforms Logistic Regression (linear baseline)
4. **Deep Learning:** Neural Network shows trade-off between precision and recall; simpler models more effective for this tabular dataset

### Evaluation Metrics Used:
- **AUC-ROC:** Primary metric (handles class imbalance well, 0.5 = random, 1.0 = perfect)
- **F1-Score:** Harmonic mean of precision-recall (sensitive to class imbalance)
- **Average Precision:** Area under Precision-Recall curve (suitable for imbalanced data)
- **Classification Reports:** Detailed precision, recall, and support by class

### Predictions on Test Set (XGBoost):
- **True Negatives (No Readmission):** 9,784 correctly identified
- **True Positives (Readmission):** ~9 correctly identified (low due to threshold)
- **False Positives:** Minimal false alarms
- **False Negatives:** High (conservative predictions)

---

## Technical Stack & Environment

### GPU Acceleration Status ✓
- **GPU Device:** NVIDIA RTX A5000 Laptop (16 GB VRAM)
- **CUDA Version:** 12.8
- **PyTorch Version:** 2.11.0+cu128
- **Status:** ✓ GPU acceleration successfully enabled and tested

### Environment Setup:
```bash
# Create conda environment
conda create -n myenv python=3.11

# Activate environment
conda activate myenv

# Install PyTorch with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### Key Libraries:
- **Data:** pandas, numpy, scikit-learn
- **ML Models:** scikit-learn, xgboost, torch
- **Visualization:** matplotlib, seaborn
- **Utils:** joblib, pickle

---

## Recommendations & Next Steps

### Phase 4: Explainability (04_explainability.ipynb)
- Implement SHAP values for model feature importance
- Generate LIME explanations for individual predictions
- Identify top 10-15 features driving readmission risk
- Create clinical decision support visualizations

### Phase 5: Fairness Analysis (05_fairness.ipynb)
- Evaluate model performance across demographic groups (race, gender, age)
- Measure and mitigate fairness disparities
- Implement fairness constraints if necessary
- Document demographic parity metrics

### Model Deployment:
- Package best model (XGBoost) for production
- Create REST API for real-time predictions
- Build Streamlit dashboard for monitoring
- Establish performance baselines and drift detection

---

## Running the Analysis

```bash
# Clone repository
git clone [repo-url]

# Install dependencies
conda activate myenv
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/

# Notebooks:
# 01_EDA.ipynb → Exploratory Data Analysis
# 02_preprocessing.ipynb → Feature Engineering & SMOTE
# 03_modeling.ipynb → Model Training & Evaluation
# 04_explainability.ipynb → SHAP & LIME Explanations
# 05_fairness.ipynb → Demographic Fairness Analysis
```

## Key Findings Summary

✅ **Data Quality:** 101K+ records with manageable missing values (dropped weight feature)
✅ **Class Imbalance Handled:** SMOTE successfully balanced training data
✅ **GPU Acceleration:** Verified NVIDIA RTX A5000 with CUDA 12.8 support
✅ **Model Performance:** XGBoost achieves 0.6721 AUC-ROC (best discrimination)
⚠️ **Low F1-Score:** Imbalanced test set results in conservative predictions
✅ **Feature Engineering:** 2,343 features engineered through one-hot encoding

---

## Authors

Data Science Team | Hospital Readmission Prediction Project 2025-2026
