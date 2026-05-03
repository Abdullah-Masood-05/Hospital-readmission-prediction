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

## Phase 1: EDA (01_EDA.ipynb)

Explore:
- Target distribution and class imbalance
- Missing value patterns
- Key feature correlations
- Demographic readmission disparities

## Setup

```bash
conda create -n readmission python=3.10
conda activate readmission
pip install -r requirements.txt
```

## Running the Analysis

```bash
# Run notebooks in order
jupyter notebook notebooks/

# Run Streamlit app
streamlit run app/app.py
```

## Authors

Data Science Team
