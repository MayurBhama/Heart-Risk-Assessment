# CardioPredict: Heart Disease Risk Assessment System

*Production-ready cardiovascular disease prediction with XGBoost, MLOps pipeline, and clinical-grade explainability.*

**[Live Demo](https://heart-risk-assessment-hm2ufzw9arcd2klkwap5ls.streamlit.app/)**

---

## Abstract

Cardiovascular disease (CVD) remains the **leading cause of death globally**, accounting for 17.9 million deaths annually. Early detection and risk assessment can significantly reduce mortality through timely intervention. This project presents **CardioPredict**, an end-to-end machine learning system that:

1. **Predicts cardiovascular disease risk** using an optimized XGBoost classifier
2. **Provides personalized recommendations** based on modifiable risk factors
3. **Implements clinical-grade interpretations** following NHS blood pressure guidelines
4. **Deploys via robust MLOps practices** with DVC pipeline, MLflow tracking, and FastAPI/Streamlit architecture

**Key Contributions:**
1. Feature engineering with clinical domain knowledge (BMI, blood pressure categories, risk scores)
2. Hyperparameter optimization via GridSearchCV with stratified cross-validation
3. NHS-compliant blood pressure interpretation for clinical accuracy
4. What-If analysis for lifestyle change impact simulation
5. Production-ready microservices architecture (FastAPI backend + Streamlit frontend)

---

## Table of Contents

1. [Introduction: The Cardiovascular Disease Challenge](#1-introduction-the-cardiovascular-disease-challenge)
2. [Dataset Analysis: Understanding Our Data](#2-dataset-analysis-understanding-our-data)
3. [Feature Engineering: Clinical Domain Knowledge](#3-feature-engineering-clinical-domain-knowledge)
4. [Model Architecture: Why XGBoost](#4-model-architecture-why-xgboost)
5. [Training Pipeline: MLOps Best Practices](#5-training-pipeline-mlops-best-practices)
6. [Evaluation Metrics: Healthcare-Focused Analysis](#6-evaluation-metrics-healthcare-focused-analysis)
7. [Clinical Interpretations: NHS Guidelines](#7-clinical-interpretations-nhs-guidelines)
8. [System Architecture: Production Deployment](#8-system-architecture-production-deployment)
9. [Results and Analysis](#9-results-and-analysis)
10. [Lessons Learned](#10-lessons-learned)
11. [Technical Documentation](#11-technical-documentation)

---

## 1. Introduction: The Cardiovascular Disease Challenge

### 1.1 Why Cardiovascular Disease Prediction Matters

```
GLOBAL CVD STATISTICS

Deaths per year:                    17.9 million (31% of all global deaths)
People living with CVD:             523 million worldwide
Preventable cases:                  80% of CVD deaths are preventable
Early detection impact:             70% risk reduction with lifestyle changes
Healthcare cost (US alone):         $363 billion annually
```

**The Critical Problem**: Many people are unaware of their cardiovascular risk until it's too late. Traditional risk assessment requires clinical visits and lab tests, limiting accessibility.

### 1.2 What This Project Does

CardioPredict is a **web-based risk assessment tool** that:

| Feature | Description |
|---------|-------------|
| **Risk Prediction** | Estimates cardiovascular disease probability using ML |
| **Risk Contributors** | Identifies and ranks modifiable vs non-modifiable factors |
| **Clinical Interpretations** | Blood pressure, cholesterol, glucose analysis |
| **Personalized Action Plan** | Priority-ranked recommendations with measurable targets |
| **What-If Analysis** | Simulates impact of lifestyle changes on risk |
| **PDF Report Generation** | Downloadable clinical report for healthcare providers |

### 1.3 The Challenges We Addressed

| Challenge | Why It's Hard | Our Solution |
|-----------|--------------|--------------|
| **Class Imbalance** | ~50% negative class in dataset | Stratified sampling + class-weighted evaluation |
| **Feature Engineering** | Raw data lacks clinical context | Domain-driven features (BMI, risk scores) |
| **Hyperparameter Tuning** | Preventing overfitting | GridSearchCV with cross-validation |
| **Clinical Accuracy** | ML must match medical guidelines | NHS BP categories implementation |
| **Deployment Complexity** | ML models hard to productionize | FastAPI + Streamlit microservices |

---

## 2. Dataset Analysis: Understanding Our Data

### 2.1 Cardiovascular Disease Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | [Kaggle Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| **Total Records** | 70,000 patient records |
| **Features** | 11 input features + 1 target |
| **Target** | Binary (0: No CVD, 1: CVD Present) |
| **Data Collection** | Medical examinations |

### 2.2 Feature Description

| Feature | Type | Description | Clinical Significance |
|---------|------|-------------|----------------------|
| `age` | Numeric | Age in days | Converted to years; major risk factor |
| `gender` | Binary | 1: Female, 2: Male | Males have higher CVD risk |
| `height` | Numeric | Height in cm | Used for BMI calculation |
| `weight` | Numeric | Weight in kg | Used for BMI calculation |
| `ap_hi` | Numeric | Systolic BP (mmHg) | Key hypertension indicator |
| `ap_lo` | Numeric | Diastolic BP (mmHg) | Key hypertension indicator |
| `cholesterol` | Ordinal | 1: Normal, 2: Above Normal, 3: High | Lipid-based risk factor |
| `gluc` | Ordinal | 1: Normal, 2: Above Normal, 3: High | Metabolic health indicator |
| `smoke` | Binary | 0: No, 1: Yes | Modifiable lifestyle factor |
| `alco` | Binary | 0: No, 1: Yes | Modifiable lifestyle factor |
| `active` | Binary | 0: No, 1: Yes | Physical activity status |

### 2.3 Class Distribution

```
TARGET VARIABLE DISTRIBUTION

No CVD (0):  ################################  ~35,000 (50%)
CVD (1):     ################################  ~35,000 (50%)

BALANCED DATASET:
- Unlike many medical datasets, this is reasonably balanced
- No extreme oversampling needed
- Stratified splitting preserves this balance
```

### 2.4 Data Quality Issues Addressed

```
DATA CLEANING PIPELINE

1. IMPOSSIBLE VALUES:
   - Blood pressure < 0 or > 300? -> Removed
   - Age < 0 or > 120 years? -> Removed
   - Height < 100cm or > 250cm? -> Flagged/removed

2. OUTLIER DETECTION:
   - IQR method with 1.5 multiplier
   - Z-score threshold of 3
   - Domain-specific bounds for medical values

3. MISSING VALUES:
   - Dataset has no missing values
   - But we handle nulls in API for robustness

4. FEATURE CONVERSION:
   - Age: days -> years
   - Gender: 1/2 -> 0/1 encoding
```

---

## 3. Feature Engineering: Clinical Domain Knowledge

### 3.1 The Philosophy

> "Raw data captures measurements. **Engineered features capture clinical knowledge.**"

We created 9 additional features based on medical domain expertise:

### 3.2 Engineered Features

#### BMI (Body Mass Index)

```python
# BMI Calculation
bmi = weight / (height/100) ** 2

# BMI Categories (WHO Standards)
BMI_CATEGORIES = {
    0: "Underweight",    # < 18.5
    1: "Normal",         # 18.5 - 25
    2: "Overweight",     # 25 - 30
    3: "Obese"           # > 30
}
```

**Clinical Significance**: BMI > 25 increases CVD risk by 30%. BMI > 30 doubles the risk.

#### Blood Pressure Features

```python
# Pulse Pressure (indicator of arterial stiffness)
pulse_pressure = ap_hi - ap_lo

# Mean Arterial Pressure (average BP during cardiac cycle)
map_pressure = ap_lo + (pulse_pressure / 3)

# NHS Blood Pressure Categories
BP_CATEGORIES = {
    0: "Normal",              # ≤120/80
    1: "High Normal",         # 121-139/81-89
    2: "Stage 1 Hypertension", # 140-159/90-99
    3: "Stage 2 Hypertension", # 160-179/100-119
    4: "Severe Hypertension"   # ≥180/≥120
}
```

#### Risk Scores

```python
# Lifestyle Risk Score (modifiable factors)
lifestyle_risk = (smoke * 3) + (alco * 2) + ((1 - active) * 2)
# Range: 0-7, higher = worse lifestyle

# Metabolic Risk Score (clinical indicators)
metabolic_risk = (cholesterol - 1) * 10 + (gluc - 1) * 10 + bp_category * 10
# Range: 0-50, higher = worse metabolic health

# Combined Risk Score (overall assessment)
combined_risk = lifestyle_risk * 5 + metabolic_risk
# Weighted combination for comprehensive view
```

### 3.3 Feature Engineering Pipeline

```
FEATURE ENGINEERING FLOW

[Raw Data: 11 features]
         |
         v
+-------------------+
| Age Conversion    |  age_days -> age_years
+-------------------+
         |
         v
+-------------------+
| BMI Calculation   |  bmi, bmi_category
+-------------------+
         |
         v
+-------------------+
| BP Features       |  pulse_pressure, map, bp_category
+-------------------+
         |
         v
+-------------------+
| Age Groups        |  young/middle/senior
+-------------------+
         |
         v
+-------------------+
| Risk Scores       |  lifestyle, metabolic, combined
+-------------------+
         |
         v
[Engineered Data: 20 features]
```

---

## 4. Model Architecture: Why XGBoost

### 4.1 Model Selection Process

We evaluated 4 different algorithms using 5-fold stratified cross-validation:

```
MODEL COMPARISON (Cross-Validation Results)

Model                  |   ROC-AUC   |  Std Dev  |  Training Time
-----------------------|-------------|-----------|---------------
Logistic Regression    |    0.78     |   ±0.01   |    Fast
Random Forest          |    0.82     |   ±0.02   |    Medium
Gradient Boosting      |    0.84     |   ±0.01   |    Slow
XGBoost               |    0.86     |   ±0.01   |    Medium

WINNER: XGBoost with 8% improvement over logistic regression baseline
```

### 4.2 Why XGBoost Won

| Advantage | Explanation |
|-----------|-------------|
| **Gradient Boosting** | Learns from mistakes of previous trees |
| **Regularization** | L1/L2 prevents overfitting |
| **Handles Missing Values** | Built-in handling (useful for production) |
| **Feature Importance** | Interpretable risk factor ranking |
| **Efficient** | Parallelized training on CPU |
| **Production Ready** | Well-documented, stable library |

### 4.3 Final Model Configuration

```yaml
# Optimized XGBoost Configuration
xgboost:
  n_estimators: 200          # Number of boosting rounds
  learning_rate: 0.05        # Step size shrinkage
  max_depth: 5               # Maximum tree depth
  min_child_weight: 3        # Minimum sum of instance weight
  subsample: 0.8             # Subsample ratio of training data
  colsample_bytree: 0.8      # Subsample ratio of columns
  
  # CRITICAL FOR DEPLOYMENT:
  eval_metric: "logloss"     # Prevents warnings in FastAPI
  objective: "binary:logistic"
  random_state: 42           # Reproducibility
```

### 4.4 Hyperparameter Tuning

```python
# GridSearchCV Parameter Grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
}

# Total combinations: 3 × 3 × 3 × 3 × 2 = 162
# With 3-fold CV: 486 model fits

grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=3,                    # 3-fold cross-validation
    scoring="roc_auc",       # Optimize for ranking ability
    n_jobs=-1,               # Use all CPU cores
)
```

---

## 5. Training Pipeline: MLOps Best Practices

### 5.1 DVC Pipeline Architecture

We use **Data Version Control (DVC)** for reproducible ML pipelines:

```
DVC PIPELINE STAGES

+----------------+     +----------------+     +----------------+     +----------------+
|    INGEST      | --> |     CLEAN      | --> |   FEATURES     | --> |    TRAIN       |
|                |     |                |     |                |     |                |
| Load raw CSV   |     | Remove outliers|     | Engineer 9 new |     | Train XGBoost  |
| Validate data  |     | Fix values     |     | clinical feats |     | GridSearch CV  |
| Save processed |     | Normalize      |     | Calculate risk |     | Save model.pkl |
+----------------+     +----------------+     +----------------+     +----------------+
```

```yaml
# dvc.yaml
stages:
  ingest:
    cmd: python scripts/stage_ingest.py
    deps:
      - src/data_ingestion/ingestion.py
      - data/raw/cardio_train.csv
    outs:
      - data/processed/cardio_ingested.csv

  clean:
    cmd: python scripts/stage_clean.py
    deps:
      - src/data_processing/cleaning.py
      - data/processed/cardio_ingested.csv
    outs:
      - data/processed/cardio_cleaned.csv

  features:
    cmd: python scripts/stage_features.py
    deps:
      - src/feature_engineering/features.py
      - data/processed/cardio_cleaned.csv
    outs:
      - data/processed/cardio_featured.csv

  train:
    cmd: python scripts/stage_train.py
    deps:
      - src/model_training/train.py
      - data/processed/cardio_featured.csv
    outs:
      - models/trained_models/best_model.pkl
      - models/trained_models/scaler.pkl
```

### 5.2 MLflow Experiment Tracking

```python
# MLflow tracking for reproducibility
mlflow.set_experiment("cardiovascular_disease_prediction")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    })
    mlflow.sklearn.log_model(model, "xgboost_model")
```

### 5.3 Cross-Validation Strategy

```
STRATIFIED K-FOLD CROSS-VALIDATION

Why Stratified?
- Preserves class distribution in each fold
- Prevents bias from unbalanced splits
- More reliable performance estimates

Fold 1: Train[################] Test[####]  -> ROC-AUC: 0.857
Fold 2: Train[################] Test[####]  -> ROC-AUC: 0.862
Fold 3: Train[################] Test[####]  -> ROC-AUC: 0.851
Fold 4: Train[################] Test[####]  -> ROC-AUC: 0.859
Fold 5: Train[################] Test[####]  -> ROC-AUC: 0.868

Mean: 0.859 ± 0.006 (very stable!)
```

---

## 6. Evaluation Metrics: Healthcare-Focused Analysis

### 6.1 Why We Prioritize Recall

```
THE COST OF ERRORS IN HEALTHCARE

FALSE NEGATIVE (Missing a sick patient):
- Patient leaves thinking they're healthy
- Disease progresses without treatment
- Potentially fatal outcome
- VERY EXPENSIVE MISTAKE

FALSE POSITIVE (Flagging a healthy patient):
- Patient gets additional tests
- Some anxiety and inconvenience
- Doctor confirms they're fine
- MANAGEABLE MISTAKE

IN HEALTHCARE: We optimize for HIGH RECALL
(Catch as many true positives as possible, even at cost of some false positives)
```

### 6.2 Our Metric Suite

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.86 | Excellent ranking ability |
| **Accuracy** | 73% | Overall correctness |
| **Recall (Sensitivity)** | 74% | % of actual CVD cases detected |
| **Precision** | 72% | % of predicted CVD that are true |
| **F1-Score** | 0.73 | Balance of precision/recall |

### 6.3 Confusion Matrix Analysis

```
CONFUSION MATRIX (Test Set: 14,000 samples)

                    Predicted
                 No CVD    CVD
Actual  No CVD    5,040    1,960   (TN=5040, FP=1960)
        CVD       1,820    5,180   (FN=1820, TP=5180)

INTERPRETATION:
- True Negatives:  5,040 correctly identified as healthy
- False Positives: 1,960 healthy flagged as at-risk (acceptable)
- False Negatives: 1,820 at-risk missed (we want to minimize this)
- True Positives:  5,180 correctly identified as at-risk
```

---

## 7. Clinical Interpretations: NHS Guidelines

### 7.1 Blood Pressure Categories

We implement **NHS Blood Pressure Guidelines** for clinical accuracy:

```
NHS BLOOD PRESSURE CLASSIFICATION

Category              | Systolic (mmHg)  | Diastolic (mmHg)
----------------------|------------------|------------------
Normal                | ≤ 120            | ≤ 80
High Normal           | 121 - 139        | 81 - 89
Stage 1 Hypertension  | 140 - 159        | 90 - 99
Stage 2 Hypertension  | 160 - 179        | 100 - 119
Severe Hypertension   | ≥ 180            | ≥ 120
```

### 7.2 Risk Contributor Ranking

```
RISK FACTOR PRIORITIZATION

HIGH PRIORITY (Immediate Action):
┌─────────────────────────────────────────────────────────────┐
│ ● Smoking (Active smoker)        - Modifiable, Very High Impact │
│ ● Stage 2+ Hypertension          - Treatable, Medical Review     │
│ ● Very High Cholesterol          - Modifiable, Diet/Medication   │
│ ● Obese BMI (>30)                - Modifiable, Lifestyle Change  │
└─────────────────────────────────────────────────────────────┘

MEDIUM PRIORITY (30-Day Action):
┌─────────────────────────────────────────────────────────────┐
│ ● Stage 1 Hypertension           - Monitor + Lifestyle           │
│ ● Above Normal Cholesterol       - Dietary Changes               │
│ ● Overweight BMI (25-30)         - Weight Management             │
│ ● Regular Alcohol Consumption    - Reduce Intake                 │
└─────────────────────────────────────────────────────────────┘

ONGOING/NON-MODIFIABLE:
┌─────────────────────────────────────────────────────────────┐
│ ● Age (55+)                      - Non-modifiable, Monitor       │
│ ● Sedentary Lifestyle            - Begin Activity Gradually      │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Personalized Action Plan

Each user receives **priority-ranked recommendations** with measurable targets:

| Priority | Recommendation | Target |
|----------|---------------|--------|
| **High** | Consider smoking cessation program | Smoke-free in 3-6 months |
| **High** | Discuss cholesterol management with doctor | LDL < 100 mg/dL |
| **Medium** | Monitor blood pressure and reduce sodium | < 2g sodium/day |
| **Medium** | Work on gradual weight reduction | -5 kg in 3 months |
| **Long-term** | Begin regular physical activity | 30 min walking, 5x/week |
| **Long-term** | Schedule regular health monitoring | Check-up every 6 months |

---

## 8. System Architecture: Production Deployment

### 8.1 Microservices Architecture

```
SYSTEM ARCHITECTURE

                         ┌─────────────────────┐
                         │    User Browser     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  STREAMLIT FRONTEND │
                         │   (streamlit.app)   │
                         │                     │
                         │  • Input Form       │
                         │  • Risk Display     │
                         │  • Action Plan      │
                         │  • What-If Analysis │
                         │  • PDF Report       │
                         └──────────┬──────────┘
                                    │ HTTP/REST
                                    ▼
                         ┌─────────────────────┐
                         │   FASTAPI BACKEND   │
                         │  (Hugging Face)     │
                         │                     │
                         │  • /predict         │
                         │  • Model Inference  │
                         │  • Risk Calculations│
                         │  • Recommendations  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
               ┌─────────────────┐   ┌─────────────────┐
               │  XGBoost Model  │   │  Standard       │
               │  (best_model.pkl│   │  Scaler         │
               │   ~2MB)         │   │  (scaler.pkl)   │
               └─────────────────┘   └─────────────────┘
```

### 8.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check + API status |
| `/predict` | POST | Main prediction endpoint |

**Prediction Request:**
```json
{
  "age": 55,
  "gender": "male",
  "height": 170,
  "weight": 80,
  "ap_hi": 140,
  "ap_lo": 90,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}
```

**Prediction Response:**
```json
{
  "prediction": 1,
  "probability": 0.67,
  "risk_level": "High",
  "summary": "Estimated risk: 67.0% (High).",
  "heart_health": {"score": 45.0, "status": "Fair"},
  "bp_interpretation": "High blood pressure (Stage 1).",
  "cholesterol_interpretation": "Cholesterol above normal.",
  "glucose_interpretation": "Glucose levels within normal range.",
  "risk_factors": {
    "bmi": 27.7,
    "bmi_category": 2,
    "blood_pressure_category": 2,
    "combined_risk_score": 42.5
  },
  "recommendations": [
    "High blood pressure detected. Reduce salt and fried foods.",
    "Cholesterol elevated. Increase fiber, avoid fried foods.",
    "Moderate risk. Lifestyle changes recommended."
  ]
}
```

### 8.3 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ML Framework** | scikit-learn, XGBoost | Model training and inference |
| **Backend API** | FastAPI | REST API endpoints |
| **Frontend** | Streamlit | Interactive web interface |
| **Pipeline** | DVC | Reproducible ML pipelines |
| **Tracking** | MLflow | Experiment tracking |
| **Deployment** | Hugging Face + Streamlit Cloud | Production hosting |
| **Version Control** | Git + GitHub | Code management |

---

## 9. Results and Analysis

### 9.1 Model Performance Summary

| Metric | Our Model | Baseline (LogReg) | Improvement |
|--------|-----------|-------------------|-------------|
| **ROC-AUC** | 0.86 | 0.78 | +10.3% |
| **Accuracy** | 73% | 70% | +4.3% |
| **Recall** | 74% | 68% | +8.8% |
| **F1-Score** | 0.73 | 0.69 | +5.8% |

### 9.2 Feature Importance

```
TOP 10 MOST IMPORTANT FEATURES

1. combined_risk_score    ████████████████████  0.18
2. age_years              ████████████████      0.15
3. ap_hi (systolic BP)    ██████████████        0.13
4. bmi                    ████████████          0.11
5. metabolic_risk_score   ██████████            0.10
6. cholesterol            ████████              0.08
7. ap_lo (diastolic BP)   ██████                0.07
8. weight                 ██████                0.06
9. bp_category            ████                  0.05
10. lifestyle_risk_score  ████                  0.04

KEY INSIGHT:
Engineered features (combined_risk_score, metabolic_risk_score)
rank higher than raw features, validating our domain knowledge approach.
```

### 9.3 Clinical Validation

| Test Case | Expected | Predicted | Correct? |
|-----------|----------|-----------|----------|
| 35yo, healthy, active, 120/80 | Low Risk | 8% (Low) | ✅ |
| 55yo, smoker, obese, 160/100 | High Risk | 89% (Very High) | ✅ |
| 45yo, high cholesterol, sedentary | Moderate | 52% (Moderate) | ✅ |
| 65yo, diabetic, hypertensive | Very High | 91% (Very High) | ✅ |

---

## 10. Lessons Learned

### 10.1 Technical Lessons

| Lesson | What We Learned |
|--------|-----------------|
| **Feature engineering > bigger models** | Domain-driven features improved ROC-AUC by 5%+ |
| **GridSearchCV validation** | Cross-validation prevents overfitting to single split |
| **API field naming matters** | Mismatch between API and frontend caused bugs |
| **NHS guidelines > custom rules** | Clinical standards provide immediate credibility |
| **Microservices separation** | Backend + frontend split enables independent deployment |

### 10.2 Medical AI Lessons

| Lesson | What We Learned |
|--------|-----------------|
| **Safe language is critical** | "Elevated risk" vs "You have heart disease" |
| **Relative risk, not absolute probability** | ML probability ≠ clinical probability |
| **Recall > Precision for screening** | Missing a patient is worse than extra tests |
| **Explainability builds trust** | Showing WHY helps users and doctors |
| **Clinical input essential** | NHS guidelines prevent embarrassing mistakes |

### 10.3 Future Improvements

| Improvement | Expected Impact | Priority |
|-------------|-----------------|----------|
| Add laboratory values (LDL, HDL, triglycerides) | +5% accuracy | High |
| Implement SHAP for individual explanations | Better explainability | Medium |
| Add Framingham Risk Score comparison | Clinical validation | Medium |
| Time-series risk tracking | Trend analysis | Low |
| Mobile app version | Accessibility | Low |

---

## 11. Technical Documentation

### 11.1 Project Structure

```
cardio-mlops/
├── configs/                    # Configuration files
│   ├── data_config.yaml        # Data preprocessing settings
│   └── model_config.yaml       # Model hyperparameters
├── data/                       # Data directory (DVC tracked)
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and featured data
│   └── test/                   # Test split for evaluation
├── models/                     # Trained model artifacts
│   ├── trained_models/         # Production models
│   │   ├── best_model.pkl      # XGBoost classifier
│   │   ├── scaler.pkl          # StandardScaler
│   │   └── best_params.json    # Optimal hyperparameters
│   └── model_artifacts/        # MLflow artifacts
├── scripts/                    # DVC pipeline scripts
│   ├── stage_ingest.py         # Data ingestion
│   ├── stage_clean.py          # Data cleaning
│   ├── stage_features.py       # Feature engineering
│   ├── stage_train.py          # Model training
│   └── retrain_model.py        # Full retraining script
├── src/                        # Source code
│   ├── data_ingestion/         # Data loading utilities
│   ├── data_processing/        # Cleaning functions
│   ├── feature_engineering/    # Feature creation
│   ├── model_training/         # Training logic
│   ├── model_evaluation/       # Evaluation metrics
│   └── deployment/             # FastAPI + Streamlit apps
│       ├── fastapi_app.py      # Backend API
│       └── streamlit_app.py    # Frontend UI
├── tests/                      # Unit tests
├── utils/                      # Logger and exceptions
├── streamlit_local.py          # Local development UI
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC pipeline lock
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### 11.2 Installation

```bash
# Clone the repository
git clone https://github.com/MayurBhama/Heart-Risk-Assessment.git
cd Heart-Risk-Assessment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Pull data with DVC
dvc pull
```

### 11.3 Running Locally

```bash
# Start FastAPI backend (Terminal 1)
python -m uvicorn src.deployment.fastapi_app:app --reload --port 8000

# Start Streamlit frontend (Terminal 2)
streamlit run streamlit_local.py --server.port 8501

# Access the application
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

### 11.4 Running the DVC Pipeline

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train

# View pipeline DAG
dvc dag
```

### 11.5 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API health check |
| `/predict` | POST | Make CVD risk prediction |

```bash
# Example API call
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": "male",
    "height": 170,
    "weight": 80,
    "ap_hi": 140,
    "ap_lo": 90,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
  }'
```

---

## Known Limitations

| Limitation | Explanation |
|------------|-------------|
| **Self-reported data** | Lifestyle factors may not reflect actual behavior |
| **Categorical clinical values** | Cholesterol/glucose not laboratory-grade |
| **Population-level model** | Not personalized to individual medical history |
| **No time component** | Single point-in-time assessment |
| **Not calibrated** | Probability is relative risk, not absolute |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
