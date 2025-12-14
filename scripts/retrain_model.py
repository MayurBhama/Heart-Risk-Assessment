"""
Industry-Grade Model Retraining Script - Simplified Version

Performs cross-validation and GridSearch-style tuning for robustness.
"""

import pandas as pd
import numpy as np
import sys
import joblib
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import logger


def main():
    print("=" * 70)
    print("INDUSTRY-GRADE MODEL RETRAINING")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv("data/processed/cardio_featured.csv")
    print(f"      Dataset shape: {df.shape}")
    
    features = [
        "age_years", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc",
        "smoke", "alco", "active", "bmi", "bmi_category",
        "pulse_pressure", "mean_arterial_pressure",
        "bp_category", "age_group", "lifestyle_risk_score",
        "metabolic_risk_score", "combined_risk_score",
    ]
    
    X = df[features].copy()
    y = df["cardio"].copy()
    
    # Split data
    print("\n[2/6] Preparing train/test split...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"      Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    # Cross-validation baseline
    print("\n[3/6] Cross-validation baseline (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1),
    }
    
    baseline_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        baseline_results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"      {name}: ROC-AUC = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    # GridSearch for best model (XGBoost)
    print("\n[4/6] Hyperparameter tuning with GridSearchCV...")
    
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 1.0],
    }
    
    xgb_base = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    
    print("      Running GridSearchCV (this may take a few minutes)...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n      Best parameters: {grid_search.best_params_}")
    print(f"      Best CV score: {grid_search.best_score_:.4f}")
    
    # Train final model
    print("\n[5/6] Training final model with best parameters...")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "=" * 70)
    print("FINAL MODEL PERFORMANCE")
    print("=" * 70)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Model: {type(best_model).__name__}")
    print(f"\nAccuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:5d}  |  FP: {cm[0][1]:5d}")
    print(f"  FN: {cm[1][0]:5d}  |  TP: {cm[1][1]:5d}")
    
    # Improvement
    old_auc = 0.7997
    improvement = (auc - old_auc) / old_auc * 100
    print(f"\n  Improvement over previous: {improvement:+.2f}%")
    
    # Save
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    Path("models/trained_models").mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, "models/trained_models/best_model.pkl")
    joblib.dump(scaler, "models/trained_models/scaler.pkl")
    
    with open("models/trained_models/best_params.json", "w") as f:
        json.dump({
            "model_name": "XGBClassifier",
            "params": grid_search.best_params_,
            "cv_score": float(grid_search.best_score_),
            "test_metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "roc_auc": float(auc),
            }
        }, f, indent=4)
    
    # Save test data for future evaluation
    test_df = X_test.copy()
    test_df["cardio"] = y_test.values
    test_df.to_csv("data/test/test.csv", index=False)
    
    print("  Model: models/trained_models/best_model.pkl")
    print("  Scaler: models/trained_models/scaler.pkl")
    print("  Params: models/trained_models/best_params.json")
    
    print("\n" + "=" * 70)
    print("RETRAINING COMPLETE!")
    print("=" * 70)
    
    return best_model


if __name__ == "__main__":
    main()
