"""
LoanWise AI - Machine Learning Model Training Pipeline
Trains a Scikit-Learn Pipeline (StandardScaler + OneHotEncoder + RandomForestClassifier)
on loan applicant financial records with strict train-test separation and full evaluation.
"""

import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Add project root to path for config access
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import DATASET_PATH, MODEL_PATH, METRICS_PATH


def train_loan_model():
    print(f"Loading dataset from: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run data/generate_data.py first.")

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset successfully loaded with shape: {df.shape}")

    # Define feature sets
    numeric_features = [
        "age",
        "work_experience",
        "monthly_income",
        "coapplicant_income",
        "total_monthly_income",
        "credit_score",
        "existing_emi",
        "requested_amount",
        "loan_tenure_months",
        "proposed_emi",
        "dti_ratio",
        "loan_to_income_ratio"
    ]

    categorical_features = [
        "employment_type",
        "loan_purpose"
    ]

    feature_cols = numeric_features + categorical_features
    target_col = "loan_approved"

    X = df[feature_cols]
    y = df[target_col]

    # Train / Test Split (80% Train, 20% Test) with Stratification to prevent class imbalance skew
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split size: {len(X_train)} samples")
    print(f"Test split size:  {len(X_test)} samples")

    # Build Preprocessor (StandardScaler for numerics, OneHotEncoder for categoricals)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    # Initialize genuine Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Combine into unified Scikit-Learn Pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_classifier)
    ])

    print("\nTraining Random Forest Pipeline...")
    model_pipeline.fit(X_train, y_train)
    print("[OK] Model training completed successfully.")

    # Evaluate on held-out test set
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]

    # Extract Feature Importances for presentation explainability
    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
    fitted_classifier = model_pipeline.named_steps["classifier"]
    
    cat_encoder = fitted_preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names
    importances = fitted_classifier.feature_importances_

    feature_importance_dict = dict(
        sorted(
            {name: round(float(imp), 4) for name, imp in zip(all_feature_names, importances)}.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    metrics = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 150,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1],
            "raw_matrix": cm
        },
        "feature_importances": feature_importance_dict,
        "features_list": {
            "numeric": numeric_features,
            "categorical": categorical_features
        }
    }

    # Save Pipeline and Metrics
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"\n[OK] Model pipeline saved to:\n     {MODEL_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Evaluation metrics saved to:\n     {METRICS_PATH}")

    # Display evaluation summary for CIA-2
    print("\n" + "=" * 50)
    print("      CIA-2 MODEL EVALUATION METRICS REPORT")
    print("=" * 50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc * 100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  [TN={cm[0][0]}, FP={cm[0][1]}]")
    print(f"  [FN={cm[1][0]}, TP={cm[1][1]}]")
    print(f"\nTop 5 Most Influential Features:")
    for rank, (feat, imp) in enumerate(list(feature_importance_dict.items())[:5], start=1):
        print(f"  {rank}. {feat}: {imp * 100:.2f}% importance")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    train_loan_model()

