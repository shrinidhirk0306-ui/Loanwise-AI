"""
LoanWise AI - Machine Learning Prediction & Risk Assessment Engine
Loads the trained Random Forest Pipeline and performs inference on applicant profiles.
Computes calibrated approval probabilities, risk tiers (Low, Medium, High),
and key risk/positive drivers.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import pandas as pd
import numpy as np

from src.config import MODEL_PATH, DEFAULT_INTEREST_RATES
from src.financial_engine import calculate_emi, calculate_dti_ratio


class LoanPredictor:
    """Encapsulates ML pipeline loading, feature formulation, and risk evaluation."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = self._load_model()
        self.expected_numeric = [
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
        self.expected_categorical = [
            "employment_type",
            "loan_purpose"
        ]
        self.feature_columns = self.expected_numeric + self.expected_categorical

    def _load_model(self):
        """Safely load serialized Scikit-learn pipeline."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Ensure you have executed models/train.py before running predictions."
            )
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize model at {self.model_path}: {e}")

    def prepare_features(self, applicant_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts raw user inputs into the exact DataFrame schema
        required by the trained ColumnTransformer pipeline.
        """
        # Extract inputs with sensible defaults
        age = int(applicant_data.get("age", 30))
        work_experience = int(applicant_data.get("work_experience", 3))
        monthly_income = float(applicant_data.get("monthly_income", 0.0))
        coapplicant_income = float(applicant_data.get("coapplicant_income", 0.0))
        credit_score = int(applicant_data.get("credit_score", 650))
        existing_emi = float(applicant_data.get("existing_emi", 0.0))
        requested_amount = float(applicant_data.get("requested_amount", 0.0))
        loan_tenure_months = int(applicant_data.get("loan_tenure_months", 12))
        employment_type = str(applicant_data.get("employment_type", "Salaried"))
        loan_purpose = str(applicant_data.get("loan_purpose", "Personal Loan"))

        # Lookup or use custom interest rate
        interest_rate = float(
            applicant_data.get("interest_rate", DEFAULT_INTEREST_RATES.get(loan_purpose, 10.0))
        )

        # Derived Financial Metrics
        total_monthly_income = monthly_income + coapplicant_income
        proposed_emi = calculate_emi(requested_amount, interest_rate, loan_tenure_months)
        total_debt = existing_emi + proposed_emi
        dti_ratio = calculate_dti_ratio(total_monthly_income, total_debt)

        annual_income = total_monthly_income * 12.0
        loan_to_income_ratio = (
            round(requested_amount / annual_income, 2) if annual_income > 0 else 99.0
        )

        # Build single-row DataFrame with exact columns expected by preprocessor
        features_dict = {
            "age": [age],
            "work_experience": [work_experience],
            "monthly_income": [monthly_income],
            "coapplicant_income": [coapplicant_income],
            "total_monthly_income": [total_monthly_income],
            "credit_score": [credit_score],
            "existing_emi": [existing_emi],
            "requested_amount": [requested_amount],
            "loan_tenure_months": [loan_tenure_months],
            "proposed_emi": [proposed_emi],
            "dti_ratio": [dti_ratio],
            "loan_to_income_ratio": [loan_to_income_ratio],
            "employment_type": [employment_type],
            "loan_purpose": [loan_purpose]
        }

        df_input = pd.DataFrame(features_dict)
        return df_input[self.feature_columns]

    def classify_risk_tier(
        self,
        approval_prob: float,
        credit_score: int,
        dti_ratio: float,
        monthly_income: float
    ) -> Tuple[str, str, List[str], List[str]]:
        """
        Classifies risk into Low, Medium, or High based on calibrated model probabilities
        combined with prudential retail banking underwriting rules.
        """
        risk_factors = []
        positive_factors = []

        # Analyze Credit Score
        if credit_score >= 750:
            positive_factors.append(f"Excellent Credit Score ({credit_score}): Strong repayment history.")
        elif credit_score >= 650:
            positive_factors.append(f"Fair Credit Score ({credit_score}): Within standard approval threshold.")
        else:
            risk_factors.append(f"Sub-prime Credit Score ({credit_score}): High historical delinquency risk.")

        # Analyze DTI / FOIR Ratio
        if dti_ratio <= 35.0:
            positive_factors.append(f"Healthy DTI Ratio ({dti_ratio:.1f}%): Ample financial breathing room.")
        elif dti_ratio <= 50.0:
            risk_factors.append(f"Moderate DTI Ratio ({dti_ratio:.1f}%): Leaves moderate cash reserve buffer.")
        else:
            risk_factors.append(f"High DTI Ratio ({dti_ratio:.1f}%): Exceeds safe 50% banking obligation limit.")

        # Analyze Income
        if monthly_income >= 80000:
            positive_factors.append("Strong primary monthly income provides robust debt-servicing capability.")
        elif monthly_income < 25000:
            risk_factors.append("Constrained monthly take-home income may limit emergency cushion.")

        # Determine Risk Level
        # High Risk conditions: sub-prime credit, excessive debt, or low model approval probability
        if credit_score < 620 or dti_ratio > 55.0 or approval_prob < 0.40:
            risk_level = "High"
            badge_color = "#EF4444"  # Red
        # Low Risk conditions: strong credit, comfortable DTI, high approval probability
        elif credit_score >= 740 and dti_ratio <= 38.0 and approval_prob >= 0.70:
            risk_level = "Low"
            badge_color = "#10B981"  # Emerald Green
        # Medium Risk: Moderate or borderline profile
        else:
            risk_level = "Medium"
            badge_color = "#F59E0B"  # Amber

        return risk_level, badge_color, risk_factors, positive_factors

    def predict(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end prediction:
        1. Feature formulation and validation
        2. ML Pipeline inference (predict & predict_proba)
        3. Risk tier categorization & driver identification
        """
        # Generate feature matrix
        df_features = self.prepare_features(applicant_data)

        # Model Inference
        raw_pred = self.pipeline.predict(df_features)[0]
        prob_array = self.pipeline.predict_proba(df_features)[0]

        prob_rejected = float(prob_array[0])
        prob_approved = float(prob_array[1])
        is_eligible = bool(raw_pred == 1)

        # Extracted values for risk assessment
        credit_score = int(df_features["credit_score"].iloc[0])
        dti_ratio = float(df_features["dti_ratio"].iloc[0])
        monthly_income = float(df_features["monthly_income"].iloc[0])
        proposed_emi = float(df_features["proposed_emi"].iloc[0])
        total_monthly_income = float(df_features["total_monthly_income"].iloc[0])
        requested_amount = float(df_features["requested_amount"].iloc[0])

        risk_level, badge_color, risk_factors, positive_factors = self.classify_risk_tier(
            prob_approved, credit_score, dti_ratio, monthly_income
        )

        return {
            "is_eligible": is_eligible,
            "decision": "Eligible / Approved" if is_eligible else "Not Eligible / High Risk",
            "approval_probability": round(prob_approved * 100.0, 2),
            "rejection_probability": round(prob_rejected * 100.0, 2),
            "risk_level": risk_level,
            "badge_color": badge_color,
            "risk_factors": risk_factors,
            "positive_factors": positive_factors,
            "engineered_features": {
                "proposed_emi": proposed_emi,
                "dti_ratio": dti_ratio,
                "total_monthly_income": total_monthly_income,
                "loan_to_income_ratio": float(df_features["loan_to_income_ratio"].iloc[0]),
                "requested_amount": requested_amount
            }
        }


# Lazy global instance helper
_global_predictor = None

def get_predictor() -> LoanPredictor:
    """Returns singleton instance of LoanPredictor."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = LoanPredictor()
    return _global_predictor

