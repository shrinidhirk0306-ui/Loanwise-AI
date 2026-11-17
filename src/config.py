"""
Configuration module for LoanWise AI.
Contains financial constants, benchmark interest rates, credit score bands,
and file path configurations.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "loan_dataset.csv"
MODEL_PATH = MODELS_DIR / "loan_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# UI & Currency Settings
APP_TITLE = "LoanWise AI – Loan Eligibility & Risk Assessment"
APP_SUBTITLE = "Smart financial profiling powered by Machine Learning and Generative AI"
CURRENCY_SYMBOL = "₹"

# Benchmark Interest Rates (% per annum) by Loan Purpose
DEFAULT_INTEREST_RATES = {
    "Home Loan": 8.5,
    "Personal Loan": 12.0,
    "Auto Loan": 9.5,
    "Education Loan": 9.0,
    "Business Loan": 13.5
}

# Credit Score (CIBIL/Experian scale) Reference Ranges
CREDIT_SCORE_RANGES = {
    "Excellent": (800, 900),
    "Good": (750, 799),
    "Fair": (650, 749),
    "Poor": (550, 649),
    "Very Poor": (300, 549)
}

# Debt-To-Income (FOIR) Benchmarks
MAX_SAFE_DTI_PERCENT = 35.0      # Ideal healthy debt ratio
MAX_ACCEPTABLE_DTI_PERCENT = 50.0 # Upper limit used by most banks
CRITICAL_DTI_PERCENT = 60.0       # Severe financial distress threshold

# Minimum Monthly Income threshold for formal credit
MIN_MONTHLY_INCOME = 15000

# Academic & Legal Disclaimer
DISCLAIMER_TEXT = (
    "Disclaimer: LoanWise AI is an academic decision-support and educational tool developed for "
    "college project evaluation (CIA-2). The predictions and risk scores are statistical estimates "
    "and do not constitute an official loan approval or legal financial advice from any banking institution."
)

