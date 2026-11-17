"""
LoanWise AI - Synthetic Dataset Generator
Generates a realistic, statistically grounded loan applicant dataset of 2,500 records
reflecting retail banking underwriting rules (CIBIL score, FOIR/DTI, employment, loan tenure).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path for config access
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import DATASET_PATH, DEFAULT_INTEREST_RATES


def calculate_monthly_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Calculate monthly Equated Monthly Installment (EMI) using PMT formula."""
    if principal <= 0 or tenure_months <= 0:
        return 0.0
    r = (annual_rate_pct / 100.0) / 12.0
    if r == 0:
        return principal / tenure_months
    emi = principal * r * ((1.0 + r) ** tenure_months) / (((1.0 + r) ** tenure_months) - 1.0)
    return round(emi, 2)


def generate_loan_dataset(num_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic applicant financial records with realistic credit distributions.
    Target variable 'loan_approved' (0 or 1) is derived from genuine risk factors
    plus calibrated noise to simulate real-world underwriting variance.
    """
    np.random.seed(random_seed)

    applicant_ids = [f"APP-{10000 + i}" for i in range(num_samples)]
    
    # Age: Normal distribution bounded between 21 and 65
    age = np.clip(np.random.normal(loc=36, scale=9, size=num_samples).astype(int), 21, 65)

    # Employment Type: 60% Salaried, 25% Self-Employed, 15% Business
    emp_choices = ["Salaried", "Self-Employed", "Business"]
    emp_probs = [0.60, 0.25, 0.15]
    employment_type = np.random.choice(emp_choices, size=num_samples, p=emp_probs)

    # Work Experience in years (cannot exceed age - 20)
    max_exp = np.maximum(age - 21, 1)
    work_experience = np.clip(
        np.random.exponential(scale=6, size=num_samples).astype(int), 0, max_exp
    )

    # Primary Monthly Income (in INR, lognormal distribution, ~ ₹25k to ₹250k)
    # Median ~ ₹55,000, mean ~ ₹68,000
    monthly_income = np.round(
        np.clip(np.random.lognormal(mean=10.9, sigma=0.45, size=num_samples), 20000, 350000),
        -2  # round to nearest 100
    )

    # Co-applicant Income: 45% have a co-applicant (spouse/parent)
    has_coapplicant = np.random.binomial(1, 0.45, size=num_samples)
    raw_coapplicant_income = np.round(
        np.clip(np.random.lognormal(mean=10.2, sigma=0.5, size=num_samples), 15000, 150000),
        -2
    )
    coapplicant_income = has_coapplicant * raw_coapplicant_income

    total_monthly_income = monthly_income + coapplicant_income

    # Credit Score (CIBIL 300 - 900)
    # Mixture of two normal distributions: prime borrowers (70%) and subprime/near-prime (30%)
    prime_scores = np.random.normal(loc=745, scale=45, size=num_samples)
    subprime_scores = np.random.normal(loc=610, scale=60, size=num_samples)
    is_prime = np.random.binomial(1, 0.72, size=num_samples)
    credit_score = np.where(is_prime == 1, prime_scores, subprime_scores)
    credit_score = np.clip(np.round(credit_score).astype(int), 300, 900)

    # Existing Monthly Debt / EMIs (scaled logically with income)
    debt_prop = np.clip(np.random.beta(a=2, b=8, size=num_samples), 0.0, 0.55)
    existing_emi = np.round(total_monthly_income * debt_prop, -2)

    # Loan Purpose
    loan_purposes = ["Home Loan", "Personal Loan", "Auto Loan", "Education Loan", "Business Loan"]
    purpose_probs = [0.35, 0.25, 0.20, 0.10, 0.10]
    loan_purpose = np.random.choice(loan_purposes, size=num_samples, p=purpose_probs)

    # Requested Loan Amount and Tenure by Purpose
    requested_amount = np.zeros(num_samples)
    loan_tenure_months = np.zeros(num_samples, dtype=int)
    proposed_emi = np.zeros(num_samples)

    for i in range(num_samples):
        purpose = loan_purpose[i]
        inc = total_monthly_income[i]

        if purpose == "Home Loan":
            # 10 to 60 times monthly income
            mult = np.random.uniform(15, 55)
            requested_amount[i] = np.clip(round(inc * mult, -3), 500000, 12000000)
            loan_tenure_months[i] = int(np.random.choice([120, 180, 240, 300, 360]))
            rate = DEFAULT_INTEREST_RATES["Home Loan"]

        elif purpose == "Personal Loan":
            # 2 to 12 times monthly income
            mult = np.random.uniform(2, 12)
            requested_amount[i] = np.clip(round(inc * mult, -3), 50000, 1500000)
            loan_tenure_months[i] = int(np.random.choice([12, 24, 36, 48, 60]))
            rate = DEFAULT_INTEREST_RATES["Personal Loan"]

        elif purpose == "Auto Loan":
            mult = np.random.uniform(4, 18)
            requested_amount[i] = np.clip(round(inc * mult, -3), 150000, 3000000)
            loan_tenure_months[i] = int(np.random.choice([24, 36, 48, 60, 84]))
            rate = DEFAULT_INTEREST_RATES["Auto Loan"]

        elif purpose == "Education Loan":
            mult = np.random.uniform(5, 25)
            requested_amount[i] = np.clip(round(inc * mult, -3), 200000, 4000000)
            loan_tenure_months[i] = int(np.random.choice([60, 84, 120, 180]))
            rate = DEFAULT_INTEREST_RATES["Education Loan"]

        else:  # Business Loan
            mult = np.random.uniform(6, 30)
            requested_amount[i] = np.clip(round(inc * mult, -3), 300000, 5000000)
            loan_tenure_months[i] = int(np.random.choice([24, 36, 60, 84]))
            rate = DEFAULT_INTEREST_RATES["Business Loan"]

        proposed_emi[i] = calculate_monthly_emi(requested_amount[i], rate, loan_tenure_months[i])

    # Financial Ratios
    # DTI (Debt-to-Income / FOIR): (existing_emi + proposed_emi) / total_monthly_income
    total_monthly_obligations = existing_emi + proposed_emi
    dti_ratio = np.round((total_monthly_obligations / total_monthly_income) * 100.0, 2)

    # Loan-to-Income Ratio (LTI): requested_amount / annual_income
    annual_income = total_monthly_income * 12.0
    loan_to_income_ratio = np.round(requested_amount / annual_income, 2)

    # Realistic Underwriting Rule + Logistic Probability Model for ground truth labels
    # Banking standard logit score:
    # 1. Credit Score: >750 gives large positive bonus, <620 severe penalty
    cs_factor = (credit_score - 680) / 45.0

    # 2. DTI / FOIR: <= 35% is safe, 35-50% neutral, >50% heavy penalty
    dti_factor = -1.8 * ((dti_ratio - 42.0) / 12.0)

    # 3. LTI factor: LTI > 6 hurts
    lti_factor = -0.6 * ((loan_to_income_ratio - 3.5) / 1.5)

    # 4. Work Experience & Employment stability
    exp_factor = np.clip(work_experience / 5.0, 0, 1.5)
    emp_factor = np.where(employment_type == "Salaried", 0.4, np.where(employment_type == "Business", 0.1, -0.1))

    # Base logit formula
    logit = 0.5 + (1.6 * cs_factor) + (1.5 * dti_factor) + lti_factor + (0.5 * exp_factor) + emp_factor

    # Hard institutional cutoff checks (e.g. CIBIL < 550 or DTI > 65% almost always rejected)
    for i in range(num_samples):
        if credit_score[i] < 540 or dti_ratio[i] > 65.0:
            logit[i] -= 4.0
        elif credit_score[i] >= 780 and dti_ratio[i] <= 35.0:
            logit[i] += 3.0

    # Convert logit to probability via sigmoid
    prob_approval = 1.0 / (1.0 + np.exp(-logit))

    # Draw binary outcome with realistic noise
    loan_approved = np.random.binomial(1, prob_approval)

    # Construct DataFrame
    df = pd.DataFrame({
        "applicant_id": applicant_ids,
        "age": age,
        "employment_type": employment_type,
        "work_experience": work_experience,
        "monthly_income": monthly_income,
        "coapplicant_income": coapplicant_income,
        "total_monthly_income": total_monthly_income,
        "credit_score": credit_score,
        "existing_emi": existing_emi,
        "loan_purpose": loan_purpose,
        "requested_amount": requested_amount,
        "loan_tenure_months": loan_tenure_months,
        "proposed_emi": proposed_emi,
        "dti_ratio": dti_ratio,
        "loan_to_income_ratio": loan_to_income_ratio,
        "loan_approved": loan_approved
    })

    return df


def main():
    print(f"Generating realistic loan applicant dataset (2,500 records)...")
    df = generate_loan_dataset(num_samples=2500, random_seed=42)
    
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    
    print(f"[OK] Successfully generated {len(df)} records and saved to:\n     {DATASET_PATH}")
    print(f"\n--- Quick Dataset Summary ---")
    print(f"Total Applicants: {len(df)}")
    approval_rate = df['loan_approved'].mean() * 100
    print(f"Approval Rate: {approval_rate:.1f}% ({df['loan_approved'].sum()} Approved, {len(df) - df['loan_approved'].sum()} Rejected)")
    print(f"Average Credit Score: {df['credit_score'].mean():.0f}")
    print(f"Average Monthly Income: INR {df['monthly_income'].mean():,.0f}")
    print(f"Average DTI (FOIR): {df['dti_ratio'].mean():.1f}%")


if __name__ == "__main__":
    main()

