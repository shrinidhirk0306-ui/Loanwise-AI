"""
LoanWise AI - Comprehensive Verification Test Suite
Tests financial formulas (manual benchmarks) and ML inference engine across realistic profiles.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.financial_engine import (
    calculate_emi,
    calculate_loan_repayment_summary,
    calculate_dti_ratio,
    calculate_disposable_income,
    estimate_max_affordable_loan
)
from src.ml_engine import get_predictor


def test_financial_calculations():
    print("=" * 60)
    print(" 1. TESTING FINANCIAL ENGINE CALCULATIONS")
    print("=" * 60)

    # Benchmark Test 1: Home Loan ₹1,000,000 at 8.5% for 20 years (240 months)
    # Theoretical PMT = 8,678.23
    p1, r1, n1 = 1000000, 8.5, 240
    emi1 = calculate_emi(p1, r1, n1)
    print(f"Test 1 - Home Loan: P={p1}, Rate={r1}%, Tenure={n1}m")
    print(f"  Calculated EMI: INR {emi1:,.2f} (Expected: ~8,678.23)")
    assert abs(emi1 - 8678.23) < 0.1, f"EMI mismatch: got {emi1}"

    # Benchmark Test 2: Personal Loan ₹100,000 at 12.0% for 1 year (12 months)
    # Theoretical PMT = 8,884.88
    p2, r2, n2 = 100000, 12.0, 12
    emi2 = calculate_emi(p2, r2, n2)
    print(f"Test 2 - Personal Loan: P={p2}, Rate={r2}%, Tenure={n2}m")
    print(f"  Calculated EMI: INR {emi2:,.2f} (Expected: ~8,884.88)")
    assert abs(emi2 - 8884.88) < 0.1, f"EMI mismatch: got {emi2}"

    # Test 3: Repayment summary breakdown
    summary = calculate_loan_repayment_summary(p1, r1, n1)
    print(f"Test 3 - Repayment Summary:")
    print(f"  Total Repayment: INR {summary['total_repayment']:,.2f}")
    print(f"  Total Interest:  INR {summary['total_interest']:,.2f}")
    assert summary["total_repayment"] > p1, "Total repayment must exceed principal"

    # Test 4: DTI Ratio
    # Income 80,000, existing 10,000, proposed 14,000 -> Debt 24,000 -> DTI = (24/80)*100 = 30.0%
    dti = calculate_dti_ratio(80000, 24000)
    print(f"Test 4 - DTI Ratio: Income=80,000, Debt=24,000 -> DTI={dti}% (Expected: 30.0%)")
    assert dti == 30.0, f"DTI mismatch: got {dti}"

    # Test 5: Maximum Affordable Loan
    # Income 70,000, existing 10,000, rate 8.5%, tenure 240m, 50% cap
    # Max total debt = 35,000. Available for new EMI = 25,000.
    max_loan = estimate_max_affordable_loan(
        monthly_income=70000,
        coapplicant_income=0,
        existing_emi=10000,
        annual_rate_pct=8.5,
        tenure_months=240,
        max_dti_cap_pct=50.0
    )
    print(f"Test 5 - Max Affordable Loan:")
    print(f"  Max Loan Sanctionable: INR {max_loan['max_loan_amount']:,.0f}")
    print(f"  Max Allowable EMI:     INR {max_loan['max_allowable_emi']:,.2f}")
    assert max_loan["max_loan_amount"] > 0, "Max loan should be greater than 0"

    # Edge cases
    assert calculate_emi(0, 10, 12) == 0.0, "Zero principal must return 0 EMI"
    assert calculate_emi(100000, 0, 10) == 10000.0, "Zero interest must equal P/n"
    assert calculate_dti_ratio(0, 5000) == 100.0, "Zero income with debt must return 100% DTI"

    print("[PASS] All Financial Engine unit tests passed!\n")


def test_ml_prediction_engine():
    print("=" * 60)
    print(" 2. TESTING MACHINE LEARNING INFERENCE ENGINE")
    print("=" * 60)

    predictor = get_predictor()

    # Profile A: Prime Borrower (High Income, 800 CIBIL, low DTI, Home Loan)
    profile_a = {
        "age": 35,
        "employment_type": "Salaried",
        "work_experience": 8,
        "monthly_income": 120000,
        "coapplicant_income": 40000,
        "credit_score": 810,
        "existing_emi": 15000,
        "loan_purpose": "Home Loan",
        "requested_amount": 2500000,
        "loan_tenure_months": 240,
        "interest_rate": 8.5
    }

    res_a = predictor.predict(profile_a)
    print("Profile A (Prime Borrower - Home Loan):")
    print(f"  Decision: {res_a['decision']}")
    print(f"  Approval Probability: {res_a['approval_probability']}%")
    print(f"  Risk Level: {res_a['risk_level']} (Badge: {res_a['badge_color']})")
    print(f"  DTI Ratio: {res_a['engineered_features']['dti_ratio']}%")
    assert res_a["is_eligible"] is True, "Profile A should be eligible"
    assert res_a["risk_level"] == "Low", f"Profile A should be Low Risk, got {res_a['risk_level']}"

    # Profile B: Borderline / Moderate Risk (Average Income, 675 CIBIL, moderate DTI)
    profile_b = {
        "age": 28,
        "employment_type": "Self-Employed",
        "work_experience": 3,
        "monthly_income": 45000,
        "coapplicant_income": 0,
        "credit_score": 675,
        "existing_emi": 8000,
        "loan_purpose": "Personal Loan",
        "requested_amount": 350000,
        "loan_tenure_months": 36,
        "interest_rate": 12.0
    }

    res_b = predictor.predict(profile_b)
    print("\nProfile B (Borderline Borrower - Personal Loan):")
    print(f"  Decision: {res_b['decision']}")
    print(f"  Approval Probability: {res_b['approval_probability']}%")
    print(f"  Risk Level: {res_b['risk_level']} (Badge: {res_b['badge_color']})")
    print(f"  DTI Ratio: {res_b['engineered_features']['dti_ratio']}%")
    assert res_b["risk_level"] in ["Low", "Medium"], f"Profile B expected Low/Medium, got {res_b['risk_level']}"

    # Profile C: Distressed Applicant (Low income, low CIBIL, high requested amount, DTI > 60%)
    profile_c = {
        "age": 25,
        "employment_type": "Salaried",
        "work_experience": 1,
        "monthly_income": 25000,
        "coapplicant_income": 0,
        "credit_score": 530,
        "existing_emi": 12000,
        "loan_purpose": "Personal Loan",
        "requested_amount": 600000,
        "loan_tenure_months": 24,
        "interest_rate": 12.0
    }

    res_c = predictor.predict(profile_c)
    print("\nProfile C (Distressed Borrower - High DTI & Poor CIBIL):")
    print(f"  Decision: {res_c['decision']}")
    print(f"  Approval Probability: {res_c['approval_probability']}%")
    print(f"  Risk Level: {res_c['risk_level']} (Badge: {res_c['badge_color']})")
    print(f"  DTI Ratio: {res_c['engineered_features']['dti_ratio']}%")
    print(f"  Identified Risk Factors: {res_c['risk_factors']}")
    assert res_c["is_eligible"] is False, "Profile C should NOT be eligible"
    assert res_c["risk_level"] == "High", f"Profile C must be High Risk, got {res_c['risk_level']}"

    print("\n[PASS] All ML inference engine unit tests passed successfully!\n")


if __name__ == "__main__":
    test_financial_calculations()
    test_ml_prediction_engine()

