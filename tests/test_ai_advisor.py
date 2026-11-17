"""
LoanWise AI - Test Suite for AI Financial Advisory Module
Validates offline fallback generation, structured section completeness,
and error handling when API key is missing or invalid.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ai_advisor import generate_ai_financial_advice, _generate_rule_based_advice
from src.financial_engine import calculate_loan_repayment_summary, estimate_max_affordable_loan
from src.ml_engine import get_predictor


def run_ai_advisor_tests():
    print("=" * 60)
    print(" TESTING AI FINANCIAL ADVISOR MODULE")
    print("=" * 60)

    # 1. Setup sample applicant data
    applicant = {
        "name": "Arun Kumar",
        "age": 32,
        "employment_type": "Salaried",
        "work_experience": 6,
        "monthly_income": 75000,
        "coapplicant_income": 25000,
        "credit_score": 780,
        "existing_emi": 12000,
        "requested_amount": 1800000,
        "loan_tenure_months": 180,
        "loan_purpose": "Home Loan",
        "interest_rate": 8.5
    }

    # Generate actual financial summary and ML prediction
    financial_summary = calculate_loan_repayment_summary(1800000, 8.5, 180)
    max_loan = estimate_max_affordable_loan(75000, 25000, 12000, 8.5, 180, 50.0)
    predictor = get_predictor()
    ml_results = predictor.predict(applicant)

    # Test 1: Explicit Offline Rule-Based Advisor
    print("\n--- Test 1: Built-in Offline Expert Advisor ---")
    offline_advice = _generate_rule_based_advice(applicant, financial_summary, ml_results, max_loan)
    print(f"Source: {offline_advice['source']}")
    print(f"Is AI Live: {offline_advice['is_ai_live']}")
    print(f"Executive Summary preview: {offline_advice['executive_summary'][:120]}...")
    print(f"Number of Strengths: {len(offline_advice['financial_strengths'])}")
    print(f"Number of Risks: {len(offline_advice['risk_factors'])}")
    print(f"Number of Recommendations: {len(offline_advice['personalized_recommendations'])}")

    assert len(offline_advice["financial_strengths"]) > 0, "Must have strengths"
    assert len(offline_advice["personalized_recommendations"]) > 0, "Must have recommendations"
    assert "Executive Summary" in offline_advice["full_markdown_text"]

    # Test 2: Call generate_ai_financial_advice with NO API KEY (should safely fallback)
    print("\n--- Test 2: Fallback when API Key is absent ---")
    res_no_key = generate_ai_financial_advice(
        applicant, financial_summary, ml_results, max_loan, api_key_override=""
    )
    print(f"Result Source: {res_no_key['source']}")
    print(f"Is AI Live: {res_no_key['is_ai_live']}")
    assert res_no_key["is_ai_live"] is False
    assert "LoanWise Rule-Based Expert Advisor" in res_no_key["source"]

    # Test 3: Call generate_ai_financial_advice with an INVALID API KEY (should catch error & fallback gracefully)
    print("\n--- Test 3: Graceful Fallback with Invalid API Key ---")
    res_invalid_key = generate_ai_financial_advice(
        applicant, financial_summary, ml_results, max_loan, api_key_override="invalid_dummy_key_12345"
    )
    print(f"Result Source: {res_invalid_key['source']}")
    print(f"Is AI Live: {res_invalid_key['is_ai_live']}")
    print(f"Caught Error Note: {res_invalid_key.get('api_error_note')}")
    assert res_invalid_key["is_ai_live"] is False
    assert "Offline Fallback" in res_invalid_key["source"]

    # Test 4: Distressed Applicant Profile (High Risk, Low CIBIL)
    print("\n--- Test 4: Advisory for High-Risk Distressed Applicant ---")
    distressed_applicant = {
        "name": "Rohan Gupta",
        "age": 26,
        "employment_type": "Self-Employed",
        "work_experience": 2,
        "monthly_income": 30000,
        "coapplicant_income": 0,
        "credit_score": 580,
        "existing_emi": 15000,
        "requested_amount": 500000,
        "loan_tenure_months": 24,
        "loan_purpose": "Personal Loan",
        "interest_rate": 12.0
    }
    fin_d = calculate_loan_repayment_summary(500000, 12.0, 24)
    max_d = estimate_max_affordable_loan(30000, 0, 15000, 12.0, 24, 50.0)
    ml_d = predictor.predict(distressed_applicant)

    advice_d = generate_ai_financial_advice(
        distressed_applicant, fin_d, ml_d, max_d, api_key_override=""
    )
    print(f"Decision: {ml_d['decision']}")
    print(f"Risk Level: {ml_d['risk_level']}")
    print(f"Executive Summary preview: {advice_d['executive_summary'][:140]}...")
    print(f"Identified Vulnerabilities: {advice_d['risk_factors']}")
    print(f"Targeted Recommendations: {advice_d['personalized_recommendations']}")

    assert ml_d["risk_level"] == "High"
    assert any("580" in r or "Sub-prime" in r for r in advice_d["risk_factors"]), "Should flag credit score"

    print("\n[PASS] All AI Advisor module unit tests passed successfully!\n")


if __name__ == "__main__":
    run_ai_advisor_tests()

