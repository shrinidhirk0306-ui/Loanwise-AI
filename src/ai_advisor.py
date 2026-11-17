"""
LoanWise AI - AI-Powered Financial Advisory Module
Uses Google Gemini API via the google-genai SDK to generate personalized,
explainable financial insights and recommendations.
Includes an intelligent rule-based offline fallback if an API key is absent or unreachable.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Add project root to path and load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _generate_rule_based_advice(
    applicant_data: Dict[str, Any],
    financial_summary: Dict[str, Any],
    ml_results: Dict[str, Any],
    max_loan_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Intelligent, deterministic rule-based financial advisor.
    Generates structured, professional recommendations based on genuine banking
    underwriting principles without requiring any external network connection.
    """
    name = applicant_data.get("name", "Applicant")
    credit_score = int(applicant_data.get("credit_score", 650))
    monthly_income = float(applicant_data.get("monthly_income", 0.0))
    coapplicant_income = float(applicant_data.get("coapplicant_income", 0.0))
    total_income = monthly_income + coapplicant_income
    existing_emi = float(applicant_data.get("existing_emi", 0.0))
    requested_amount = float(applicant_data.get("requested_amount", 0.0))
    tenure_months = int(applicant_data.get("loan_tenure_months", 12))
    purpose = applicant_data.get("loan_purpose", "Personal Loan")

    proposed_emi = financial_summary.get("monthly_emi", 0.0)
    dti_ratio = ml_results.get("engineered_features", {}).get("dti_ratio", 0.0)
    approval_prob = ml_results.get("approval_probability", 50.0)
    is_eligible = ml_results.get("is_eligible", False)
    risk_level = ml_results.get("risk_level", "Medium")

    # 1. Executive Summary
    if is_eligible and risk_level == "Low":
        exec_summary = (
            f"Based on our machine learning risk model, {name} presents a **strong, low-risk credit profile** "
            f"with an estimated **{approval_prob}% approval probability** for the requested {purpose} of ₹{requested_amount:,.0f}. "
            f"The profile demonstrates disciplined debt management and sufficient disposable surplus."
        )
    elif is_eligible and risk_level == "Medium":
        exec_summary = (
            f"{name}'s profile qualifies under moderate underwriting criteria with an estimated **{approval_prob}% approval probability**. "
            f"While baseline eligibility is met, certain financial parameters (such as debt ratio or credit tier) "
            f"indicate a **moderate risk exposure** that lenders may offset with higher interest rates or collateral requirements."
        )
    else:
        exec_summary = (
            f"Our statistical assessment indicates that {name}'s application currently faces **elevated risk barriers** "
            f"(approval probability: **{approval_prob}%**). The primary constraining factors are the high debt burden "
            f"(DTI of {dti_ratio:.1f}%) and credit score positioning. Improving key financial metrics will substantially enhance approval odds."
        )

    # 2. Financial Strengths
    strengths: List[str] = []
    if credit_score >= 750:
        strengths.append(f"**High Credit Score ({credit_score})**: Demonstrates exceptional historical repayment discipline and qualifies for preferential interest rates.")
    elif credit_score >= 670:
        strengths.append(f"**Satisfactory Credit Score ({credit_score})**: Meets baseline institutional underwriting benchmarks.")

    if dti_ratio <= 35.0:
        strengths.append(f"**Healthy Debt-to-Income Ratio ({dti_ratio:.1f}%)**: Well below the 50% banking risk ceiling, ensuring substantial uncommitted cash flow.")
    elif dti_ratio <= 45.0:
        strengths.append(f"**Manageable Obligation Level ({dti_ratio:.1f}%)**: Total monthly liabilities remain within tolerable operational limits.")

    if coapplicant_income > 0:
        strengths.append(f"**Co-borrower Support**: Adding ₹{coapplicant_income:,.0f} secondary monthly income expands household borrowing bandwidth.")

    if total_income >= 80000:
        strengths.append(f"**Robust Take-Home Income**: Combined net income of ₹{total_income:,.0f}/month provides solid debt-servicing resilience.")

    if not strengths:
        strengths.append("Regular employment status provides steady, predictable monthly cash inflows.")

    # 3. Risk Factors & Vulnerabilities
    risks: List[str] = []
    if dti_ratio > 50.0:
        risks.append(f"**Critical Debt-to-Income Ratio ({dti_ratio:.1f}%)**: Monthly debt obligations exceed 50% of take-home income, posing elevated default vulnerability in case of unexpected expenses.")
    elif dti_ratio > 40.0:
        risks.append(f"**Elevated FOIR Buffer ({dti_ratio:.1f}%)**: Fixed obligations leave limited cushion for emergency savings or variable living costs.")

    if credit_score < 650:
        risks.append(f"**Sub-prime Credit Rating ({credit_score})**: Historical delinquency or high credit utilization creates friction with automated underwriting algorithms.")

    if existing_emi > (total_income * 0.35):
        risks.append(f"**Pre-existing Debt Load**: Existing monthly EMIs (₹{existing_emi:,.0f}) already consume a substantial share of monthly earnings.")

    if requested_amount > (total_income * 12 * 5) and purpose != "Home Loan":
        risks.append(f"**High Loan-to-Income Exposure**: Loan amount is high relative to annual income, which elevates the lender's risk exposure.")

    if not risks:
        risks.append("No immediate red flags detected. Maintain current repayment discipline and monitor credit report annually.")

    # 4. Actionable, Personalized Recommendations
    recommendations: List[str] = []
    if dti_ratio > 45.0:
        # Calculate impact of extending tenure
        extended_tenure = tenure_months + 12
        recommendations.append(
            f"**Extend Loan Tenure**: Increasing tenure from {tenure_months} months to {extended_tenure} months would reduce monthly EMI burden and lower DTI into a safer zone."
        )

    if existing_emi > 0:
        recommendations.append(
            "**Accelerate Existing Debt Paydown**: Clearing smaller high-interest credit card balances or short-term personal loans will instantly improve your debt-to-income ratio."
        )

    if credit_score < 750:
        recommendations.append(
            "**Credit Score Optimization**: Keep revolving credit card utilization below 30% of total limit and ensure zero delayed payments for the next 6-9 months to push score above 750."
        )

    if not is_eligible and coapplicant_income == 0:
        recommendations.append(
            "**Introduce an Earning Co-applicant**: Applying jointly with an employed spouse or parent adds eligible income, which can turn an ineligible profile into an approved loan."
        )

    if max_loan_summary and max_loan_summary.get("max_loan_amount", 0) > 0:
        max_amt = max_loan_summary["max_loan_amount"]
        if requested_amount > max_amt:
            recommendations.append(
                f"**Realign Loan Quantum**: Based on standard 50% FOIR constraints, your maximum prudential borrowing capacity is approximately ₹{max_amt:,.0f}."
            )

    recommendations.append(
        "**Shop with Pre-qualification**: Compare formal pre-approved offers across multiple lenders before formal submission to avoid multiple hard credit inquiries."
    )

    # Build formatted markdown string
    markdown_output = f"""### Executive Summary
{exec_summary}

### Key Financial Strengths
""" + "\n".join(f"- {s}" for s in strengths) + """

### Risk Factors & Areas of Caution
""" + "\n".join(f"- {r}" for r in risks) + """

### Personalized Action Plan & Recommendations
""" + "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))

    return {
        "executive_summary": exec_summary,
        "financial_strengths": strengths,
        "risk_factors": risks,
        "personalized_recommendations": recommendations,
        "full_markdown_text": markdown_output,
        "source": "LoanWise Rule-Based Expert Advisor (Offline Mode)",
        "is_ai_live": False
    }


def generate_ai_financial_advice(
    applicant_data: Dict[str, Any],
    financial_summary: Dict[str, Any],
    ml_results: Dict[str, Any],
    max_loan_summary: Optional[Dict[str, Any]] = None,
    api_key_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates personalized financial advisory insights.
    Prioritizes Google Gemini API when an API key is available;
    seamlessly falls back to the deterministic banking expert system if unavailable or offline.
    """
    # Check for API key (priority: parameter override > environment variable)
    api_key = api_key_override or os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        return _generate_rule_based_advice(
            applicant_data, financial_summary, ml_results, max_loan_summary
        )

    # If API key is present, attempt Gemini API call
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = f"""
You are an expert Senior Credit Underwriter and Financial Advisor for "LoanWise AI", an educational fintech decision system.
Analyze the following applicant profile based STRICTLY on their actual financial metrics and machine learning prediction.

APPLICANT PROFILE:
- Name: {applicant_data.get('name', 'Applicant')}
- Age: {applicant_data.get('age', 'N/A')}
- Employment Type: {applicant_data.get('employment_type', 'N/A')} ({applicant_data.get('work_experience', 'N/A')} yrs experience)
- Primary Monthly Income: INR {applicant_data.get('monthly_income', 0):,.0f}
- Co-applicant Monthly Income: INR {applicant_data.get('coapplicant_income', 0):,.0f}
- Credit Score (CIBIL): {applicant_data.get('credit_score', 'N/A')}
- Existing Monthly EMIs: INR {applicant_data.get('existing_emi', 0):,.0f}
- Requested Loan: INR {applicant_data.get('requested_amount', 0):,.0f} ({applicant_data.get('loan_purpose', 'N/A')})
- Loan Tenure: {applicant_data.get('loan_tenure_months', 0)} months

CALCULATED FINANCIAL METRICS:
- Proposed Monthly EMI: INR {financial_summary.get('monthly_emi', 0):,.2f}
- Total Interest Payable: INR {financial_summary.get('total_interest', 0):,.2f}
- Debt-to-Income / FOIR Ratio: {ml_results.get('engineered_features', {}).get('dti_ratio', 0):.1f}%
- Maximum Prudential Loan Capacity: INR {max_loan_summary.get('max_loan_amount', 0) if max_loan_summary else 0:,.0f}

MACHINE LEARNING PREDICTION:
- Model Decision: {ml_results.get('decision', 'N/A')}
- Model Approval Probability: {ml_results.get('approval_probability', 0)}%
- Model Risk Level: {ml_results.get('risk_level', 'N/A')}

INSTRUCTIONS:
1. Ground your advice STRICTLY in the real numbers provided above. Never hallucinate or assume additional unlisted debts.
2. Structure your response with EXACTLY the following 4 markdown headings:
   ### Executive Summary
   ### Key Financial Strengths
   ### Risk Factors & Areas of Caution
   ### Personalized Action Plan & Recommendations
3. In 'Executive Summary', synthesize the ML probability and financial health concisely.
4. In 'Key Financial Strengths', list 2-4 bullet points highlighting specific positive metrics.
5. In 'Risk Factors & Areas of Caution', list 2-4 bullet points analyzing vulnerabilities (DTI, credit tier, debt load).
6. In 'Personalized Action Plan & Recommendations', give 3-5 numbered, highly actionable tips (e.g., tenure adjustment, debt consolidation, co-applicant strategy, or score building).
7. Maintain an empowering, professional tone. Never guarantee an official bank approval, noting this is an educational assessment.
"""

        # Generate response using gemini-2.5-flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        response_text = response.text.strip() if response.text else ""

        if not response_text:
            raise ValueError("Empty response received from Gemini API.")

        return {
            "full_markdown_text": response_text,
            "source": "Google Gemini AI (Live)",
            "is_ai_live": True
        }

    except Exception as e:
        # Gracefully handle any failure (invalid key, rate limit, quota, network error)
        # and fallback to offline expert advisor
        fallback_res = _generate_rule_based_advice(
            applicant_data, financial_summary, ml_results, max_loan_summary
        )
        fallback_res["source"] = f"LoanWise Rule-Based Expert Advisor (Offline Fallback: {type(e).__name__})"
        fallback_res["api_error_note"] = str(e)
        return fallback_res

