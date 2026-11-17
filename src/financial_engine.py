"""
LoanWise AI - Financial Calculation Engine
Implements retail banking mathematical equations:
1. Equated Monthly Installment (EMI) using standard PMT formula
2. Amortization and total interest/repayment summary
3. Debt-to-Income (DTI / FOIR) ratio
4. Net disposable monthly income
5. Maximum affordable borrowing capacity
"""

from typing import Dict, Any


def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """
    Calculate the monthly Equated Monthly Installment (EMI) using the banking PMT formula:
    EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)

    Parameters:
        principal: Requested loan amount (P)
        annual_rate_pct: Annual interest rate in percent (e.g. 8.5 for 8.5%)
        tenure_months: Repayment tenure in months (n)

    Returns:
        Monthly EMI rounded to 2 decimal places. Returns 0.0 if inputs are invalid.
    """
    try:
        p = float(principal)
        r_annual = float(annual_rate_pct)
        n = int(tenure_months)

        if p <= 0 or n <= 0:
            return 0.0

        # Handle 0% interest edge case (e.g. promotional interest-free loans)
        if r_annual <= 0:
            return round(p / n, 2)

        # Monthly interest rate (r)
        r = (r_annual / 100.0) / 12.0

        # Standard compound factor (1 + r)^n
        comp = (1.0 + r) ** n
        emi = p * r * comp / (comp - 1.0)

        return round(float(emi), 2)
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def calculate_loan_repayment_summary(
    principal: float, annual_rate_pct: float, tenure_months: int
) -> Dict[str, Any]:
    """
    Compute full loan repayment breakdown including total principal,
    total interest payable, and overall payout amount.
    """
    p = max(0.0, float(principal))
    emi = calculate_emi(p, annual_rate_pct, tenure_months)
    n = max(0, int(tenure_months))

    total_repayment = round(emi * n, 2) if n > 0 else 0.0
    total_interest = round(max(0.0, total_repayment - p), 2)
    interest_ratio_pct = round((total_interest / p * 100.0), 2) if p > 0 else 0.0

    return {
        "principal": round(p, 2),
        "monthly_emi": emi,
        "tenure_months": n,
        "tenure_years": round(n / 12.0, 1),
        "annual_interest_rate": float(annual_rate_pct),
        "total_interest": total_interest,
        "total_repayment": total_repayment,
        "interest_ratio_pct": interest_ratio_pct
    }


def calculate_dti_ratio(total_monthly_income: float, total_monthly_debt: float) -> float:
    """
    Calculate Debt-to-Income (DTI / FOIR - Fixed Obligation to Income Ratio):
    DTI = (Total Monthly Debt / Total Monthly Income) * 100

    Parameters:
        total_monthly_income: Combined monthly take-home income
        total_monthly_debt: Sum of all existing EMIs + proposed EMI

    Returns:
        DTI percentage rounded to 2 decimal places. Capped at 100.0 for negative or zero income.
    """
    inc = float(total_monthly_income)
    debt = max(0.0, float(total_monthly_debt))

    if inc <= 0:
        return 100.0 if debt > 0 else 0.0

    dti = (debt / inc) * 100.0
    return round(float(dti), 2)


def calculate_disposable_income(
    total_monthly_income: float,
    existing_emi: float,
    proposed_emi: float,
    estimated_living_cost_pct: float = 30.0
) -> Dict[str, float]:
    """
    Calculates monthly cash surplus after servicing debt obligations
    and accounting for estimated essential living expenses.
    """
    income = max(0.0, float(total_monthly_income))
    existing = max(0.0, float(existing_emi))
    proposed = max(0.0, float(proposed_emi))
    living_pct = max(0.0, min(80.0, float(estimated_living_cost_pct)))

    total_debt = existing + proposed
    estimated_living_expenses = round(income * (living_pct / 100.0), 2)
    net_surplus = round(income - total_debt - estimated_living_expenses, 2)
    uncommitted_cash_flow = round(income - total_debt, 2)

    return {
        "total_income": income,
        "total_debt_obligations": total_debt,
        "estimated_living_expenses": estimated_living_expenses,
        "uncommitted_cash_flow": uncommitted_cash_flow,
        "net_surplus": net_surplus
    }


def estimate_max_affordable_loan(
    monthly_income: float,
    coapplicant_income: float,
    existing_emi: float,
    annual_rate_pct: float,
    tenure_months: int,
    max_dti_cap_pct: float = 50.0
) -> Dict[str, Any]:
    """
    Estimates maximum loan amount an applicant can qualify for under standard
    banking prudential FOIR/DTI guidelines.

    Formula (Reverse PMT):
    Max Allowable Total Debt = Total Income * (max_dti_cap_pct / 100)
    Max Allowable New EMI = Max Allowable Total Debt - Existing EMI
    P_max = (EMI_max * ((1 + r)^n - 1)) / (r * (1 + r)^n)
    """
    total_income = max(0.0, float(monthly_income) + float(coapplicant_income))
    existing = max(0.0, float(existing_emi))
    r_annual = max(0.1, float(annual_rate_pct))
    n = max(1, int(tenure_months))
    cap_pct = max(10.0, min(70.0, float(max_dti_cap_pct)))

    max_allowable_debt = total_income * (cap_pct / 100.0)
    max_allowable_emi = max(0.0, max_allowable_debt - existing)

    if max_allowable_emi <= 0 or total_income <= 0:
        return {
            "max_loan_amount": 0.0,
            "max_allowable_emi": 0.0,
            "limiting_factor": "Existing debt obligations already exceed or equal the prudential DTI limit (50%)."
        }

    r = (r_annual / 100.0) / 12.0
    comp = (1.0 + r) ** n
    max_principal = max_allowable_emi * (comp - 1.0) / (r * comp)

    # Round down to nearest 10,000 for realistic banking sanction limit
    sanctionable_amount = max(0.0, round(max_principal / 10000.0) * 10000.0)

    return {
        "max_loan_amount": sanctionable_amount,
        "max_allowable_emi": round(max_allowable_emi, 2),
        "tenure_months": n,
        "assumed_interest_rate": r_annual,
        "dti_cap_applied": cap_pct,
        "limiting_factor": "Prudential 50% FOIR/DTI ceiling"
    }

