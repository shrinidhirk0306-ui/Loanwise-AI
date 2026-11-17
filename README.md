# LoanWise AI – AI-Based Loan Eligibility & Risk Assessment System

> **College CIA-2 Academic Project**  
> A modern fintech decision-support platform combining **Scikit-Learn Machine Learning** for quantitative credit underwriting and **Google Gemini Generative AI** for qualitative financial advisory.

---

## 🌟 Executive Summary

Retail banks face high default risks when underwriting credit applications, while consumers struggle to understand why loans are approved, rejected, or burdened with high interest rates. **LoanWise AI** bridges this gap:

1. **Interactive Financial Profiler**: Captures applicant demographics, employment details, income, existing liabilities, and credit score.
2. **Real-Time Banking Math Engine**: Calculates monthly EMI (standard PMT formula), total interest outlay, Debt-to-Income / FOIR (Fixed Obligation to Income Ratio), and net disposable cash flow.
3. **Genuine Machine Learning Engine**: An end-to-end Scikit-Learn `Pipeline` (`StandardScaler` + `OneHotEncoder` + `RandomForestClassifier`) trained on 2,500 realistic loan records, delivering **90.20% accuracy** and **97.74% ROC-AUC**.
4. **Three-Tier Risk Stratification**: Dynamically classifies borrower profiles into **Low Risk** (Emerald), **Medium Risk** (Amber), or **High Risk** (Red).
5. **AI Financial Advisor (Google Gemini + Offline Fallback)**: Generates structured, natural-language executive summaries, strengths, vulnerabilities, and personalized action plans. Works 100% offline even without an API key!
6. **Academic Model Transparency**: Live confusion matrix, Gini feature importance charts, and dataset explorer for CIA-2 examination defense.

---

## 🏗️ Architecture & Technology Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Frontend & Dashboard** | Streamlit + Plotly | Responsive fintech UI with custom CSS, dynamic cards, and interactive charts |
| **Data Processing** | Pandas 2.2.3 & NumPy | Data manipulation and statistical transformations |
| **Machine Learning** | Scikit-Learn 1.6.1 | Random Forest (150 estimators), Pipeline, ColumnTransformer |
| **Model Persistence** | Joblib | Binary serialization of the fitted ML pipeline |
| **Generative AI** | Google Gemini API (`google-genai`) | AI financial advisory with deterministic offline expert fallback |
| **Configuration** | Python-Dotenv | Environment variable management for API keys |

---

## 📁 Project Directory Structure

```text
LoanWise-AI/
├── .env.example               # Template for optional GEMINI_API_KEY
├── requirements.txt           # Verified dependencies (compatible with Windows Smart App Control)
├── README.md                  # Comprehensive documentation and viva presentation guide
├── app.py                     # Main Streamlit application dashboard
│
├── data/
│   ├── generate_data.py       # Realistic 2,500-sample banking dataset generator
│   └── loan_dataset.csv       # Training & validation dataset
│
├── models/
│   ├── train.py               # Model training, stratified evaluation, and metric generation
│   ├── loan_model.joblib      # Serialized Scikit-learn Pipeline
│   └── model_metrics.json     # Saved metrics, confusion matrix, and feature importances
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Benchmark rates, CIBIL bands, and credit thresholds
│   ├── financial_engine.py    # PMT EMI math, DTI/FOIR, and max borrowing capacity
│   ├── ml_engine.py           # ML inference, feature synthesis, and risk tiers
│   ├── ai_advisor.py          # Gemini AI client with offline rule-based fallback
│   └── ui_components.py       # Custom fintech CSS, metric cards, and Plotly charts
│
└── tests/
    ├── test_engines.py        # Unit tests for financial formulas and ML predictions
    └── test_ai_advisor.py     # Unit tests for AI advisor and offline fallbacks
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- PowerShell or Terminal

### 2. Setup Virtual Environment & Install Dependencies
```powershell
# Open terminal in the project directory
cd LoanWise-AI

# Create virtual environment (already configured if cloned)
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. (Optional) Configure Google Gemini API Key
To enable live Gemini AI generation:
1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` and insert your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
*Note: If no API key is provided, the application automatically runs its built-in rule-based expert advisor. The app is 100% functional offline!*

### 4. Run the Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Machine Learning Model Evaluation Report

Evaluated on an unseen **500-sample stratified test set**:

- **Accuracy**: `90.20%`
- **Precision**: `92.19%` (Minimizes bad loan sanctions)
- **Recall**: `89.86%` (Maximizes credit access for deserving borrowers)
- **F1 Score**: `91.01%`
- **ROC-AUC**: `97.74%`

### Confusion Matrix
| | Predicted Negative (0) | Predicted Positive (1) |
| :--- | :---: | :---: |
| **Actual Negative (Rejected)** | **203 (True Negatives)** | 21 (False Positives) |
| **Actual Positive (Approved)** | 28 (False Negatives) | **248 (True Positives)** |

### Top 5 Most Influential Features
1. **`dti_ratio` (48.17%)**: Monthly debt obligations relative to net take-home pay.
2. **`credit_score` (15.37%)**: Historical repayment record (CIBIL scale 300–900).
3. **`proposed_emi` (9.04%)**: Impact of the new loan installment on monthly cash flow.
4. **`existing_emi` (5.13%)**: Existing liability load.
5. **`loan_to_income_ratio` (5.11%)**: Total borrowing leverage relative to annual income.

---

## 🎯 Viva Demonstration Guide (For CIA-2 Presentation)

When presenting to examiners, use the **Quick Demo Profiles** dropdown in the sidebar to showcase distinct model behaviors:

1. **🌟 Profile 1: Prime Borrower (Low Risk)**:
   - *Data*: High income (₹1,20,000 + ₹40,000 co-applicant), 810 CIBIL score, low existing obligations, 20-year Home Loan.
   - *Outcome*: **Approved / Eligible** (98.8% confidence), **Low Risk (Emerald Badge)**, DTI 22.9%.

2. **⚖️ Profile 2: Borderline Applicant (Medium Risk)**:
   - *Data*: Moderate income (₹52,000), 675 CIBIL score, 5-year Auto Loan.
   - *Outcome*: **Approved / Eligible** (91.7% confidence), **Medium Risk (Amber Badge)** due to DTI at 43.6%.

3. **⚠️ Profile 3: Distressed Borrower (High Risk)**:
   - *Data*: Entry-level income (₹28,000), 540 CIBIL score, existing EMI ₹14,000, requesting ₹6,00,000 Personal Loan for 2 years.
   - *Outcome*: **Not Eligible** (0.6% approval probability), **High Risk (Red Badge)** due to DTI exceeding 100%.
   - *AI Advice*: Recommends tenure extension, debt consolidation, and credit repair strategies.

---

## ⚖️ Academic Disclaimer
*This software is an educational decision-support prototype created for college project evaluation (CIA-2). The predictions and risk scores are statistical estimates and do not constitute an official loan approval or legal financial advice from any banking institution.*

