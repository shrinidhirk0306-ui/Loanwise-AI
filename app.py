"""
LoanWise AI – AI-Based Loan Eligibility and Risk Assessment System
A modern fintech application built with Streamlit, Scikit-Learn, and Google Gemini API.
College CIA-2 Academic Project.
"""

import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd

# Internal Application Imports
from src.config import (
    APP_TITLE,
    DEFAULT_INTEREST_RATES,
    CREDIT_SCORE_RANGES,
    DATASET_PATH,
    METRICS_PATH,
    MODEL_PATH
)
from src.financial_engine import (
    calculate_emi,
    calculate_loan_repayment_summary,
    calculate_dti_ratio,
    calculate_disposable_income,
    estimate_max_affordable_loan
)
from src.ml_engine import get_predictor
from src.ai_advisor import generate_ai_financial_advice
from src.ui_components import (
    apply_fintech_styles,
    render_hero_header,
    render_disclaimer,
    render_metric_card,
    render_decision_banner,
    create_loan_breakdown_pie,
    create_dti_gauge_chart,
    create_cash_flow_bar_chart,
    create_feature_importance_chart,
    create_confusion_matrix_heatmap
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="LoanWise AI – Credit & Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Modern Fintech CSS
apply_fintech_styles()

# Initialize session state for pre-fills and results
if "applicant_form" not in st.session_state:
    st.session_state["applicant_form"] = {
        "name": "Priya Sharma",
        "age": 32,
        "employment_type": "Salaried",
        "work_experience": 6,
        "monthly_income": 85000,
        "coapplicant_income": 25000,
        "credit_score": 780,
        "existing_emi": 12000,
        "loan_purpose": "Home Loan",
        "requested_amount": 2800000,
        "loan_tenure_months": 240,
        "interest_rate": 8.5
    }

if "assessment_result" not in st.session_state:
    st.session_state["assessment_result"] = None


# -----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION & PRE-SET PROFILES
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏦 System Settings")
    st.caption("College CIA-2 Academic Demonstration System")

    # Demo Profiles for Quick Evaluation
    st.markdown("#### ⚡ Quick Demo Profiles")
    preset_choice = st.selectbox(
        "Load Pre-configured Profile:",
        [
            "Custom (Manual Input)",
            "🌟 Prime Borrower (Low Risk)",
            "⚖️ Borderline Borrower (Medium Risk)",
            "⚠️ Distressed Applicant (High Risk)"
        ],
        index=0,
        help="Select a benchmark profile to instantly demonstrate model behavior during viva."
    )

    if preset_choice == "🌟 Prime Borrower (Low Risk)":
        st.session_state["applicant_form"] = {
            "name": "Arjun Mehta",
            "age": 36,
            "employment_type": "Salaried",
            "work_experience": 10,
            "monthly_income": 120000,
            "coapplicant_income": 40000,
            "credit_score": 810,
            "existing_emi": 15000,
            "loan_purpose": "Home Loan",
            "requested_amount": 3500000,
            "loan_tenure_months": 240,
            "interest_rate": 8.5
        }
    elif preset_choice == "⚖️ Borderline Borrower (Medium Risk)":
        st.session_state["applicant_form"] = {
            "name": "Kavita Reddy",
            "age": 29,
            "employment_type": "Self-Employed",
            "work_experience": 4,
            "monthly_income": 52000,
            "coapplicant_income": 0,
            "credit_score": 675,
            "existing_emi": 12000,
            "loan_purpose": "Auto Loan",
            "requested_amount": 750000,
            "loan_tenure_months": 60,
            "interest_rate": 9.5
        }
    elif preset_choice == "⚠️ Distressed Applicant (High Risk)":
        st.session_state["applicant_form"] = {
            "name": "Rahul Verma",
            "age": 25,
            "employment_type": "Salaried",
            "work_experience": 1,
            "monthly_income": 28000,
            "coapplicant_income": 0,
            "credit_score": 540,
            "existing_emi": 14000,
            "loan_purpose": "Personal Loan",
            "requested_amount": 600000,
            "loan_tenure_months": 24,
            "interest_rate": 12.0
        }

    st.divider()

    # Optional Gemini API Key
    st.markdown("#### 🤖 Google Gemini AI Key")
    custom_gemini_key = st.text_input(
        "API Key (Optional):",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="If left blank, LoanWise AI automatically uses its built-in rule-based expert advisor. No key required!"
    )
    if custom_gemini_key:
        st.success("Gemini API Key active", icon="✅")
    else:
        st.info("Offline rule-based advisor active (No key required)", icon="💡")

    st.divider()
    st.markdown("#### 📋 System Specifications")
    st.markdown(
        """
        - **ML Model:** Random Forest Classifier (150 trees)
        - **Dataset:** 2,500 Underwriting Records
        - **Model Accuracy:** 90.20%
        - **Framework:** Streamlit + Scikit-Learn
        """
    )


# -----------------------------------------------------------------------------
# MAIN APP HEADER & TABS
# -----------------------------------------------------------------------------
render_hero_header()

tab1, tab2, tab3 = st.tabs([
    "📝 1. Loan Assessment & Calculator",
    "📊 2. Assessment Results & AI Insights",
    "🔬 3. ML Model Transparency & Metrics"
])


# =============================================================================
# TAB 1: APPLICANT FORM & REAL-TIME CALCULATOR
# =============================================================================
with tab1:
    render_disclaimer()

    st.markdown("### Applicant Financial Profile")
    st.caption("Enter the prospective borrower's employment, income, and requested credit terms below.")

    col_form, col_calc = st.columns([1.1, 0.9], gap="large")

    current_form = st.session_state["applicant_form"]

    with col_form:
        st.markdown("#### Personal & Employment Details")
        f_name = st.text_input("Applicant Full Name", value=current_form["name"])

        c_age, c_emp, c_exp = st.columns(3)
        with c_age:
            f_age = st.number_input("Age (Years)", min_value=21, max_value=65, value=int(current_form["age"]))
        with c_emp:
            emp_options = ["Salaried", "Self-Employed", "Business"]
            emp_index = emp_options.index(current_form["employment_type"]) if current_form["employment_type"] in emp_options else 0
            f_emp = st.selectbox("Employment Type", emp_options, index=emp_index)
        with c_exp:
            f_exp = st.number_input("Experience (Years)", min_value=0, max_value=40, value=int(current_form["work_experience"]))

        st.markdown("#### Income & Existing Obligations")
        c_inc1, c_inc2 = st.columns(2)
        with c_inc1:
            f_inc = st.number_input(
                "Primary Monthly Net Income (₹)",
                min_value=10000,
                max_value=2000000,
                value=int(current_form["monthly_income"]),
                step=5000
            )
        with c_inc2:
            f_coinc = st.number_input(
                "Co-Applicant Monthly Income (₹)",
                min_value=0,
                max_value=1000000,
                value=int(current_form["coapplicant_income"]),
                step=5000
            )

        c_cibil, c_emi = st.columns(2)
        with c_cibil:
            f_cibil = st.slider(
                "Credit Score (CIBIL Scale)",
                min_value=300,
                max_value=900,
                value=int(current_form["credit_score"]),
                help="Standard bureau credit score from 300 to 900."
            )
            # Display credit bracket label
            if f_cibil >= 800:
                cibil_label = "🟢 Excellent (800+)"
            elif f_cibil >= 750:
                cibil_label = "🟢 Good (750-799)"
            elif f_cibil >= 650:
                cibil_label = "🟡 Fair (650-749)"
            elif f_cibil >= 550:
                cibil_label = "🟠 Poor (550-649)"
            else:
                cibil_label = "🔴 Very Poor (<550)"
            st.caption(f"Score Rating: **{cibil_label}**")

        with c_emi:
            f_exist_emi = st.number_input(
                "Existing Monthly EMIs (₹)",
                min_value=0,
                max_value=500000,
                value=int(current_form["existing_emi"]),
                step=2000,
                help="Sum of ongoing personal loans, credit cards, or auto EMIs."
            )

        st.markdown("#### Loan Requirements")
        c_purp, c_rate = st.columns(2)
        with c_purp:
            purpose_options = list(DEFAULT_INTEREST_RATES.keys())
            purp_index = purpose_options.index(current_form["loan_purpose"]) if current_form["loan_purpose"] in purpose_options else 0
            f_purpose = st.selectbox("Loan Purpose", purpose_options, index=purp_index)
        with c_rate:
            default_rate = DEFAULT_INTEREST_RATES.get(f_purpose, 10.0)
            f_rate = st.number_input(
                "Interest Rate (% p.a.)",
                min_value=4.0,
                max_value=25.0,
                value=float(default_rate),
                step=0.25
            )

        c_amt, c_tenure = st.columns(2)
        with c_amt:
            f_amt = st.number_input(
                "Requested Loan Amount (₹)",
                min_value=50000,
                max_value=20000000,
                value=int(current_form["requested_amount"]),
                step=50000
            )
        with c_tenure:
            f_tenure = st.slider(
                "Repayment Tenure (Months)",
                min_value=6,
                max_value=360,
                value=int(current_form["loan_tenure_months"]),
                step=6
            )
            st.caption(f"Equivalent to **{f_tenure / 12:.1f} years**")

    # Right Column: Real-Time Dynamic EMI Preview
    with col_calc:
        st.markdown("#### ⚡ Live Loan Cost Preview")
        st.caption("Instant mathematical computation using the standard banking PMT equation.")

        # Compute preview values
        preview_summary = calculate_loan_repayment_summary(f_amt, f_rate, f_tenure)
        preview_emi = preview_summary["monthly_emi"]
        preview_tot_income = f_inc + f_coinc
        preview_dti = calculate_dti_ratio(preview_tot_income, f_exist_emi + preview_emi)

        st.markdown(
            f"""
            <div class="lw-card" style="background: #F8FAFC; border: 1.5px solid #CBD5E1;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;">
                    Estimated Monthly EMI
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #0F172A; font-family: 'JetBrains Mono', monospace; margin: 4px 0 10px 0;">
                    ₹{preview_emi:,.2f}
                </div>
                <div style="font-size: 0.9rem; color: #475569;">
                    Total Repayment: <b>₹{preview_summary['total_repayment']:,.2f}</b><br>
                    Total Interest: <b style="color: #059669;">₹{preview_summary['total_interest']:,.2f}</b><br>
                    Projected DTI (FOIR): <b>{preview_dti:.1f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Small Donut Chart for Preview
        fig_pie = create_loan_breakdown_pie(f_amt, preview_summary["total_interest"])
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # Submit Assessment Action
    col_btn, _ = st.columns([2, 3])
    with col_btn:
        assess_clicked = st.button(
            "🚀 Assess Loan Eligibility & Generate AI Report",
            type="primary",
            use_container_width=True
        )

    if assess_clicked:
        with st.spinner("Evaluating profile against Random Forest pipeline & generating AI insights..."):
            applicant_payload = {
                "name": f_name,
                "age": f_age,
                "employment_type": f_emp,
                "work_experience": f_exp,
                "monthly_income": f_inc,
                "coapplicant_income": f_coinc,
                "credit_score": f_cibil,
                "existing_emi": f_exist_emi,
                "loan_purpose": f_purpose,
                "requested_amount": f_amt,
                "loan_tenure_months": f_tenure,
                "interest_rate": f_rate
            }

            # 1. Financial calculations
            rep_summary = calculate_loan_repayment_summary(f_amt, f_rate, f_tenure)
            max_loan = estimate_max_affordable_loan(f_inc, f_coinc, f_exist_emi, f_rate, f_tenure)
            disp_income = calculate_disposable_income(f_inc + f_coinc, f_exist_emi, rep_summary["monthly_emi"])

            # 2. Genuine ML prediction
            predictor = get_predictor()
            ml_res = predictor.predict(applicant_payload)

            # 3. AI / Rule-based advisory
            ai_res = generate_ai_financial_advice(
                applicant_data=applicant_payload,
                financial_summary=rep_summary,
                ml_results=ml_res,
                max_loan_summary=max_loan,
                api_key_override=custom_gemini_key
            )

            # Save full assessment in session state
            st.session_state["assessment_result"] = {
                "applicant": applicant_payload,
                "repayment_summary": rep_summary,
                "max_loan": max_loan,
                "disposable_income": disp_income,
                "ml_results": ml_res,
                "ai_advisor": ai_res
            }

            st.success("✅ Assessment completed successfully! View your detailed results in **Tab 2: Assessment Results & AI Insights**.")


# =============================================================================
# TAB 2: RESULTS & AI INSIGHTS DASHBOARD
# =============================================================================
with tab2:
    render_disclaimer()

    res = st.session_state["assessment_result"]

    if res is None:
        st.info("ℹ️ No active assessment yet. Enter applicant details in **Tab 1: Loan Assessment & Calculator** and click **Assess Loan Eligibility** to see your results.", icon="💡")
    else:
        applicant = res["applicant"]
        rep = res["repayment_summary"]
        max_l = res["max_loan"]
        disp = res["disposable_income"]
        ml = res["ml_results"]
        ai = res["ai_advisor"]

        st.markdown(f"### Assessment Dossier: {applicant['name']}")
        st.caption(f"Profile: {applicant['employment_type']} &bull; Age {applicant['age']} &bull; Requested ₹{applicant['requested_amount']:,.0f} ({applicant['loan_purpose']})")

        # Top Executive Decision Banner
        render_decision_banner(
            is_eligible=ml["is_eligible"],
            approval_prob=ml["approval_probability"],
            risk_level=ml["risk_level"],
            badge_color=ml["badge_color"]
        )

        # 4 Core KPI Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            render_metric_card(
                label="Monthly EMI",
                value=f"₹{rep['monthly_emi']:,.0f}",
                subtext=f"{rep['tenure_months']} months @ {rep['annual_interest_rate']}%"
            )
        with col_m2:
            dti_val = ml["engineered_features"]["dti_ratio"]
            render_metric_card(
                label="Debt-to-Income (FOIR)",
                value=f"{dti_val:.1f}%",
                subtext="Banking ceiling: 50%"
            )
        with col_m3:
            render_metric_card(
                label="Net Monthly Surplus",
                value=f"₹{disp['net_surplus']:,.0f}",
                subtext="After debt & living costs"
            )
        with col_m4:
            render_metric_card(
                label="Max Prudential Loan",
                value=f"₹{max_l['max_loan_amount']:,.0f}",
                subtext="Based on 50% FOIR cap"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Two-Column Visual Analytics Section
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(create_dti_gauge_chart(dti_val), use_container_width=True)
            st.plotly_chart(create_cash_flow_bar_chart(disp), use_container_width=True)

        with col_g2:
            st.plotly_chart(create_loan_breakdown_pie(applicant["requested_amount"], rep["total_interest"]), use_container_width=True)
            
            # Key Ratio & Repayment Summary Card
            st.markdown(
                f"""
                <div class="lw-card" style="background: #F8FAFC;">
                    <h4 style="margin-top: 0; color: #0F172A;">Repayment & Liability Summary</h4>
                    <ul style="color: #334155; line-height: 1.8; font-size: 0.92rem; padding-left: 20px;">
                        <li><b>Principal Borrowed:</b> ₹{rep['principal']:,.2f}</li>
                        <li><b>Total Cumulative Interest:</b> ₹{rep['total_interest']:,.2f}</li>
                        <li><b>Overall Repayment Outlay:</b> ₹{rep['total_repayment']:,.2f}</li>
                        <li><b>Interest-to-Principal Cost:</b> {rep['interest_ratio_pct']:.1f}%</li>
                        <li><b>Existing Debt Burden:</b> ₹{applicant['existing_emi']:,.2f}/month</li>
                        <li><b>Loan-to-Income Ratio:</b> {ml['engineered_features']['loan_to_income_ratio']:.2f}x annual income</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        # AI-Generated Insights Box
        st.markdown("### 🤖 AI Financial Advisor Dossier")
        source_label = ai.get("source", "LoanWise Advisor")
        is_live = ai.get("is_ai_live", False)
        badge_style = "background: #DCFCE7; color: #15803D;" if is_live else "background: #F1F5F9; color: #475569;"
        
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span style="font-size: 0.8rem; font-weight: 700; padding: 4px 10px; border-radius: 6px; {badge_style}">
                    Source: {source_label}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="lw-ai-box">
                {ai['full_markdown_text']}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Download / Export Summary Report
        st.markdown("#### 📥 Export Assessment Dossier")
        summary_text = (
            f"LOANWISE AI - CREDIT & RISK ASSESSMENT REPORT\n"
            f"Applicant: {applicant['name']}\n"
            f"Decision: {ml['decision']} (Approval Probability: {ml['approval_probability']}%)\n"
            f"Risk Tier: {ml['risk_level']}\n"
            f"Requested Loan: INR {applicant['requested_amount']:,.2f} ({applicant['loan_purpose']})\n"
            f"Monthly EMI: INR {rep['monthly_emi']:,.2f}\n"
            f"Total Repayment: INR {rep['total_repayment']:,.2f}\n"
            f"DTI Ratio: {dti_val:.1f}%\n"
            f"CIBIL Score: {applicant['credit_score']}\n\n"
            f"--- AI ADVISORY ---\n"
            f"{ai['full_markdown_text']}\n"
        )
        st.download_button(
            label="📄 Download Assessment Summary (.txt)",
            data=summary_text,
            file_name=f"LoanWise_Assessment_{applicant['name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )


# =============================================================================
# TAB 3: MACHINE LEARNING MODEL TRANSPARENCY & CIA-2 AUDIT
# =============================================================================
with tab3:
    render_disclaimer()

    st.markdown("### Machine Learning Model Architecture & Transparency")
    st.caption("Detailed statistical evaluation and feature explainability for CIA-2 academic evaluation.")

    # Load Saved Model Metrics
    metrics_data = None
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    if metrics_data:
        # Display 4 Evaluation KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric_card("Accuracy Score", f"{metrics_data['accuracy'] * 100:.2f}%", "On 500 test samples")
        with k2:
            render_metric_card("Precision Score", f"{metrics_data['precision'] * 100:.2f}%", "Minimizes bad loans")
        with k3:
            render_metric_card("Recall Score", f"{metrics_data['recall'] * 100:.2f}%", "Catches eligible borrowers")
        with k4:
            render_metric_card("ROC-AUC Score", f"{metrics_data['roc_auc'] * 100:.2f}%", "Class separability")

        st.markdown("<br>", unsafe_allow_html=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            # Confusion Matrix Heatmap
            cm_matrix = metrics_data["confusion_matrix"]["raw_matrix"]
            st.plotly_chart(create_confusion_matrix_heatmap(cm_matrix), use_container_width=True)
            st.caption(
                f"**Confusion Matrix Breakdown:** True Negatives (TN) = {cm_matrix[0][0]}, "
                f"False Positives (FP) = {cm_matrix[0][1]}, False Negatives (FN) = {cm_matrix[1][0]}, "
                f"True Positives (TP) = {cm_matrix[1][1]}."
            )

        with col_t2:
            # Feature Importance Bar Chart
            feat_imp = metrics_data.get("feature_importances", {})
            st.plotly_chart(create_feature_importance_chart(feat_imp), use_container_width=True)
            st.caption(
                "**Feature Influence:** Debt-to-Income (FOIR) and Credit Score dominate 63%+ of the model's "
                "underwriting decisions, mirroring actual institutional retail banking risk policies."
            )

    st.divider()

    # Dataset Exploration Section
    st.markdown("### Sample Training Dataset Explorer")
    if DATASET_PATH.exists():
        df_sample = pd.read_csv(DATASET_PATH)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Total Records", f"{len(df_sample):,}")
        with col_d2:
            st.metric("Baseline Approval Rate", f"{df_sample['loan_approved'].mean() * 100:.1f}%")
        with col_d3:
            st.metric("Mean Credit Score", f"{df_sample['credit_score'].mean():.0f}")

        # Interactive Filter
        filter_purpose = st.multiselect(
            "Filter by Loan Purpose:",
            options=df_sample["loan_purpose"].unique(),
            default=df_sample["loan_purpose"].unique()
        )
        filtered_df = df_sample[df_sample["loan_purpose"].isin(filter_purpose)]

        st.dataframe(
            filtered_df[[
                "applicant_id",
                "employment_type",
                "monthly_income",
                "credit_score",
                "existing_emi",
                "loan_purpose",
                "requested_amount",
                "proposed_emi",
                "dti_ratio",
                "loan_approved"
            ]].head(25),
            use_container_width=True
        )
    else:
        st.warning("Sample dataset not found. Run `data/generate_data.py` to create the dataset.")

    st.divider()

    # Academic Project Defense Notes
    st.markdown("### 🎓 Academic Project Viva Highlights")
    st.markdown(
        """
        - **Pipeline Integrity:** The system uses a Scikit-Learn `Pipeline` with `ColumnTransformer`, standardizing numerical features with `StandardScaler` and encoding categoricals via `OneHotEncoder(handle_unknown='ignore')`.
        - **Zero Data Leakage:** Train/Test split (80/20) was strictly applied *before* scaling and encoding.
        - **Dual Decision Architecture:** Combines quantitative ML probability estimates with institutional prudential debt ratio guardrails.
        - **Explainable AI:** Feature importances extractable from ensemble trees satisfy modern credit compliance guidelines (Fair Lending / Responsible AI).
        """
    )

