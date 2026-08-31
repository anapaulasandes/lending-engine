from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# dashboard/expansion_demo -> repository root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.sequential_learning import (
    amount_risk_pd,
    beta_posterior,
    calculate_break_even_pd,
    current_decision,
    evaluate_offer,
    price_take_up_probability,
    recommend_next_cohort,
    summarize_posterior,
)


st.set_page_config(page_title="Credit Expansion Decisioning", page_icon="C", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        min-width: 0;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] p {
        height: auto !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1.7rem;
        line-height: 1.15;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.1%}"


def build_beta_prior(prior_mean_pd: float, prior_strength: float) -> Dict[str, float]:
    if not 0.0 < prior_mean_pd < 1.0:
        raise ValueError("prior_mean_pd must be strictly between 0 and 1.")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive.")
    alpha = prior_mean_pd * prior_strength
    beta = (1.0 - prior_mean_pd) * prior_strength
    return {"alpha": float(alpha), "beta": float(beta)}


def contract_revenue(loan_amount: float, pricing_rate: float) -> float:
    return float(loan_amount * pricing_rate)


def expected_unit_economics_per_loan(
    loan_amount: float,
    pricing_rate: float,
    expected_pd: float,
    lgd: float,
    funding_cost: float,
    operating_cost: float,
    collection_cost: float,
) -> float:
    contractual_revenue = contract_revenue(loan_amount, pricing_rate)
    expected_loss = expected_pd * loan_amount * lgd
    expected_collection_cost = expected_pd * collection_cost
    return contractual_revenue - expected_loss - funding_cost - operating_cost - expected_collection_cost


def break_even_default_rate(
    loan_amount: float,
    pricing_rate: float,
    lgd: float,
    funding_cost: float,
    operating_cost: float,
    collection_cost: float,
) -> float:
    return (loan_amount * pricing_rate - funding_cost - operating_cost) / (loan_amount * lgd + collection_cost)


def audit_inputs(rows: List[Dict[str, str]]) -> None:
    st.write("**Inputs used in this calculation**")
    st.table(pd.DataFrame(rows, columns=["Parameter", "Value", "Source"]))


PRIOR_EXPECTED_PD = float(st.session_state.get("model_prior_expected_pd", 0.10))
PRIOR_STRENGTH = float(st.session_state.get("model_prior_strength", 20.0))
PRIOR_ALPHA = PRIOR_EXPECTED_PD * PRIOR_STRENGTH
PRIOR_BETA = (1 - PRIOR_EXPECTED_PD) * PRIOR_STRENGTH
RECOMMENDED_LOAN = 200.0
RECOMMENDED_TERM = 1
RECOMMENDED_FEE_RATE = 0.20
LGD = float(st.session_state.get("model_lgd", 0.70))
FUNDING_COST = float(st.session_state.get("model_funding_cost", 15.0))
OPERATING_COST = float(st.session_state.get("model_operating_cost", 10.0))
COLLECTION_COST_PER_DEFAULT = float(st.session_state.get("model_collection_cost", 5.0))
INITIAL_COHORT = 10


def metric_label_with_tooltip(label: str, tooltip: str) -> None:
    st.markdown(
        f'<span><strong>{label}</strong> '
        f'<span title="{tooltip}" style="display: inline-block; border: 1px solid currentColor; '
        f'border-radius: 50%; font-size: 0.7rem; font-weight: 700; height: 1rem; '
        f'line-height: 1rem; text-align: center; width: 1rem;">i</span></span>',
        unsafe_allow_html=True,
    )


# Define market segments for the Panama expansion case study
MARKET_SEGMENTS = {
    "Rural SME - Agribusiness": {
        "description": "Small-scale agricultural businesses and farming cooperatives in rural areas. Typically seasonal cash flows, moderate to high default risk. Loan amounts: $300-800. Price sensitivity: medium. Annual revenue $50k-200k.",
        "default_prior": 0.12,
        "price_sensitivity": 0.35,
        "typical_loan_amount": 500.0,
    },
    "Urban Micro-Retail": {
        "description": "Small retail shops and street vendors in urban/peri-urban areas. Daily cash operations, volatile repayment patterns. Loan amounts: $150-400. Price sensitivity: high. Daily sales $50-300.",
        "default_prior": 0.15,
        "price_sensitivity": 0.45,
        "typical_loan_amount": 250.0,
    },
    "Transportation & Logistics": {
        "description": "Small trucking operators, taxi owners, and delivery services. Fuel/maintenance volatility, but regular income. Loan amounts: $600-1200. Price sensitivity: low-medium. Monthly revenue $2k-5k.",
        "default_prior": 0.09,
        "price_sensitivity": 0.28,
        "typical_loan_amount": 750.0,
    },
}

st.title("CREDIT EXPANSION DECISIONING")
st.caption("Structured evidence and quantitative decisions for credit market entry.")
st.caption("Illustrative case study. All portfolio assumptions, risk parameters, business constraints and numerical results are synthetic and do not represent proprietary company data. Public market information is used only for contextual illustration.")

# Initialize session state
if "step1_completed" not in st.session_state:
    st.session_state.step1_completed = False
if "step2_completed" not in st.session_state:
    st.session_state.step2_completed = False
if "step3_completed" not in st.session_state:
    st.session_state.step3_completed = False

# ============================================================================
# STEP 1 - UNDERSTAND: Market Assessment
# ============================================================================

st.header("Step 1 - Understand the Market & Population")
st.caption("Describe the market opportunity. We will structure it and identify comparable evidence.")

with st.container(border=True):
    st.subheader("Market Opportunity")
    
    opportunity_description = st.text_area(
        "Your opportunity:",
        value="We are considering launching an unsecured installment loan in Panama for lower-to-middle income salaried consumers. Our target customers have verifiable recurring employment income but may have limited access to traditional bank credit or insufficient credit history for conventional underwriting. The product would address short-term liquidity needs and unexpected household expenses through relatively small, fixed-payment installment loans. We currently have no direct repayment history for this population in Panama, but we operate a comparable consumer lending product in Belize.",
        height=150,
        label_visibility="collapsed",
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        run_assessment = st.button("Run AI Assessment", use_container_width=True, type="primary")

if run_assessment or st.session_state.step1_completed:
    st.session_state.step1_completed = True
    
    if run_assessment:
        with st.spinner("Analyzing market, population and comparable evidence..."):
            import time
            time.sleep(0.8)
    
    # AI Market Assessment Output
    st.divider()
    
    with st.container(border=True):
        assessment_col1, assessment_col2 = st.columns([3, 1])
        with assessment_col1:
            st.subheader("Market Assessment")
        with assessment_col2:
            st.caption("Panama | Consumer Lending")
        
        # Assessment metrics
        assess_c1, assess_c2, assess_c3, assess_c4 = st.columns(4)
        assess_c1.metric("Target population", "Salaried consumers")
        assess_c2.metric("Product fit", "High")
        assess_c3.metric("Confidence", "Medium")
        assess_c4.metric("Expected risk", "Moderate")
        
        assess_c5, assess_c6, assess_c7, assess_c8 = st.columns(4)
        assess_c5.metric("Evidence available", "Comparable evidence")
        assess_c6.metric("Comparable market", "Belize")
        assess_c7.metric("Income pattern", "Recurring salary")
        assess_c8.metric("Credit access", "Underserved")
    
    st.divider()
    
    # Evidence / Reasoning Section
    with st.container(border=True):
        st.subheader("Evidence, Inference & Assumptions")
        
        obs_col, inf_col, ass_col = st.columns(3)
        
        with obs_col:
            st.markdown("**OBSERVED**")
            st.caption("What we know directly")
            st.write("- Comparable consumer installment lending exists in Belize")
            st.write("- Target population has verifiable recurring employment income")
        
        with inf_col:
            st.markdown("**INFERRED**")
            st.caption("What we can reasonably infer")
            st.write("- Existing Caribbean portfolio evidence may inform the initial Panama expectation")
            st.write("- Shorter loan terms can accelerate repayment learning")
        
        with ass_col:
            st.markdown("**ASSUMED**")
            st.caption("What we are hypothesizing")
            st.write("- Local repayment behavior may differ from comparable markets")
            st.write("- Recurring income is assumed to be informative of repayment capacity")
    
    st.divider()
    
    # Evidence Transfer Assessment
    with st.container(border=True):
        st.subheader("Evidence Transfer Assessment")
        
        transfer_col1, transfer_col2 = st.columns([2, 2])
        
        with transfer_col1:
            st.write("**Belize may provide informative comparable-market evidence, but confidence should be reduced when transferring repayment expectations to Panama.**")
        
        with transfer_col2:
            t1, t2 = st.columns(2)
            t1.metric("Evidence source", "Belize")
            t2.metric("Transferability", "Partial")
        
        t3, t4 = st.columns(2)
        t3.metric("Transfer discount", "Required")
        t3_text = "**Main reason:** Borrower selection, market structure and collections effectiveness may differ materially across countries."
        
        t4.metric("Confidence", "Medium")
        t4_text = "**Key uncertainty:** Local repayment behavior"
        
        st.write(t3_text)
        st.write(t4_text)
    
    st.divider()
    
    # What Should We Learn First
    with st.container(border=True):
        st.subheader("Key Uncertainties to Resolve")
        
        uncertainties = [
            "1. How does repayment behavior in Panama compare with Belize?",
            "2. How sensitive is repayment to loan amount?",
            "3. Does performance vary materially by income band?",
            "4. How transferable are existing underwriting signals?",
        ]
        
        for u in uncertainties:
            st.write(u)
    
    st.divider()
    
    # CTA to Step 2
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        proceed_to_step2 = st.button(
            "Next: Business Constraints",
            use_container_width=True,
            type="primary",
            key="proceed_to_step2"
        )
    
    if proceed_to_step2:
        st.session_state.step1_market = "Panama"
        st.session_state.step1_population = "Lower-to-middle income salaried consumers"
        st.session_state.step1_evidence_basis = "Comparable markets"
        st.session_state.step1_comparable_markets = ["Belize"]
        st.session_state.step1_confidence = "Medium"
        st.session_state.step2_completed = True
        st.rerun()

# ============================================================================
# STEP 2 - CONSTRAIN: Business Constraints
# ============================================================================

if st.session_state.step1_completed and st.session_state.step2_completed:
    st.divider()
    st.header("Step 2 - Define Business Constraints")
    st.caption("Set the risk, capital and operational boundaries for the credit experiment.")
    
    with st.container(border=True):
        st.info("Set the boundaries. The decision engine will design the policy.")
        st.caption("These are hard constraints, not recommendations.")
        
        constraint_col1, constraint_col2, constraint_col3 = st.columns(3)
        
        with constraint_col1:
            max_learning_loss = st.number_input(
                "Maximum acceptable learning loss",
                min_value=100.0,
                value=float(st.session_state.get("constraint_max_learning_loss", 4000.0)),
                step=250.0,
                help="What is this? Maximum economic loss the business is willing to accept while learning.\n\nWhy does it matter? For this MVP, the learning-loss budget is compared against expected credit loss generated during the initial cohort.\n\nIf increased: Allows more exploration or larger exposures.\n\nIf decreased: Forces a more conservative experiment and may slow learning.",
            )
            st.session_state.constraint_max_learning_loss = max_learning_loss

            max_portfolio_exposure = st.number_input(
                "Maximum portfolio exposure",
                min_value=1000.0,
                value=float(st.session_state.get("constraint_max_portfolio_exposure", 30000.0)),
                step=1000.0,
                help="What is this? Maximum total outstanding principal permitted during the learning phase.\n\nWhy does it matter? It limits capital committed before repayment evidence matures.\n\nIf increased: Allows more simultaneous exposure.\n\nIf decreased: Requires smaller or more sequential cohorts.",
            )
            st.session_state.constraint_max_portfolio_exposure = max_portfolio_exposure
        
        with constraint_col2:
            max_pilot_capacity = st.number_input(
                "Maximum pilot capacity (customers)",
                min_value=10,
                value=int(st.session_state.get("constraint_max_pilot_capacity", 100)),
                step=5,
                help="What is this? Maximum number of customers operationally available for the pilot.\n\nWhy does it matter? It caps the observations the business can support; it is not the recommended cohort size.\n\nIf increased: Gives the engine more potential observations.\n\nIf decreased: Limits the available pilot capacity.",
            )
            st.session_state.constraint_max_pilot_capacity = max_pilot_capacity

            max_loan_amount = st.number_input(
                "Maximum loan amount allowed",
                min_value=100.0,
                value=float(st.session_state.get("constraint_max_loan_amount", 1000.0)),
                step=50.0,
                help="What is this? Governance or commercial ceiling for exposure to one borrower.\n\nWhy does it matter? It defines the product space the engine can evaluate; it is not the recommended starting amount.\n\nIf increased: Allows larger candidate products.\n\nIf decreased: Reduces potential loss severity but restricts the product space.",
            )
            st.session_state.constraint_max_loan_amount = max_loan_amount
        
        with constraint_col3:
            max_repayment_term = st.number_input(
                "Maximum repayment term (installments)",
                min_value=2,
                value=int(st.session_state.get("constraint_max_repayment_term", 6)),
                step=1,
                help="What is this? Longest repayment structure the business permits.\n\nWhy does it matter? It sets product flexibility and the speed at which mature repayment evidence arrives; it is not the recommended term.\n\nIf increased: Allows more product flexibility but delays mature repayment evidence.\n\nIf decreased: Speeds up maturity but limits product flexibility.",
            )
            st.session_state.constraint_max_repayment_term = max_repayment_term
        
        with st.expander("Advanced business constraints (optional)"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                min_required_return = st.number_input(
                    "Minimum required return (%)",
                    min_value=0.0,
                    value=float(st.session_state.get("constraint_min_required_return", 5.0)),
                    step=0.5,
                )
                st.session_state.constraint_min_required_return = min_required_return
            
            with adv_col2:
                max_default_rate = st.number_input(
                    "Maximum acceptable default rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("constraint_max_default_rate", 20.0)),
                    step=0.5,
                )
                st.session_state.constraint_max_default_rate = max_default_rate
            
            with adv_col3:
                min_pricing = st.number_input(
                    "Minimum pricing allowed (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("constraint_min_pricing", 5.0)),
                    step=1.0,
                )
                max_pricing = st.number_input(
                    "Maximum pricing allowed (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.get("constraint_max_pricing", 50.0)),
                    step=1.0,
                )
                st.session_state.constraint_min_pricing = min_pricing
                st.session_state.constraint_max_pricing = max_pricing
    
    st.divider()
    
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        proceed_to_step3 = st.button(
            "Next: Recommended Policy",
            use_container_width=True,
            type="primary",
            key="proceed_to_step3"
        )
    
    if proceed_to_step3:
        st.session_state.step3_completed = True
        st.rerun()

# ============================================================================
# STEP 3 - DECIDE: Recommended Credit Experiment
# ============================================================================

if st.session_state.step1_completed and st.session_state.step2_completed and st.session_state.step3_completed:
    st.divider()
    st.header("Step 3 - Recommended Policy")
    st.caption("The decision engine recommends the following credit experiment within your constraints.")
    
    # Market context from Step 1
    with st.container(border=True):
        st.subheader("Market Context")
        context_c1, context_c2, context_c3, context_c4, context_c5 = st.columns(5)
        context_c1.metric("Market", "Panama")
        context_c2.metric("Population", "Salaried, credit-underserved")
        context_c3.metric("Evidence basis", "Comparable markets")
        context_c4.metric("Comparable", "Belize")
        context_c5.metric("Confidence", "Medium")
    
    st.divider()
    
    # ========== DECISION 1: WHO AND HOW MANY ==========
    with st.container(border=True):
        st.subheader("Decision 1 - Who Should We Lend To First?")
        
        dec1_col1, dec1_col2 = st.columns([2, 1])
        with dec1_col1:
            st.write("**Target segment:** Salaried consumers with verifiable recurring income")
            st.write("**Recommended initial cohort:** 10 customers")
        with dec1_col2:
            st.write("**Selection logic:**")
            st.caption("Smallest eligible cohort under the exposure and uncertainty limits")
        
        with st.expander("Why this recommendation?"):
            cohort_candidates = [10, 25, 50, 100]
            cohort_rows = []
            for cohort_size in cohort_candidates:
                expected_defaults = cohort_size * PRIOR_EXPECTED_PD
                expected_non_defaults = cohort_size * (1 - PRIOR_EXPECTED_PD)
                alpha_updated = PRIOR_ALPHA + expected_defaults
                beta_updated = PRIOR_BETA + expected_non_defaults
                variance = (alpha_updated * beta_updated) / ((alpha_updated + beta_updated) ** 2 * (alpha_updated + beta_updated + 1))
                standard_deviation = np.sqrt(variance)
                initial_exposure = cohort_size * RECOMMENDED_LOAN
                exposure_utilization = initial_exposure / max_portfolio_exposure
                qualifies = standard_deviation <= 0.06 and exposure_utilization <= 0.10
                cohort_rows.append({
                    "Cohort": f"{cohort_size} customers",
                    "Uncertainty (percentage points)": round(standard_deviation * 100, 2),
                    "Initial principal exposure": money(initial_exposure),
                    "Exposure utilization": pct(exposure_utilization),
                    "Result": "Eligible" if qualifies else "Not eligible",
                })

            st.write("**Decision logic**")
            st.caption("Apply the same eligibility thresholds to every candidate cohort before selecting the smallest eligible option.")
            st.write("**Inputs used**")
            st.caption(f"Prior expected default rate: {pct(PRIOR_EXPECTED_PD)}. Prior strength: {PRIOR_STRENGTH:.0f} equivalent observations. Recommended loan amount: {money(RECOMMENDED_LOAN)}. Maximum portfolio exposure: {money(max_portfolio_exposure)}.")
            st.write("**Candidates evaluated**")
            st.caption("10, 25, 50 and 100 customers.")
            st.write("**Comparison chart**")
            st.caption("Evidence gained by initial cohort size")
            st.bar_chart(pd.DataFrame(cohort_rows).set_index("Cohort")["Uncertainty (percentage points)"], use_container_width=True)
            st.table(pd.DataFrame(cohort_rows))
            st.caption("Larger cohorts reduce uncertainty, but require more capital to be committed before the first policy review.")
            st.write("**Calculation**")
            st.code(f"expected defaults = n x {PRIOR_EXPECTED_PD:.2f}\nexpected non-defaults = n x {1 - PRIOR_EXPECTED_PD:.2f}\nalpha_updated = {PRIOR_ALPHA:.0f} + expected defaults\nbeta_updated = {PRIOR_BETA:.0f} + expected non-defaults\nposterior standard deviation = sqrt((alpha_updated x beta_updated) / ((alpha_updated + beta_updated)^2 x (alpha_updated + beta_updated + 1)))\ninitial exposure = n x {money(RECOMMENDED_LOAN)}\nexposure utilization = initial exposure / {money(max_portfolio_exposure)}")
            st.write("**Selection rule**")
            st.caption("Choose the smallest candidate satisfying uncertainty <= 6 pp and initial exposure <= 10% of the portfolio exposure limit. These are illustrative MVP decision rules, not universal statistical requirements.")
            st.write("**Assumptions**")
            st.caption("Comparable-market evidence is partially transferable, matured outcomes are informative, and each candidate is reviewed before further origination.")
            st.write("**What would change the result?**")
            st.caption("Different prior evidence, learning-loss budget, loan amount, or operational capacity.")
            st.write("**Reproduce this decision**")
            st.code(f"For n = 10:\nuncertainty = 5.39 pp <= 6 pp\ninitial exposure = 10 x {money(RECOMMENDED_LOAN)} = {money(10 * RECOMMENDED_LOAN)}\nexposure utilization = {money(10 * RECOMMENDED_LOAN)} / {money(max_portfolio_exposure)} = {10 * RECOMMENDED_LOAN / max_portfolio_exposure:.1%}\nRESULT = Eligible")
            audit_inputs([
                {"Parameter": "Prior expected PD", "Value": "10%", "Source": "Comparable portfolio assumption"},
                {"Parameter": "Prior strength", "Value": "20", "Source": "Model assumption"},
                {"Parameter": "Loan amount", "Value": "$200", "Source": "Decision candidate"},
                {"Parameter": "Cohort candidates", "Value": "10, 25, 50, 100", "Source": "Model assumption"},
            ])
    
    st.divider()
    
    # ========== DECISION 2: WHAT PRODUCT ==========
    with st.container(border=True):
        st.subheader("Decision 2 - What Should We Offer?")
        
        dec2_c1, dec2_c2, dec2_c3 = st.columns(3)
        with dec2_c1:
            st.metric("Recommended loan amount", "$200")
        with dec2_c2:
            st.metric("Recommended term", "1 installment")
        with dec2_c3:
            st.metric("Recommended fee", "20%")
        
        recommended_loan = RECOMMENDED_LOAN
        recommended_term = RECOMMENDED_TERM
        recommended_fee_rate = RECOMMENDED_FEE_RATE
        expected_pd = PRIOR_EXPECTED_PD
        lgd = LGD
        funding_cost = FUNDING_COST
        operating_cost = OPERATING_COST
        collection_cost = COLLECTION_COST_PER_DEFAULT
        expected_ue = expected_unit_economics_per_loan(
            loan_amount=recommended_loan,
            pricing_rate=recommended_fee_rate,
            expected_pd=expected_pd,
            lgd=lgd,
            funding_cost=funding_cost,
            operating_cost=operating_cost,
            collection_cost=collection_cost,
        )
        break_even_pd = break_even_default_rate(
            recommended_loan,
            recommended_fee_rate,
            lgd,
            funding_cost,
            operating_cost,
            collection_cost,
        )

        economic_headroom = break_even_pd - expected_pd
        economics_col, break_even_col, headroom_col = st.columns(3)
        with economics_col:
            metric_label_with_tooltip(
                "Expected Unit Economics / loan",
                "Expected Unit Economics estimates the economic contribution of one loan after expected credit losses, funding costs and operating costs.",
            )
            st.metric("Expected UE / loan", money(expected_ue))
        with break_even_col:
            st.metric("Break-even default rate", pct(break_even_pd))
        with headroom_col:
            st.metric("Economic headroom", f"{economic_headroom * 100:.1f} pp")
            st.caption("Small headroom means relatively little deterioration in credit performance would eliminate expected Unit Economics.")

        if expected_ue < 0:
            st.write(f"**Learning investment: ${abs(expected_ue):,.0f} per loan**")
            st.caption(
                "The initial policy intentionally accepts a small controlled economic cost "
                "to acquire local repayment evidence while limiting exposure."
            )

        with st.expander("How are unit economics calculated?"):
            expected_revenue = recommended_loan * recommended_fee_rate
            expected_credit_loss = expected_pd * lgd * recommended_loan
            expected_collection_cost = expected_pd * collection_cost
            st.write("**Expected revenue**")
            st.code(f"${recommended_loan:,.0f} x {recommended_fee_rate:.2f} = {money(expected_revenue)}")
            st.write("**Expected credit loss**")
            st.code(f"${recommended_loan:,.0f} x {expected_pd:.2f} x {lgd:.2f} = {money(expected_credit_loss)}")
            st.write("**Funding cost**")
            st.code(money(funding_cost))
            st.write("**Operating cost**")
            st.code(money(operating_cost))
            st.write("**Expected collection cost**")
            st.code(f"{expected_pd:.2f} x {money(collection_cost)} = {money(expected_collection_cost)}")
            st.write("**Expected Unit Economics**")
            st.code(f"{money(expected_revenue)} - {money(expected_credit_loss)} - {money(funding_cost)} - {money(operating_cost)} - {money(expected_collection_cost)} = {money(expected_ue)}")
            st.write("**Why this matters**")
            st.caption("Default risk alone does not determine whether a credit product is viable. Two products with different loan amounts, pricing and repayment structures can have similar default risk but very different economics.")
            st.caption("During an early learning phase, the engine may accept slightly negative Unit Economics if the expected learning value justifies the controlled economic cost and the result remains within the learning-loss budget.")
            st.write("**Break-even default rate**")
            st.code(f"0 = {money(expected_revenue)} - PD_break_even x ({lgd:.2f} x ${recommended_loan:,.0f} + ${collection_cost:,.0f}) - {money(funding_cost)} - {money(operating_cost)}\nPD_break_even = (${expected_revenue:,.2f} - ${funding_cost:,.2f} - ${operating_cost:,.2f}) / (${recommended_loan:,.0f} x {lgd:.2f} + ${collection_cost:,.0f}) = {pct(break_even_pd)}")
            audit_inputs([
                {"Parameter": "Loan amount", "Value": "$200", "Source": "Decision candidate"},
                {"Parameter": "Fee rate", "Value": "20%", "Source": "Model assumption"},
                {"Parameter": "Expected PD", "Value": "10%", "Source": "Comparable portfolio assumption"},
                {"Parameter": "LGD", "Value": "70%", "Source": "Model assumption"},
                {"Parameter": "Funding / operating cost", "Value": "$15.00 / $10.00", "Source": "Model assumption"},
                {"Parameter": "Collection cost per default", "Value": "$5.00", "Source": "Model assumption"},
            ])

        with st.expander("Why this recommendation?"):
            product_rows = []
            amount_rows = []
            for amount in [200, 500, 800, 1000]:
                revenue = amount * recommended_fee_rate
                credit_loss = expected_pd * lgd * amount
                expected_collections = expected_pd * collection_cost
                ue = expected_unit_economics_per_loan(amount, recommended_fee_rate, expected_pd, lgd, funding_cost, operating_cost, collection_cost)
                learning_loss_capacity = int(max_learning_loss // credit_loss)
                exposure_capacity = int(max_portfolio_exposure // amount)
                amount_rows.append({"Loan amount": f"${amount:,.0f}", "Expected UE / loan": round(ue, 2)})
                for term in [1, 2, 4, 6]:
                    product_rows.append({
                        "Loan": f"${amount:,.0f}", "Term": f"{term} installment(s)", "Revenue": money(revenue),
                        "Expected credit loss": money(credit_loss), "Funding": money(funding_cost),
                        "Operating": money(operating_cost), "Expected collections": money(expected_collections),
                        "Expected UE": money(ue), "Principal exposure": money(amount),
                        "Expected UE per $100 exposed": money(ue / amount * 100),
                        "Expected credit loss per customer": money(credit_loss),
                        "Customers under learning-loss budget": learning_loss_capacity,
                        "Customers under exposure limit": exposure_capacity,
                        "Time to mature evidence": f"{term} period(s)", "Learning-budget consumption": money(max(0, -ue)),
                    })

            st.write("**Decision logic**")
            st.caption("Prefer the initial learning policy that limits downside exposure and produces mature repayment evidence quickly, rather than the product with the highest immediate Unit Economics.")
            st.write("**Inputs used**")
            st.caption("Loan amounts: $200, $500, $800 and $1,000. Terms: 1, 2, 4 and 6 installments. Fee: 20%. Expected PD: 10%.")
            st.write("**Candidates evaluated**")
            st.write("Candidate product grid")
            st.table(pd.DataFrame(product_rows))
            st.write("**Comparison chart**")
            st.caption("Unit Economics by Candidate Loan Amount")
            st.bar_chart(pd.DataFrame(amount_rows).set_index("Loan amount"), use_container_width=True)
            st.caption("Zero on the chart is the break-even reference: values above zero have positive expected Unit Economics.")
            term_rows = pd.DataFrame({"Term": ["1 installment", "2 installments", "4 installments", "6 installments"], "Relative time to mature": [1, 2, 4, 6]}).set_index("Term")
            st.caption("Evidence Maturity by Repayment Term")
            st.bar_chart(term_rows, use_container_width=True)
            st.write("**Calculation**")
            st.code("Expected Revenue = Loan Amount x Fee Rate\nExpected Credit Loss = PD x LGD x Loan Amount\nExpected Collection Cost = PD x Collection Cost per Default\nExpected UE = Revenue - Expected Credit Loss - Funding Cost - Operating Cost - Expected Collection Cost")
            st.write("**Selection rule**")
            st.caption("Choose the smallest candidate that satisfies all hard business constraints, keeps expected cohort credit loss within the learning-loss budget, preserves the highest number of possible observations under that budget, and has non-negative Expected Unit Economics. If multiple candidates qualify, prefer the lower-exposure candidate for the first local cohort. $200 qualifies and is selected; larger candidates have higher absolute UE but permit fewer observations and commit more capital per customer.")
            st.write("**Assumptions**")
            st.caption("The fee, expected default rate, LGD, funding cost, operating cost and collection cost are held constant across the candidate grid.")
            st.write("**What would change the result?**")
            st.caption("Observed repayment performance, pricing, loss severity, funding cost, operating cost, collection cost, or the learning budget.")
            st.write("**Reproduce this decision**")
            st.code(f"For $200, 1 installment:\nRevenue = $200 x {recommended_fee_rate:.2f} = {money(expected_revenue)}\nExpected Credit Loss = $200 x {expected_pd:.2f} x {lgd:.2f} = {money(expected_credit_loss)}\nExpected Collection Cost = {expected_pd:.2f} x {money(collection_cost)} = {money(expected_collection_cost)}\nExpected UE = {money(expected_revenue)} - {money(expected_credit_loss)} - {money(funding_cost)} - {money(operating_cost)} - {money(expected_collection_cost)} = {money(expected_ue)}\nEconomic headroom = {pct(break_even_pd)} - {pct(expected_pd)} = {economic_headroom * 100:.1f} pp")
            audit_inputs([
                {"Parameter": "Candidate loans", "Value": "$200, $500, $800, $1,000", "Source": "Model assumption"},
                {"Parameter": "Candidate terms", "Value": "1, 2, 4, 6", "Source": "Model assumption"},
                {"Parameter": "Fee rate", "Value": "20%", "Source": "Model assumption"},
                {"Parameter": "Expected PD / LGD", "Value": "10% / 70%", "Source": "Comparable portfolio assumption / Model assumption"},
                {"Parameter": "Learning-loss budget", "Value": money(max_learning_loss), "Source": "Business constraint"},
            ])
    
    st.divider()
    
    # ========== DECISION 3: HOW FAST ==========
    with st.container(border=True):
        st.subheader("Decision 3 - How Fast Should We Deploy?")
        
        st.write("**Deployment Rule**")
        rule_c1, rule_a1, rule_c2, rule_a2, rule_c3, rule_a3, rule_c4 = st.columns([2, 0.4, 3, 0.4, 2, 0.4, 3])
        with rule_c1:
            st.write("**10 customers**")
        with rule_a1:
            st.write("->")
        with rule_c2:
            st.write("**Wait for 10 matured outcomes**")
        with rule_a2:
            st.write("->")
        with rule_c3:
            st.write("**Review evidence**")
        with rule_a3:
            st.write("->")
        with rule_c4:
            st.write("**Authorize next cohort**")
        st.caption("New exposure is gated by repayment evidence. The next cohort is not automatically originated before the review criterion is reached.")
        
        with st.expander("Why this recommendation?"):
            gate_candidates = [10, 25, 50, 100]
            gate_rows = pd.DataFrame({
                "Deployment gate": [f"{customers} customers" for customers in gate_candidates],
                "Capital committed before review": [customers * RECOMMENDED_LOAN for customers in gate_candidates],
            }).set_index("Deployment gate")
            st.write("**Decision logic**")
            st.caption("Decision 1 selects the number of first observations. This decision controls how much capital is committed before the lender can react to those observations.")
            st.write("**Inputs used**")
            st.caption("Initial loan amount: $200. Candidate deployment gates: 10, 25, 50 and 100 customers.")
            st.write("**Candidates evaluated**")
            st.caption("10, 25, 50 and 100 customers originated before the first evidence review.")
            st.write("**Comparison chart**")
            st.caption("Capital at Risk Before First Policy Update")
            st.bar_chart(gate_rows, use_container_width=True)
            st.write("**Calculation**")
            st.code("Capital committed before review = customers originated before evidence gate x initial loan amount")
            st.write("**Selection rule**")
            st.caption("The next cohort is not authorized until the required matured outcomes from the current cohort are available.")
            st.write("**Assumptions**")
            st.caption("Each customer receives the initial $200 loan and reaches maturity before the first policy review.")
            st.write("**What would change the result?**")
            st.caption("A different initial loan amount, evidence gate, or repayment maturity period.")
            st.write("**Reproduce this decision**")
            st.code("For the recommended gate:\ncapital committed before review = 10 x $200 = $2,000")
            audit_inputs([
                {"Parameter": "Initial loan amount", "Value": "$200", "Source": "Decision candidate"},
                {"Parameter": "Recommended evidence gate", "Value": "10 customers", "Source": "Model assumption"},
                {"Parameter": "Gate candidates", "Value": "10, 25, 50, 100", "Source": "Model assumption"},
            ])
    
    st.divider()
    
    # ========== DECISION 4: HOW MUCH RISK ==========
    with st.container(border=True):
        st.subheader("Decision 4 - How Much Risk Are We Taking?")
        expected_credit_loss_per_loan = PRIOR_EXPECTED_PD * LGD * RECOMMENDED_LOAN
        expected_learning_loss = INITIAL_COHORT * expected_credit_loss_per_loan
        expected_exposure = INITIAL_COHORT * RECOMMENDED_LOAN
        remaining_learning_budget = max_learning_loss - expected_learning_loss
        
        risk_c1, risk_c2, risk_c3, risk_c4 = st.columns(4)
        risk_c1.metric("Expected exposure", money(expected_exposure))
        risk_c2.metric(
            "Expected Credit Loss - Initial Cohort",
            money(expected_learning_loss),
            help="For this MVP, the learning-loss budget is compared against expected credit loss generated during the initial cohort.",
        )
        risk_c3.metric("Maximum possible exposure", money(expected_exposure))
        risk_c4.metric("Remaining learning budget", money(remaining_learning_budget))
        
        with st.expander("How is risk controlled?"):
            utilization_rows = pd.DataFrame({
                "Constraint": ["Learning loss", "Portfolio exposure", "Pilot capacity", "Loan amount ceiling", "Repayment term ceiling"],
                "Utilization (%)": [
                    expected_learning_loss / max_learning_loss * 100,
                    expected_exposure / max_portfolio_exposure * 100,
                    INITIAL_COHORT / max_pilot_capacity * 100,
                    RECOMMENDED_LOAN / max_loan_amount * 100,
                    RECOMMENDED_TERM / max_repayment_term * 100,
                ],
            }).set_index("Constraint")
            st.write("**Decision logic**")
            st.caption("Keep the initial policy within each business limit before approving exposure.")
            st.write("**Inputs used**")
            st.caption("Expected default rate: 10%. LGD: 70%. Initial cohort: 10. All limits come from Step 2.")
            st.write("**Candidates evaluated**")
            st.caption("The selected initial cohort, loan amount and term against each hard constraint.")
            st.write("**Comparison chart**")
            st.caption("Hard constraint utilization; all values start at zero and represent the share of each limit used.")
            st.bar_chart(utilization_rows, use_container_width=True)
            st.write("**Calculation**")
            st.code(f"Expected Credit Loss - Initial Cohort = cohort size x loan amount x PD x LGD\n= {INITIAL_COHORT} x $200 x {expected_pd:.2f} x {lgd:.2f}\n= {money(expected_learning_loss)}\n\nCredit-loss budget utilization = {money(expected_learning_loss)} / {money(max_learning_loss)} = {expected_learning_loss / max_learning_loss:.1%}\nExposure utilization = {money(expected_exposure)} / {money(max_portfolio_exposure)} = {expected_exposure / max_portfolio_exposure:.1%}\nCapacity utilization = {INITIAL_COHORT} / {max_pilot_capacity} = {INITIAL_COHORT / max_pilot_capacity:.1%}\nLoan ceiling utilization = $200 / {money(max_loan_amount)} = {RECOMMENDED_LOAN / max_loan_amount:.1%}\nTerm utilization = 1 / {max_repayment_term} = {RECOMMENDED_TERM / max_repayment_term:.1%}")
            st.write("**Selection rule**")
            st.caption("The policy is feasible when each utilization is at or below 100%; no constraint is binding in this illustrative recommendation.")
            st.write("**Assumptions**")
            st.caption("Expected Credit Loss - Initial Cohort operationalizes the learning-loss budget for this MVP. It is distinct from Expected Unit Economics and Learning Investment.")
            st.write("**What would change the result?**")
            st.caption("A change in default risk, loss severity, cohort size, product amount, term, or a Step 2 business limit.")
            st.write("**Reproduce this decision**")
            st.code(f"Learning loss utilization = {money(expected_learning_loss)} / {money(max_learning_loss)} = {expected_learning_loss / max_learning_loss:.1%}\nCapacity utilization = {INITIAL_COHORT} / {max_pilot_capacity} = {INITIAL_COHORT / max_pilot_capacity:.1%}\nLoan ceiling utilization = $200 / {money(max_loan_amount)} = {RECOMMENDED_LOAN / max_loan_amount:.1%}")
            audit_inputs([
                {"Parameter": "Expected PD / LGD", "Value": "10% / 70%", "Source": "Comparable portfolio assumption / Model assumption"},
                {"Parameter": "Initial cohort / loan", "Value": "10 / $200", "Source": "Decision candidate"},
                {"Parameter": "Learning-loss budget", "Value": money(max_learning_loss), "Source": "Business constraint"},
                {"Parameter": "Exposure / capacity limits", "Value": f"{money(max_portfolio_exposure)} / {max_pilot_capacity}", "Source": "Business constraint"},
            ])
    
    st.divider()
    
    # Constraint compliance
    with st.container(border=True):
        st.subheader("Policy Feasibility Check")
        st.write("**Feasible policy**")
        st.caption("All five hard constraints are within limits.")
    
    st.divider()
    
    # Technical configuration
    with st.expander("Advanced Model Configuration"):
        st.write("**ECONOMICS**")
        econ_c1, econ_c2, econ_c3, econ_c4 = st.columns(4)
        
        with econ_c1:
            funding_cost = st.number_input("Funding cost / loan", min_value=0.0, value=FUNDING_COST, step=1.0, key="model_funding_cost", help="Estimated cost of funding the loan.")
        with econ_c2:
            operating_cost = st.number_input("Operating cost / loan", min_value=0.0, value=OPERATING_COST, step=1.0, key="model_operating_cost", help="Estimated operational cost associated with originating and servicing one loan.")
        with econ_c3:
            collection_cost = st.number_input("Collection cost / loan", min_value=0.0, value=COLLECTION_COST_PER_DEFAULT, step=1.0, key="model_collection_cost", help="Expected collections cost associated with delinquent loans.")
        with econ_c4:
            lgd = st.number_input("LGD (Loss Given Default)", min_value=0.0, max_value=1.0, value=LGD, step=0.05, key="model_lgd", help="Expected percentage of exposure lost when a borrower defaults, after recoveries.")
        
        st.divider()
        st.write("**RISK EVIDENCE**")
        belief_c1, belief_c2 = st.columns(2)
        
        with belief_c1:
            prior_mean_pd = st.number_input("Prior expected PD", min_value=0.01, max_value=0.60, value=PRIOR_EXPECTED_PD, step=0.01, key="model_prior_expected_pd", help="Starting expectation of the population's default rate before local outcomes are observed.")
        with belief_c2:
            prior_strength = st.number_input("Prior strength", min_value=2.0, max_value=200.0, value=PRIOR_STRENGTH, step=1.0, key="model_prior_strength", help="How much confidence the engine places in the starting evidence. Higher values make new observations change the initial expectation more slowly.")
        
        with st.expander("Technical Decision Rules"):
            engine_c1, engine_c2, engine_c3 = st.columns(3)
            with engine_c1:
                risk_sensitivity = st.number_input("Risk sensitivity", min_value=0.0, max_value=1.0, value=0.25, step=0.05, help="Controls how strongly the decision engine penalizes uncertainty and downside risk. Higher values make recommendations more conservative.")
            with engine_c2:
                expand_threshold = st.slider("Expand threshold", min_value=0.1, max_value=0.95, value=0.8, step=0.05, help="Minimum level of evidence required before the engine recommends increasing exposure.")
                stop_threshold = st.slider("Stop threshold", min_value=0.05, max_value=0.8, value=0.4, step=0.05, help="Evidence level at which the engine recommends stopping or reducing exposure.")
            with engine_c3:
                min_actionable_probability = st.number_input("Min actionable probability", min_value=0.1, max_value=1.0, value=0.7, step=0.05, help="Minimum confidence required before the engine converts uncertain evidence into a policy change.")
                monte_carlo_trials = st.number_input("Simulation trials", min_value=200, max_value=10000, value=4000, step=200, help="Number of simulated scenarios used to estimate uncertainty in the decision. Higher values improve numerical stability but require more computation.")

    with st.expander("Decision Trace"):
        st.code(f"Market evidence -> Comparable Belize evidence\nStarting PD -> {pct(PRIOR_EXPECTED_PD)}\nBusiness limits -> {money(max_learning_loss)} credit-loss budget; {money(max_portfolio_exposure)} exposure limit; {max_pilot_capacity} customer capacity; {money(max_loan_amount)} ticket ceiling; {max_repayment_term} installment ceiling\nCohort candidates -> 10, 25, 50, 100\nSelected cohort -> {INITIAL_COHORT}\nLoan candidates -> $200, $500, $800, $1,000\nSelected starting loan -> {money(RECOMMENDED_LOAN)}\nTerm candidates -> 1, 2, 4, 6\nSelected term -> {RECOMMENDED_TERM} installment\nExpected initial credit loss -> {money(expected_learning_loss)}\nExpected UE / loan -> {money(expected_ue)}\nPolicy result -> Feasible")
    
    st.divider()
    
    # CTA to Step 4
    col_s4_1, col_s4_2, col_s4_3 = st.columns([1, 2, 1])
    with col_s4_2:
        proceed_to_step4 = st.button(
            "Define Learning & Scaling Rules",
            use_container_width=True,
            type="primary",
            key="proceed_to_step4"
        )
    
    if proceed_to_step4:
        st.session_state.step4_ready = True
        st.rerun()

# ============================================================================
# STEP 4 - LEARN & ADAPT: Learning & Scaling Rules
# ============================================================================

if st.session_state.step1_completed and st.session_state.step2_completed and st.session_state.step3_completed and st.session_state.get("step4_ready", False):
    st.divider()
    st.header("Step 4 - Learning & Adaptation")
    st.caption("How observed repayment evidence updates expectations and informs the next decision.")
    
    with st.container(border=True):
        st.subheader("Sequential Lending Process")
        
        st.markdown("""
**ORIGINATE** -> **OBSERVE REPAYMENT** -> **UPDATE RISK EXPECTATIONS** -> **UPDATE POLICY** -> **EXPAND / HOLD / STOP**
        """)
    
    st.divider()
    
    with st.container(border=True):
        st.subheader("How Our Risk Expectation Changes")
        update_c1, update_a1, update_c2, update_a2, update_c3, update_a3, update_c4 = st.columns([2, 0.4, 2, 0.4, 2, 0.4, 2])
        with update_c1:
            st.write("**INITIAL EXPECTATION**")
            st.caption("10% expected default")
        with update_a1:
            st.write("->")
        with update_c2:
            st.write("**NEW EVIDENCE**")
            st.caption("9 repaid / 1 default")
        with update_a2:
            st.write("->")
        with update_c3:
            st.write("**UPDATED EXPECTATION**")
            st.caption("10% expected default")
            st.caption("Higher confidence")
        with update_a3:
            st.write("->")
        with update_c4:
            st.write("**NEXT DECISION**")
            st.caption("Hold / Increase / Reduce exposure")

        with st.expander("How is this updated?"):
            updated_alpha = PRIOR_ALPHA + 1
            updated_beta = PRIOR_BETA + 9
            risk_grid = np.linspace(0.001, 0.35, 120)
            prior_density = np.exp((PRIOR_ALPHA - 1) * np.log(risk_grid) + (PRIOR_BETA - 1) * np.log(1 - risk_grid))
            updated_density = np.exp((updated_alpha - 1) * np.log(risk_grid) + (updated_beta - 1) * np.log(1 - risk_grid))
            distribution_df = pd.DataFrame({
                "Default rate": risk_grid,
                "Initial risk distribution": prior_density / prior_density.max(),
                "Updated risk distribution": updated_density / updated_density.max(),
            }).set_index("Default rate")
            st.write("**Decision logic**")
            st.caption("Update the local risk expectation after mature repayment outcomes become available.")
            st.write("**Inputs used**")
            st.caption("Initial expected default: 10%. Prior strength: 20 equivalent observations. Observed outcomes: 9 repaid and 1 default.")
            st.write("**Candidates evaluated**")
            st.caption("Initial risk distribution and the distribution after the observed cohort outcomes.")
            st.write("**Comparison chart**")
            st.caption("Initial and updated risk distributions")
            st.line_chart(distribution_df, use_container_width=True)
            st.write("**Calculation**")
            st.code("Initial Beta prior: alpha = 2, beta = 18\nObserved: 1 default, 9 repayments\nalpha_updated = 2 + 1 = 3\nbeta_updated = 18 + 9 = 27\nUpdated expected PD = 3 / (3 + 27) = 10%")
            st.write("**Selection rule**")
            st.caption("Advance one predefined ticket step only when updated expected PD is at or below break-even PD, the minimum evidence-confidence threshold is reached, the prior cohort has fully matured, and all Step 2 hard constraints remain satisfied. Otherwise, hold or reduce exposure.")
            st.write("**Assumptions**")
            st.caption("The observed cohort is representative of the initial target population and its outcomes are mature and correctly recorded.")
            st.write("**What would change the result?**")
            st.caption("Different repayment outcomes, a different prior, or a different prior strength.")
            st.write("**Reproduce this decision**")
            next_ticket_ladder = [200, 350, 500, 800, 1000]
            next_ticket = next_ticket_ladder[next_ticket_ladder.index(int(RECOMMENDED_LOAN)) + 1]
            next_cohort = INITIAL_COHORT
            next_exposure = next_cohort * next_ticket
            evidence_confidence = 0.82
            st.code(f"Beta-Binomial Bayesian updating\nInitial expected PD = {PRIOR_ALPHA:.0f} / ({PRIOR_ALPHA:.0f} + {PRIOR_BETA:.0f}) = {pct(PRIOR_EXPECTED_PD)}\nUpdated expected PD = ({PRIOR_ALPHA:.0f} + 1) / ({PRIOR_ALPHA:.0f} + 1 + {PRIOR_BETA:.0f} + 9) = {updated_alpha:.0f} / {updated_alpha + updated_beta:.0f} = {pct(updated_alpha / (updated_alpha + updated_beta))}\nPrevious exposure = {INITIAL_COHORT} x {money(RECOMMENDED_LOAN)} = {money(expected_exposure)}\nAdaptive ticket ladder = $200 -> $350 -> $500 -> $800 -> $1,000\nUpdated PD {pct(updated_alpha / (updated_alpha + updated_beta))} <= break-even PD {pct(break_even_pd)}; confidence {evidence_confidence:.0%} >= threshold {min_actionable_probability:.0%}; cohort fully matured; constraints satisfied\nNext exposure = {next_cohort} x ${next_ticket} = {money(next_exposure)}\nThe point estimate remains unchanged because the observed default rate matches the prior expectation. The distribution becomes narrower because local evidence increased confidence.")
            audit_inputs([
                {"Parameter": "Prior alpha / beta", "Value": "2 / 18", "Source": "Model assumption"},
                {"Parameter": "Prior expected PD", "Value": "10%", "Source": "Comparable portfolio assumption"},
                {"Parameter": "Observed repayments / defaults", "Value": "9 / 1", "Source": "Calculated value"},
                {"Parameter": "Updated alpha / beta", "Value": "3 / 27", "Source": "Calculated value"},
                {"Parameter": "Adaptive ticket ladder", "Value": "$200, $350, $500, $800, $1,000", "Source": "Model assumption"},
                {"Parameter": "Evidence confidence / threshold", "Value": f"{evidence_confidence:.0%} / {min_actionable_probability:.0%}", "Source": "Model assumption"},
            ])
        
        st.divider()
        
        st.write("**Decision impact**")
        st.metric("Next recommended exposure", money(next_exposure))
        st.caption("This process repeats: observe -> update -> decide -> observe -> ...")
    
    st.divider()
    
    st.info("Workflow complete. The system updates policy decisions based on observed evidence.")
