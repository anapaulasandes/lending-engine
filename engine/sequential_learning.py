from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def calculate_break_even_pd(
    revenue_per_loan: float,
    exposure_per_loan: float,
    loss_given_default: float = 1.0,
    funding_cost_per_loan: float = 0.0,
    operating_cost_per_loan: float = 0.0,
    collection_cost_per_loan: float = 0.0,
) -> float:
    """Return the PD at which expected profit becomes zero for a single loan.

    The model follows the project convention:

    expected_profit = expected_revenue - expected_credit_loss - funding_cost
                      - operating_cost - collection_cost

    with expected_credit_loss = PD * exposure * LGD.
    """
    if exposure_per_loan <= 0:
        raise ValueError("exposure_per_loan must be positive.")
    if loss_given_default < 0 or loss_given_default > 1:
        raise ValueError("loss_given_default must be between 0 and 1.")

    fixed_costs = (
        funding_cost_per_loan + operating_cost_per_loan + collection_cost_per_loan
    )
    numerator = revenue_per_loan - fixed_costs
    denominator = exposure_per_loan * loss_given_default
    if denominator <= 0:
        raise ValueError("denominator is not positive; check exposure and LGD inputs.")

    break_even_pd = numerator / denominator
    return float(np.clip(break_even_pd, 0.0, 1.0))


def beta_posterior(
    prior_alpha: float,
    prior_beta: float,
    observed_defaults: int,
    observed_matured_loans: int,
) -> Dict[str, float]:
    """Return the Beta posterior parameters after incorporating matured outcomes."""
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be positive.")
    if observed_defaults < 0:
        raise ValueError("observed_defaults cannot be negative.")
    if observed_matured_loans < 0:
        raise ValueError("observed_matured_loans cannot be negative.")
    if observed_defaults > observed_matured_loans:
        raise ValueError("observed_defaults cannot exceed observed_matured_loans.")

    posterior_alpha = prior_alpha + observed_defaults
    posterior_beta = prior_beta + observed_matured_loans - observed_defaults

    return {
        "posterior_alpha": float(posterior_alpha),
        "posterior_beta": float(posterior_beta),
        "posterior_mean_pd": float(posterior_alpha / (posterior_alpha + posterior_beta)),
    }


def amount_risk_pd(
    base_pd: float,
    loan_amount: float,
    reference_amount: float = 500.0,
    risk_sensitivity: float = 0.25,
) -> float:
    """Model how credit risk changes with loan amount relative to a reference point."""
    if not 0.0 <= base_pd <= 1.0:
        raise ValueError("base_pd must be between 0 and 1.")
    if reference_amount <= 0:
        return float(np.clip(base_pd, 0.0, 1.0))
    relative_change = (loan_amount - reference_amount) / reference_amount
    adjusted_pd = base_pd * (1.0 + risk_sensitivity * relative_change)
    return float(np.clip(adjusted_pd, 0.0, 1.0))


def price_take_up_probability(
    pricing_rate: float,
    reference_pricing_rate: float = 0.20,
    sensitivity: float = 0.35,
) -> float:
    """Model how take-up falls as pricing grows above the reference price."""
    if reference_pricing_rate <= 0:
        return 1.0
    relative_change = (pricing_rate - reference_pricing_rate) / reference_pricing_rate
    take_up = 1.0 - sensitivity * relative_change
    return float(np.clip(take_up, 0.0, 1.0))


def evaluate_offer(
    loan_amount: float,
    pricing_rate: float,
    expected_pd: float,
    lgd: float,
    funding_cost_per_loan: float = 0.0,
    operating_cost_per_loan: float = 0.0,
    collection_cost_per_loan: float = 0.0,
    take_up_probability: float = 1.0,
) -> Dict[str, float]:
    """Evaluate the economics of a single offer using a transparent business model."""
    expected_revenue = float(loan_amount * pricing_rate)
    expected_credit_loss = float(expected_pd * loan_amount * lgd)
    expected_collection_cost = float(collection_cost_per_loan)
    expected_ue_per_loan = (
        expected_revenue
        - expected_credit_loss
        - funding_cost_per_loan
        - operating_cost_per_loan
        - expected_collection_cost
    )
    expected_value_per_eligible_customer = float(
        take_up_probability * expected_ue_per_loan
    )
    return {
        "expected_revenue": float(expected_revenue),
        "expected_credit_loss": float(expected_credit_loss),
        "funding_cost": float(funding_cost_per_loan),
        "operating_cost": float(operating_cost_per_loan),
        "expected_collection_cost": float(expected_collection_cost),
        "expected_ue_per_loan": float(expected_ue_per_loan),
        "take_up_probability": float(np.clip(take_up_probability, 0.0, 1.0)),
        "expected_value_per_eligible_customer": float(expected_value_per_eligible_customer),
    }


def summarize_posterior(
    posterior_alpha: float,
    posterior_beta: float,
    break_even_pd: float,
    random_seed: Optional[int] = None,
    samples: int = 200_000,
    credible_level: float = 0.8,
) -> Dict[str, Any]:
    """Summarize a Beta posterior with a business-facing default-rate view."""
    if posterior_alpha <= 0 or posterior_beta <= 0:
        raise ValueError("posterior_alpha and posterior_beta must be positive.")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)
    draws = rng.beta(posterior_alpha, posterior_beta, size=samples)
    lower_frac = (1.0 - credible_level) / 2.0
    upper_frac = 1.0 - lower_frac
    lower_bound, upper_bound = np.quantile(draws, [lower_frac, upper_frac])

    return {
        "posterior_alpha": float(posterior_alpha),
        "posterior_beta": float(posterior_beta),
        "posterior_mean_pd": float(np.mean(draws)),
        "credible_lower": float(lower_bound),
        "credible_upper": float(upper_bound),
        "probability_pd_below_break_even": float(np.mean(draws < break_even_pd)),
    }


def evaluate_candidate_cohort(
    cohort_size: int,
    prior_alpha: float,
    prior_beta: float,
    observed_defaults: int,
    observed_matured_loans: int,
    break_even_pd: float,
    revenue_per_loan: float,
    exposure_per_loan: float,
    loss_given_default: float = 1.0,
    funding_cost_per_loan: float = 0.0,
    operating_cost_per_loan: float = 0.0,
    collection_cost_per_loan: float = 0.0,
    monte_carlo_trials: int = 4000,
    expand_threshold: float = 0.80,
    stop_threshold: float = 0.40,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a candidate cohort size under sequential learning assumptions."""
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive.")
    if monte_carlo_trials <= 0:
        raise ValueError("monte_carlo_trials must be positive.")
    if not 0.0 <= stop_threshold <= expand_threshold <= 1.0:
        raise ValueError("Use stop_threshold <= expand_threshold and values between 0 and 1.")

    posterior = beta_posterior(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        observed_defaults=observed_defaults,
        observed_matured_loans=observed_matured_loans,
    )

    rng = np.random.default_rng(random_seed)
    p_true = rng.beta(
        posterior["posterior_alpha"], posterior["posterior_beta"], size=monte_carlo_trials
    )
    defaults_next = rng.binomial(cohort_size, p_true)

    alpha_after = posterior["posterior_alpha"] + defaults_next
    beta_after = posterior["posterior_beta"] + cohort_size - defaults_next

    action_scores = []
    profit_distribution = []
    for idx in range(monte_carlo_trials):
        posterior_draws = rng.beta(alpha_after[idx], beta_after[idx], size=2000)
        prob_viable = float(np.mean(posterior_draws < break_even_pd))
        if prob_viable >= expand_threshold:
            action = "EXPAND"
        elif prob_viable <= stop_threshold:
            action = "STOP"
        else:
            action = "SAMPLE_MORE"
        action_scores.append(action)

        profit_per_loan = (
            revenue_per_loan
            - p_true[idx] * exposure_per_loan * loss_given_default
            - funding_cost_per_loan
            - operating_cost_per_loan
            - collection_cost_per_loan
        )
        cohort_profit = cohort_size * profit_per_loan
        profit_distribution.append(cohort_profit)

    profit_distribution = np.asarray(profit_distribution, dtype=float)
    action_array = np.asarray(action_scores)
    prob_expand = float(np.mean(action_array == "EXPAND"))
    prob_stop = float(np.mean(action_array == "STOP"))
    prob_sample_more = float(np.mean(action_array == "SAMPLE_MORE"))
    prob_actionable = prob_expand + prob_stop
    expected_cohort_profit = float(np.mean(profit_distribution))
    prob_negative_profit = float(np.mean(profit_distribution < 0.0))
    p05_profit = float(np.percentile(profit_distribution, 5))
    cvar_05 = float(np.mean(profit_distribution[profit_distribution <= np.percentile(profit_distribution, 5)])) if np.any(profit_distribution <= np.percentile(profit_distribution, 5)) else 0.0
    exposure = cohort_size * exposure_per_loan
    learning_efficiency = prob_actionable / max(exposure / 1000.0, 1e-9)

    return {
        "cohort_size": int(cohort_size),
        "exposure": float(exposure),
        "prob_expand": float(prob_expand),
        "prob_stop": float(prob_stop),
        "prob_sample_more": float(prob_sample_more),
        "prob_actionable": float(prob_actionable),
        "expected_cohort_profit": float(expected_cohort_profit),
        "prob_negative_profit": float(prob_negative_profit),
        "p05_profit": float(p05_profit),
        "cvar_05": float(cvar_05),
        "learning_efficiency": float(learning_efficiency),
        "break_even_pd": float(break_even_pd),
        "posterior_mean_pd": float(posterior["posterior_mean_pd"]),
        "posterior_alpha": float(posterior["posterior_alpha"]),
        "posterior_beta": float(posterior["posterior_beta"]),
    }


def recommend_next_cohort(
    prior_alpha: float,
    prior_beta: float,
    observed_defaults: int,
    observed_matured_loans: int,
    break_even_pd: float,
    revenue_per_loan: float,
    exposure_per_loan: float,
    candidate_cohort_sizes: Iterable[int],
    max_exposure: Optional[float] = None,
    max_downside_loss: Optional[float] = None,
    minimum_probability_actionable: float = 0.60,
    loss_given_default: float = 1.0,
    funding_cost_per_loan: float = 0.0,
    operating_cost_per_loan: float = 0.0,
    collection_cost_per_loan: float = 0.0,
    monte_carlo_trials: int = 4000,
    expand_threshold: float = 0.80,
    stop_threshold: float = 0.40,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Choose the smallest feasible cohort that is likely to produce actionable evidence."""
    sizes = [int(size) for size in candidate_cohort_sizes]
    if not sizes:
        raise ValueError("candidate_cohort_sizes must include at least one cohort size.")

    rows = []
    for size in sizes:
        row = evaluate_candidate_cohort(
            cohort_size=size,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            observed_defaults=observed_defaults,
            observed_matured_loans=observed_matured_loans,
            break_even_pd=break_even_pd,
            revenue_per_loan=revenue_per_loan,
            exposure_per_loan=exposure_per_loan,
            loss_given_default=loss_given_default,
            funding_cost_per_loan=funding_cost_per_loan,
            operating_cost_per_loan=operating_cost_per_loan,
            collection_cost_per_loan=collection_cost_per_loan,
            monte_carlo_trials=monte_carlo_trials,
            expand_threshold=expand_threshold,
            stop_threshold=stop_threshold,
            random_seed=random_seed,
        )
        row["within_exposure_limit"] = (
            max_exposure is None or row["exposure"] <= max_exposure
        )
        row["within_downside_limit"] = (
            max_downside_loss is None or abs(row["cvar_05"]) <= max_downside_loss
        )
        row["meets_actionability"] = row["prob_actionable"] >= minimum_probability_actionable
        rows.append(row)

    feasible = [row for row in rows if row["within_exposure_limit"] and row["within_downside_limit"] and row["meets_actionability"]]
    if not feasible:
        return {
            "decision": "NO_FEASIBLE_COHORT",
            "reason": "No candidate cohort satisfies the configured exposure and actionability constraints.",
            "candidate_results": pd.DataFrame(rows),
        }

    chosen = sorted(feasible, key=lambda row: (row["cohort_size"], -row["prob_actionable"]))[0]
    return {
        "decision": "SAMPLE_MORE",
        "recommended_cohort_size": int(chosen["cohort_size"]),
        "recommended_exposure": float(chosen["exposure"]),
        "probability_actionable": float(chosen["prob_actionable"]),
        "expected_cohort_profit": float(chosen["expected_cohort_profit"]),
        "probability_negative_profit": float(chosen["prob_negative_profit"]),
        "candidate_results": pd.DataFrame(rows),
        "selected_row": chosen,
    }


def current_decision(
    prior_alpha: float,
    prior_beta: float,
    observed_defaults: int,
    observed_matured_loans: int,
    break_even_pd: float,
    revenue_per_loan: float,
    exposure_per_loan: float,
    candidate_cohort_sizes: Optional[Iterable[int]] = None,
    max_exposure: Optional[float] = None,
    max_downside_loss: Optional[float] = None,
    minimum_probability_actionable: float = 0.60,
    loss_given_default: float = 1.0,
    funding_cost_per_loan: float = 0.0,
    operating_cost_per_loan: float = 0.0,
    collection_cost_per_loan: float = 0.0,
    expand_threshold: float = 0.80,
    stop_threshold: float = 0.40,
    monte_carlo_trials: int = 4000,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the current recommendation under the sequential learning framework."""
    posterior = beta_posterior(
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        observed_defaults=observed_defaults,
        observed_matured_loans=observed_matured_loans,
    )
    summary = summarize_posterior(
        posterior_alpha=posterior["posterior_alpha"],
        posterior_beta=posterior["posterior_beta"],
        break_even_pd=break_even_pd,
        random_seed=random_seed,
    )

    probability_viable = summary["probability_pd_below_break_even"]
    if probability_viable >= expand_threshold:
        current_action = "EXPAND"
        recommended_next_cohort = None
    elif probability_viable <= stop_threshold:
        current_action = "STOP"
        recommended_next_cohort = None
    else:
        current_action = "SAMPLE_MORE"
        candidate_sizes = list(candidate_cohort_sizes or [10, 20, 25, 30, 40, 50, 75, 100])
        recommended = recommend_next_cohort(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            observed_defaults=observed_defaults,
            observed_matured_loans=observed_matured_loans,
            break_even_pd=break_even_pd,
            revenue_per_loan=revenue_per_loan,
            exposure_per_loan=exposure_per_loan,
            candidate_cohort_sizes=candidate_sizes,
            max_exposure=max_exposure,
            max_downside_loss=max_downside_loss,
            minimum_probability_actionable=minimum_probability_actionable,
            loss_given_default=loss_given_default,
            funding_cost_per_loan=funding_cost_per_loan,
            operating_cost_per_loan=operating_cost_per_loan,
            collection_cost_per_loan=collection_cost_per_loan,
            monte_carlo_trials=monte_carlo_trials,
            expand_threshold=expand_threshold,
            stop_threshold=stop_threshold,
            random_seed=random_seed,
        )
        recommended_next_cohort = recommended

    return {
        "current_action": current_action,
        "break_even_pd": float(break_even_pd),
        "posterior_alpha": float(posterior["posterior_alpha"]),
        "posterior_beta": float(posterior["posterior_beta"]),
        "posterior_mean_pd": float(posterior["posterior_mean_pd"]),
        "probability_viable": float(probability_viable),
        "credible_interval": (
            summary["credible_lower"],
            summary["credible_upper"],
        ),
        "recommendation": recommended_next_cohort,
        "expand_if": "P(PD < break_even) >= expand_threshold",
        "stop_if": "P(PD < break_even) <= stop_threshold",
    }
