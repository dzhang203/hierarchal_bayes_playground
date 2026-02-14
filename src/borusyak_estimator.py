"""
Borusyak et al. (2024) imputation estimator for staggered DiD.

This module implements the imputation-based difference-in-differences estimator,
which is robust to heterogeneous treatment effects across time and units.

Key features:
- Two-way fixed effects (TWFE) estimation on never-treated units
- Counterfactual imputation for treated units
- Individual treatment effect (ITE) estimates
- Cohort-level standard errors (conservative)
- Adjustment for heterogeneous post-period lengths

References:
- Borusyak, Jaravel, and Spiess (2024). "Revisiting Event Study Designs:
  Robust and Efficient Estimation." Review of Economic Studies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats


def estimate_twfe(
    df: pd.DataFrame,
    outcome_col: str = 'revenue',
    unit_col: str = 'creator_id',
    time_col: str = 'week'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate two-way fixed effects (TWFE) model: y_it = α_i + λ_t + ε_it

    Uses only never-treated units to estimate fixed effects, avoiding
    contamination from treatment effects.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (should contain only never-treated units)
    outcome_col : str
        Name of outcome variable column
    unit_col : str
        Name of unit identifier column
    time_col : str
        Name of time period column

    Returns
    -------
    unit_fe : np.ndarray
        Estimated unit fixed effects (α_i), indexed by unit_col
    time_fe : np.ndarray
        Estimated time fixed effects (λ_t), indexed by time_col

    Notes
    -----
    We use the "within" transformation (demeaning) to estimate fixed effects:
    1. Demean within units to remove unit FEs
    2. Demean within time to get time FEs
    3. Iterate to convergence (Frisch-Waugh-Lovell theorem)

    This is equivalent to LSDV (Least Squares Dummy Variables) but more efficient.
    """
    # Get unique units and times
    units = df[unit_col].unique()
    times = df[time_col].unique()
    n_units = len(units)
    n_times = len(times)

    # Create mappings
    unit_to_idx = {u: i for i, u in enumerate(units)}
    time_to_idx = {t: i for i, t in enumerate(times)}

    # Convert to matrix form (units × times)
    Y = np.full((n_units, n_times), np.nan)
    for _, row in df.iterrows():
        i = unit_to_idx[row[unit_col]]
        t = time_to_idx[row[time_col]]
        Y[i, t] = row[outcome_col]

    # Initialize fixed effects
    unit_fe = np.zeros(n_units)
    time_fe = np.zeros(n_times)

    # Iterative demeaning to estimate fixed effects
    # (Converges quickly, usually 3-5 iterations)
    for _ in range(10):
        # Update time FEs: average residual in each period
        # (after removing unit FEs)
        for t in range(n_times):
            residuals = Y[:, t] - unit_fe
            time_fe[t] = np.nanmean(residuals)

        # Update unit FEs: average residual for each unit
        # (after removing time FEs)
        for i in range(n_units):
            residuals = Y[i, :] - time_fe
            unit_fe[i] = np.nanmean(residuals)

    # Normalize: set mean unit FE to zero (identification)
    # The mean is absorbed into time FEs
    unit_fe -= unit_fe.mean()

    return unit_fe, time_fe, unit_to_idx, time_to_idx


def borusyak_imputation_estimator(
    df: pd.DataFrame,
    never_treated_ids: set,
    outcome_col: str = 'revenue',
    unit_col: str = 'creator_id',
    time_col: str = 'week',
    treatment_col: str = 'treated',
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Estimate individual treatment effects using Borusyak imputation method.

    Algorithm:
    1. Fit TWFE on never-treated units: y_it = α_i + λ_t + ε_it
    2. For each treated unit, impute counterfactual: ŷ_i(0) = α̂_i + λ̂_t
    3. Compute ITE: τ̂_i = y_i - ŷ_i(0)
    4. Aggregate: ATT = mean(τ̂_i), with robust SEs

    Parameters
    ----------
    df : pd.DataFrame
        Full panel data (treated + never-treated)
    never_treated_ids : set
        Set of unit IDs that are never treated (control pool)
    outcome_col : str
        Name of outcome variable
    unit_col : str
        Name of unit identifier
    time_col : str
        Name of time period
    treatment_col : str
        Name of treatment indicator (0/1)
    verbose : bool
        Whether to print progress

    Returns
    -------
    ite_estimates : pd.DataFrame
        Individual treatment effect estimates with columns:
        - creator_id: unit identifier
        - effect_hat: estimated ITE (τ̂_i)
        - se: standard error (cohort-level, conservative)
        - n_post: number of post-treatment periods
        - adoption_week: week of treatment adoption

    diagnostics : dict
        Diagnostic information:
        - att: average treatment effect on treated
        - se_att: standard error of ATT
        - n_treated: number of treated units
        - n_never_treated: number of control units
        - cohort_atts: ATT by adoption cohort
        - cohort_ses: SE by adoption cohort
    """
    if verbose:
        print("="*70)
        print("BORUSYAK IMPUTATION ESTIMATOR")
        print("="*70)

    # Separate never-treated and treated units
    never_treated_df = df[df[unit_col].isin(never_treated_ids)].copy()
    treated_df = df[~df[unit_col].isin(never_treated_ids)].copy()

    n_never_treated = len(never_treated_ids)
    n_treated = df[~df[unit_col].isin(never_treated_ids)][unit_col].nunique()

    if verbose:
        print(f"Never-treated units: {n_never_treated}")
        print(f"Treated units: {n_treated}")
        print()

    # =====================================================================
    # STEP 1: Estimate TWFE on never-treated units
    # =====================================================================

    if verbose:
        print("Step 1: Estimating two-way fixed effects on never-treated...")

    unit_fe, time_fe, unit_to_idx, time_to_idx = estimate_twfe(
        never_treated_df,
        outcome_col=outcome_col,
        unit_col=unit_col,
        time_col=time_col
    )

    if verbose:
        print(f"  ✓ Estimated {len(unit_fe)} unit FEs and {len(time_fe)} time FEs")
        print()

    # =====================================================================
    # STEP 2: Impute counterfactuals for treated units
    # =====================================================================

    if verbose:
        print("Step 2: Imputing counterfactuals for treated units...")

    # For each treated unit, we need to:
    # 1. Estimate their unit FE using pre-treatment data
    # 2. Impute post-treatment counterfactual using unit FE + time FE

    treated_units = treated_df[unit_col].unique()
    ite_results = []

    for creator_id in treated_units:
        creator_data = treated_df[treated_df[unit_col] == creator_id].copy()

        # Get adoption week (first week with treatment = 1)
        adoption_week = creator_data[creator_data[treatment_col] == 1][time_col].min()

        # Pre-treatment data
        pre_treatment = creator_data[creator_data[time_col] < adoption_week]

        # Post-treatment data
        post_treatment = creator_data[creator_data[time_col] >= adoption_week]

        if len(pre_treatment) == 0 or len(post_treatment) == 0:
            # Skip if no pre or post data
            continue

        # Estimate unit FE for this creator using pre-treatment data
        # α̂_i = mean(y_it - λ̂_t) over pre-treatment periods
        pre_residuals = []
        for _, row in pre_treatment.iterrows():
            t = row[time_col]
            if t in time_to_idx:
                time_idx = time_to_idx[t]
                residual = row[outcome_col] - time_fe[time_idx]
                pre_residuals.append(residual)

        if len(pre_residuals) == 0:
            continue

        creator_fe = np.mean(pre_residuals)

        # Impute counterfactual for post-treatment periods
        # ŷ_it(0) = α̂_i + λ̂_t
        observed_post = []
        imputed_post = []

        for _, row in post_treatment.iterrows():
            t = row[time_col]
            if t in time_to_idx:
                time_idx = time_to_idx[t]
                y_observed = row[outcome_col]
                y_imputed = creator_fe + time_fe[time_idx]

                observed_post.append(y_observed)
                imputed_post.append(y_imputed)

        # Compute ITE: τ̂_i = mean(y_it) - mean(ŷ_it(0))
        ite = np.mean(observed_post) - np.mean(imputed_post)
        n_post = len(observed_post)

        ite_results.append({
            'creator_id': creator_id,
            'effect_hat': ite,
            'n_post': n_post,
            'adoption_week': adoption_week
        })

    ite_df = pd.DataFrame(ite_results)

    if verbose:
        print(f"  ✓ Computed ITEs for {len(ite_df)} treated units")
        print()

    # =====================================================================
    # STEP 3: Compute cohort-level standard errors
    # =====================================================================

    if verbose:
        print("Step 3: Computing cohort-level standard errors...")

    # Group by adoption cohort
    cohort_atts = {}
    cohort_ses = {}

    for adoption_week in ite_df['adoption_week'].unique():
        cohort_ites = ite_df[ite_df['adoption_week'] == adoption_week]['effect_hat']

        # Cohort ATT (average across cohort)
        cohort_att = cohort_ites.mean()

        # Cohort SE (standard error of the mean)
        # Conservative: use sample SD / sqrt(n)
        if len(cohort_ites) > 1:
            cohort_se = cohort_ites.std() / np.sqrt(len(cohort_ites))
        else:
            # Single-member cohort: use fallback SE based on global variation
            # This is conservative - we'll replace it with median cohort SE after all cohorts are processed
            cohort_se = np.nan

        cohort_atts[adoption_week] = cohort_att
        cohort_ses[adoption_week] = cohort_se

    # For single-member cohorts, use median SE from multi-member cohorts
    valid_ses = [se for se in cohort_ses.values() if not np.isnan(se)]
    if valid_ses:
        median_se = np.median(valid_ses)
        for week, se in cohort_ses.items():
            if np.isnan(se):
                cohort_ses[week] = median_se
                if verbose:
                    print(f"  ⚠️  Week {week}: single-member cohort, using median SE = {median_se:.3f}")
    else:
        # All cohorts are single-member (edge case)
        # Use global std as fallback
        global_se = ite_df['effect_hat'].std() / np.sqrt(len(ite_df))
        for week in cohort_ses.keys():
            if np.isnan(cohort_ses[week]):
                cohort_ses[week] = global_se

    # Assign cohort SEs to individuals
    ite_df['se_cohort'] = ite_df['adoption_week'].map(cohort_ses)

    # Adjust SE by post-period length (creators with more data → smaller SE)
    # SE_i = SE_cohort × sqrt(n_post_cohort_avg / n_post_i)
    cohort_avg_n_post = ite_df.groupby('adoption_week')['n_post'].transform('mean')
    ite_df['se'] = ite_df['se_cohort'] * np.sqrt(cohort_avg_n_post / ite_df['n_post'])

    if verbose:
        print(f"  ✓ Computed SEs for {len(cohort_atts)} cohorts")
        print()

    # =====================================================================
    # STEP 4: Aggregate ATT
    # =====================================================================

    # Overall ATT (average across all treated units)
    att = ite_df['effect_hat'].mean()

    # Overall SE (accounting for cohort structure)
    # Use weighted average of cohort SEs
    cohort_sizes = ite_df.groupby('adoption_week').size()
    cohort_weights = cohort_sizes / cohort_sizes.sum()

    # Weighted variance
    weighted_var = sum(
        cohort_weights.get(week, 0) * cohort_ses[week]**2
        for week in cohort_atts.keys()
        if week in cohort_weights.index
    )
    se_att = np.sqrt(weighted_var) if weighted_var > 0 else np.nan

    if verbose:
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"Average Treatment Effect (ATT): {att:.3f}")
        print(f"Standard Error: {se_att:.3f}")
        print(f"95% CI: [{att - 1.96*se_att:.3f}, {att + 1.96*se_att:.3f}]")
        print()
        print(f"Cohort-level ATTs:")
        for week in sorted(cohort_atts.keys())[:5]:
            print(f"  Week {week}: {cohort_atts[week]:.3f} (SE: {cohort_ses[week]:.3f})")
        if len(cohort_atts) > 5:
            print(f"  ... ({len(cohort_atts)} cohorts total)")
        print("="*70)

    # Package diagnostics
    diagnostics = {
        'att': att,
        'se_att': se_att,
        'n_treated': n_treated,
        'n_never_treated': n_never_treated,
        'cohort_atts': cohort_atts,
        'cohort_ses': cohort_ses,
        'n_cohorts': len(cohort_atts)
    }

    # Return estimates with only necessary columns
    ite_estimates = ite_df[['creator_id', 'effect_hat', 'se', 'n_post', 'adoption_week']].copy()

    return ite_estimates, diagnostics


def validate_borusyak_estimates(
    ite_estimates: pd.DataFrame,
    true_effects: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Validate Borusyak estimates against ground truth.

    Parameters
    ----------
    ite_estimates : pd.DataFrame
        Estimated ITEs from Borusyak estimator
    true_effects : np.ndarray
        True treatment effects (ground truth)
    verbose : bool
        Whether to print validation results

    Returns
    -------
    metrics : dict
        Validation metrics:
        - bias: mean(estimated - true)
        - mae: mean absolute error
        - rmse: root mean squared error
        - coverage: fraction of 95% CIs containing truth
    """
    # Get true effects for treated units
    creator_ids = ite_estimates['creator_id'].values

    # Handle both dict and Series/array
    if isinstance(true_effects, dict):
        true_treated = np.array([true_effects[cid] for cid in creator_ids])
    else:
        # Assume Series or array-like with indexing
        true_treated = true_effects[creator_ids].values if hasattr(true_effects[creator_ids], 'values') else true_effects[creator_ids]

    # Compute errors
    errors = ite_estimates['effect_hat'].values - true_treated
    bias = errors.mean()
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())

    # Coverage: fraction of 95% CIs containing truth
    ci_lower = ite_estimates['effect_hat'] - 1.96 * ite_estimates['se']
    ci_upper = ite_estimates['effect_hat'] + 1.96 * ite_estimates['se']
    covered = (true_treated >= ci_lower) & (true_treated <= ci_upper)
    coverage = covered.mean()

    metrics = {
        'bias': bias,
        'mae': mae,
        'rmse': rmse,
        'coverage': coverage
    }

    if verbose:
        print("="*70)
        print("BORUSYAK VALIDATION")
        print("="*70)
        print(f"Bias: {bias:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Coverage (95% CI): {coverage:.1%}")
        print("="*70)

    return metrics


if __name__ == "__main__":
    # Test with synthetic data
    from data_generation_staggered import generate_staggered_adoption_data

    print("Generating synthetic data...")
    df, truth = generate_staggered_adoption_data(
        n_genres=5,
        n_creators_per_genre=40,
        n_weeks=52,
        seed=42
    )

    print("\nRunning Borusyak estimator...")
    ite_estimates, diagnostics = borusyak_imputation_estimator(
        df,
        never_treated_ids=truth['never_treated'],
        verbose=True
    )

    print("\nValidating against ground truth...")
    metrics = validate_borusyak_estimates(
        ite_estimates,
        truth['creator_effects'],
        verbose=True
    )

    print("\n✓ Borusyak estimator test complete!")
