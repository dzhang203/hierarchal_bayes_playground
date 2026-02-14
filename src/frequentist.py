"""
Frequentist baseline estimators for A/B experiments.

This module implements two frequentist approaches:
1. No pooling: Independent per-creator estimates (high variance)
2. Complete pooling: Genre-level averages applied to all (high bias)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


def no_pooling_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute independent per-creator treatment effect estimates.

    This is the standard frequentist approach: analyze each creator's experiment
    independently using a two-sample t-test. Unbiased but high-variance,
    especially for small sample sizes.

    Parameters
    ----------
    df : pd.DataFrame
        User-level data with columns [creator_id, genre, genre_idx, group, revenue]

    Returns
    -------
    pd.DataFrame
        One row per creator with columns:
        - creator_id: creator identifier
        - genre: genre name
        - genre_idx: numeric genre index
        - effect_hat: estimated treatment effect (mean difference)
        - se: standard error of the estimate
        - ci_lower: lower bound of 95% confidence interval
        - ci_upper: upper bound of 95% confidence interval
        - p_value: two-sided p-value from Welch's t-test
        - n_total: total sample size
        - n_treatment: treatment group size
        - n_control: control group size
    """
    results = []

    for creator_id in df['creator_id'].unique():
        creator_df = df[df['creator_id'] == creator_id]

        # Get treatment and control revenues
        treatment = creator_df[creator_df['group'] == 'treatment']['revenue'].values
        control = creator_df[creator_df['group'] == 'control']['revenue'].values

        # Compute summary statistics
        mean_treatment = treatment.mean()
        mean_control = control.mean()
        var_treatment = treatment.var(ddof=1)
        var_control = control.var(ddof=1)
        n_treatment = len(treatment)
        n_control = len(control)

        # Treatment effect estimate
        effect_hat = mean_treatment - mean_control

        # Standard error (Welch's formula for unequal variances)
        se = np.sqrt(var_treatment / n_treatment + var_control / n_control)

        # 95% confidence interval
        ci_lower = effect_hat - 1.96 * se
        ci_upper = effect_hat + 1.96 * se

        # P-value from Welch's t-test
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        # Get genre info
        genre = creator_df['genre'].iloc[0]
        genre_idx = creator_df['genre_idx'].iloc[0]

        results.append({
            'creator_id': creator_id,
            'genre': genre,
            'genre_idx': genre_idx,
            'effect_hat': effect_hat,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'n_total': n_treatment + n_control,
            'n_treatment': n_treatment,
            'n_control': n_control
        })

    return pd.DataFrame(results).sort_values('creator_id').reset_index(drop=True)


def complete_pooling_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute genre-level pooled estimates applied to all creators.

    This is the opposite extreme from no pooling: completely ignore individual
    variation and assign each creator their genre's average treatment effect.
    Low variance but high bias (doesn't account for real individual differences).

    Parameters
    ----------
    df : pd.DataFrame
        User-level data with columns [creator_id, genre, genre_idx, group, revenue]

    Returns
    -------
    pd.DataFrame
        One row per creator with the same columns as no_pooling_estimates.
        All creators within a genre get identical estimates.
    """
    # First compute per-creator sample sizes (needed for individual SEs)
    creator_info = df.groupby('creator_id').agg({
        'genre': 'first',
        'genre_idx': 'first',
        'revenue': 'size'
    }).rename(columns={'revenue': 'n_total'}).reset_index()

    # Compute treatment/control sizes per creator
    group_sizes = df.groupby(['creator_id', 'group']).size().unstack(fill_value=0)
    creator_info['n_treatment'] = group_sizes.get('treatment', 0).values
    creator_info['n_control'] = group_sizes.get('control', 0).values

    # Compute genre-level pooled estimates
    genre_results = []

    for genre_idx in df['genre_idx'].unique():
        genre_df = df[df['genre_idx'] == genre_idx]

        # Pool all users in this genre
        treatment = genre_df[genre_df['group'] == 'treatment']['revenue'].values
        control = genre_df[genre_df['group'] == 'control']['revenue'].values

        # Compute genre-level statistics
        mean_treatment = treatment.mean()
        mean_control = control.mean()
        var_treatment = treatment.var(ddof=1)
        var_control = control.var(ddof=1)
        n_treatment = len(treatment)
        n_control = len(control)

        # Genre-level treatment effect
        effect_hat = mean_treatment - mean_control

        # Genre-level standard error
        se_genre = np.sqrt(var_treatment / n_treatment + var_control / n_control)

        # P-value
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        genre_name = genre_df['genre'].iloc[0]

        genre_results.append({
            'genre_idx': genre_idx,
            'genre': genre_name,
            'effect_hat': effect_hat,
            'se_genre': se_genre,
            'p_value': p_value,
            'var_treatment_pooled': var_treatment,
            'var_control_pooled': var_control
        })

    genre_df_results = pd.DataFrame(genre_results)

    # Now assign each creator their genre's estimate
    # But compute individual-level SEs based on their own sample size
    results = []

    for _, creator in creator_info.iterrows():
        genre_idx = creator['genre_idx']
        genre_stats = genre_df_results[genre_df_results['genre_idx'] == genre_idx].iloc[0]

        # Use genre-level effect estimate
        effect_hat = genre_stats['effect_hat']

        # Compute SE for this specific creator using pooled variances
        # This gives the uncertainty we'd have if we ran a study with this creator's n
        # but assuming the genre-level variance
        var_treatment = genre_stats['var_treatment_pooled']
        var_control = genre_stats['var_control_pooled']
        n_treatment = creator['n_treatment']
        n_control = creator['n_control']

        se = np.sqrt(var_treatment / n_treatment + var_control / n_control)

        # 95% CI
        ci_lower = effect_hat - 1.96 * se
        ci_upper = effect_hat + 1.96 * se

        results.append({
            'creator_id': creator['creator_id'],
            'genre': creator['genre'],
            'genre_idx': genre_idx,
            'effect_hat': effect_hat,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': genre_stats['p_value'],
            'n_total': creator['n_total'],
            'n_treatment': n_treatment,
            'n_control': n_control
        })

    return pd.DataFrame(results).sort_values('creator_id').reset_index(drop=True)


def compare_estimates(
    no_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
    truth: dict
) -> pd.DataFrame:
    """
    Compare no-pooling and complete-pooling estimates to ground truth.

    Parameters
    ----------
    no_pool : pd.DataFrame
        No-pooling estimates
    complete_pool : pd.DataFrame
        Complete-pooling estimates
    truth : dict
        Ground truth parameters

    Returns
    -------
    pd.DataFrame
        Comparison table with errors and diagnostics
    """
    true_effects = truth['creator_effects']

    comparison = pd.DataFrame({
        'creator_id': no_pool['creator_id'],
        'genre_idx': no_pool['genre_idx'],
        'n_total': no_pool['n_total'],
        'true_effect': true_effects,
        'no_pool_est': no_pool['effect_hat'],
        'complete_pool_est': complete_pool['effect_hat'],
        'no_pool_error': no_pool['effect_hat'] - true_effects,
        'complete_pool_error': complete_pool['effect_hat'] - true_effects,
        'no_pool_se': no_pool['se'],
        'complete_pool_se': complete_pool['se'],
        'no_pool_width': no_pool['ci_upper'] - no_pool['ci_lower'],
        'complete_pool_width': complete_pool['ci_upper'] - complete_pool['ci_lower']
    })

    return comparison


if __name__ == "__main__":
    # Test with synthetic data
    import sys
    sys.path.append('/Users/davidzhang/github/hierarchal_bayes_playground')
    from src.data_generation import generate_experiment_data

    print("Generating synthetic data...")
    df, truth = generate_experiment_data(seed=42)

    print("\nComputing no-pooling estimates...")
    no_pool = no_pooling_estimates(df)
    print(f"✓ Computed estimates for {len(no_pool)} creators")

    print("\nComputing complete-pooling estimates...")
    complete_pool = complete_pooling_estimates(df)
    print(f"✓ Computed estimates for {len(complete_pool)} creators")

    # Compute errors
    true_effects = truth['creator_effects']
    no_pool_mse = np.mean((no_pool['effect_hat'] - true_effects) ** 2)
    complete_pool_mse = np.mean((complete_pool['effect_hat'] - true_effects) ** 2)

    print(f"\nMean Squared Error:")
    print(f"  No pooling: {no_pool_mse:.4f}")
    print(f"  Complete pooling: {complete_pool_mse:.4f}")

    # Coverage
    no_pool_coverage = np.mean(
        (true_effects >= no_pool['ci_lower']) & (true_effects <= no_pool['ci_upper'])
    )
    complete_pool_coverage = np.mean(
        (true_effects >= complete_pool['ci_lower']) & (true_effects <= complete_pool['ci_upper'])
    )

    print(f"\n95% Interval Coverage:")
    print(f"  No pooling: {no_pool_coverage:.3f}")
    print(f"  Complete pooling: {complete_pool_coverage:.3f}")

    # Average interval width
    no_pool_width = (no_pool['ci_upper'] - no_pool['ci_lower']).mean()
    complete_pool_width = (complete_pool['ci_upper'] - complete_pool['ci_lower']).mean()

    print(f"\nAverage 95% CI Width:")
    print(f"  No pooling: {no_pool_width:.3f}")
    print(f"  Complete pooling: {complete_pool_width:.3f}")
