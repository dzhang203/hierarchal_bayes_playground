"""
Validation and comparison of estimation methods.

This module compares three approaches:
1. No pooling (frequentist per-creator)
2. Complete pooling (genre-level average)
3. Partial pooling (hierarchical Bayesian model)

The key metrics are:
- Mean Squared Error (MSE) to ground truth
- Coverage of confidence/credible intervals
- Interval width (precision)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_mse(estimates: np.ndarray, truth: np.ndarray) -> float:
    """
    Compute Mean Squared Error between estimates and ground truth.

    Parameters
    ----------
    estimates : np.ndarray
        Estimated treatment effects
    truth : np.ndarray
        True treatment effects

    Returns
    -------
    float
        Mean squared error
    """
    return np.mean((estimates - truth) ** 2)


def compute_mae(estimates: np.ndarray, truth: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    estimates : np.ndarray
        Estimated treatment effects
    truth : np.ndarray
        True treatment effects

    Returns
    -------
    float
        Mean absolute error
    """
    return np.mean(np.abs(estimates - truth))


def compute_coverage(
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    truth: np.ndarray
) -> float:
    """
    Compute coverage rate of confidence/credible intervals.

    For a well-calibrated 95% interval, this should be ≈ 0.95.

    Parameters
    ----------
    ci_lower : np.ndarray
        Lower bounds of intervals
    ci_upper : np.ndarray
        Upper bounds of intervals
    truth : np.ndarray
        True values

    Returns
    -------
    float
        Fraction of intervals that contain the true value
    """
    covered = (truth >= ci_lower) & (truth <= ci_upper)
    return covered.mean()


def compute_avg_interval_width(
    ci_lower: np.ndarray,
    ci_upper: np.ndarray
) -> float:
    """
    Compute average width of confidence/credible intervals.

    Narrower intervals with maintained coverage = better precision.

    Parameters
    ----------
    ci_lower : np.ndarray
        Lower bounds of intervals
    ci_upper : np.ndarray
        Upper bounds of intervals

    Returns
    -------
    float
        Average interval width
    """
    return np.mean(ci_upper - ci_lower)


def stratified_metrics(
    estimates_df: pd.DataFrame,
    truth: np.ndarray,
    bins: list = None
) -> pd.DataFrame:
    """
    Compute metrics stratified by sample size.

    HBM should help most for small sample sizes, so we want to see
    the performance breakdown across different creator sizes.

    Parameters
    ----------
    estimates_df : pd.DataFrame
        Estimates with columns [effect_hat, ci_lower, ci_upper, n_total]
    truth : np.ndarray
        True treatment effects
    bins : list, optional
        Sample size bin edges. Default: [0, 100, 500, np.inf]

    Returns
    -------
    pd.DataFrame
        Metrics for each sample size bin
    """
    if bins is None:
        bins = [0, 100, 500, np.inf]

    bin_labels = [f"n={bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    bin_labels[-1] = f"n>{bins[-2]}"

    estimates_df = estimates_df.copy()
    estimates_df['size_bin'] = pd.cut(
        estimates_df['n_total'],
        bins=bins,
        labels=bin_labels
    )

    results = []
    for bin_label in bin_labels:
        mask = estimates_df['size_bin'] == bin_label
        if mask.sum() == 0:
            continue

        bin_estimates = estimates_df[mask]['effect_hat'].values
        bin_truth = truth[mask]
        bin_ci_lower = estimates_df[mask]['ci_lower'].values
        bin_ci_upper = estimates_df[mask]['ci_upper'].values

        results.append({
            'size_bin': bin_label,
            'n_creators': mask.sum(),
            'mse': compute_mse(bin_estimates, bin_truth),
            'mae': compute_mae(bin_estimates, bin_truth),
            'coverage': compute_coverage(bin_ci_lower, bin_ci_upper, bin_truth),
            'avg_width': compute_avg_interval_width(bin_ci_lower, bin_ci_upper)
        })

    return pd.DataFrame(results)


def compare_all_methods(
    no_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
    hbm: pd.DataFrame,
    truth: dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare all three estimation methods.

    Parameters
    ----------
    no_pool : pd.DataFrame
        No-pooling estimates
    complete_pool : pd.DataFrame
        Complete-pooling estimates
    hbm : pd.DataFrame
        HBM estimates
    truth : dict
        Ground truth parameters
    verbose : bool
        Whether to print comparison table

    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics
    """
    true_effects = truth['creator_effects']

    # Overall metrics
    methods = {
        'No Pooling': no_pool,
        'Complete Pooling': complete_pool,
        'HBM (Partial Pooling)': hbm
    }

    overall_results = []
    for method_name, df in methods.items():
        estimates = df['effect_hat'].values
        ci_lower = df['ci_lower'].values
        ci_upper = df['ci_upper'].values

        overall_results.append({
            'Method': method_name,
            'MSE': compute_mse(estimates, true_effects),
            'MAE': compute_mae(estimates, true_effects),
            'Coverage': compute_coverage(ci_lower, ci_upper, true_effects),
            'Avg CI Width': compute_avg_interval_width(ci_lower, ci_upper)
        })

    overall_df = pd.DataFrame(overall_results)

    if verbose:
        print("=" * 80)
        print("OVERALL COMPARISON")
        print("=" * 80)
        print(overall_df.to_string(index=False))
        print("=" * 80)

        # Print winner for each metric
        print("\nKey Findings:")
        mse_winner = overall_df.loc[overall_df['MSE'].idxmin(), 'Method']
        print(f"  • Lowest MSE: {mse_winner}")

        # Coverage closest to 0.95
        coverage_diff = (overall_df['Coverage'] - 0.95).abs()
        coverage_winner = overall_df.loc[coverage_diff.idxmin(), 'Method']
        print(f"  • Best calibrated intervals: {coverage_winner}")

        width_winner = overall_df.loc[overall_df['Avg CI Width'].idxmin(), 'Method']
        print(f"  • Narrowest intervals: {width_winner}")

    return overall_df


def stratified_comparison(
    no_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
    hbm: pd.DataFrame,
    truth: dict,
    bins: list = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Compare methods stratified by sample size.

    Parameters
    ----------
    no_pool : pd.DataFrame
        No-pooling estimates
    complete_pool : pd.DataFrame
        Complete-pooling estimates
    hbm : pd.DataFrame
        HBM estimates
    truth : dict
        Ground truth parameters
    bins : list, optional
        Sample size bin edges
    verbose : bool
        Whether to print results

    Returns
    -------
    dict
        Dictionary mapping method names to stratified results DataFrames
    """
    true_effects = truth['creator_effects']

    methods = {
        'No Pooling': no_pool,
        'Complete Pooling': complete_pool,
        'HBM': hbm
    }

    stratified_results = {}
    for method_name, df in methods.items():
        stratified_results[method_name] = stratified_metrics(df, true_effects, bins)

    if verbose:
        print("\n" + "=" * 80)
        print("STRATIFIED COMPARISON (by sample size)")
        print("=" * 80)

        # Combine results for easier comparison
        for i, (method_name, df) in enumerate(stratified_results.items()):
            print(f"\n{method_name}:")
            print(df.to_string(index=False))

        # Print MSE comparison table
        print("\n" + "=" * 80)
        print("MSE COMPARISON BY SAMPLE SIZE")
        print("=" * 80)

        mse_comparison = pd.DataFrame({
            method: df.set_index('size_bin')['mse']
            for method, df in stratified_results.items()
        })
        print(mse_comparison.to_string())

    return stratified_results


def compute_shrinkage_metrics(
    no_pool: pd.DataFrame,
    hbm: pd.DataFrame,
    truth: dict
) -> pd.DataFrame:
    """
    Analyze shrinkage behavior of HBM.

    Shrinkage is the amount that HBM pulls noisy estimates toward
    the genre mean. More shrinkage for small-sample creators.

    Parameters
    ----------
    no_pool : pd.DataFrame
        No-pooling estimates
    hbm : pd.DataFrame
        HBM estimates
    truth : dict
        Ground truth parameters

    Returns
    -------
    pd.DataFrame
        Shrinkage analysis with columns:
        - creator_id
        - n_total
        - genre_idx
        - no_pool_est
        - hbm_est
        - shrinkage: (no_pool - hbm)
        - shrinkage_pct: shrinkage as % of no_pool deviation from genre mean
    """
    # Get genre means from HBM
    # We'll compute them from the true genre assignments
    genre_means = []
    for g in range(truth['n_genres']):
        mask = truth['creator_genre'] == g
        genre_means.append(hbm[mask]['effect_hat'].mean())

    creator_genre_means = np.array([genre_means[g] for g in truth['creator_genre']])

    shrinkage_df = pd.DataFrame({
        'creator_id': no_pool['creator_id'],
        'n_total': no_pool['n_total'],
        'genre_idx': no_pool['genre_idx'],
        'no_pool_est': no_pool['effect_hat'],
        'hbm_est': hbm['effect_hat'],
        'true_effect': truth['creator_effects'],
        'genre_mean': creator_genre_means
    })

    # Compute shrinkage metrics
    shrinkage_df['shrinkage'] = shrinkage_df['no_pool_est'] - shrinkage_df['hbm_est']

    # Shrinkage as a fraction of the no-pool deviation from genre mean
    no_pool_deviation = shrinkage_df['no_pool_est'] - shrinkage_df['genre_mean']
    shrinkage_df['shrinkage_pct'] = np.where(
        np.abs(no_pool_deviation) > 0.01,
        shrinkage_df['shrinkage'] / no_pool_deviation,
        0
    )

    return shrinkage_df


def validate_genre_recovery(
    genre_estimates: pd.DataFrame,
    truth: dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Check how well HBM recovers true genre-level effects.

    Parameters
    ----------
    genre_estimates : pd.DataFrame
        HBM genre estimates (from extract_genre_estimates)
    truth : dict
        Ground truth parameters
    verbose : bool
        Whether to print results

    Returns
    -------
    pd.DataFrame
        Comparison of estimated vs. true genre effects
    """
    comparison = pd.DataFrame({
        'genre': truth['genre_names'],
        'true_effect': truth['genre_effects'],
        'estimated_effect': genre_estimates['mu_genre_mean'],
        'posterior_sd': genre_estimates['mu_genre_std'],
        'ci_lower': genre_estimates['ci_lower'],
        'ci_upper': genre_estimates['ci_upper']
    })

    comparison['error'] = comparison['estimated_effect'] - comparison['true_effect']
    comparison['abs_error'] = comparison['error'].abs()

    # Check if true value is in credible interval
    comparison['covered'] = (
        (comparison['true_effect'] >= comparison['ci_lower']) &
        (comparison['true_effect'] <= comparison['ci_upper'])
    )

    if verbose:
        print("\n" + "=" * 80)
        print("GENRE-LEVEL EFFECT RECOVERY")
        print("=" * 80)
        print(comparison[['genre', 'true_effect', 'estimated_effect', 'error', 'covered']].to_string(index=False))
        print("=" * 80)
        print(f"Coverage: {comparison['covered'].mean():.2%}")
        print(f"Mean Absolute Error: {comparison['abs_error'].mean():.4f}")

    return comparison


if __name__ == "__main__":
    # Test with synthetic data
    import sys
    sys.path.append('/Users/davidzhang/github/hierarchal_bayes_playground')
    from src.data_generation import generate_experiment_data
    from src.frequentist import no_pooling_estimates, complete_pooling_estimates
    from src.hierarchical_model import (
        prepare_creator_summaries,
        fit_hierarchical_model,
        extract_hbm_estimates,
        extract_genre_estimates
    )

    print("Generating synthetic data...")
    df, truth = generate_experiment_data(seed=42)

    print("\nComputing frequentist baselines...")
    no_pool = no_pooling_estimates(df)
    complete_pool = complete_pooling_estimates(df)

    print("\nFitting HBM (this may take a minute)...")
    summaries = prepare_creator_summaries(df)
    idata = fit_hierarchical_model(
        summaries,
        n_genres=truth['n_genres'],
        draws=1000,
        tune=500,
        chains=2
    )

    hbm = extract_hbm_estimates(idata, summaries)
    genre_est = extract_genre_estimates(idata, truth['n_genres'])

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    # Overall comparison
    overall = compare_all_methods(no_pool, complete_pool, hbm, truth)

    # Stratified comparison
    stratified = stratified_comparison(no_pool, complete_pool, hbm, truth)

    # Genre recovery
    genre_validation = validate_genre_recovery(genre_est, truth)

    # Shrinkage analysis
    print("\n" + "="*80)
    print("SHRINKAGE ANALYSIS")
    print("="*80)
    shrinkage = compute_shrinkage_metrics(no_pool, hbm, truth)

    # Show examples: small vs large creators
    small_creators = shrinkage.nsmallest(5, 'n_total')[['creator_id', 'n_total', 'shrinkage_pct']]
    large_creators = shrinkage.nlargest(5, 'n_total')[['creator_id', 'n_total', 'shrinkage_pct']]

    print("\nShrinkage for smallest creators:")
    print(small_creators.to_string(index=False))

    print("\nShrinkage for largest creators:")
    print(large_creators.to_string(index=False))

    print(f"\nAverage shrinkage % by size:")
    print(f"  Small (n<100): {shrinkage[shrinkage['n_total'] < 100]['shrinkage_pct'].mean():.2%}")
    print(f"  Medium (100-500): {shrinkage[(shrinkage['n_total'] >= 100) & (shrinkage['n_total'] < 500)]['shrinkage_pct'].mean():.2%}")
    print(f"  Large (n>500): {shrinkage[shrinkage['n_total'] >= 500]['shrinkage_pct'].mean():.2%}")
