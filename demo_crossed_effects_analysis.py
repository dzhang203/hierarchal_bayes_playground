"""
Comprehensive crossed-effects analysis with stratified performance by sample size.

This demo:
1. Generates crossed-effects data (genre × ARPU with higher variance)
2. Estimates ITEs with Borusyak
3. Compares grouping strategies: genre-only, ARPU-only, genre×ARPU crossed
4. Shows HBM benefits more for small-sample creators
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generation_staggered_crossed import (
    generate_staggered_adoption_data_crossed,
    summarize_crossed_data
)
from src.borusyak_estimator import (
    borusyak_imputation_estimator,
    validate_borusyak_estimates
)
from src.hierarchical_model_robust import (
    fit_hierarchical_model_robust,
    extract_hbm_estimates_robust
)
from src.hierarchical_model_crossed import (
    fit_crossed_hbm,
    extract_crossed_estimates,
    extract_cell_effects
)
from src.validation import compute_mse, compute_coverage, compute_avg_interval_width
from src.grouping_selection import create_grouping


def stratified_metrics(
    estimates: pd.DataFrame,
    true_effects: pd.Series,
    strata_col: str,
    strata_labels: dict = None
) -> pd.DataFrame:
    """
    Compute metrics stratified by a column (e.g., sample size bins).

    Parameters
    ----------
    estimates : pd.DataFrame
        Estimates with columns: creator_id, effect_hat, ci_lower, ci_upper
    true_effects : pd.Series
        True treatment effects (indexed by creator_id)
    strata_col : str
        Column to stratify by
    strata_labels : dict, optional
        Labels for strata values

    Returns
    -------
    pd.DataFrame
        Metrics by stratum
    """
    results = []

    for stratum in sorted(estimates[strata_col].unique()):
        subset = estimates[estimates[strata_col] == stratum]
        creator_ids = subset['creator_id'].values

        # Handle dict properly
        if isinstance(true_effects, dict):
            true_subset = np.array([true_effects[cid] for cid in creator_ids])
        elif hasattr(true_effects, 'loc'):
            true_subset = true_effects.loc[creator_ids].values
        else:
            true_subset = true_effects[creator_ids]

        mse = compute_mse(subset['effect_hat'].values, true_subset)
        coverage = compute_coverage(
            subset['ci_lower'].values,
            subset['ci_upper'].values,
            true_subset
        )
        width = compute_avg_interval_width(
            subset['ci_lower'].values,
            subset['ci_upper'].values
        )

        label = strata_labels.get(stratum, str(stratum)) if strata_labels else str(stratum)

        results.append({
            'Stratum': label,
            'N': len(subset),
            'MSE': mse,
            'Coverage': coverage,
            'Avg Width': width
        })

    return pd.DataFrame(results)


def main():
    print("="*80)
    print("CROSSED-EFFECTS ANALYSIS WITH STRATIFICATION")
    print("="*80)
    print()

    # =========================================================================
    # Step 1: Generate crossed-effects data
    # =========================================================================

    print("[1/7] Generating crossed-effects data...")
    df, truth = generate_staggered_adoption_data_crossed(
        n_genres=5,
        n_arpu_quintiles=5,
        n_creators_per_cell=8,  # 200 total creators
        n_weeks=52,
        noise_scale=2.0,  # High variance (realistic)
        seed=42
    )
    summarize_crossed_data(df, truth)
    print()

    # =========================================================================
    # Step 2: Estimate ITEs with Borusyak
    # =========================================================================

    print("[2/7] Estimating ITEs with Borusyak...")
    ite_estimates, diagnostics = borusyak_imputation_estimator(
        df,
        never_treated_ids=truth['never_treated'],
        verbose=True
    )

    # Validate
    print()
    print("Borusyak validation:")
    borusyak_metrics = validate_borusyak_estimates(
        ite_estimates,
        truth['creator_effects'],
        verbose=True
    )
    print()

    # =========================================================================
    # Step 3: Add sample size bins for stratification
    # =========================================================================

    print("[3/7] Creating sample size bins...")

    # Post-treatment periods as proxy for sample size
    ite_estimates['n_post_bin'] = pd.cut(
        ite_estimates['n_post'],
        bins=[0, 15, 25, 100],
        labels=['Small (≤15 weeks)', 'Medium (16-25 weeks)', 'Large (>25 weeks)']
    )

    print("  Sample size distribution:")
    print(ite_estimates.groupby('n_post_bin').size())
    print()

    # =========================================================================
    # Step 4: Fit HBM with GENRE-ONLY grouping
    # =========================================================================

    print("[4/7] Fitting HBM with GENRE-ONLY grouping...")
    genre_data = create_grouping(
        ite_estimates,
        truth['creator_features'],
        'genre_idx'
    )

    idata_genre = fit_hierarchical_model_robust(
        genre_data,
        n_genres=5,
        draws=1000,
        tune=1000,
        chains=2,
        random_seed=42,
        target_accept=0.95
    )

    hbm_genre_estimates = extract_hbm_estimates_robust(idata_genre, genre_data)
    hbm_genre_estimates['n_post_bin'] = genre_data['n_post_bin'].values
    print("  ✓ Genre-only HBM complete!")
    print()

    # =========================================================================
    # Step 5: Fit HBM with ARPU-ONLY grouping
    # =========================================================================

    print("[5/7] Fitting HBM with ARPU-ONLY grouping...")
    arpu_data = create_grouping(
        ite_estimates,
        truth['creator_features'],
        'arpu_quintile'
    )

    idata_arpu = fit_hierarchical_model_robust(
        arpu_data,
        n_genres=5,  # Actually 5 ARPU quintiles, but reusing same param
        draws=1000,
        tune=1000,
        chains=2,
        random_seed=43
    )

    hbm_arpu_estimates = extract_hbm_estimates_robust(idata_arpu, arpu_data)
    hbm_arpu_estimates['n_post_bin'] = arpu_data['n_post_bin'].values
    print("  ✓ ARPU-only HBM complete!")
    print()

    # =========================================================================
    # Step 6: Fit CROSSED HBM (genre × ARPU)
    # =========================================================================

    print("[6/7] Fitting CROSSED HBM (genre × ARPU)...")

    # Merge ITEs with features
    crossed_data = ite_estimates.merge(
        truth['creator_features'][['creator_id', 'genre_idx', 'arpu_quintile']],
        on='creator_id',
        how='left'
    )

    # Check for missing values before fitting
    if crossed_data[['effect_hat', 'se', 'genre_idx', 'arpu_quintile']].isnull().any().any():
        print("  ⚠️  WARNING: Found NaN values. Dropping...")
        crossed_data = crossed_data.dropna(subset=['effect_hat', 'se', 'genre_idx', 'arpu_quintile'])

    idata_crossed = fit_crossed_hbm(
        crossed_data,
        n_genres=5,
        n_arpu_quintiles=5,
        draws=1000,
        tune=1000,
        chains=2,
        random_seed=44,
        target_accept=0.95
    )

    hbm_crossed_estimates = extract_crossed_estimates(idata_crossed, crossed_data)
    hbm_crossed_estimates['n_post_bin'] = crossed_data['n_post_bin'].values
    cell_effects = extract_cell_effects(idata_crossed, n_genres=5, n_arpu_quintiles=5)
    print("  ✓ Crossed HBM complete!")
    print()

    # =========================================================================
    # Step 7: Compare results
    # =========================================================================

    print("[7/7] Comparing results...")
    print()

    # Get true effects for treated creators
    creator_ids = ite_estimates['creator_id'].values
    # Handle dict properly
    if isinstance(truth['creator_effects'], dict):
        true_treated = np.array([truth['creator_effects'][cid] for cid in creator_ids])
    else:
        true_treated = truth['creator_effects'][creator_ids]

    # Overall comparison
    print("="*80)
    print("OVERALL COMPARISON")
    print("="*80)
    print(f"{'Method':<25} {'MSE':<12} {'Coverage':<12} {'Avg CI Width':<12}")
    print("-"*80)

    # Borusyak
    bor_mse = compute_mse(ite_estimates['effect_hat'].values, true_treated)
    bor_cov = compute_coverage(
        ite_estimates['effect_hat'] - 1.96 * ite_estimates['se'],
        ite_estimates['effect_hat'] + 1.96 * ite_estimates['se'],
        true_treated
    )
    bor_width = compute_avg_interval_width(
        ite_estimates['effect_hat'] - 1.96 * ite_estimates['se'],
        ite_estimates['effect_hat'] + 1.96 * ite_estimates['se']
    )
    print(f"{'Borusyak (no pooling)':<25} {bor_mse:<12.4f} {bor_cov:<12.3f} {bor_width:<12.3f}")

    # Genre-only HBM
    genre_mse = compute_mse(hbm_genre_estimates['effect_hat'].values, true_treated)
    genre_cov = compute_coverage(
        hbm_genre_estimates['ci_lower'].values,
        hbm_genre_estimates['ci_upper'].values,
        true_treated
    )
    genre_width = compute_avg_interval_width(
        hbm_genre_estimates['ci_lower'].values,
        hbm_genre_estimates['ci_upper'].values
    )
    print(f"{'HBM (genre only)':<25} {genre_mse:<12.4f} {genre_cov:<12.3f} {genre_width:<12.3f}")

    # ARPU-only HBM
    arpu_mse = compute_mse(hbm_arpu_estimates['effect_hat'].values, true_treated)
    arpu_cov = compute_coverage(
        hbm_arpu_estimates['ci_lower'].values,
        hbm_arpu_estimates['ci_upper'].values,
        true_treated
    )
    arpu_width = compute_avg_interval_width(
        hbm_arpu_estimates['ci_lower'].values,
        hbm_arpu_estimates['ci_upper'].values
    )
    print(f"{'HBM (ARPU only)':<25} {arpu_mse:<12.4f} {arpu_cov:<12.3f} {arpu_width:<12.3f}")

    # Crossed HBM
    crossed_mse = compute_mse(hbm_crossed_estimates['effect_hat'].values, true_treated)
    crossed_cov = compute_coverage(
        hbm_crossed_estimates['ci_lower'].values,
        hbm_crossed_estimates['ci_upper'].values,
        true_treated
    )
    crossed_width = compute_avg_interval_width(
        hbm_crossed_estimates['ci_lower'].values,
        hbm_crossed_estimates['ci_upper'].values
    )
    print(f"{'HBM (genre × ARPU)':<25} {crossed_mse:<12.4f} {crossed_cov:<12.3f} {crossed_width:<12.3f}")

    print("="*80)
    print()

    # MSE improvements
    print("MSE Improvements over Borusyak:")
    print(f"  Genre-only:    {(1 - genre_mse/bor_mse)*100:>6.1f}%")
    print(f"  ARPU-only:     {(1 - arpu_mse/bor_mse)*100:>6.1f}%")
    print(f"  Genre × ARPU:  {(1 - crossed_mse/bor_mse)*100:>6.1f}%")
    print()

    # =========================================================================
    # Stratified analysis by sample size
    # =========================================================================

    print("="*80)
    print("STRATIFIED ANALYSIS: Performance by Sample Size")
    print("="*80)
    print()

    # Prepare data for stratified analysis
    ite_estimates['ci_lower'] = ite_estimates['effect_hat'] - 1.96 * ite_estimates['se']
    ite_estimates['ci_upper'] = ite_estimates['effect_hat'] + 1.96 * ite_estimates['se']

    # Borusyak stratified
    print("Borusyak (no pooling):")
    bor_strat = stratified_metrics(
        ite_estimates,
        truth['creator_effects'],
        'n_post_bin'
    )
    print(bor_strat.to_string(index=False))
    print()

    # Genre-only HBM stratified
    print("HBM (genre only):")
    genre_strat = stratified_metrics(
        hbm_genre_estimates,
        truth['creator_effects'],
        'n_post_bin'
    )
    print(genre_strat.to_string(index=False))
    print()

    # Crossed HBM stratified
    print("HBM (genre × ARPU crossed):")
    crossed_strat = stratified_metrics(
        hbm_crossed_estimates,
        truth['creator_effects'],
        'n_post_bin'
    )
    print(crossed_strat.to_string(index=False))
    print()

    # =========================================================================
    # Key findings
    # =========================================================================

    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()

    # Find best overall method
    best_overall = min([
        ('Borusyak', bor_mse),
        ('Genre-only', genre_mse),
        ('ARPU-only', arpu_mse),
        ('Genre×ARPU', crossed_mse)
    ], key=lambda x: x[1])

    print(f"✓ Best overall method: {best_overall[0]} (MSE={best_overall[1]:.4f})")
    print()

    # Compare small vs large creators
    small_idx = bor_strat[bor_strat['Stratum'] == 'Small (≤15 weeks)'].index[0]
    large_idx = bor_strat[bor_strat['Stratum'] == 'Large (>25 weeks)'].index[0]

    bor_small_mse = bor_strat.loc[small_idx, 'MSE']
    bor_large_mse = bor_strat.loc[large_idx, 'MSE']
    crossed_small_mse = crossed_strat.loc[small_idx, 'MSE']
    crossed_large_mse = crossed_strat.loc[large_idx, 'MSE']

    small_improvement = (1 - crossed_small_mse / bor_small_mse) * 100
    large_improvement = (1 - crossed_large_mse / bor_large_mse) * 100

    print("✓ HBM benefits by sample size:")
    print(f"  Small creators (≤15 weeks):  {small_improvement:>6.1f}% MSE reduction")
    print(f"  Large creators (>25 weeks):  {large_improvement:>6.1f}% MSE reduction")
    print()

    if small_improvement > large_improvement:
        print("  → HBM helps MORE for small-sample creators! ✓")
    else:
        print("  → HBM benefits are similar across sample sizes")
    print()

    # Crossed vs genre-only
    if crossed_mse < genre_mse:
        improvement = (1 - crossed_mse / genre_mse) * 100
        print(f"✓ Crossed effects improve over genre-only by {improvement:.1f}%")
    else:
        print("✓ Genre-only performs similarly to crossed effects")
    print()

    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("Cell-level effects (genre × ARPU):")
    print(cell_effects.pivot(index='genre_idx', columns='arpu_quintile', values='effect_mean'))


if __name__ == "__main__":
    main()
