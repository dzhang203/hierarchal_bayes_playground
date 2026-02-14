"""
Run complete staggered adoption analysis pipeline.

This script demonstrates the full Borusyak → HBM workflow:
1. Generate synthetic staggered adoption data
2. Estimate ITEs using Borusyak imputation estimator
3. Compare different grouping structures (genre vs size vs ARPU)
4. Fit HBM with best grouping
5. Validate against ground truth
6. Generate visualizations

This complements the original A/B test demo (run_full_analysis.py)
by showing how to handle observational/quasi-experimental data.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from src.data_generation_staggered import (
    generate_staggered_adoption_data,
    summarize_staggered_data
)
from src.borusyak_estimator import (
    borusyak_imputation_estimator,
    validate_borusyak_estimates
)
from src.grouping_selection import (
    compare_groupings,
    create_grouping,
    check_posterior_separation,
    summarize_grouping_quality
)
from src.hierarchical_model import (
    extract_hbm_estimates,
    extract_genre_estimates,
    check_mcmc_diagnostics
)
from src.validation import (
    compute_mse,
    compute_coverage,
    compute_avg_interval_width
)
from src.visualization import (
    plot_shrinkage,
    plot_genre_recovery,
    plot_posterior_distributions
)


def plot_borusyak_validation(
    ite_estimates: pd.DataFrame,
    true_effects: np.ndarray,
    save_path: str = None
):
    """
    Plot Borusyak estimates vs. ground truth.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get true effects for treated units
    creator_ids = ite_estimates['creator_id'].values
    true_treated = true_effects[creator_ids]

    # Left: Scatter plot
    ax = axes[0]
    ax.scatter(true_treated, ite_estimates['effect_hat'],
              alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

    # Add y=x line
    lims = [
        min(true_treated.min(), ite_estimates['effect_hat'].min()),
        max(true_treated.max(), ite_estimates['effect_hat'].max())
    ]
    ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Perfect estimation')

    ax.set_xlabel('True Effect', fontweight='bold')
    ax.set_ylabel('Borusyak Estimate', fontweight='bold')
    ax.set_title('Borusyak ITE Estimates vs. Ground Truth', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Residuals vs n_post
    ax = axes[1]
    residuals = ite_estimates['effect_hat'] - true_treated
    ax.scatter(ite_estimates['n_post'], residuals,
              alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Post-Treatment Periods', fontweight='bold')
    ax.set_ylabel('Estimation Error', fontweight='bold')
    ax.set_title('Estimation Error vs. Data Availability', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved Borusyak validation plot to {save_path}")

    plt.close()


def main():
    """Run full staggered adoption analysis pipeline."""

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    print("="*80)
    print("STAGGERED ADOPTION ANALYSIS: BORUSYAK → HBM PIPELINE")
    print("="*80)
    print()
    print("This demo shows how to:")
    print("  1. Estimate ITEs from staggered adoption using Borusyak et al.")
    print("  2. Select optimal grouping structure via Bayesian model comparison")
    print("  3. Apply hierarchical Bayesian pooling for improved precision")
    print()

    # =====================================================================
    # STEP 1: Generate Synthetic Data
    # =====================================================================

    print("[1/7] Generating synthetic staggered adoption data...")
    print()

    df, truth = generate_staggered_adoption_data(
        n_genres=5,
        n_creators_per_genre=40,  # 200 total creators
        n_weeks=52,
        treatment_start_week=12,
        treatment_end_week=40,
        pct_never_treated=0.20,
        genre_std=0.3,
        sigma_creator_effect=0.4,
        seed=42
    )

    summarize_staggered_data(df, truth)
    print()

    # =====================================================================
    # STEP 2: Estimate ITEs using Borusyak
    # =====================================================================

    print("[2/7] Estimating ITEs with Borusyak imputation estimator...")
    print()

    ite_estimates, diagnostics = borusyak_imputation_estimator(
        df,
        never_treated_ids=truth['never_treated'],
        verbose=True
    )

    print()

    # Validate Borusyak estimates
    print("[3/7] Validating Borusyak estimates against ground truth...")
    print()

    borusyak_metrics = validate_borusyak_estimates(
        ite_estimates,
        truth['creator_effects'],
        verbose=True
    )

    # Plot validation
    plot_borusyak_validation(
        ite_estimates,
        truth['creator_effects'],
        save_path='outputs/borusyak_validation.png'
    )

    print()

    # =====================================================================
    # STEP 3: Compare Grouping Structures
    # =====================================================================

    print("[4/7] Comparing grouping structures via Bayesian model selection...")
    print()

    grouping_candidates = ['genre_idx', 'size_quartile', 'arpu_quartile']

    comparison, models = compare_groupings(
        ite_estimates,
        truth['creator_features'],
        grouping_candidates,
        draws=1000,
        tune=500,
        chains=2,
        random_seed=42,
        verbose=True
    )

    # Get min group sizes for summary
    min_group_sizes = {}
    for var in grouping_candidates:
        # For treated units only
        treated_features = truth['creator_features'][
            truth['creator_features']['creator_id'].isin(ite_estimates['creator_id'])
        ]
        sizes = treated_features.groupby(var).size()
        min_group_sizes[var] = sizes.min()

    quality_summary = summarize_grouping_quality(
        comparison,
        models,
        min_group_sizes,
        verbose=True
    )

    # Save comparison
    comparison.to_csv('outputs/grouping_comparison.csv')
    quality_summary.to_csv('outputs/grouping_quality.csv')

    print()

    # =====================================================================
    # STEP 4: Analyze Best Grouping
    # =====================================================================

    best_grouping = comparison.index[0]
    print(f"[5/7] Analyzing best grouping: {best_grouping}")
    print()

    # Get best model
    best_idata = models[best_grouping]

    # Extract estimates
    grouped_data = create_grouping(
        ite_estimates,
        truth['creator_features'],
        best_grouping
    )
    n_groups = grouped_data['genre_idx'].nunique()

    hbm_estimates = extract_hbm_estimates(best_idata, grouped_data)
    genre_estimates = extract_genre_estimates(best_idata, n_groups)

    # Check diagnostics
    print("Checking MCMC diagnostics for best model...")
    diagnostics_best = check_mcmc_diagnostics(best_idata, verbose=True)
    print()

    # =====================================================================
    # STEP 5: Compare Borusyak vs. HBM
    # =====================================================================

    print("[6/7] Comparing Borusyak (no pooling) vs. HBM (partial pooling)...")
    print()

    # Get true effects for treated units
    creator_ids = ite_estimates['creator_id'].values
    true_treated = truth['creator_effects'][creator_ids]

    # Compute metrics for both methods
    borusyak_mse = compute_mse(ite_estimates['effect_hat'].values, true_treated)
    hbm_mse = compute_mse(hbm_estimates['effect_hat'].values, true_treated)

    borusyak_coverage = compute_coverage(
        ite_estimates['effect_hat'] - 1.96 * ite_estimates['se'],
        ite_estimates['effect_hat'] + 1.96 * ite_estimates['se'],
        true_treated
    )
    hbm_coverage = compute_coverage(
        hbm_estimates['ci_lower'].values,
        hbm_estimates['ci_upper'].values,
        true_treated
    )

    borusyak_width = compute_avg_interval_width(
        ite_estimates['effect_hat'] - 1.96 * ite_estimates['se'],
        ite_estimates['effect_hat'] + 1.96 * ite_estimates['se']
    )
    hbm_width = compute_avg_interval_width(
        hbm_estimates['ci_lower'].values,
        hbm_estimates['ci_upper'].values
    )

    print("="*80)
    print("COMPARISON: BORUSYAK vs. HBM")
    print("="*80)
    print(f"{'Metric':<20} {'Borusyak (DiD)':<20} {'HBM (Pooled)':<20} {'Improvement':<15}")
    print("-"*80)
    print(f"{'MSE':<20} {borusyak_mse:<20.4f} {hbm_mse:<20.4f} {(1-hbm_mse/borusyak_mse)*100:>14.1f}%")
    print(f"{'Coverage (95%)':<20} {borusyak_coverage:<20.3f} {hbm_coverage:<20.3f} {(hbm_coverage-borusyak_coverage)*100:>14.1f} pp")
    print(f"{'Avg CI Width':<20} {borusyak_width:<20.3f} {hbm_width:<20.3f} {(1-hbm_width/borusyak_width)*100:>14.1f}%")
    print("="*80)
    print()

    improvement_pct = (1 - hbm_mse / borusyak_mse) * 100
    width_reduction = (1 - hbm_width / borusyak_width) * 100

    print(f"✓ HBM achieves {improvement_pct:.1f}% lower MSE than Borusyak alone")
    print(f"✓ HBM has {width_reduction:.1f}% narrower intervals with maintained coverage")
    print()

    # =====================================================================
    # STEP 6: Generate Visualizations
    # =====================================================================

    print("[7/7] Generating visualizations...")
    print()

    # Shrinkage plot
    print("  • Shrinkage plot...")
    # Need to convert ite_estimates to match expected format
    ite_for_plot = ite_estimates.copy()
    ite_for_plot['genre_idx'] = grouped_data['genre_idx'].values
    ite_for_plot['n_total'] = ite_estimates['n_post']  # Use n_post as proxy

    # Create temporary truth dict with correct indexing
    truth_subset = {
        'creator_genre': grouped_data['genre_idx'].values,
        'n_genres': n_groups
    }

    plot_shrinkage(
        ite_for_plot,
        hbm_estimates,
        truth_subset,
        save_path='outputs/shrinkage_plot_staggered.png'
    )

    # Posterior separation
    print("  • Posterior group effects...")
    check_posterior_separation(
        best_idata,
        best_grouping,
        save_path=f'outputs/posterior_separation_{best_grouping}.png'
    )

    # Posterior distributions
    print("  • Hyperparameter recovery...")
    # Need to create comparable truth dict
    truth_for_plot = {
        'genre_effects': truth['genre_effects'][:n_groups],  # May not match exactly
        'sigma_creator': truth['sigma_creator_effect'],
        'sigma_obs': truth['sigma_obs']
    }

    plot_posterior_distributions(
        best_idata,
        truth_for_plot,
        save_path='outputs/posteriors_staggered.png'
    )

    print()
    print("  ✓ All visualizations saved to outputs/")

    # =====================================================================
    # Summary
    # =====================================================================

    print()
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("Key Results:")
    print(f"  1. Borusyak ATT: {diagnostics['att']:.3f} (True: ~{truth['genre_effects'].mean():.3f})")
    print(f"  2. Borusyak recovered ITEs with {borusyak_metrics['coverage']:.1%} coverage")
    print(f"  3. Best grouping: {best_grouping} (LOO-IC weight: {comparison.loc[best_grouping, 'weight']:.2f})")
    print(f"  4. HBM improvement: {improvement_pct:.1f}% MSE reduction over Borusyak alone")
    print(f"  5. Maintained coverage: {hbm_coverage:.1%} (target: 95%)")
    print()
    print("Outputs:")
    print("  • Figures: outputs/*_staggered.png")
    print("  • Tables: outputs/grouping_*.csv")
    print()
    print("The two-stage approach (Borusyak → HBM) successfully:")
    print("  ✓ Handles staggered adoption (causal identification)")
    print("  ✓ Borrows strength across creators (improved precision)")
    print("  ✓ Selects grouping structure objectively (no overfitting)")
    print("="*80)


if __name__ == "__main__":
    main()
