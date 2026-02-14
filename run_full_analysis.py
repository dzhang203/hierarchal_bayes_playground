"""
Run complete hierarchical Bayesian modeling analysis.

This script:
1. Generates synthetic data
2. Computes frequentist baselines
3. Fits hierarchical Bayesian model
4. Validates and compares all methods
5. Generates all visualizations
6. Saves results to outputs/
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generation import generate_experiment_data, summarize_data
from src.frequentist import no_pooling_estimates, complete_pooling_estimates
from src.hierarchical_model import (
    prepare_creator_summaries,
    fit_hierarchical_model,
    extract_hbm_estimates,
    extract_genre_estimates,
    check_mcmc_diagnostics
)
from src.validation import (
    compare_all_methods,
    stratified_comparison,
    compute_shrinkage_metrics,
    validate_genre_recovery
)
from src.visualization import (
    plot_shrinkage,
    plot_mse_comparison,
    plot_coverage_vs_width,
    plot_individual_creators,
    plot_genre_recovery,
    plot_posterior_distributions,
    plot_trace
)


def main():
    """Run full analysis pipeline."""

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    print("=" * 80)
    print("HIERARCHICAL BAYESIAN MODELING FOR CREATOR EXPERIMENTS")
    print("=" * 80)

    # 1. Generate data
    print("\n[1/7] Generating synthetic data...")
    df, truth = generate_experiment_data(seed=42)
    summarize_data(df, truth)

    # 2. Frequentist baselines
    print("\n[2/7] Computing frequentist baselines...")
    no_pool = no_pooling_estimates(df)
    print(f"  ✓ No-pooling: {len(no_pool)} creators")

    complete_pool = complete_pooling_estimates(df)
    print(f"  ✓ Complete-pooling: {len(complete_pool)} creators")

    # 3. Fit HBM
    print("\n[3/7] Fitting hierarchical Bayesian model...")
    print("  (This will take 2-5 minutes)")

    summaries = prepare_creator_summaries(df)
    idata = fit_hierarchical_model(
        summaries,
        n_genres=truth['n_genres'],
        draws=2000,
        tune=1000,
        chains=4,
        random_seed=42
    )

    # 4. Check diagnostics
    print("\n[4/7] Checking MCMC diagnostics...")
    diagnostics = check_mcmc_diagnostics(idata, verbose=True)

    if not (diagnostics['all_rhat_ok'] and diagnostics['all_ess_ok'] and diagnostics['n_divergences'] == 0):
        print("\n⚠ WARNING: MCMC diagnostics indicate potential sampling issues.")
        print("Results may still be useful, but interpret with caution.")

    # 5. Extract estimates
    print("\n[5/7] Extracting posterior estimates...")
    hbm = extract_hbm_estimates(idata, summaries)
    genre_est = extract_genre_estimates(idata, truth['n_genres'])
    print(f"  ✓ Extracted estimates for {len(hbm)} creators")
    print(f"  ✓ Extracted estimates for {len(genre_est)} genres")

    # 6. Validation
    print("\n[6/7] Comparing methods and validating results...")
    overall = compare_all_methods(no_pool, complete_pool, hbm, truth, verbose=True)

    print("\n" + "="*80)
    stratified = stratified_comparison(no_pool, complete_pool, hbm, truth, verbose=True)

    print("\n" + "="*80)
    genre_validation = validate_genre_recovery(genre_est, truth, verbose=True)

    # 7. Generate visualizations
    print("\n[7/7] Generating visualizations...")

    plots = [
        ("Shrinkage plot", lambda: plot_shrinkage(no_pool, hbm, truth, save_path='outputs/shrinkage_plot.png')),
        ("MSE comparison", lambda: plot_mse_comparison(no_pool, complete_pool, hbm, truth, save_path='outputs/mse_comparison.png')),
        ("Coverage vs width", lambda: plot_coverage_vs_width(no_pool, complete_pool, hbm, truth, save_path='outputs/coverage_vs_width.png')),
        ("Individual creators", lambda: plot_individual_creators(no_pool, hbm, truth, save_path='outputs/individual_creators.png')),
        ("Genre recovery", lambda: plot_genre_recovery(genre_est, truth, save_path='outputs/genre_recovery.png')),
        ("Posterior distributions", lambda: plot_posterior_distributions(idata, truth, save_path='outputs/posteriors.png')),
        ("Trace plots", lambda: plot_trace(idata, save_path='outputs/trace_plots.png'))
    ]

    for plot_name, plot_func in plots:
        print(f"  • {plot_name}...")
        plot_func()
        plt.close('all')  # Close to save memory

    print(f"\n  ✓ All plots saved to outputs/")

    # Save results
    print("\nSaving results...")

    comparison_df = pd.DataFrame({
        'creator_id': no_pool['creator_id'],
        'genre': no_pool['genre'],
        'n_total': no_pool['n_total'],
        'true_effect': truth['creator_effects'],
        'no_pool_est': no_pool['effect_hat'],
        'no_pool_se': no_pool['se'],
        'complete_pool_est': complete_pool['effect_hat'],
        'hbm_est': hbm['effect_hat'],
        'hbm_se': hbm['se']
    })

    comparison_df.to_csv('outputs/comparison_results.csv', index=False)
    overall.to_csv('outputs/overall_metrics.csv', index=False)
    genre_validation.to_csv('outputs/genre_recovery.csv', index=False)

    print("  ✓ Results saved to outputs/")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey Results:")
    print(f"  • HBM MSE: {overall[overall['Method'] == 'HBM (Partial Pooling)']['MSE'].values[0]:.4f}")
    print(f"  • No-pooling MSE: {overall[overall['Method'] == 'No Pooling']['MSE'].values[0]:.4f}")
    print(f"  • HBM improvement: {((overall[overall['Method'] == 'No Pooling']['MSE'].values[0] - overall[overall['Method'] == 'HBM (Partial Pooling)']['MSE'].values[0]) / overall[overall['Method'] == 'No Pooling']['MSE'].values[0] * 100):.1f}% reduction in MSE")

    print(f"\n  • HBM Coverage: {overall[overall['Method'] == 'HBM (Partial Pooling)']['Coverage'].values[0]:.3f}")
    print(f"  • HBM Avg CI Width: {overall[overall['Method'] == 'HBM (Partial Pooling)']['Avg CI Width'].values[0]:.3f}")

    print("\nOutputs:")
    print("  • Figures: outputs/*.png")
    print("  • Data: outputs/*.csv")
    print("  • Notebook: notebooks/walkthrough.ipynb")

    print("\nNext step: Open notebooks/walkthrough.ipynb for interactive analysis!")
    print("=" * 80)


if __name__ == "__main__":
    main()
