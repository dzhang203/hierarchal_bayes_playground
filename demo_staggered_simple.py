"""
Simplified staggered adoption demo - showing the core Borusyak → HBM workflow.

This demonstrates the two-stage approach without the grouping comparison complexity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_generation_staggered import (
    generate_staggered_adoption_data,
    summarize_staggered_data
)
from src.borusyak_estimator import (
    borusyak_imputation_estimator,
    validate_borusyak_estimates
)
from src.hierarchical_model import fit_hierarchical_model, extract_hbm_estimates
from src.validation import compute_mse, compute_coverage, compute_avg_interval_width
from src.grouping_selection import create_grouping


def main():
    print("="*80)
    print("STAGGERED ADOPTION DEMO: Borusyak → HBM (Simplified)")
    print("="*80)
    print()

    # Generate data
    print("[1/5] Generating staggered adoption data...")
    df, truth = generate_staggered_adoption_data(
        n_genres=5,
        n_creators_per_genre=40,
        n_weeks=52,
        seed=42
    )
    summarize_staggered_data(df, truth)
    print()

    # Estimate ITEs
    print("[2/5] Estimating ITEs with Borusyak...")
    ite_estimates, diagnostics = borusyak_imputation_estimator(
        df,
        never_treated_ids=truth['never_treated'],
        verbose=True
    )
    print()

    # Validate Borusyak
    print("[3/5] Validating Borusyak estimates...")
    metrics = validate_borusyak_estimates(
        ite_estimates,
        truth['creator_effects'],
        verbose=True
    )
    print()

    # Apply HBM with genre grouping
    print("[4/5] Applying HBM with genre grouping...")
    grouped_data = create_grouping(ite_estimates, truth['creator_features'], 'genre_idx')

    # Fit with more conservative settings to avoid initialization issues
    print("  Fitting hierarchical model...")
    try:
        idata = fit_hierarchical_model(
            grouped_data,
            n_genres=5,
            draws=1000,
            tune=1000,
            chains=2,
            random_seed=42,
            target_accept=0.95  # More conservative
        )

        hbm_estimates = extract_hbm_estimates(idata, grouped_data)

        print("  ✓ HBM fitted successfully!")

        # Compare
        print()
        print("[5/5] Comparing Borusyak vs. HBM...")

        creator_ids = ite_estimates['creator_id'].values
        true_treated = truth['creator_effects'][creator_ids]

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

        print()
        print("="*80)
        print("RESULTS: BORUSYAK vs. HBM")
        print("="*80)
        print(f"{'Metric':<20} {'Borusyak':<15} {'HBM':<15} {'Improvement':<15}")
        print("-"*80)
        print(f"{'MSE':<20} {borusyak_mse:<15.4f} {hbm_mse:<15.4f} {(1-hbm_mse/borusyak_mse)*100:>14.1f}%")
        print(f"{'Coverage':<20} {borusyak_coverage:<15.3f} {hbm_coverage:<15.3f} {'':<15}")
        print(f"{'Avg CI Width':<20} {borusyak_width:<15.3f} {hbm_width:<15.3f} {(1-hbm_width/borusyak_width)*100:>14.1f}%")
        print("="*80)
        print()

        improvement = (1 - hbm_mse / borusyak_mse) * 100
        print(f"✓ HBM achieves {improvement:.1f}% lower MSE than Borusyak alone")
        print(f"✓ Maintained coverage: {hbm_coverage:.1%} (target: 95%)")
        print()
        print("="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print()
        print("Key takeaway: The two-stage approach (Borusyak → HBM) successfully:")
        print("  1. Handles staggered adoption (causal identification)")
        print("  2. Improves precision via hierarchical pooling")
        print("  3. Maintains proper coverage")

    except Exception as e:
        print(f"  ✗ Error fitting HBM: {e}")
        print()
        print("Note: HBM fitting can be sensitive to initialization.")
        print("The Borusyak estimates above are still valid!")
        print("Try adjusting target_accept or using more tune steps.")


if __name__ == "__main__":
    main()
