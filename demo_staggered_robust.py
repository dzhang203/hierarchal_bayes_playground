"""
Staggered adoption demo with ROBUST HBM fitting.

This uses the robust HBM version that handles small SEs from Borusyak better.
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
from src.hierarchical_model_robust import (
    fit_hierarchical_model_robust,
    extract_hbm_estimates_robust
)
from src.validation import compute_mse, compute_coverage, compute_avg_interval_width
from src.grouping_selection import create_grouping


def main():
    print("="*80)
    print("STAGGERED ADOPTION DEMO: Borusyak → Robust HBM")
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

    # Apply ROBUST HBM with genre grouping
    print("[4/5] Applying ROBUST HBM with genre grouping...")
    grouped_data = create_grouping(ite_estimates, truth['creator_features'], 'genre_idx')

    # Fit with robust version
    print("  Fitting robust hierarchical model...")
    idata = fit_hierarchical_model_robust(
        grouped_data,
        n_genres=5,
        draws=1000,
        tune=1000,
        chains=2,
        random_seed=42,
        target_accept=0.95,
        min_se=0.05,  # Floor for very small SEs
        use_informative_prior=True  # Use data-driven prior
    )

    hbm_estimates = extract_hbm_estimates_robust(idata, grouped_data)

    print("  ✓ Robust HBM fitted successfully!")

    # Compare
    print()
    print("[5/5] Comparing Borusyak vs. Robust HBM...")

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
    print("RESULTS: BORUSYAK vs. ROBUST HBM")
    print("="*80)
    print(f"{'Metric':<20} {'Borusyak':<15} {'HBM':<15} {'Improvement':<15}")
    print("-"*80)
    print(f"{'MSE':<20} {borusyak_mse:<15.4f} {hbm_mse:<15.4f} {(1-hbm_mse/borusyak_mse)*100:>14.1f}%")
    print(f"{'Coverage':<20} {borusyak_coverage:<15.3f} {hbm_coverage:<15.3f} {'':<15}")
    print(f"{'Avg CI Width':<20} {borusyak_width:<15.3f} {hbm_width:<15.3f} {(1-hbm_width/borusyak_width)*100:>14.1f}%")
    print("="*80)
    print()

    improvement_mse = (1 - hbm_mse / borusyak_mse) * 100
    improvement_width = (1 - hbm_width / borusyak_width) * 100

    print(f"✓ Robust HBM achieves {improvement_mse:.1f}% lower MSE than Borusyak alone")
    print(f"✓ Confidence intervals are {abs(improvement_width):.1f}% {'narrower' if improvement_width > 0 else 'wider'}")
    print(f"✓ Coverage: {hbm_coverage:.1%} (target: 95%)")
    print()

    # Check if HBM is adding value
    if hbm_mse < borusyak_mse and abs(hbm_coverage - 0.95) < abs(borusyak_coverage - 0.95):
        print("="*80)
        print("✅ HBM IS ADDING VALUE!")
        print("="*80)
        print("The two-stage approach (Borusyak → HBM) successfully:")
        print("  1. Handles staggered adoption (causal identification)")
        print("  2. Improves precision via hierarchical pooling (lower MSE)")
        print("  3. Maintains or improves coverage calibration")
    elif hbm_mse > borusyak_mse:
        print("="*80)
        print("⚠️  HBM NOT IMPROVING MSE")
        print("="*80)
        print("Possible reasons:")
        print("  - Treatment effects don't follow genre structure")
        print("  - Sample sizes are large enough that pooling doesn't help")
        print("  - Borusyak SEs are already very precise")
    else:
        print("="*80)
        print("✓ MIXED RESULTS")
        print("="*80)
        print("HBM shows some improvement but may not be dramatic.")

    print()
    print("="*80)
    print("DEMO COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
