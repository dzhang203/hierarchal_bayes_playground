"""
Calibrated Borusyak estimator with improved standard errors.

The standard Borusyak implementation underestimates SEs with high noise
because it doesn't account for TWFE estimation uncertainty.

This version adds an empirical calibration factor based on coverage validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Set
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.borusyak_estimator import borusyak_imputation_estimator


def borusyak_imputation_estimator_calibrated(
    df: pd.DataFrame,
    never_treated_ids: Set[int],
    calibration_factor: float = 1.5,
    verbose: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Borusyak estimator with calibrated (inflated) standard errors.

    The base Borusyak estimator underestimates SEs because it only captures
    within-cohort variation, not uncertainty in TWFE estimates.

    Empirical findings:
    - noise_scale=1.0 → 75% coverage (should be 95%) → inflation needed: ~1.3x
    - noise_scale=2.0 → 62% coverage (should be 95%) → inflation needed: ~1.5x
    - noise_scale=3.0 → 59% coverage (should be 95%) → inflation needed: ~1.6x

    This function applies a calibration factor to inflate SEs.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    never_treated_ids : Set[int]
        Never-treated creator IDs
    calibration_factor : float
        Multiplier for standard errors (default: 1.5 for noise_scale≈2.0)
        Recommendations:
        - Low noise environment: 1.3
        - Medium noise (typical revenue): 1.5
        - High noise: 1.6-1.8
    verbose : bool
        Print details
    **kwargs
        Additional arguments passed to borusyak_imputation_estimator

    Returns
    -------
    ite_estimates : pd.DataFrame
        ITEs with calibrated SEs
    diagnostics : dict
        Diagnostics from Borusyak estimation
    """
    # Run standard Borusyak
    ite_estimates, diagnostics = borusyak_imputation_estimator(
        df,
        never_treated_ids,
        verbose=verbose,
        **kwargs
    )

    # Inflate standard errors
    ite_estimates['se_uncalibrated'] = ite_estimates['se'].copy()
    ite_estimates['se'] = ite_estimates['se'] * calibration_factor

    if verbose:
        print()
        print("="*70)
        print("CALIBRATED STANDARD ERRORS")
        print("="*70)
        print(f"Calibration factor: {calibration_factor}x")
        print(f"SE range before: [{ite_estimates['se_uncalibrated'].min():.3f}, {ite_estimates['se_uncalibrated'].max():.3f}]")
        print(f"SE range after:  [{ite_estimates['se'].min():.3f}, {ite_estimates['se'].max():.3f}]")
        print()
        print("Why calibration is needed:")
        print("  • Borusyak SE only captures within-cohort variation")
        print("  • Doesn't account for TWFE estimation uncertainty")
        print("  • With high noise, TWFE estimates are uncertain")
        print("  • Results in under-coverage (60-75% instead of 95%)")
        print("="*70)
        print()

    # Update diagnostics
    diagnostics['calibration_factor'] = calibration_factor
    diagnostics['se_calibrated'] = True

    return ite_estimates, diagnostics


def validate_calibration(
    df: pd.DataFrame,
    truth: Dict,
    calibration_factors: list = [1.0, 1.3, 1.5, 1.7, 2.0],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Test different calibration factors to find optimal coverage.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    truth : dict
        Ground truth (must have 'never_treated' and 'creator_effects')
    calibration_factors : list
        Factors to test
    verbose : bool
        Print results

    Returns
    -------
    pd.DataFrame
        Results for each calibration factor
    """
    from src.borusyak_estimator import validate_borusyak_estimates

    results = []

    for factor in calibration_factors:
        ites, _ = borusyak_imputation_estimator_calibrated(
            df,
            truth['never_treated'],
            calibration_factor=factor,
            verbose=False
        )

        metrics = validate_borusyak_estimates(
            ites,
            truth['creator_effects'],
            verbose=False
        )

        results.append({
            'Calibration Factor': factor,
            'Coverage': metrics['coverage'],
            'RMSE': metrics['rmse'],
            'Avg SE': ites['se'].mean()
        })

    results_df = pd.DataFrame(results)

    if verbose:
        print("="*70)
        print("CALIBRATION VALIDATION")
        print("="*70)
        print(results_df.to_string(index=False))
        print()

        # Find closest to 95%
        results_df['dist_from_95'] = abs(results_df['Coverage'] - 0.95)
        best_idx = results_df['dist_from_95'].idxmin()
        best_factor = results_df.loc[best_idx, 'Calibration Factor']
        best_coverage = results_df.loc[best_idx, 'Coverage']

        print(f"✓ Recommended calibration factor: {best_factor}")
        print(f"  Achieves {best_coverage:.1%} coverage (target: 95%)")
        print("="*70)

    return results_df


if __name__ == "__main__":
    import sys
    sys.path.append('/Users/davidzhang/github/hierarchal_bayes_playground')

    print("Testing calibrated Borusyak estimator...\n")

    from src.data_generation_staggered_crossed import generate_staggered_adoption_data_crossed

    # Test with high noise
    df, truth = generate_staggered_adoption_data_crossed(
        n_genres=5,
        n_arpu_quintiles=5,
        n_creators_per_cell=8,
        noise_scale=2.0,
        seed=42
    )

    print("\nValidating calibration factors...")
    validate_calibration(df, truth, verbose=True)

    print("\n✓ Test complete!")
