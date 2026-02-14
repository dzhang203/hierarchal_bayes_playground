"""
Grouping selection for hierarchical models using Bayesian model comparison.

This module implements principled selection of grouping structures for HBM
using cross-validated predictive performance (LOO-IC).

Key principle: Avoid overfitting by comparing groupings using out-of-sample
predictive accuracy, not in-sample fit.
"""

import numpy as np
import pandas as pd
import arviz as az
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from src.hierarchical_model import fit_hierarchical_model, prepare_creator_summaries


def create_grouping(
    ite_estimates: pd.DataFrame,
    creator_features: pd.DataFrame,
    grouping_var: str
) -> pd.DataFrame:
    """
    Create grouped estimates by merging ITEs with creator features.

    Parameters
    ----------
    ite_estimates : pd.DataFrame
        ITE estimates with columns [creator_id, effect_hat, se]
    creator_features : pd.DataFrame
        Creator features with [creator_id, genre_idx, size_quartile, arpu_quartile, ...]
    grouping_var : str
        Which feature to use for grouping: 'genre_idx', 'size_quartile', 'arpu_quartile'

    Returns
    -------
    pd.DataFrame
        Merged data with grouping variable
    """
    # Merge ITEs with features
    grouped_data = ite_estimates.merge(
        creator_features[['creator_id', grouping_var]],
        on='creator_id',
        how='left'
    )

    # Rename grouping var to 'genre_idx' for compatibility with existing HBM code
    # (The HBM expects 'genre_idx' as the grouping variable)
    grouped_data = grouped_data.rename(columns={grouping_var: 'genre_idx'})

    return grouped_data


def compare_groupings(
    ite_estimates: pd.DataFrame,
    creator_features: pd.DataFrame,
    grouping_candidates: List[str],
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare different grouping structures using Bayesian model selection.

    Uses LOO-IC (Leave-One-Out Information Criterion) to compare groupings
    based on out-of-sample predictive performance. Lower LOO-IC = better.

    Parameters
    ----------
    ite_estimates : pd.DataFrame
        ITE estimates from Stage 1 (Borusyak)
    creator_features : pd.DataFrame
        Creator features for grouping
    grouping_candidates : List[str]
        List of grouping variables to compare:
        e.g., ['genre_idx', 'size_quartile', 'arpu_quartile']
    draws : int
        MCMC draws per chain
    tune : int
        MCMC tuning steps
    chains : int
        Number of MCMC chains
    random_seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress

    Returns
    -------
    comparison : pd.DataFrame
        Model comparison table with columns:
        - rank: rank by LOO-IC (0 = best)
        - loo: LOO-IC value
        - loo_se: SE of LOO-IC
        - d_loo: difference from best model
        - weight: Bayesian model averaging weight

    models : dict
        Dictionary of fitted models {grouping_name: idata}
    """
    if verbose:
        print("="*70)
        print("GROUPING SELECTION VIA BAYESIAN MODEL COMPARISON")
        print("="*70)
        print(f"Candidates: {grouping_candidates}")
        print()

    models = {}

    # Fit HBM for each grouping candidate
    for grouping_var in grouping_candidates:
        if verbose:
            print(f"Fitting HBM with grouping: {grouping_var}")

        # Create grouped data
        grouped_data = create_grouping(ite_estimates, creator_features, grouping_var)

        # Count groups
        n_groups = grouped_data['genre_idx'].nunique()

        # Check minimum group size
        group_sizes = grouped_data.groupby('genre_idx').size()
        min_group_size = group_sizes.min()

        if verbose:
            print(f"  Number of groups: {n_groups}")
            print(f"  Group sizes: min={min_group_size}, "
                  f"median={group_sizes.median():.0f}, max={group_sizes.max()}")

        if min_group_size < 5:
            print(f"  ⚠️  WARNING: Some groups have < 5 observations. "
                  f"Estimates may be unstable.")

        # Fit hierarchical model
        try:
            idata = fit_hierarchical_model(
                grouped_data,
                n_genres=n_groups,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed
            )
            models[grouping_var] = idata

            if verbose:
                print(f"  ✓ Model fitted successfully")
                print()

        except Exception as e:
            print(f"  ✗ Error fitting model: {e}")
            print()
            continue

    if len(models) == 0:
        raise ValueError("No models were successfully fitted!")

    # =====================================================================
    # Compare models using LOO-IC
    # =====================================================================

    if verbose:
        print("="*70)
        print("MODEL COMPARISON (LOO-IC)")
        print("="*70)
        print()

    # Compute LOO-IC for each model
    comparison = az.compare(models)

    if verbose:
        print(comparison)
        print()
        print("Interpretation:")
        print("  • rank: 0 = best model")
        print("  • loo: LOO-IC (lower = better out-of-sample prediction)")
        print("  • d_loo: difference from best (> 4 is meaningful)")
        print("  • weight: Bayesian model averaging weight")
        print()

        # Identify best model
        best_grouping = comparison.index[0]
        print(f"✓ Best grouping: {best_grouping}")

        # Check if difference is meaningful
        if len(comparison) > 1:
            second_best = comparison.index[1]
            d_loo = comparison.loc[second_best, 'd_loo']
            d_loo_se = comparison.loc[second_best, 'd_se']

            print(f"  Difference from 2nd best: {d_loo:.1f} (SE: {d_loo_se:.1f})")

            if d_loo > 4:
                print(f"  → Clear winner! {best_grouping} is substantially better.")
            elif d_loo > 2:
                print(f"  → {best_grouping} is likely better, but not decisive.")
            else:
                print(f"  → {best_grouping} and {second_best} are roughly equivalent.")

        print("="*70)

    return comparison, models


def check_posterior_separation(
    idata,
    grouping_name: str,
    var_name: str = 'mu_genre',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Check if group-level effects are meaningfully separated in posterior.

    If posteriors overlap heavily, the grouping doesn't capture real
    differences in treatment effects.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model
    grouping_name : str
        Name of grouping (for plot title)
    var_name : str
        Name of group-level parameter to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Forest plot of group-level effects
    az.plot_forest(
        idata,
        var_names=[var_name],
        combined=True,
        hdi_prob=0.95,
        ax=ax
    )

    ax.set_title(f'Posterior Group Effects: {grouping_name}\n'
                f'(95% HDI intervals)',
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Treatment Effect', fontweight='bold')
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved posterior separation plot to {save_path}")

    plt.show()


def summarize_grouping_quality(
    comparison: pd.DataFrame,
    models: Dict,
    min_group_sizes: Dict[str, int],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create summary table of grouping quality metrics.

    Parameters
    ----------
    comparison : pd.DataFrame
        Model comparison from az.compare()
    models : dict
        Fitted models
    min_group_sizes : dict
        Minimum group size for each grouping
    verbose : bool
        Whether to print summary

    Returns
    -------
    pd.DataFrame
        Summary table with quality metrics
    """
    summary_rows = []

    for grouping_name in comparison.index:
        loo = comparison.loc[grouping_name, 'loo']
        d_loo = comparison.loc[grouping_name, 'd_loo']
        weight = comparison.loc[grouping_name, 'weight']
        rank = comparison.loc[grouping_name, 'rank']

        # Check MCMC diagnostics
        idata = models[grouping_name]
        summary = az.summary(idata, var_names=['mu_global', 'sigma_genre', 'sigma_creator'])

        max_rhat = summary['r_hat'].max()
        min_ess = summary['ess_bulk'].min()

        summary_rows.append({
            'Grouping': grouping_name,
            'Rank': int(rank),
            'LOO-IC': loo,
            'Δ LOO': d_loo,
            'Weight': weight,
            'Min Group Size': min_group_sizes.get(grouping_name, None),
            'Max R̂': max_rhat,
            'Min ESS': min_ess
        })

    summary_df = pd.DataFrame(summary_rows).set_index('Grouping')

    if verbose:
        print("="*70)
        print("GROUPING QUALITY SUMMARY")
        print("="*70)
        print(summary_df.to_string())
        print()
        print("Quality criteria:")
        print("  ✓ LOO-IC: Lower is better")
        print("  ✓ Weight: Higher = more predictive power")
        print("  ✓ Min Group Size: Should be > 10")
        print("  ✓ Max R̂: Should be < 1.01")
        print("  ✓ Min ESS: Should be > 400")
        print("="*70)

    return summary_df


if __name__ == "__main__":
    # Test with synthetic data
    import sys
    sys.path.append('/Users/davidzhang/github/hierarchal_bayes_playground')

    from src.data_generation_staggered import generate_staggered_adoption_data
    from src.borusyak_estimator import borusyak_imputation_estimator

    print("Generating synthetic data...")
    df, truth = generate_staggered_adoption_data(
        n_genres=5,
        n_creators_per_genre=40,
        seed=42
    )

    print("\nEstimating ITEs with Borusyak...")
    ite_estimates, diagnostics = borusyak_imputation_estimator(
        df,
        never_treated_ids=truth['never_treated'],
        verbose=False
    )

    print("\nComparing groupings...")
    grouping_candidates = ['genre_idx', 'size_quartile', 'arpu_quartile']

    comparison, models = compare_groupings(
        ite_estimates,
        truth['creator_features'],
        grouping_candidates,
        draws=500,
        tune=250,
        chains=2,
        verbose=True
    )

    # Get min group sizes
    min_group_sizes = {}
    for var in grouping_candidates:
        sizes = truth['creator_features'].groupby(var).size()
        min_group_sizes[var] = sizes.min()

    summary = summarize_grouping_quality(comparison, models, min_group_sizes)

    print("\n✓ Grouping selection test complete!")
