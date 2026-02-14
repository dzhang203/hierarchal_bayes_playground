"""
Visualize the coverage-precision trade-off for small creators.

This script creates a detailed visualization showing WHY HBM is better
for small creators: it's not about coverage (both are ~95%), but about
achieving that coverage with much narrower (more useful) intervals.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_generation import generate_experiment_data
from src.frequentist import no_pooling_estimates
from src.hierarchical_model import (
    prepare_creator_summaries,
    fit_hierarchical_model,
    extract_hbm_estimates
)

# Set style
sns.set_style("whitegrid")

def main():
    print("Generating data and fitting models...")
    df, truth = generate_experiment_data(seed=42)

    # Get estimates
    no_pool = no_pooling_estimates(df)
    summaries = prepare_creator_summaries(df)
    idata = fit_hierarchical_model(summaries, n_genres=truth['n_genres'],
                                   draws=1000, tune=500, chains=2, random_seed=42)
    hbm = extract_hbm_estimates(idata, summaries)

    # Focus on small creators
    small_mask = no_pool['n_total'] < 100
    small_no_pool = no_pool[small_mask].copy()
    small_hbm = hbm[small_mask].copy()
    true_effects_small = truth['creator_effects'][small_mask]

    # Select 20 random small creators for visualization
    np.random.seed(42)
    sample_indices = np.random.choice(len(small_no_pool), size=20, replace=False)

    sample_no_pool = small_no_pool.iloc[sample_indices].reset_index(drop=True)
    sample_hbm = small_hbm.iloc[sample_indices].reset_index(drop=True)
    sample_truth = true_effects_small[sample_indices]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left panel: No Pooling
    ax = axes[0]
    x_pos = np.arange(len(sample_no_pool))

    # Plot intervals
    for i in range(len(sample_no_pool)):
        ax.plot([i, i],
               [sample_no_pool.iloc[i]['ci_lower'], sample_no_pool.iloc[i]['ci_upper']],
               'o-', color='#FF6B6B', alpha=0.6, linewidth=2, markersize=8)

    # Plot true effects
    ax.scatter(x_pos, sample_truth, color='red', s=100, marker='*',
              zorder=10, label='True Effect', edgecolors='darkred', linewidth=1.5)

    # Count coverage
    covered = ((sample_truth >= sample_no_pool['ci_lower']) &
              (sample_truth <= sample_no_pool['ci_upper']))
    coverage_rate = covered.mean()
    avg_width = (sample_no_pool['ci_upper'] - sample_no_pool['ci_lower']).mean()

    ax.set_xlabel('Creator Index', fontweight='bold', fontsize=12)
    ax.set_ylabel('Treatment Effect ($)', fontweight='bold', fontsize=12)
    ax.set_title(f'No Pooling (Frequentist)\n' +
                f'Coverage: {coverage_rate:.1%} | Avg Width: ${avg_width:.2f}',
                fontweight='bold', fontsize=14)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Annotate a few intervals
    for i in [0, 5, 10]:
        width = sample_no_pool.iloc[i]['ci_upper'] - sample_no_pool.iloc[i]['ci_lower']
        ax.annotate(f'±${width/2:.2f}',
                   xy=(i, sample_no_pool.iloc[i]['ci_upper']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7)

    # Right panel: HBM
    ax = axes[1]

    # Plot intervals
    for i in range(len(sample_hbm)):
        ax.plot([i, i],
               [sample_hbm.iloc[i]['ci_lower'], sample_hbm.iloc[i]['ci_upper']],
               'o-', color='#45B7D1', alpha=0.6, linewidth=2, markersize=8)

    # Plot true effects
    ax.scatter(x_pos, sample_truth, color='red', s=100, marker='*',
              zorder=10, label='True Effect', edgecolors='darkred', linewidth=1.5)

    # Count coverage
    covered = ((sample_truth >= sample_hbm['ci_lower']) &
              (sample_truth <= sample_hbm['ci_upper']))
    coverage_rate = covered.mean()
    avg_width = (sample_hbm['ci_upper'] - sample_hbm['ci_lower']).mean()

    ax.set_xlabel('Creator Index', fontweight='bold', fontsize=12)
    ax.set_ylabel('Treatment Effect ($)', fontweight='bold', fontsize=12)
    ax.set_title(f'HBM (Partial Pooling)\n' +
                f'Coverage: {coverage_rate:.1%} | Avg Width: ${avg_width:.2f}',
                fontweight='bold', fontsize=14)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Annotate a few intervals
    for i in [0, 5, 10]:
        width = sample_hbm.iloc[i]['ci_upper'] - sample_hbm.iloc[i]['ci_lower']
        ax.annotate(f'±${width/2:.2f}',
                   xy=(i, sample_hbm.iloc[i]['ci_upper']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.7)

    # Set same y-axis limits for both panels for fair comparison
    all_lower = min(sample_no_pool['ci_lower'].min(), sample_hbm['ci_lower'].min())
    all_upper = max(sample_no_pool['ci_upper'].max(), sample_hbm['ci_upper'].max())
    y_margin = (all_upper - all_lower) * 0.1
    for ax in axes:
        ax.set_ylim(all_lower - y_margin, all_upper + y_margin)

    plt.suptitle('Coverage-Precision Trade-off for Small Creators (n < 100)\n' +
                'Same coverage, but HBM intervals are much narrower!',
                fontweight='bold', fontsize=16, y=0.98)

    plt.tight_layout()
    plt.savefig('outputs/coverage_precision_tradeoff.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization to outputs/coverage_precision_tradeoff.png")

    # Print detailed statistics
    print("\n" + "="*80)
    print("COVERAGE-PRECISION TRADE-OFF ANALYSIS")
    print("="*80)

    # For all small creators (not just sample)
    no_pool_coverage = ((true_effects_small >= small_no_pool['ci_lower']) &
                       (true_effects_small <= small_no_pool['ci_upper'])).mean()
    hbm_coverage = ((true_effects_small >= small_hbm['ci_lower']) &
                   (true_effects_small <= small_hbm['ci_upper'])).mean()

    no_pool_width = (small_no_pool['ci_upper'] - small_no_pool['ci_lower']).mean()
    hbm_width = (small_hbm['ci_upper'] - small_hbm['ci_lower']).mean()

    print(f"\nFor ALL small creators (n < 100, total = {len(small_no_pool)}):\n")
    print(f"No Pooling:")
    print(f"  Coverage:     {no_pool_coverage:.1%}")
    print(f"  Avg CI Width: ${no_pool_width:.3f}")
    print(f"\nHBM:")
    print(f"  Coverage:     {hbm_coverage:.1%}")
    print(f"  Avg CI Width: ${hbm_width:.3f}")
    print(f"\nImprovement:")
    print(f"  Coverage change: {(hbm_coverage - no_pool_coverage)*100:+.1f} percentage points")
    print(f"  Width reduction: {(1 - hbm_width/no_pool_width)*100:.1f}%")
    print(f"\n✓ Same coverage, {(1 - hbm_width/no_pool_width)*100:.0f}% narrower intervals!")
    print("="*80)

    plt.show()

if __name__ == "__main__":
    main()
