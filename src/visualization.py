"""
Visualization functions for hierarchical Bayesian model analysis.

This module creates all plots for understanding and validating the HBM approach:
1. Shrinkage plot (the key diagnostic)
2. MSE comparison
3. Coverage vs. width analysis
4. Individual creator examples
5. Genre effect recovery
6. Hyperparameter recovery
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from typing import Dict, Optional, Tuple

# Set style
sns.set_style("whitegrid")
sns.set_context("notebook")


def plot_shrinkage(
    no_pool: pd.DataFrame,
    hbm: pd.DataFrame,
    truth: dict,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    THE key plot: visualize how HBM shrinks estimates toward genre means.

    Shows:
    - x-axis: no-pooling (frequentist) estimate
    - y-axis: HBM estimate
    - Color: genre
    - Size: sample size (larger = more data)
    - Reference line: y=x (no shrinkage)
    - Horizontal lines: genre means

    Small creators should be pulled more toward their genre mean.
    Large creators should stay near the y=x line.

    Parameters
    ----------
    no_pool : pd.DataFrame
        No-pooling estimates
    hbm : pd.DataFrame
        HBM estimates
    truth : dict
        Ground truth (for genre info)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    df = pd.DataFrame({
        'no_pool': no_pool['effect_hat'],
        'hbm': hbm['effect_hat'],
        'genre_idx': no_pool['genre_idx'],
        'n_total': no_pool['n_total']
    })

    # Normalize sample sizes for marker size
    size_normalized = (df['n_total'] - df['n_total'].min()) / (df['n_total'].max() - df['n_total'].min())
    marker_sizes = 20 + 200 * size_normalized  # Range from 20 to 220

    # Compute genre means from HBM
    genre_means = []
    for g in range(truth['n_genres']):
        mask = df['genre_idx'] == g
        genre_means.append(df[mask]['hbm'].mean())

    # Plot by genre
    colors = sns.color_palette("husl", truth['n_genres'])
    genre_names = truth['genre_names']

    for g in range(truth['n_genres']):
        mask = df['genre_idx'] == g
        ax.scatter(
            df[mask]['no_pool'],
            df[mask]['hbm'],
            s=marker_sizes[mask],
            alpha=0.6,
            color=colors[g],
            label=genre_names[g],
            edgecolors='white',
            linewidth=0.5
        )

    # Add y=x reference line (no shrinkage)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plot_min = min(xlim[0], ylim[0])
    plot_max = max(xlim[1], ylim[1])
    ax.plot([plot_min, plot_max], [plot_min, plot_max],
            'k--', alpha=0.3, linewidth=2, label='No shrinkage (y=x)')

    # Add genre mean horizontal lines
    for g, genre_mean in enumerate(genre_means):
        ax.axhline(genre_mean, color=colors[g], linestyle=':', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('No-Pooling Estimate (Frequentist)', fontsize=12, fontweight='bold')
    ax.set_ylabel('HBM Estimate (Partial Pooling)', fontsize=12, fontweight='bold')
    ax.set_title('Shrinkage Plot: How HBM Adjusts Noisy Estimates\n'
                 'Small points (small n) are pulled more toward genre means',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved shrinkage plot to {save_path}")

    return fig


def plot_mse_comparison(
    no_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
    hbm: pd.DataFrame,
    truth: dict,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare MSE across methods, overall and stratified by sample size.

    Parameters
    ----------
    no_pool : pd.DataFrame
        No-pooling estimates
    complete_pool : pd.DataFrame
        Complete-pooling estimates
    hbm : pd.DataFrame
        HBM estimates
    truth : dict
        Ground truth
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    from src.validation import stratified_metrics

    true_effects = truth['creator_effects']
    bins = [0, 100, 500, np.inf]

    # Compute stratified MSE for each method
    methods = {
        'No Pooling': no_pool,
        'Complete Pooling': complete_pool,
        'HBM': hbm
    }

    stratified_results = []
    for method_name, df in methods.items():
        strat = stratified_metrics(df, true_effects, bins)
        strat['method'] = method_name
        stratified_results.append(strat)

    # Add overall MSE
    for method_name, df in methods.items():
        mse_overall = np.mean((df['effect_hat'].values - true_effects) ** 2)
        stratified_results.append(pd.DataFrame([{
            'size_bin': 'Overall',
            'method': method_name,
            'mse': mse_overall
        }]))

    df_plot = pd.concat(stratified_results, ignore_index=True)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    # Define bin order
    bin_order = ['n=0-100', 'n=100-500', 'n>500', 'Overall']
    df_plot['size_bin'] = pd.Categorical(df_plot['size_bin'], categories=bin_order, ordered=True)
    df_plot = df_plot.sort_values('size_bin')

    # Plot
    x = np.arange(len(bin_order))
    width = 0.25

    colors = {'No Pooling': '#FF6B6B', 'Complete Pooling': '#4ECDC4', 'HBM': '#45B7D1'}

    for i, method in enumerate(['No Pooling', 'Complete Pooling', 'HBM']):
        method_data = df_plot[df_plot['method'] == method]
        mse_values = [method_data[method_data['size_bin'] == bin]['mse'].values[0]
                      if len(method_data[method_data['size_bin'] == bin]) > 0 else 0
                      for bin in bin_order]

        ax.bar(x + i * width, mse_values, width, label=method, color=colors[method], alpha=0.8)

    ax.set_xlabel('Sample Size Bin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title('MSE Comparison: HBM Wins Especially for Small Creators',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(bin_order)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved MSE comparison to {save_path}")

    return fig


def plot_coverage_vs_width(
    no_pool: pd.DataFrame,
    complete_pool: pd.DataFrame,
    hbm: pd.DataFrame,
    truth: dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot coverage vs. interval width for each method.

    Ideal: high coverage (~95%) with narrow intervals.

    Parameters
    ----------
    no_pool : pd.DataFrame
        No-pooling estimates
    complete_pool : pd.DataFrame
        Complete-pooling estimates
    hbm : pd.DataFrame
        HBM estimates
    truth : dict
        Ground truth
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    from src.validation import compute_coverage, compute_avg_interval_width

    true_effects = truth['creator_effects']

    methods = {
        'No Pooling': no_pool,
        'Complete Pooling': complete_pool,
        'HBM': hbm
    }

    results = []
    for method_name, df in methods.items():
        coverage = compute_coverage(
            df['ci_lower'].values,
            df['ci_upper'].values,
            true_effects
        )
        width = compute_avg_interval_width(
            df['ci_lower'].values,
            df['ci_upper'].values
        )
        results.append({
            'method': method_name,
            'coverage': coverage,
            'avg_width': width
        })

    df_plot = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=figsize)

    colors = {'No Pooling': '#FF6B6B', 'Complete Pooling': '#4ECDC4', 'HBM': '#45B7D1'}

    for _, row in df_plot.iterrows():
        ax.scatter(row['avg_width'], row['coverage'],
                  s=500, alpha=0.7, color=colors[row['method']],
                  edgecolors='white', linewidth=2,
                  label=row['method'])

    # Add target coverage line
    ax.axhline(0.95, color='green', linestyle='--', linewidth=2,
               alpha=0.5, label='Target coverage (95%)')

    # Add annotations
    for _, row in df_plot.iterrows():
        ax.annotate(row['method'],
                   (row['avg_width'], row['coverage']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[row['method']], alpha=0.3))

    ax.set_xlabel('Average Interval Width', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
    ax.set_title('Coverage vs. Precision Trade-off\n'
                 'HBM achieves 95% coverage with narrower intervals than No Pooling',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved coverage plot to {save_path}")

    return fig


def plot_individual_creators(
    no_pool: pd.DataFrame,
    hbm: pd.DataFrame,
    truth: dict,
    n_examples: int = 12,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Show detailed comparison for individual creators.

    Pick examples spanning small/medium/large sample sizes.

    Parameters
    ----------
    no_pool : pd.DataFrame
        No-pooling estimates
    hbm : pd.DataFrame
        HBM estimates
    truth : dict
        Ground truth
    n_examples : int
        Number of creators to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    # Select diverse examples
    df = pd.DataFrame({
        'creator_id': no_pool['creator_id'],
        'n_total': no_pool['n_total'],
        'true_effect': truth['creator_effects']
    })

    # Get examples from different size bins
    bins = [0, 100, 500, np.inf]
    examples_per_bin = n_examples // 3

    selected_ids = []
    for i in range(len(bins) - 1):
        mask = (df['n_total'] >= bins[i]) & (df['n_total'] < bins[i+1])
        if mask.sum() > 0:
            bin_df = df[mask]
            sample_ids = bin_df.sample(min(examples_per_bin, len(bin_df)), random_state=42)['creator_id'].values
            selected_ids.extend(sample_ids)

    selected_ids = selected_ids[:n_examples]

    # Create subplots
    n_rows = (n_examples + 2) // 3
    n_cols = min(3, n_examples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_examples > 1 else [axes]

    for idx, creator_id in enumerate(selected_ids):
        ax = axes[idx]

        # Get data for this creator
        no_pool_row = no_pool[no_pool['creator_id'] == creator_id].iloc[0]
        hbm_row = hbm[hbm['creator_id'] == creator_id].iloc[0]
        true_effect = truth['creator_effects'][creator_id]
        genre_idx = no_pool_row['genre_idx']

        # Compute genre mean
        genre_mask = truth['creator_genre'] == genre_idx
        genre_mean = hbm[hbm['genre_idx'] == genre_idx]['effect_hat'].mean()

        # Plot frequentist estimate
        ax.errorbar([1], [no_pool_row['effect_hat']],
                   yerr=[[no_pool_row['effect_hat'] - no_pool_row['ci_lower']],
                         [no_pool_row['ci_upper'] - no_pool_row['effect_hat']]],
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color='#FF6B6B', label='No Pooling', alpha=0.8)

        # Plot HBM estimate
        ax.errorbar([2], [hbm_row['effect_hat']],
                   yerr=[[hbm_row['effect_hat'] - hbm_row['ci_lower']],
                         [hbm_row['ci_upper'] - hbm_row['effect_hat']]],
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color='#45B7D1', label='HBM', alpha=0.8)

        # True effect
        ax.axhline(true_effect, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='True Effect')

        # Genre mean
        ax.axhline(genre_mean, color='gray', linestyle=':', linewidth=2,
                  alpha=0.5, label='Genre Mean')

        ax.set_xlim([0.5, 2.5])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Freq.', 'HBM'])
        ax.set_ylabel('Treatment Effect', fontsize=9)
        ax.set_title(f'Creator {creator_id} (n={no_pool_row["n_total"]})',
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    # Hide unused subplots
    for idx in range(len(selected_ids), len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Individual Creator Examples: How HBM Improves Small-Sample Estimates',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved individual creators plot to {save_path}")

    return fig


def plot_genre_recovery(
    genre_estimates: pd.DataFrame,
    truth: dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize how well HBM recovers genre-level effects.

    Parameters
    ----------
    genre_estimates : pd.DataFrame
        HBM genre estimates
    truth : dict
        Ground truth
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_genres = truth['n_genres']
    x = np.arange(n_genres)

    # Plot HBM estimates with credible intervals
    ax.errorbar(x, genre_estimates['mu_genre_mean'],
               yerr=[genre_estimates['mu_genre_mean'] - genre_estimates['ci_lower'],
                     genre_estimates['ci_upper'] - genre_estimates['mu_genre_mean']],
               fmt='o', markersize=10, capsize=5, capthick=2,
               color='#45B7D1', label='HBM Estimate', alpha=0.8,
               linewidth=2)

    # Plot true values
    ax.scatter(x, truth['genre_effects'], s=200, color='red',
              marker='*', label='True Effect', alpha=0.8,
              edgecolors='darkred', linewidth=2, zorder=5)

    ax.set_xlabel('Genre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Treatment Effect', fontsize=12, fontweight='bold')
    ax.set_title('Genre-Level Effect Recovery: HBM vs. Ground Truth',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(truth['genre_names'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved genre recovery plot to {save_path}")

    return fig


def plot_posterior_distributions(
    idata: az.InferenceData,
    truth: dict,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot posterior distributions for key hyperparameters.

    Shows how well the model recovers:
    - sigma_genre (between-genre variance)
    - sigma_creator (within-genre variance)
    - mu_global (global mean)

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model
    truth : dict
        Ground truth
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # sigma_genre
    az.plot_posterior(idata, var_names=['sigma_genre'], ax=axes[0],
                     ref_val=truth['genre_effects'].std())
    axes[0].set_title('sigma_genre (Between-Genre SD)', fontweight='bold')
    axes[0].axvline(truth['genre_effects'].std(), color='red', linestyle='--',
                   linewidth=2, label='Empirical True Value')

    # sigma_creator
    az.plot_posterior(idata, var_names=['sigma_creator'], ax=axes[1],
                     ref_val=truth['sigma_creator'])
    axes[1].set_title('sigma_creator (Within-Genre SD)', fontweight='bold')
    axes[1].axvline(truth['sigma_creator'], color='red', linestyle='--',
                   linewidth=2, label='True Value')

    # mu_global
    az.plot_posterior(idata, var_names=['mu_global'], ax=axes[2],
                     ref_val=truth['genre_effects'].mean())
    axes[2].set_title('mu_global (Global Mean Effect)', fontweight='bold')
    axes[2].axvline(truth['genre_effects'].mean(), color='red', linestyle='--',
                   linewidth=2, label='True Mean')

    fig.suptitle('Hyperparameter Recovery: Posterior vs. True Values',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved posterior distributions to {save_path}")

    return fig


def plot_trace(
    idata: az.InferenceData,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot MCMC trace plots for diagnostics.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig = az.plot_trace(
        idata,
        var_names=['mu_global', 'sigma_genre', 'sigma_creator'],
        figsize=figsize
    )

    plt.suptitle('MCMC Trace Plots: Convergence Diagnostics',
                fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trace plots to {save_path}")

    return fig


if __name__ == "__main__":
    # Test visualizations
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

    print("Generating data and fitting models...")
    df, truth = generate_experiment_data(seed=42)
    no_pool = no_pooling_estimates(df)
    complete_pool = complete_pooling_estimates(df)

    summaries = prepare_creator_summaries(df)
    idata = fit_hierarchical_model(summaries, n_genres=truth['n_genres'],
                                   draws=1000, tune=500, chains=2)
    hbm = extract_hbm_estimates(idata, summaries)
    genre_est = extract_genre_estimates(idata, truth['n_genres'])

    print("\nGenerating plots...")

    # Create all plots
    plot_shrinkage(no_pool, hbm, truth)
    plot_mse_comparison(no_pool, complete_pool, hbm, truth)
    plot_coverage_vs_width(no_pool, complete_pool, hbm, truth)
    plot_individual_creators(no_pool, hbm, truth)
    plot_genre_recovery(genre_est, truth)
    plot_posterior_distributions(idata, truth)

    plt.show()
    print("\n✓ All plots generated successfully!")
