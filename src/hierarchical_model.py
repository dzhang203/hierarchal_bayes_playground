"""
Hierarchical Bayesian Model for A/B experiment analysis.

This module implements a partial pooling approach using PyMC. The model
borrows strength across creators within genres to improve estimates,
especially for small sample sizes.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Tuple, Optional


def prepare_creator_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sufficient statistics for each creator's experiment.

    For a normal-normal model, we only need:
    - Observed mean difference (treatment - control)
    - Standard error of that difference

    This avoids feeding raw user-level data to the model, massively
    speeding up computation.

    Parameters
    ----------
    df : pd.DataFrame
        User-level data with columns [creator_id, genre_idx, group, revenue]

    Returns
    -------
    pd.DataFrame
        One row per creator with columns:
        - creator_id
        - genre_idx
        - effect_hat: observed treatment - control difference
        - se: standard error of the difference
        - n_total: total sample size
    """
    summaries = []

    for creator_id in df['creator_id'].unique():
        creator_df = df[df['creator_id'] == creator_id]

        treatment = creator_df[creator_df['group'] == 'treatment']['revenue'].values
        control = creator_df[creator_df['group'] == 'control']['revenue'].values

        # Compute sufficient statistics
        mean_treatment = treatment.mean()
        mean_control = control.mean()
        var_treatment = treatment.var(ddof=1)
        var_control = control.var(ddof=1)
        n_treatment = len(treatment)
        n_control = len(control)

        effect_hat = mean_treatment - mean_control
        se = np.sqrt(var_treatment / n_treatment + var_control / n_control)

        genre_idx = creator_df['genre_idx'].iloc[0]

        summaries.append({
            'creator_id': creator_id,
            'genre_idx': genre_idx,
            'effect_hat': effect_hat,
            'se': se,
            'n_total': n_treatment + n_control
        })

    return pd.DataFrame(summaries).sort_values('creator_id').reset_index(drop=True)


def fit_hierarchical_model(
    creator_summaries: pd.DataFrame,
    n_genres: int,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
    target_accept: float = 0.9
) -> az.InferenceData:
    """
    Fit hierarchical Bayesian model for treatment effects.

    Model specification:
        # Hyperpriors (platform-level)
        mu_global ~ Normal(0, 1)
        sigma_genre ~ HalfNormal(1)

        # Genre-level priors
        mu_genre[g] ~ Normal(mu_global, sigma_genre) for g in 1..G

        # Creator-level priors
        sigma_creator ~ HalfNormal(1)
        tau[i] ~ Normal(mu_genre[genre_of_i], sigma_creator) for i in 1..N

        # Likelihood
        effect_hat[i] ~ Normal(tau[i], se[i]) for i in 1..N

    Uses non-centered parameterization for efficient MCMC sampling.

    Parameters
    ----------
    creator_summaries : pd.DataFrame
        Summary statistics per creator (from prepare_creator_summaries)
    n_genres : int
        Number of genres
    draws : int
        Number of posterior samples per chain (after tune)
    tune : int
        Number of tuning/warmup steps (discarded)
    chains : int
        Number of independent MCMC chains
    random_seed : int
        Random seed for reproducibility
    target_accept : float
        Target acceptance rate for step size adaptation (0.9 is conservative)

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object containing posterior samples and diagnostics
    """
    # Extract data
    genre_idx = creator_summaries['genre_idx'].values
    observed_effects = creator_summaries['effect_hat'].values
    observed_se = creator_summaries['se'].values
    n_creators = len(creator_summaries)

    print(f"Fitting hierarchical model with {n_creators} creators in {n_genres} genres...")

    with pm.Model() as model:
        # Hyperpriors (platform-level)
        mu_global = pm.Normal('mu_global', mu=0, sigma=1)
        sigma_genre = pm.HalfNormal('sigma_genre', sigma=1)

        # Genre effects (non-centered parameterization)
        # This is critical for MCMC efficiency!
        # Instead of: mu_genre ~ Normal(mu_global, sigma_genre)
        # We write: mu_genre = mu_global + sigma_genre * offset
        # where offset ~ Normal(0, 1)
        genre_offset = pm.Normal('genre_offset', mu=0, sigma=1, shape=n_genres)
        mu_genre = pm.Deterministic('mu_genre', mu_global + sigma_genre * genre_offset)

        # Creator-level variance within genres
        sigma_creator = pm.HalfNormal('sigma_creator', sigma=1)

        # Creator effects (non-centered parameterization)
        creator_offset = pm.Normal('creator_offset', mu=0, sigma=1, shape=n_creators)
        tau = pm.Deterministic('tau', mu_genre[genre_idx] + sigma_creator * creator_offset)

        # Likelihood: observed effect ~ Normal(true effect, known SE)
        # The SE is treated as fixed/known (computed from data)
        obs = pm.Normal('obs', mu=tau, sigma=observed_se, observed=observed_effects)

        # Sample from posterior
        print(f"Sampling: {draws} draws × {chains} chains (+ {tune} tune steps per chain)")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            target_accept=target_accept,
            return_inferencedata=True
        )

    print("✓ Sampling complete!")
    return idata


def extract_hbm_estimates(
    idata: az.InferenceData,
    creator_summaries: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract point estimates and credible intervals from posterior.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model posterior
    creator_summaries : pd.DataFrame
        Creator summary statistics (for metadata)

    Returns
    -------
    pd.DataFrame
        Posterior estimates with columns:
        - creator_id
        - genre_idx
        - effect_hat: posterior mean of tau[i]
        - se: posterior standard deviation of tau[i]
        - ci_lower: 2.5th percentile (95% credible interval)
        - ci_upper: 97.5th percentile
        - n_total: sample size
    """
    # Extract tau samples: shape (chains, draws, n_creators)
    tau_samples = idata.posterior['tau'].values

    # Flatten across chains and draws: shape (total_samples, n_creators)
    n_creators = tau_samples.shape[-1]
    tau_flat = tau_samples.reshape(-1, n_creators)

    # Compute posterior statistics
    posterior_mean = tau_flat.mean(axis=0)
    posterior_std = tau_flat.std(axis=0)
    ci_lower = np.percentile(tau_flat, 2.5, axis=0)
    ci_upper = np.percentile(tau_flat, 97.5, axis=0)

    # Package into DataFrame
    results = pd.DataFrame({
        'creator_id': creator_summaries['creator_id'],
        'genre_idx': creator_summaries['genre_idx'],
        'effect_hat': posterior_mean,
        'se': posterior_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_total': creator_summaries['n_total']
    })

    return results


def extract_genre_estimates(idata: az.InferenceData, n_genres: int) -> pd.DataFrame:
    """
    Extract posterior estimates for genre-level effects.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model posterior
    n_genres : int
        Number of genres

    Returns
    -------
    pd.DataFrame
        Genre-level estimates with columns:
        - genre_idx
        - mu_genre_mean: posterior mean
        - mu_genre_std: posterior SD
        - ci_lower: 95% credible interval bounds
        - ci_upper
    """
    mu_genre_samples = idata.posterior['mu_genre'].values
    mu_genre_flat = mu_genre_samples.reshape(-1, n_genres)

    results = pd.DataFrame({
        'genre_idx': range(n_genres),
        'mu_genre_mean': mu_genre_flat.mean(axis=0),
        'mu_genre_std': mu_genre_flat.std(axis=0),
        'ci_lower': np.percentile(mu_genre_flat, 2.5, axis=0),
        'ci_upper': np.percentile(mu_genre_flat, 97.5, axis=0)
    })

    return results


def check_mcmc_diagnostics(idata: az.InferenceData, verbose: bool = True) -> dict:
    """
    Verify that MCMC sampling worked correctly.

    Before trusting any results, we MUST check:
    1. R-hat < 1.01 (chains converged)
    2. Effective sample size > 400 (enough independent samples)
    3. No divergent transitions (sampler didn't fail)

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model posterior
    verbose : bool
        Whether to print diagnostic messages

    Returns
    -------
    dict
        Diagnostic results with keys:
        - all_rhat_ok: bool
        - all_ess_ok: bool
        - n_divergences: int
        - max_rhat: float
        - min_ess_bulk: float
    """
    # Compute summary statistics
    summary = az.summary(
        idata,
        var_names=['mu_global', 'sigma_genre', 'sigma_creator', 'mu_genre']
    )

    # Check R-hat (convergence diagnostic)
    max_rhat = summary['r_hat'].max()
    all_rhat_ok = max_rhat < 1.01

    # Check effective sample size
    min_ess = summary['ess_bulk'].min()
    all_ess_ok = min_ess > 400

    # Check for divergences
    n_divergences = int(idata.sample_stats['diverging'].sum().values)

    diagnostics = {
        'all_rhat_ok': all_rhat_ok,
        'all_ess_ok': all_ess_ok,
        'n_divergences': n_divergences,
        'max_rhat': max_rhat,
        'min_ess_bulk': min_ess
    }

    if verbose:
        print("=" * 60)
        print("MCMC DIAGNOSTICS")
        print("=" * 60)

        status = "✓" if all_rhat_ok else "✗"
        print(f"{status} R-hat check: max R-hat = {max_rhat:.4f} (should be < 1.01)")

        status = "✓" if all_ess_ok else "✗"
        print(f"{status} ESS check: min ESS = {min_ess:.0f} (should be > 400)")

        status = "✓" if n_divergences == 0 else "✗"
        print(f"{status} Divergences: {n_divergences} (should be 0)")

        if all_rhat_ok and all_ess_ok and n_divergences == 0:
            print("\n✓ All diagnostics passed! Model fit is trustworthy.")
        else:
            print("\n⚠ WARNING: Some diagnostics failed. Results may be unreliable.")
            if not all_rhat_ok:
                print("  - Try increasing number of tune/draw steps")
            if n_divergences > 0:
                print("  - Try increasing target_accept to 0.95 or 0.99")

        print("=" * 60)

    return diagnostics


if __name__ == "__main__":
    # Test with synthetic data
    import sys
    sys.path.append('/Users/davidzhang/github/hierarchal_bayes_playground')
    from src.data_generation import generate_experiment_data

    print("Generating synthetic data...")
    df, truth = generate_experiment_data(seed=42)

    print("\nPreparing creator summaries...")
    summaries = prepare_creator_summaries(df)
    print(f"✓ Prepared summaries for {len(summaries)} creators")

    print("\nFitting hierarchical model...")
    idata = fit_hierarchical_model(
        summaries,
        n_genres=truth['n_genres'],
        draws=1000,  # Use fewer draws for quick test
        tune=500,
        chains=2,
        random_seed=42
    )

    print("\nChecking diagnostics...")
    diagnostics = check_mcmc_diagnostics(idata)

    print("\nExtracting estimates...")
    hbm_estimates = extract_hbm_estimates(idata, summaries)
    print(f"✓ Extracted estimates for {len(hbm_estimates)} creators")

    genre_estimates = extract_genre_estimates(idata, truth['n_genres'])
    print(f"✓ Extracted genre estimates for {len(genre_estimates)} genres")

    # Quick validation
    true_effects = truth['creator_effects']
    hbm_mse = np.mean((hbm_estimates['effect_hat'] - true_effects) ** 2)
    print(f"\nHBM Mean Squared Error: {hbm_mse:.4f}")

    # Coverage
    hbm_coverage = np.mean(
        (true_effects >= hbm_estimates['ci_lower']) &
        (true_effects <= hbm_estimates['ci_upper'])
    )
    print(f"HBM 95% Credible Interval Coverage: {hbm_coverage:.3f}")
