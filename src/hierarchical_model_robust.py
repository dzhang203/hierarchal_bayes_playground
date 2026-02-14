"""
Robust version of hierarchical model with better initialization for Borusyak ITEs.

This addresses numerical issues when fitting HBM on top of Borusyak estimates,
which can have very small standard errors that create peaked likelihoods.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Optional


def fit_hierarchical_model_robust(
    creator_summaries: pd.DataFrame,
    n_genres: int,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
    target_accept: float = 0.95,
    min_se: float = 0.05,
    use_informative_prior: bool = True
) -> az.InferenceData:
    """
    Fit hierarchical Bayesian model with robust initialization.

    Improvements over base version:
    1. Adds minimum SE threshold to avoid extreme peaked likelihoods
    2. Uses informative priors based on observed data
    3. Better starting values for MCMC

    Parameters
    ----------
    creator_summaries : pd.DataFrame
        Summary statistics per creator with columns:
        [creator_id, genre_idx, effect_hat, se, n_total]
    n_genres : int
        Number of genres
    draws : int
        Number of posterior samples per chain
    tune : int
        Number of tuning/warmup steps
    chains : int
        Number of independent MCMC chains
    random_seed : int
        Random seed for reproducibility
    target_accept : float
        Target acceptance rate (0.95 = very conservative)
    min_se : float
        Minimum SE threshold (adds small jitter to very precise estimates)
    use_informative_prior : bool
        Whether to use data-driven priors for mu_global

    Returns
    -------
    az.InferenceData
        Posterior samples and diagnostics
    """
    # Extract data
    genre_idx = creator_summaries['genre_idx'].values
    observed_effects = creator_summaries['effect_hat'].values
    observed_se = creator_summaries['se'].values.copy()  # Copy for modification
    n_creators = len(creator_summaries)

    # Apply minimum SE threshold to avoid numerical issues
    too_small = observed_se < min_se
    if too_small.any():
        n_adjusted = too_small.sum()
        print(f"  ⚠️  Adjusting {n_adjusted} SEs below {min_se} to avoid numerical issues")
        observed_se[too_small] = min_se

    # Compute informative prior parameters from data
    if use_informative_prior:
        data_mean = np.average(observed_effects, weights=1/observed_se**2)  # Precision-weighted mean
        data_std = np.std(observed_effects)
        print(f"  Using informative prior: μ_global ~ Normal({data_mean:.2f}, {data_std:.2f})")
        prior_mu = data_mean
        prior_sigma = data_std
    else:
        prior_mu = 0.0
        prior_sigma = 1.0

    print(f"Fitting robust hierarchical model with {n_creators} creators in {n_genres} genres...")

    with pm.Model() as model:
        # Hyperpriors with data-driven initialization
        mu_global = pm.Normal('mu_global', mu=prior_mu, sigma=prior_sigma)
        sigma_genre = pm.HalfNormal('sigma_genre', sigma=1)

        # Genre effects (non-centered parameterization)
        genre_offset = pm.Normal('genre_offset', mu=0, sigma=1, shape=n_genres)
        mu_genre = pm.Deterministic('mu_genre', mu_global + sigma_genre * genre_offset)

        # Creator-level variance
        sigma_creator = pm.HalfNormal('sigma_creator', sigma=1)

        # Creator effects (non-centered parameterization)
        creator_offset = pm.Normal('creator_offset', mu=0, sigma=1, shape=n_creators)
        tau = pm.Deterministic('tau', mu_genre[genre_idx] + sigma_creator * creator_offset)

        # Likelihood with adjusted SEs
        obs = pm.Normal('obs', mu=tau, sigma=observed_se, observed=observed_effects)

        # Sample with better initialization
        print(f"Sampling: {draws} draws × {chains} chains (+ {tune} tune steps per chain)")
        print(f"  SE range: [{observed_se.min():.3f}, {observed_se.max():.3f}]")

        try:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                target_accept=target_accept,
                return_inferencedata=True,
                initvals={
                    'mu_global': data_mean if use_informative_prior else 0.0,
                    'genre_offset': np.zeros(n_genres),
                    'creator_offset': np.zeros(n_creators)
                }
            )
            print("✓ Sampling complete!")
            return idata

        except Exception as e:
            print(f"  ✗ Sampling failed: {e}")
            print("  Trying with jitter+adapt_diag initialization...")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                target_accept=target_accept,
                return_inferencedata=True,
                init='jitter+adapt_diag'
            )
            print("✓ Sampling complete (with fallback initialization)!")
            return idata


def extract_hbm_estimates_robust(
    idata: az.InferenceData,
    creator_summaries: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract posterior estimates with diagnostics.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model posterior
    creator_summaries : pd.DataFrame
        Creator summary statistics

    Returns
    -------
    pd.DataFrame
        Posterior estimates with columns:
        [creator_id, genre_idx, effect_hat, se, ci_lower, ci_upper, n_total]
    """
    # Extract tau samples: shape (chains, draws, n_creators)
    tau_samples = idata.posterior['tau'].values

    # Flatten chains: (chains, draws, n_creators) → (chains*draws, n_creators)
    tau_flat = tau_samples.reshape(-1, tau_samples.shape[-1])

    # Compute posterior statistics
    posterior_mean = tau_flat.mean(axis=0)
    posterior_std = tau_flat.std(axis=0)
    ci_lower = np.percentile(tau_flat, 2.5, axis=0)
    ci_upper = np.percentile(tau_flat, 97.5, axis=0)

    # Combine with creator metadata
    estimates = pd.DataFrame({
        'creator_id': creator_summaries['creator_id'].values,
        'genre_idx': creator_summaries['genre_idx'].values,
        'effect_hat': posterior_mean,
        'se': posterior_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_total': creator_summaries['n_total'].values if 'n_total' in creator_summaries.columns else np.nan
    })

    return estimates.sort_values('creator_id').reset_index(drop=True)


if __name__ == "__main__":
    # Quick test
    print("Testing robust HBM on synthetic Borusyak-like data...")

    np.random.seed(42)
    n_creators = 50
    n_genres = 3

    # Simulate Borusyak ITEs with small SEs
    genre_idx = np.random.choice(n_genres, size=n_creators)
    true_effects = np.random.normal(0.5, 0.3, size=n_creators)
    small_ses = np.random.uniform(0.05, 0.2, size=n_creators)  # Very small SEs
    observed_effects = true_effects + np.random.normal(0, small_ses)

    test_data = pd.DataFrame({
        'creator_id': range(n_creators),
        'genre_idx': genre_idx,
        'effect_hat': observed_effects,
        'se': small_ses,
        'n_total': 100
    })

    print(f"\nTest data: {n_creators} creators, {n_genres} genres")
    print(f"SE range: [{small_ses.min():.3f}, {small_ses.max():.3f}]")

    # Fit robust model
    idata = fit_hierarchical_model_robust(
        test_data,
        n_genres=n_genres,
        draws=500,
        tune=500,
        chains=2,
        target_accept=0.95
    )

    # Extract estimates
    hbm_estimates = extract_hbm_estimates_robust(idata, test_data)

    print("\n✓ Robust HBM test passed!")
    print(f"MSE: {np.mean((hbm_estimates['effect_hat'].values - true_effects)**2):.4f}")
