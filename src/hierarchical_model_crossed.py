"""
Hierarchical Bayesian model with CROSSED effects (genre × ARPU).

This extends the base HBM to handle interactions between grouping variables.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Optional


def fit_crossed_hbm(
    creator_summaries: pd.DataFrame,
    n_genres: int,
    n_arpu_quintiles: int,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42,
    target_accept: float = 0.95,
    min_se: float = 0.05,
    use_informative_prior: bool = True
) -> az.InferenceData:
    """
    Fit hierarchical model with genre × ARPU crossed effects.

    Model specification:
        # Hyperpriors
        mu_global ~ Normal(data_mean, data_std)  # If informative
        sigma_cell ~ HalfNormal(1)               # Between-cell variation
        sigma_creator ~ HalfNormal(1)            # Within-cell variation

        # Cell-level effects (genre × ARPU combinations)
        For each cell c in 1...(n_genres × n_arpu_quintiles):
            mu_cell[c] ~ Normal(mu_global, sigma_cell)

        # Creator-level effects
        For each creator i:
            cell_i = genre[i] * n_arpu + arpu[i]
            tau[i] ~ Normal(mu_cell[cell_i], sigma_creator)

        # Likelihood
        effect_hat[i] ~ Normal(tau[i], se[i])

    Parameters
    ----------
    creator_summaries : pd.DataFrame
        Must contain: creator_id, genre_idx, arpu_quintile, effect_hat, se
    n_genres : int
        Number of genres
    n_arpu_quintiles : int
        Number of ARPU quintiles
    draws : int
        MCMC draws per chain
    tune : int
        MCMC tuning steps
    chains : int
        Number of MCMC chains
    random_seed : int
        Random seed
    target_accept : float
        Target acceptance rate (0.95 = conservative)
    min_se : float
        Minimum SE threshold to avoid numerical issues
    use_informative_prior : bool
        Use data-driven prior for mu_global

    Returns
    -------
    az.InferenceData
        Posterior samples and diagnostics
    """
    # Validate inputs
    required_cols = ['creator_id', 'genre_idx', 'arpu_quintile', 'effect_hat', 'se']
    missing = [c for c in required_cols if c not in creator_summaries.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for NaN values
    if creator_summaries[['effect_hat', 'se']].isnull().any().any():
        print("⚠️  WARNING: Found NaN values in effect_hat or se. Dropping affected rows.")
        creator_summaries = creator_summaries.dropna(subset=['effect_hat', 'se'])

    # Extract data
    genre_idx = creator_summaries['genre_idx'].values.astype(int)
    arpu_idx = creator_summaries['arpu_quintile'].values.astype(int)
    observed_effects = creator_summaries['effect_hat'].values
    observed_se = creator_summaries['se'].values.copy()
    n_creators = len(creator_summaries)
    n_cells = n_genres * n_arpu_quintiles

    # Validate indices
    if genre_idx.max() >= n_genres or genre_idx.min() < 0:
        raise ValueError(f"genre_idx out of range: [{genre_idx.min()}, {genre_idx.max()}], expected [0, {n_genres-1}]")
    if arpu_idx.max() >= n_arpu_quintiles or arpu_idx.min() < 0:
        raise ValueError(f"arpu_quintile out of range: [{arpu_idx.min()}, {arpu_idx.max()}], expected [0, {n_arpu_quintiles-1}]")

    # Compute cell index: cell = genre * n_arpu + arpu
    cell_idx = genre_idx * n_arpu_quintiles + arpu_idx

    # Verify cell indices
    if cell_idx.max() >= n_cells or cell_idx.min() < 0:
        raise ValueError(f"cell_idx out of range: [{cell_idx.min()}, {cell_idx.max()}], expected [0, {n_cells-1}]")

    # Apply minimum SE threshold
    too_small = observed_se < min_se
    if too_small.any():
        n_adjusted = too_small.sum()
        print(f"  ⚠️  Adjusting {n_adjusted} SEs below {min_se} to avoid numerical issues")
        observed_se[too_small] = min_se

    # Compute informative prior
    if use_informative_prior:
        data_mean = np.average(observed_effects, weights=1/observed_se**2)
        data_std = np.std(observed_effects)
        print(f"  Using informative prior: μ_global ~ Normal({data_mean:.2f}, {data_std:.2f})")
        prior_mu = data_mean
        prior_sigma = data_std
    else:
        prior_mu = 0.0
        prior_sigma = 1.0

    print(f"Fitting crossed HBM: {n_creators} creators, {n_genres} genres × {n_arpu_quintiles} ARPU = {n_cells} cells")

    # Check cell occupancy
    cell_counts = pd.Series(cell_idx).value_counts()
    min_cell_size = cell_counts.min()
    empty_cells = n_cells - len(cell_counts)

    print(f"  Cell sizes: min={min_cell_size}, median={cell_counts.median():.0f}, max={cell_counts.max()}")
    if empty_cells > 0:
        print(f"  ⚠️  WARNING: {empty_cells} empty cells (will have weak posteriors)")
    if min_cell_size < 3:
        print(f"  ⚠️  WARNING: Some cells have < 3 creators (estimates may be unstable)")

    with pm.Model() as model:
        # Hyperpriors (global level)
        mu_global = pm.Normal('mu_global', mu=prior_mu, sigma=prior_sigma)
        sigma_cell = pm.HalfNormal('sigma_cell', sigma=1)  # Between-cell variation
        sigma_creator = pm.HalfNormal('sigma_creator', sigma=1)  # Within-cell variation

        # Cell-level effects (non-centered parameterization)
        cell_offset = pm.Normal('cell_offset', mu=0, sigma=1, shape=n_cells)
        mu_cell = pm.Deterministic('mu_cell', mu_global + sigma_cell * cell_offset)

        # Creator-level effects (non-centered parameterization)
        creator_offset = pm.Normal('creator_offset', mu=0, sigma=1, shape=n_creators)
        tau = pm.Deterministic('tau', mu_cell[cell_idx] + sigma_creator * creator_offset)

        # Likelihood
        obs = pm.Normal('obs', mu=tau, sigma=observed_se, observed=observed_effects)

        # Sample
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
                    'cell_offset': np.zeros(n_cells),
                    'creator_offset': np.zeros(n_creators)
                }
            )
            print("✓ Sampling complete!")
            return idata

        except Exception as e:
            print(f"  ✗ Sampling failed with custom init: {e}")
            print("  Trying with jitter+adapt_diag...")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                target_accept=target_accept,
                return_inferencedata=True,
                init='jitter+adapt_diag'
            )
            print("✓ Sampling complete (with fallback init)!")
            return idata


def extract_crossed_estimates(
    idata: az.InferenceData,
    creator_summaries: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract posterior estimates from crossed HBM.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model posterior
    creator_summaries : pd.DataFrame
        Creator metadata

    Returns
    -------
    pd.DataFrame
        Posterior estimates with columns:
        [creator_id, genre_idx, arpu_quintile, effect_hat, se, ci_lower, ci_upper]
    """
    # Extract tau samples
    tau_samples = idata.posterior['tau'].values

    # Flatten chains
    tau_flat = tau_samples.reshape(-1, tau_samples.shape[-1])

    # Posterior statistics
    posterior_mean = tau_flat.mean(axis=0)
    posterior_std = tau_flat.std(axis=0)
    ci_lower = np.percentile(tau_flat, 2.5, axis=0)
    ci_upper = np.percentile(tau_flat, 97.5, axis=0)

    # Combine with metadata
    estimates = pd.DataFrame({
        'creator_id': creator_summaries['creator_id'].values,
        'genre_idx': creator_summaries['genre_idx'].values,
        'arpu_quintile': creator_summaries['arpu_quintile'].values,
        'effect_hat': posterior_mean,
        'se': posterior_std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })

    return estimates.sort_values('creator_id').reset_index(drop=True)


def extract_cell_effects(
    idata: az.InferenceData,
    n_genres: int,
    n_arpu_quintiles: int
) -> pd.DataFrame:
    """
    Extract cell-level (genre × ARPU) treatment effects from posterior.

    Parameters
    ----------
    idata : az.InferenceData
        Fitted model posterior
    n_genres : int
        Number of genres
    n_arpu_quintiles : int
        Number of ARPU quintiles

    Returns
    -------
    pd.DataFrame
        Cell-level effects with columns:
        [genre_idx, arpu_quintile, effect_mean, effect_std, ci_lower, ci_upper]
    """
    # Extract mu_cell samples
    mu_cell_samples = idata.posterior['mu_cell'].values

    # Flatten chains
    mu_cell_flat = mu_cell_samples.reshape(-1, mu_cell_samples.shape[-1])

    # Posterior statistics
    cell_mean = mu_cell_flat.mean(axis=0)
    cell_std = mu_cell_flat.std(axis=0)
    cell_lower = np.percentile(mu_cell_flat, 2.5, axis=0)
    cell_upper = np.percentile(mu_cell_flat, 97.5, axis=0)

    # Convert cell index to genre × ARPU
    cell_effects = []
    for cell_id in range(len(cell_mean)):
        genre_idx = cell_id // n_arpu_quintiles
        arpu_idx = cell_id % n_arpu_quintiles

        cell_effects.append({
            'genre_idx': genre_idx,
            'arpu_quintile': arpu_idx,
            'cell_idx': cell_id,
            'effect_mean': cell_mean[cell_id],
            'effect_std': cell_std[cell_id],
            'ci_lower': cell_lower[cell_id],
            'ci_upper': cell_upper[cell_id]
        })

    return pd.DataFrame(cell_effects)


if __name__ == "__main__":
    # Quick test
    print("Testing crossed HBM...\n")

    np.random.seed(42)
    n_creators = 50
    n_genres = 3
    n_arpu = 3

    # Simulate data with crossed structure
    genre_idx = np.random.choice(n_genres, size=n_creators)
    arpu_idx = np.random.choice(n_arpu, size=n_creators)

    # True cell effects (genre × ARPU)
    true_cell_effects = np.random.normal(0.5, 0.2, size=n_genres * n_arpu)
    cell_idx = genre_idx * n_arpu + arpu_idx
    true_effects = true_cell_effects[cell_idx] + np.random.normal(0, 0.1, size=n_creators)

    # Observed effects with noise
    small_ses = np.random.uniform(0.08, 0.15, size=n_creators)
    observed_effects = true_effects + np.random.normal(0, small_ses)

    test_data = pd.DataFrame({
        'creator_id': range(n_creators),
        'genre_idx': genre_idx,
        'arpu_quintile': arpu_idx,
        'effect_hat': observed_effects,
        'se': small_ses
    })

    print(f"Test data: {n_creators} creators, {n_genres} genres × {n_arpu} ARPU = {n_genres*n_arpu} cells")

    # Fit model
    idata = fit_crossed_hbm(
        test_data,
        n_genres=n_genres,
        n_arpu_quintiles=n_arpu,
        draws=500,
        tune=500,
        chains=2,
        target_accept=0.95
    )

    # Extract estimates
    estimates = extract_crossed_estimates(idata, test_data)
    cell_effects = extract_cell_effects(idata, n_genres, n_arpu)

    print("\n✓ Crossed HBM test passed!")
    print(f"MSE: {np.mean((estimates['effect_hat'].values - true_effects)**2):.4f}")
    print(f"\nCell effects recovered:")
    print(cell_effects[['genre_idx', 'arpu_quintile', 'effect_mean']].head(9))
