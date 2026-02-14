"""
Synthetic data generation for creator A/B experiments.

This module generates synthetic experiment data with a known hierarchical structure,
allowing us to validate that the Bayesian model can recover ground truth parameters.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def generate_experiment_data(
    n_genres: int = 5,
    n_creators_per_genre: int = 100,
    genre_mean: float = 0.5,
    genre_std: float = 0.3,
    sigma_creator: float = 0.4,
    sigma_obs: float = 2.0,
    baseline_revenue: float = 5.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic A/B experiment data with hierarchical structure.

    Data Generating Process (DGP):
    1. Genre-level effects: mu_genre[g] ~ Normal(genre_mean, genre_std)
    2. Creator-level effects: tau[i] ~ Normal(mu_genre[genre_of_i], sigma_creator)
    3. Sample sizes: highly variable (30-5000 per creator)
    4. User-level observations: revenue ~ Normal(baseline ± tau[i], sigma_obs)

    Parameters
    ----------
    n_genres : int
        Number of content genres (e.g., comedy, music, gaming)
    n_creators_per_genre : int
        Number of creators per genre
    genre_mean : float
        Global mean treatment effect across all genres
    genre_std : float
        Standard deviation of genre-level effects
    sigma_creator : float
        Within-genre standard deviation (how much creators vary within a genre)
    sigma_obs : float
        Observation-level noise (individual user revenue variance)
    baseline_revenue : float
        Average revenue per user in control group (dollars)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    df : pd.DataFrame
        User-level data with columns:
        - creator_id: unique identifier for each creator
        - genre: genre name
        - genre_idx: numeric genre index (0 to n_genres-1)
        - group: 'treatment' or 'control'
        - revenue: revenue per user in dollars

    truth : dict
        Ground truth parameters:
        - genre_effects: array of true genre-level mean effects
        - creator_effects: array of true creator-level effects
        - sigma_creator: within-genre standard deviation
        - sigma_obs: observation noise standard deviation
        - creator_genre: mapping from creator_id to genre_idx
        - creator_n: sample size per creator
        - genre_names: list of genre names
    """
    rng = np.random.RandomState(seed)

    # Total number of creators
    n_creators = n_genres * n_creators_per_genre

    # Generate genre names
    genre_names = [f"Genre_{i}" for i in range(n_genres)]

    # Step 1: Generate true genre-level effects
    # These are the "true" average treatment effects for each genre
    true_genre_effects = rng.normal(genre_mean, genre_std, size=n_genres)

    # Step 2: Assign each creator to a genre
    creator_genre = np.repeat(np.arange(n_genres), n_creators_per_genre)

    # Step 3: Generate true creator-level treatment effects
    # Each creator's effect is drawn from their genre's distribution
    true_creator_effects = np.zeros(n_creators)
    for i in range(n_creators):
        genre_idx = creator_genre[i]
        true_creator_effects[i] = rng.normal(
            true_genre_effects[genre_idx],
            sigma_creator
        )

    # Step 4: Generate highly variable sample sizes
    # Mix of small, medium, and large creators (realistic distribution)
    creator_sample_sizes = np.zeros(n_creators, dtype=int)

    # 40% small creators (30-100 users)
    n_small = int(0.4 * n_creators)
    creator_sample_sizes[:n_small] = rng.randint(30, 101, size=n_small)

    # 35% medium creators (100-500 users)
    n_medium = int(0.35 * n_creators)
    creator_sample_sizes[n_small:n_small+n_medium] = rng.randint(100, 501, size=n_medium)

    # 25% large creators (500-5000 users)
    n_large = n_creators - n_small - n_medium
    creator_sample_sizes[n_small+n_medium:] = rng.randint(500, 5001, size=n_large)

    # Shuffle to mix small/medium/large across genres
    rng.shuffle(creator_sample_sizes)

    # Step 5: Generate user-level observations
    data_rows = []

    for creator_id in range(n_creators):
        n_total = creator_sample_sizes[creator_id]
        n_treatment = n_total // 2
        n_control = n_total - n_treatment

        genre_idx = creator_genre[creator_id]
        true_effect = true_creator_effects[creator_id]

        # Control group: revenue ~ Normal(baseline, sigma_obs)
        control_revenue = rng.normal(
            baseline_revenue,
            sigma_obs,
            size=n_control
        )

        # Treatment group: revenue ~ Normal(baseline + true_effect, sigma_obs)
        treatment_revenue = rng.normal(
            baseline_revenue + true_effect,
            sigma_obs,
            size=n_treatment
        )

        # Add control observations
        for rev in control_revenue:
            data_rows.append({
                'creator_id': creator_id,
                'genre': genre_names[genre_idx],
                'genre_idx': genre_idx,
                'group': 'control',
                'revenue': rev
            })

        # Add treatment observations
        for rev in treatment_revenue:
            data_rows.append({
                'creator_id': creator_id,
                'genre': genre_names[genre_idx],
                'genre_idx': genre_idx,
                'group': 'treatment',
                'revenue': rev
            })

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Package ground truth
    truth = {
        'genre_effects': true_genre_effects,
        'creator_effects': true_creator_effects,
        'sigma_creator': sigma_creator,
        'sigma_obs': sigma_obs,
        'creator_genre': creator_genre,
        'creator_n': creator_sample_sizes,
        'genre_names': genre_names,
        'n_genres': n_genres,
        'baseline_revenue': baseline_revenue
    }

    return df, truth


def summarize_data(df: pd.DataFrame, truth: Dict) -> None:
    """
    Print summary statistics about the generated data.

    Parameters
    ----------
    df : pd.DataFrame
        User-level experiment data
    truth : dict
        Ground truth parameters
    """
    print("=" * 60)
    print("SYNTHETIC DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal observations: {len(df):,}")
    print(f"Total creators: {df['creator_id'].nunique()}")
    print(f"Number of genres: {truth['n_genres']}")

    # Sample size distribution
    creator_sizes = df.groupby('creator_id').size()
    print(f"\nSample size distribution:")
    print(f"  Min: {creator_sizes.min()}")
    print(f"  25th percentile: {creator_sizes.quantile(0.25):.0f}")
    print(f"  Median: {creator_sizes.median():.0f}")
    print(f"  75th percentile: {creator_sizes.quantile(0.75):.0f}")
    print(f"  Max: {creator_sizes.max()}")

    # True genre effects
    print(f"\nTrue genre-level treatment effects:")
    for i, (genre_name, effect) in enumerate(zip(truth['genre_names'], truth['genre_effects'])):
        print(f"  {genre_name}: ${effect:.3f}")

    # Creator effect distribution
    print(f"\nCreator-level effect distribution:")
    print(f"  Mean: ${truth['creator_effects'].mean():.3f}")
    print(f"  Std: ${truth['creator_effects'].std():.3f}")
    print(f"  Min: ${truth['creator_effects'].min():.3f}")
    print(f"  Max: ${truth['creator_effects'].max():.3f}")

    # Signal-to-noise ratio
    print(f"\nNoise parameters:")
    print(f"  Within-genre SD (sigma_creator): ${truth['sigma_creator']:.2f}")
    print(f"  Observation noise (sigma_obs): ${truth['sigma_obs']:.2f}")
    print(f"  Baseline revenue: ${truth['baseline_revenue']:.2f}")

    # Expected standard error for a small creator
    small_n = 60
    expected_se = truth['sigma_obs'] * np.sqrt(2 / (small_n / 2))
    print(f"\nExpected SE for creator with n={small_n}: ${expected_se:.2f}")
    print(f"  (This is why small creators need hierarchical modeling!)")

    print("=" * 60)


if __name__ == "__main__":
    # Test the data generation
    df, truth = generate_experiment_data()
    summarize_data(df, truth)

    # Quick validation
    print("\nQuick validation checks:")
    print(f"✓ No missing values: {df.isnull().sum().sum() == 0}")
    print(f"✓ All creators have both groups: {df.groupby('creator_id')['group'].nunique().min() == 2}")
    print(f"✓ Revenue is numeric: {pd.api.types.is_numeric_dtype(df['revenue'])}")
