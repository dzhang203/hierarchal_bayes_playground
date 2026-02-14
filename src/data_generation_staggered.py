"""
Synthetic data generation for staggered adoption setting.

This module generates panel data (creator × time) with staggered treatment adoption,
allowing validation of DiD → HBM pipeline against known ground truth.

Key features:
- Weekly time series data
- Staggered adoption dates across creators
- Never-treated units (control pool)
- Hierarchical treatment effects by genre
- Creator and time fixed effects
- Multiple features: genre, size (followers), ARPU
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from datetime import datetime, timedelta


def generate_staggered_adoption_data(
    n_genres: int = 5,
    n_creators_per_genre: int = 40,
    n_weeks: int = 52,
    treatment_start_week: int = 12,
    treatment_end_week: int = 40,
    pct_never_treated: float = 0.20,
    genre_mean: float = 0.5,
    genre_std: float = 0.3,
    sigma_creator_effect: float = 0.4,
    sigma_time: float = 0.1,
    sigma_obs: float = 0.5,
    baseline_revenue: float = 5.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate synthetic panel data with staggered treatment adoption.

    Data Generating Process (DGP):
    ───────────────────────────────
    For creator i at week t:

    y_it = α_i + λ_t + τ_i × Post_it + ε_it

    Where:
    - α_i: Creator fixed effect (baseline revenue level)
    - λ_t: Time fixed effect (common time trends)
    - τ_i: Individual treatment effect (varies by genre)
    - Post_it: Indicator for post-treatment period
    - ε_it: Observation noise

    Treatment effects are hierarchical:
    - μ_genre[g] ~ Normal(genre_mean, genre_std)  [genre-level effects]
    - τ_i ~ Normal(μ_genre[g_i], sigma_creator_effect)  [creator-level effects]

    Parameters
    ----------
    n_genres : int
        Number of content genres
    n_creators_per_genre : int
        Number of creators per genre
    n_weeks : int
        Total number of weeks in panel
    treatment_start_week : int
        First week any creator can adopt (staggered start)
    treatment_end_week : int
        Last week any creator can adopt
    pct_never_treated : float
        Fraction of creators who never adopt (control pool)
    genre_mean : float
        Global mean treatment effect
    genre_std : float
        Between-genre standard deviation
    sigma_creator_effect : float
        Within-genre standard deviation (creator heterogeneity)
    sigma_time : float
        Magnitude of time fixed effects (common trends)
    sigma_obs : float
        Observation-level noise
    baseline_revenue : float
        Average revenue per creator-week
    seed : int
        Random seed for reproducibility

    Returns
    -------
    df : pd.DataFrame
        Panel data with columns:
        - creator_id: unique creator identifier
        - week: week number (0 to n_weeks-1)
        - date: calendar date
        - genre: genre name
        - genre_idx: numeric genre index
        - size_quartile: creator size (1=small, 4=large)
        - arpu_quartile: creator monetization (1=low, 4=high)
        - treated: 1 if treated in this week, 0 otherwise
        - weeks_since_treatment: weeks since adoption (0 if not treated)
        - revenue: observed revenue

    truth : dict
        Ground truth parameters:
        - genre_effects: true genre-level treatment effects
        - creator_effects: true individual treatment effects
        - creator_fixed_effects: baseline revenue levels (α_i)
        - time_fixed_effects: common time trends (λ_t)
        - adoption_weeks: week each creator adopted (None if never)
        - never_treated: set of never-treated creator IDs
        - creator_features: DataFrame with creator-level features
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Total creators
    n_creators = n_genres * n_creators_per_genre

    # Genre names
    genre_names = [f"Genre_{i}" for i in range(n_genres)]

    print("="*70)
    print("GENERATING STAGGERED ADOPTION DATA")
    print("="*70)
    print(f"Total creators: {n_creators}")
    print(f"Genres: {n_genres}")
    print(f"Weeks: {n_weeks}")
    print(f"Never-treated: {int(pct_never_treated * n_creators)}")
    print()

    # =====================================================================
    # STEP 1: Generate hierarchical treatment effects
    # =====================================================================

    # Genre-level effects: μ_genre[g] ~ Normal(genre_mean, genre_std)
    true_genre_effects = rng.normal(genre_mean, genre_std, size=n_genres)

    # Assign creators to genres
    creator_genre = np.repeat(np.arange(n_genres), n_creators_per_genre)

    # Creator-level effects: τ_i ~ Normal(μ_genre[g_i], sigma_creator_effect)
    true_creator_effects = np.zeros(n_creators)
    for i in range(n_creators):
        g = creator_genre[i]
        true_creator_effects[i] = rng.normal(
            true_genre_effects[g],
            sigma_creator_effect
        )

    print("True genre effects:")
    for i, (name, effect) in enumerate(zip(genre_names, true_genre_effects)):
        print(f"  {name}: {effect:.3f}")
    print()

    # =====================================================================
    # STEP 2: Generate creator features (for grouping selection)
    # =====================================================================

    # Creator fixed effects (baseline revenue level)
    # Correlated with true features to make grouping meaningful
    creator_fixed_effects = rng.normal(baseline_revenue, 1.0, size=n_creators)

    # Size (followers) - log-normal distribution
    # Larger creators tend to have higher baseline revenue
    creator_size = rng.lognormal(mean=10, sigma=1.5, size=n_creators)
    size_quartile = pd.qcut(
        creator_size,
        q=4,
        labels=[1, 2, 3, 4]
    ).astype(int)

    # ARPU (average revenue per user) - also correlated with baseline
    # Higher ARPU creators have higher monetization
    creator_arpu = rng.lognormal(mean=0, sigma=0.5, size=n_creators)
    arpu_quartile = pd.qcut(
        creator_arpu,
        q=4,
        labels=[1, 2, 3, 4]
    ).astype(int)

    # Adjust fixed effects to be correlated with size/ARPU
    # (Makes grouping selection non-trivial - size and ARPU predict baseline,
    #  but we want to know if they predict treatment effects)
    creator_fixed_effects += 0.3 * np.log(creator_size) / np.log(creator_size).std()
    creator_fixed_effects += 0.2 * np.log(creator_arpu) / np.log(creator_arpu).std()

    # =====================================================================
    # STEP 3: Assign adoption dates (staggered)
    # =====================================================================

    n_never_treated = int(pct_never_treated * n_creators)
    n_eventually_treated = n_creators - n_never_treated

    # Never-treated creators (control pool)
    never_treated_ids = set(rng.choice(n_creators, size=n_never_treated, replace=False))

    # Adoption weeks for eventually-treated creators
    # Uniform distribution between treatment_start_week and treatment_end_week
    adoption_weeks = {}
    eventually_treated_ids = [i for i in range(n_creators) if i not in never_treated_ids]

    for creator_id in eventually_treated_ids:
        adoption_week = rng.randint(treatment_start_week, treatment_end_week + 1)
        adoption_weeks[creator_id] = adoption_week

    print(f"Adoption cohorts (creators per week):")
    adoption_counts = pd.Series(adoption_weeks.values()).value_counts().sort_index()
    for week, count in adoption_counts.head(10).items():
        print(f"  Week {week}: {count} creators")
    if len(adoption_counts) > 10:
        print(f"  ... ({len(adoption_counts)} cohorts total)")
    print()

    # =====================================================================
    # STEP 4: Generate time fixed effects (common trends)
    # =====================================================================

    # Time effects: λ_t ~ Normal(0, sigma_time) with some smoothness
    # Use cumulative sum to create trending patterns
    time_shocks = rng.normal(0, sigma_time, size=n_weeks)
    time_fixed_effects = np.cumsum(time_shocks)
    time_fixed_effects -= time_fixed_effects.mean()  # Center at zero

    # =====================================================================
    # STEP 5: Generate panel data
    # =====================================================================

    start_date = datetime(2024, 1, 1)
    data_rows = []

    for creator_id in range(n_creators):
        genre_idx = creator_genre[creator_id]
        alpha_i = creator_fixed_effects[creator_id]
        tau_i = true_creator_effects[creator_id]
        adoption_week = adoption_weeks.get(creator_id, None)

        for week in range(n_weeks):
            # Treatment indicator
            if adoption_week is not None and week >= adoption_week:
                treated = 1
                weeks_since = week - adoption_week
            else:
                treated = 0
                weeks_since = 0

            # Generate outcome: y_it = α_i + λ_t + τ_i × Post_it + ε_it
            lambda_t = time_fixed_effects[week]
            epsilon_it = rng.normal(0, sigma_obs)

            revenue = alpha_i + lambda_t + tau_i * treated + epsilon_it

            data_rows.append({
                'creator_id': creator_id,
                'week': week,
                'date': start_date + timedelta(weeks=week),
                'genre': genre_names[genre_idx],
                'genre_idx': genre_idx,
                'size_quartile': size_quartile[creator_id],
                'arpu_quartile': arpu_quartile[creator_id],
                'treated': treated,
                'weeks_since_treatment': weeks_since,
                'revenue': revenue
            })

    df = pd.DataFrame(data_rows)

    # Create creator features DataFrame
    creator_features = pd.DataFrame({
        'creator_id': range(n_creators),
        'genre': [genre_names[creator_genre[i]] for i in range(n_creators)],
        'genre_idx': creator_genre,
        'size_quartile': size_quartile,
        'arpu_quartile': arpu_quartile,
        'size': creator_size,
        'arpu': creator_arpu,
        'never_treated': [i in never_treated_ids for i in range(n_creators)],
        'adoption_week': [adoption_weeks.get(i, None) for i in range(n_creators)]
    })

    # Package ground truth
    truth = {
        'genre_effects': true_genre_effects,
        'creator_effects': true_creator_effects,
        'creator_fixed_effects': creator_fixed_effects,
        'time_fixed_effects': time_fixed_effects,
        'adoption_weeks': adoption_weeks,
        'never_treated': never_treated_ids,
        'creator_features': creator_features,
        'genre_names': genre_names,
        'n_genres': n_genres,
        'sigma_creator_effect': sigma_creator_effect,
        'sigma_time': sigma_time,
        'sigma_obs': sigma_obs,
        'n_weeks': n_weeks
    }

    print(f"✓ Generated {len(df):,} observations")
    print(f"✓ {n_eventually_treated} treated, {n_never_treated} never-treated")
    print("="*70)

    return df, truth


def summarize_staggered_data(df: pd.DataFrame, truth: Dict) -> None:
    """
    Print summary statistics about the staggered adoption data.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    truth : dict
        Ground truth parameters
    """
    print("="*70)
    print("STAGGERED ADOPTION DATA SUMMARY")
    print("="*70)

    n_creators = df['creator_id'].nunique()
    n_weeks = df['week'].nunique()

    print(f"\nPanel structure:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Creators: {n_creators}")
    print(f"  Weeks: {n_weeks}")
    print(f"  Balanced panel: {len(df) == n_creators * n_weeks}")

    # Treatment status
    treated_creators = df.groupby('creator_id')['treated'].max()
    n_treated = treated_creators.sum()
    n_never_treated = len(treated_creators) - n_treated

    print(f"\nTreatment status:")
    print(f"  Eventually treated: {n_treated}")
    print(f"  Never treated: {n_never_treated}")
    print(f"  Never-treated fraction: {n_never_treated/len(treated_creators):.1%}")

    # Adoption timing
    adoption_weeks = truth['adoption_weeks']
    if adoption_weeks:
        adoption_list = list(adoption_weeks.values())
        print(f"\nAdoption timing:")
        print(f"  First adoption: Week {min(adoption_list)}")
        print(f"  Last adoption: Week {max(adoption_list)}")
        print(f"  Number of cohorts: {len(set(adoption_list))}")

    # Post-treatment periods
    treated_df = df[df['treated'] == 1]
    if len(treated_df) > 0:
        post_periods = treated_df.groupby('creator_id')['weeks_since_treatment'].max()
        print(f"\nPost-treatment periods:")
        print(f"  Min: {post_periods.min()}")
        print(f"  Median: {post_periods.median():.0f}")
        print(f"  Max: {post_periods.max()}")

    # True effects
    print(f"\nTrue treatment effects:")
    print(f"  Genre-level effects:")
    for i, (name, effect) in enumerate(zip(truth['genre_names'], truth['genre_effects'])):
        print(f"    {name}: {effect:.3f}")

    print(f"\n  Creator-level effects:")
    print(f"    Mean: {truth['creator_effects'].mean():.3f}")
    print(f"    Std: {truth['creator_effects'].std():.3f}")
    print(f"    Min: {truth['creator_effects'].min():.3f}")
    print(f"    Max: {truth['creator_effects'].max():.3f}")

    # Features
    print(f"\nCreator features:")
    features = truth['creator_features']
    print(f"  Genres: {features['genre'].nunique()}")
    print(f"  Size quartiles: {features['size_quartile'].nunique()}")
    print(f"  ARPU quartiles: {features['arpu_quartile'].nunique()}")

    # Revenue distribution
    print(f"\nRevenue distribution:")
    print(f"  Mean: ${df['revenue'].mean():.2f}")
    print(f"  Std: ${df['revenue'].std():.2f}")
    print(f"  Min: ${df['revenue'].min():.2f}")
    print(f"  Max: ${df['revenue'].max():.2f}")

    print("="*70)


if __name__ == "__main__":
    # Test data generation
    df, truth = generate_staggered_adoption_data(seed=42)
    summarize_staggered_data(df, truth)

    # Quick validation
    print("\nQuick validation:")
    print(f"✓ No missing values: {df.isnull().sum().sum() == 0}")
    print(f"✓ Balanced panel: {df.groupby('creator_id').size().nunique() == 1}")
    print(f"✓ Revenue is numeric: {pd.api.types.is_numeric_dtype(df['revenue'])}")

    # Check parallel trends visually (should be flat pre-treatment)
    print("\n✓ Data generation complete!")
