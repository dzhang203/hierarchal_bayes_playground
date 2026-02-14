"""
Enhanced staggered adoption data generation with crossed effects.

This version creates realistic hierarchical structure where:
1. Genre matters (main effect)
2. ARPU quintile matters (main effect)
3. Genre × ARPU interaction matters (crossed effect)
4. Higher variance to reflect noisy revenue data
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Set


def generate_staggered_adoption_data_crossed(
    n_genres: int = 5,
    n_arpu_quintiles: int = 5,
    n_creators_per_cell: int = 8,  # Creators per (genre, ARPU) cell
    n_weeks: int = 52,
    treatment_start_week: int = 12,
    treatment_end_week: int = 40,
    pct_never_treated: float = 0.20,
    noise_scale: float = 2.0,  # Increased from 1.0 - realistic revenue noise
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate panel data with crossed genre × ARPU quintile effects.

    Hierarchical structure:
    1. Genre-level effects: μ_genre ~ Normal(0.5, 0.2)
    2. ARPU-level effects: μ_arpu ~ Normal(0, 0.15)
    3. Interaction effects: μ_genre_arpu ~ Normal(μ_genre + μ_arpu, 0.1)
    4. Creator effects: τ_i ~ Normal(μ_genre_arpu[i], 0.3)

    This creates realistic structure where:
    - Premium users (high ARPU) respond differently than low ARPU
    - Response varies by genre (e.g., gaming vs education)
    - Interaction: premium gaming users respond very differently than premium education

    Parameters
    ----------
    n_genres : int
        Number of genres
    n_arpu_quintiles : int
        Number of ARPU quintiles (5 = quintiles)
    n_creators_per_cell : int
        Number of creators per (genre, ARPU) combination
    n_weeks : int
        Number of weeks in panel
    treatment_start_week : int
        First week where treatment can be adopted
    treatment_end_week : int
        Last week where treatment can be adopted
    pct_never_treated : float
        Fraction of creators who never adopt (control pool)
    noise_scale : float
        Scale of outcome noise (higher = more realistic revenue variance)
    seed : int
        Random seed

    Returns
    -------
    df : pd.DataFrame
        Panel data with columns:
        [creator_id, week, revenue, treated, genre_idx, arpu_quintile]
    truth : dict
        Ground truth containing:
        - genre_effects: genre-level treatment effects
        - arpu_effects: ARPU-level treatment effects
        - interaction_effects: genre × ARPU interaction effects
        - creator_effects: creator-level treatment effects
        - creator_features: creator characteristics
        - never_treated: set of never-treated creator IDs
        - adoption_weeks: dict of adoption weeks by creator
    """
    np.random.seed(seed)

    # Total creators
    n_cells = n_genres * n_arpu_quintiles
    n_total_creators = n_cells * n_creators_per_cell
    n_never_treated = int(n_total_creators * pct_never_treated)
    n_treated = n_total_creators - n_never_treated

    print("="*70)
    print("GENERATING CROSSED-EFFECTS STAGGERED ADOPTION DATA")
    print("="*70)
    print(f"Total creators: {n_total_creators}")
    print(f"Genres: {n_genres}")
    print(f"ARPU quintiles: {n_arpu_quintiles}")
    print(f"Cells (genre × ARPU): {n_cells}")
    print(f"Creators per cell: {n_creators_per_cell}")
    print(f"Weeks: {n_weeks}")
    print(f"Never-treated: {n_never_treated}")
    print(f"Noise scale: {noise_scale}x")
    print()

    # =====================================================================
    # Generate hierarchical treatment effect structure
    # =====================================================================

    # Level 1: Genre main effects
    # Some genres respond better to treatment (e.g., gaming > education)
    genre_effects = np.random.normal(0.5, 0.2, size=n_genres)

    # Level 2: ARPU main effects
    # Higher ARPU users might respond differently
    # Make it non-monotonic (middle ARPU quintiles respond best)
    arpu_effects = np.array([
        -0.1,   # Q1 (low ARPU): negative effect
        0.1,    # Q2: small positive
        0.2,    # Q3 (middle): largest positive
        0.15,   # Q4: moderate positive
        0.05    # Q5 (high ARPU): small positive
    ])[:n_arpu_quintiles]

    # Level 3: Interaction effects (genre × ARPU)
    # Some combinations work especially well/poorly
    interaction_effects = np.zeros((n_genres, n_arpu_quintiles))

    for g in range(n_genres):
        for a in range(n_arpu_quintiles):
            # Base: sum of main effects
            base_effect = genre_effects[g] + arpu_effects[a]

            # Add interaction noise (some combos are special)
            interaction = np.random.normal(0, 0.15)

            # Make certain combinations extra good/bad
            if g == 0 and a == 4:  # Genre 0, high ARPU: extra good
                interaction += 0.3
            elif g == 1 and a == 0:  # Genre 1, low ARPU: extra bad
                interaction -= 0.3
            elif g == 3 and a == 2:  # Genre 3, mid ARPU: extra good
                interaction += 0.25

            interaction_effects[g, a] = base_effect + interaction

    # Print effect structure
    print("Treatment Effect Structure:")
    print("-" * 70)
    print("Genre main effects:")
    for g in range(n_genres):
        print(f"  Genre_{g}: {genre_effects[g]:.3f}")
    print()
    print("ARPU main effects:")
    for a in range(n_arpu_quintiles):
        print(f"  ARPU_Q{a+1}: {arpu_effects[a]:.3f}")
    print()
    print("Interaction effects (genre × ARPU):")
    print("     ", end="")
    for a in range(n_arpu_quintiles):
        print(f"Q{a+1:1d}      ", end="")
    print()
    for g in range(n_genres):
        print(f"G{g}: ", end="")
        for a in range(n_arpu_quintiles):
            print(f"{interaction_effects[g, a]:6.3f}  ", end="")
        print()
    print()

    # =====================================================================
    # Generate creator-level characteristics and effects
    # =====================================================================

    creator_features = []
    creator_effects = {}
    creator_id = 0

    for genre_idx in range(n_genres):
        for arpu_idx in range(n_arpu_quintiles):
            # Creators in this cell
            for _ in range(n_creators_per_cell):
                # Creator-level treatment effect (add heterogeneity)
                cell_mean = interaction_effects[genre_idx, arpu_idx]
                creator_effect = np.random.normal(cell_mean, 0.3)

                # ARPU value (continuous, within quintile)
                # Q1: [0, 20), Q2: [20, 40), ..., Q5: [80, 100)
                arpu_min = arpu_idx * 20
                arpu_max = (arpu_idx + 1) * 20
                arpu_value = np.random.uniform(arpu_min, arpu_max)

                creator_features.append({
                    'creator_id': creator_id,
                    'genre_idx': genre_idx,
                    'arpu_quintile': arpu_idx,
                    'arpu_value': arpu_value,
                    'cell_idx': genre_idx * n_arpu_quintiles + arpu_idx
                })

                creator_effects[creator_id] = creator_effect
                creator_id += 1

    creator_features = pd.DataFrame(creator_features)

    # =====================================================================
    # Assign treatment adoption
    # =====================================================================

    # Randomly select never-treated
    all_creators = list(range(n_total_creators))
    never_treated = set(np.random.choice(all_creators, size=n_never_treated, replace=False))
    treated_creators = [c for c in all_creators if c not in never_treated]

    # Assign adoption weeks (staggered)
    adoption_weeks = {}
    available_weeks = list(range(treatment_start_week, treatment_end_week + 1))

    for creator_id in treated_creators:
        adoption_week = np.random.choice(available_weeks)
        adoption_weeks[creator_id] = adoption_week

    # Count cohorts
    cohort_counts = pd.Series(adoption_weeks.values()).value_counts().sort_index()
    n_cohorts = len(cohort_counts)

    print("Adoption cohorts:")
    for week in sorted(cohort_counts.index[:10]):
        print(f"  Week {week}: {cohort_counts[week]} creators")
    if n_cohorts > 10:
        print(f"  ... ({n_cohorts} cohorts total)")
    print()

    # =====================================================================
    # Generate panel data
    # =====================================================================

    # Fixed effects
    creator_fe = np.random.normal(5, 1.5, size=n_total_creators)  # Baseline revenue
    time_fe = np.random.normal(0, 0.3, size=n_weeks)  # Weekly trends

    # Add seasonal pattern to time FE
    for week in range(n_weeks):
        seasonal = 0.5 * np.sin(2 * np.pi * week / 52)  # Annual seasonality
        time_fe[week] += seasonal

    panel_data = []

    for creator_id in range(n_total_creators):
        for week in range(n_weeks):
            # Base revenue (fixed effects)
            revenue = creator_fe[creator_id] + time_fe[week]

            # Treatment effect
            if creator_id in adoption_weeks and week >= adoption_weeks[creator_id]:
                treatment_effect = creator_effects[creator_id]
                revenue += treatment_effect
                treated = 1
            else:
                treated = 0

            # Add noise (high variance for realism)
            revenue += np.random.normal(0, noise_scale)

            panel_data.append({
                'creator_id': creator_id,
                'week': week,
                'revenue': revenue,
                'treated': treated
            })

    df = pd.DataFrame(panel_data)

    # Merge with creator features
    df = df.merge(creator_features[['creator_id', 'genre_idx', 'arpu_quintile', 'arpu_value']],
                  on='creator_id', how='left')

    print(f"✓ Generated {len(df)} observations")
    print(f"✓ {n_treated} treated, {n_never_treated} never-treated")
    print("="*70)

    # =====================================================================
    # Return data and ground truth
    # =====================================================================

    truth = {
        'genre_effects': genre_effects,
        'arpu_effects': arpu_effects,
        'interaction_effects': interaction_effects,
        'creator_effects': creator_effects,
        'creator_features': creator_features,
        'never_treated': never_treated,
        'adoption_weeks': adoption_weeks,
        'n_genres': n_genres,
        'n_arpu_quintiles': n_arpu_quintiles
    }

    return df, truth


def summarize_crossed_data(df: pd.DataFrame, truth: Dict) -> None:
    """Print summary statistics for crossed-effects data."""

    print("="*70)
    print("CROSSED-EFFECTS DATA SUMMARY")
    print("="*70)
    print()

    # Panel structure
    print("Panel structure:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Creators: {df['creator_id'].nunique()}")
    print(f"  Weeks: {df['week'].nunique()}")
    print(f"  Balanced: {len(df) == df['creator_id'].nunique() * df['week'].nunique()}")
    print()

    # Treatment
    n_treated = len([c for c in truth['adoption_weeks'].keys()])
    n_never = len(truth['never_treated'])
    print("Treatment status:")
    print(f"  Eventually treated: {n_treated}")
    print(f"  Never treated: {n_never}")
    print(f"  Never-treated fraction: {n_never/(n_treated+n_never)*100:.1f}%")
    print()

    # Cells
    print("Crossed structure:")
    print(f"  Genres: {truth['n_genres']}")
    print(f"  ARPU quintiles: {truth['n_arpu_quintiles']}")
    print(f"  Total cells: {truth['n_genres'] * truth['n_arpu_quintiles']}")
    cell_counts = truth['creator_features'].groupby(['genre_idx', 'arpu_quintile']).size()
    print(f"  Creators per cell: min={cell_counts.min()}, median={cell_counts.median():.0f}, max={cell_counts.max()}")
    print()

    # Treatment effects
    effects = np.array(list(truth['creator_effects'].values()))
    print("True treatment effects (creator-level):")
    print(f"  Mean: {effects.mean():.3f}")
    print(f"  Std: {effects.std():.3f}")
    print(f"  Min: {effects.min():.3f}")
    print(f"  Max: {effects.max():.3f}")
    print()

    # Variance decomposition
    features = truth['creator_features']
    merged = features.merge(
        pd.Series(truth['creator_effects'], name='true_effect'),
        left_on='creator_id',
        right_index=True
    )

    total_var = merged['true_effect'].var()

    # Between-genre variance
    genre_means = merged.groupby('genre_idx')['true_effect'].mean()
    between_genre_var = genre_means.var()

    # Between-ARPU variance
    arpu_means = merged.groupby('arpu_quintile')['true_effect'].mean()
    between_arpu_var = arpu_means.var()

    # Between-cell variance (interaction)
    cell_means = merged.groupby(['genre_idx', 'arpu_quintile'])['true_effect'].mean()
    between_cell_var = cell_means.var()

    print("Variance decomposition:")
    print(f"  Total variance: {total_var:.4f}")
    print(f"  Between-genre: {between_genre_var/total_var*100:.1f}%")
    print(f"  Between-ARPU: {between_arpu_var/total_var*100:.1f}%")
    print(f"  Between-cell (genre×ARPU): {between_cell_var/total_var*100:.1f}%")
    print(f"  Within-cell: {(total_var - between_cell_var)/total_var*100:.1f}%")
    print()

    # Revenue
    print("Revenue distribution:")
    print(f"  Mean: ${df['revenue'].mean():.2f}")
    print(f"  Std: ${df['revenue'].std():.2f}")
    print(f"  Min: ${df['revenue'].min():.2f}")
    print(f"  Max: ${df['revenue'].max():.2f}")
    print("="*70)
    print()


if __name__ == "__main__":
    # Test generation
    print("Testing crossed-effects data generation...\n")

    df, truth = generate_staggered_adoption_data_crossed(
        n_genres=5,
        n_arpu_quintiles=5,
        n_creators_per_cell=8,
        n_weeks=52,
        noise_scale=2.0,
        seed=42
    )

    summarize_crossed_data(df, truth)

    # Validation checks
    print("Validation checks:")
    print(f"  ✓ No NaN in revenue: {df['revenue'].isnull().sum() == 0}")
    print(f"  ✓ No NaN in treated: {df['treated'].isnull().sum() == 0}")
    print(f"  ✓ All creators have features: {(df['genre_idx'].isnull().sum() == 0) and (df['arpu_quintile'].isnull().sum() == 0)}")
    print(f"  ✓ Balanced panel: {df.groupby('creator_id').size().nunique() == 1}")
    print("\n✓ Test complete!")
