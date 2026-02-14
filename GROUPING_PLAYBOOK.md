# Grouping Selection Playbook for HBM

## The Critical Question: How to Find Meaningful Groups?

You're absolutely right - **finding meaningful groups is the most important step** in hierarchical Bayesian modeling. Poor grouping → weak shrinkage → minimal HBM benefit. This playbook provides a systematic approach for discovering and validating group structures in your empirical data.

---

## Table of Contents

1. [Why Grouping Matters](#why-grouping-matters)
2. [Phase 1: Exploratory Analysis](#phase-1-exploratory-analysis)
3. [Phase 2: Candidate Generation](#phase-2-candidate-generation)
4. [Phase 3: Statistical Validation](#phase-3-statistical-validation)
5. [Phase 4: Model Comparison](#phase-4-model-comparison)
6. [Phase 5: Diagnostic Checks](#phase-5-diagnostic-checks)
7. [Production Workflow](#production-workflow)
8. [Common Pitfalls](#common-pitfalls)

---

## Why Grouping Matters

### The Goal

Find groupings where **between-group variance** >> **within-group variance** for treatment effects.

**Good grouping** (genre example):
```
Genre A: τ = [0.8, 0.9, 0.7,  0.85] → mean = 0.81, sd = 0.07
Genre B: τ = [0.2, 0.3, 0.25, 0.28] → mean = 0.26, sd = 0.04
→ Between-genre difference: 0.55 >> within-genre variation: 0.05
→ Shrinkage is helpful!
```

**Bad grouping** (random example):
```
Group 1: τ = [0.2, 0.9, 0.3, 0.7] → mean = 0.53, sd = 0.31
Group 2: τ = [0.8, 0.3, 0.6, 0.4] → mean = 0.53, sd = 0.22
→ No between-group difference, high within-group variation
→ Shrinkage hurts more than helps!
```

### Quantifying Grouping Quality

**Variance Decomposition:**
```python
total_var = var(all_treatment_effects)
between_group_var = var(group_means)
within_group_var = mean(group_variances)

# Good grouping: between_group_var / total_var > 0.3
grouping_quality = between_group_var / total_var
```

**Rule of Thumb:**
- 🔴 < 20%: Poor grouping (don't use HBM)
- 🟡 20-40%: Moderate (HBM helps somewhat)
- 🟢 > 40%: Good grouping (strong HBM benefits)

---

## Phase 1: Exploratory Analysis

### Step 1.1: Visualize Raw Treatment Effects

**Before you have ITEs** (during pilot or on historical data):

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Assume you have historical rollout data with:
# - creator_id
# - feature (potential grouping variable)
# - revenue_pre (before rollout)
# - revenue_post (after rollout)

df['lift'] = (df['revenue_post'] - df['revenue_pre']) / df['revenue_pre']

# Plot distributions by candidate grouping
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# By genre
sns.boxplot(data=df, x='genre', y='lift', ax=axes[0, 0])
axes[0, 0].set_title('Revenue Lift by Genre')
axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)

# By size quartile
sns.boxplot(data=df, x='size_quartile', y='lift', ax=axes[0, 1])
axes[0, 1].set_title('Revenue Lift by Creator Size')

# By ARPU quintile
sns.boxplot(data=df, x='arpu_quintile', y='lift', ax=axes[1, 0])
axes[1, 1].set_title('Revenue Lift by ARPU Quintile')

# By engagement quartile
sns.boxplot(data=df, x='engagement_quartile', y='lift', ax=axes[1, 1])
axes[1, 1].set_title('Revenue Lift by Engagement')

plt.tight_layout()
plt.savefig('grouping_exploration.png', dpi=300)
plt.show()
```

**What to look for:**
- ✅ Clear separation between boxes (different medians)
- ✅ Non-overlapping interquartile ranges
- ✅ Consistent ordering (monotonic pattern)
- ❌ All boxes overlapping → grouping doesn't matter

### Step 1.2: Compute Descriptive Statistics

```python
def analyze_grouping_candidate(df, grouping_var, outcome='lift'):
    """
    Analyze how well a grouping variable explains outcome variation.
    """
    # Group statistics
    group_stats = df.groupby(grouping_var)[outcome].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ])

    print(f"\n{'='*70}")
    print(f"Grouping Analysis: {grouping_var}")
    print(f"{'='*70}")
    print(group_stats)
    print()

    # Variance decomposition
    total_var = df[outcome].var()
    group_means = df.groupby(grouping_var)[outcome].mean()
    between_var = group_means.var()
    within_var = df.groupby(grouping_var)[outcome].var().mean()

    quality_score = between_var / total_var * 100

    print(f"Variance Decomposition:")
    print(f"  Total variance:   {total_var:.4f}")
    print(f"  Between-group:    {between_var:.4f} ({between_var/total_var*100:.1f}%)")
    print(f"  Within-group:     {within_var:.4f} ({within_var/total_var*100:.1f}%)")
    print()
    print(f"Grouping Quality Score: {quality_score:.1f}%")

    if quality_score > 40:
        print("  → ✅ EXCELLENT grouping (strong HBM benefits expected)")
    elif quality_score > 20:
        print("  → 🟡 MODERATE grouping (modest HBM benefits)")
    else:
        print("  → 🔴 POOR grouping (HBM unlikely to help)")

    print(f"{'='*70}\n")

    return quality_score

# Test all candidates
candidates = ['genre', 'size_quartile', 'arpu_quintile', 'engagement_quartile']
scores = {}

for candidate in candidates:
    scores[candidate] = analyze_grouping_candidate(df, candidate)

# Rank candidates
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print("\nCandidates Ranked by Quality:")
for rank, (name, score) in enumerate(ranked, 1):
    print(f"  {rank}. {name}: {score:.1f}%")
```

### Step 1.3: Check Group Sizes

```python
def check_group_sizes(df, grouping_var, min_size=10):
    """
    Verify all groups have sufficient sample size.
    """
    sizes = df.groupby(grouping_var).size()

    print(f"\nGroup Sizes for '{grouping_var}':")
    print(sizes)
    print(f"\n  Min: {sizes.min()}")
    print(f"  Median: {sizes.median()}")
    print(f"  Max: {sizes.max()}")

    if sizes.min() < min_size:
        print(f"\n  ⚠️  WARNING: {(sizes < min_size).sum()} groups have < {min_size} creators")
        print(f"  Consider:")
        print(f"    - Merging small groups")
        print(f"    - Using coarser grouping (e.g., tertiles instead of quintiles)")
        print(f"    - Excluding very small groups")
    else:
        print(f"\n  ✅ All groups have >= {min_size} creators")

    return sizes

for candidate in candidates:
    check_group_sizes(df, candidate, min_size=15)
```

---

## Phase 2: Candidate Generation

### Strategy 1: Domain Knowledge (Start Here!)

**Questions to ask:**

1. **Product Perspective:**
   - Which creators would respond similarly to this feature?
   - Are there natural segments in your product (e.g., gaming vs education)?
   - Do different creator types use the feature differently?

2. **Historical Patterns:**
   - In past rollouts, what segments showed different responses?
   - Are there persistent creator archetypes in your data?

3. **Mechanism Thinking:**
   - *Why* might some creators benefit more?
   - What underlying factors drive differential response?

**Example: Creator Platform**

```python
# Domain-driven candidates
grouping_candidates = {
    # Content-based
    'genre': ['Gaming', 'Education', 'Entertainment', 'Lifestyle', 'Other'],
    'content_type': ['Video', 'Livestream', 'Shorts', 'Mixed'],

    # Audience-based
    'audience_size_tier': ['Micro (<10K)', 'Mid (10K-100K)', 'Macro (100K-1M)', 'Mega (1M+)'],
    'audience_engagement': ['Low', 'Medium', 'High', 'Very High'],  # Based on percentiles

    # Monetization-based
    'arpu_tier': ['Q1 (low)', 'Q2', 'Q3', 'Q4', 'Q5 (high)'],
    'monetization_method': ['Ads', 'Subscriptions', 'Donations', 'Mixed'],

    # Activity-based
    'posting_frequency': ['Daily', 'Weekly', 'Monthly', 'Irregular'],
    'platform_tenure': ['New (<6mo)', 'Growing (6mo-2y)', 'Established (2y+)']
}
```

### Strategy 2: Data-Driven Discovery

**Unsupervised Clustering on Pre-Treatment Features:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Select pre-treatment features
feature_cols = [
    'avg_revenue_pre',
    'avg_views_pre',
    'avg_engagement_pre',
    'follower_count',
    'content_diversity_score',
    'posting_consistency'
]

X = df[feature_cols].fillna(df[feature_cols].median())
X_scaled = StandardScaler().fit_transform(X)

# Elbow method for optimal clusters
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.savefig('elbow_plot.png')

# Fit with optimal K (choose from elbow)
optimal_k = 5  # Based on elbow
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['data_driven_cluster'] = kmeans.fit_predict(X_scaled)

# Characterize clusters
print("\nCluster Characteristics:")
cluster_profiles = df.groupby('data_driven_cluster')[feature_cols].mean()
print(cluster_profiles)

# Add to candidates
candidates.append('data_driven_cluster')
```

### Strategy 3: Interaction Effects

Sometimes the best grouping is a **combination**:

```python
# Create crossed groupings
df['genre_x_size'] = df['genre'].astype(str) + '_' + df['size_tier'].astype(str)
df['genre_x_arpu'] = df['genre'].astype(str) + '_' + df['arpu_tier'].astype(str)

# But check cell sizes!
cell_sizes = df.groupby('genre_x_arpu').size()
print(f"Crossed cells: {len(cell_sizes)}")
print(f"Min cell size: {cell_sizes.min()}")

if cell_sizes.min() < 10:
    print("⚠️  Some cells too small, consider merging or using main effects only")
```

---

## Phase 3: Statistical Validation

### Test 1: ANOVA F-test

**Null hypothesis:** Group means are all equal

```python
from scipy import stats

def anova_test(df, grouping_var, outcome='lift'):
    """
    Test if group means differ significantly.
    """
    groups = [group[outcome].values for name, group in df.groupby(grouping_var)]

    f_stat, p_value = stats.f_oneway(*groups)

    print(f"\nANOVA Test: {grouping_var}")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")

    if p_value < 0.001:
        print("  → ✅ HIGHLY SIGNIFICANT group differences")
    elif p_value < 0.05:
        print("  → ✓ Significant group differences")
    else:
        print("  → ✗ No significant group differences")

    return f_stat, p_value

for candidate in candidates:
    anova_test(df, candidate)
```

### Test 2: Kruskal-Wallis (Non-Parametric)

**For non-normal distributions:**

```python
def kruskal_test(df, grouping_var, outcome='lift'):
    """
    Non-parametric test for group differences.
    """
    groups = [group[outcome].values for name, group in df.groupby(grouping_var)]

    h_stat, p_value = stats.kruskal(*groups)

    print(f"\nKruskal-Wallis Test: {grouping_var}")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")

    return h_stat, p_value

for candidate in candidates:
    kruskal_test(df, candidate)
```

### Test 3: Effect Size (Eta-squared)

**How much variance is explained?**

```python
def compute_eta_squared(df, grouping_var, outcome='lift'):
    """
    Compute η² (proportion of variance explained by grouping).
    """
    # Total sum of squares
    grand_mean = df[outcome].mean()
    ss_total = ((df[outcome] - grand_mean) ** 2).sum()

    # Between-group sum of squares
    group_means = df.groupby(grouping_var)[outcome].transform('mean')
    ss_between = ((group_means - grand_mean) ** 2).sum()

    eta_squared = ss_between / ss_total

    print(f"\n η² for {grouping_var}: {eta_squared:.4f} ({eta_squared*100:.1f}%)")

    if eta_squared > 0.14:
        print("  → Large effect size")
    elif eta_squared > 0.06:
        print("  → Medium effect size")
    else:
        print("  → Small effect size")

    return eta_squared

for candidate in candidates:
    compute_eta_squared(df, candidate)
```

---

## Phase 4: Model Comparison (The Gold Standard)

### Approach: LOO-IC on Real Data

**After you have ITEs from Borusyak:**

```python
import arviz as az
from src.hierarchical_model_robust import fit_hierarchical_model_robust
from src.grouping_selection import create_grouping

def compare_groupings_empirical(
    ite_estimates,
    creator_features,
    candidates,
    verbose=True
):
    """
    Compare groupings using LOO-IC on actual ITE estimates.

    This is the GOLD STANDARD - directly measures predictive performance.
    """
    models = {}
    loo_scores = {}

    for grouping_var in candidates:
        print(f"\n[{candidates.index(grouping_var)+1}/{len(candidates)}] Fitting HBM with {grouping_var}...")

        # Create grouped data
        grouped_data = create_grouping(
            ite_estimates,
            creator_features,
            grouping_var
        )

        n_groups = grouped_data['genre_idx'].nunique()

        # Check minimum group size
        group_sizes = grouped_data.groupby('genre_idx').size()
        min_size = group_sizes.min()

        if min_size < 5:
            print(f"  ⚠️  Skipping - min group size = {min_size} (< 5)")
            continue

        # Fit model
        try:
            idata = fit_hierarchical_model_robust(
                grouped_data,
                n_genres=n_groups,
                draws=1000,
                tune=1000,
                chains=2,
                random_seed=42
            )

            models[grouping_var] = idata

            # Compute LOO
            loo = az.loo(idata, pointwise=True)
            loo_scores[grouping_var] = loo.loo

            print(f"  ✓ LOO-IC: {loo.loo:.2f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Compare
    if len(models) > 0:
        comparison = az.compare(models)

        print("\n" + "="*70)
        print("MODEL COMPARISON (LOO-IC)")
        print("="*70)
        print(comparison)
        print()

        best_grouping = comparison.index[0]
        print(f"✅ BEST GROUPING: {best_grouping}")

        if len(comparison) > 1:
            second_best = comparison.index[1]
            d_loo = comparison.loc[second_best, 'd_loo']

            if d_loo > 4:
                print(f"  → Clear winner! {best_grouping} is substantially better.")
            elif d_loo > 2:
                print(f"  → {best_grouping} is likely better, but not decisive.")
            else:
                print(f"  → {best_grouping} and {second_best} are roughly equivalent.")

        print("="*70)

        return comparison, models
    else:
        print("\n⚠️  No models successfully fitted!")
        return None, None

# Run comparison
comparison, models = compare_groupings_empirical(
    ite_estimates,
    creator_features,
    candidates=['genre', 'size_quartile', 'arpu_quintile', 'genre_x_arpu'],
    verbose=True
)
```

### Interpreting LOO-IC

- **Lower LOO-IC = Better** out-of-sample prediction
- **d_loo > 4**: Strong evidence for better model
- **d_loo 2-4**: Moderate evidence
- **d_loo < 2**: Models are equivalent

**Important:** LOO-IC is **cross-validated** - it doesn't overfit!

---

## Phase 5: Diagnostic Checks

### Check 1: Posterior Separation

**Do group-level effects actually differ?**

```python
def check_posterior_separation(idata, grouping_name):
    """
    Visualize whether group effects are meaningfully separated.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    az.plot_forest(
        idata,
        var_names=['mu_genre'],
        combined=True,
        hdi_prob=0.95,
        ax=ax
    )

    ax.set_title(f'Group-Level Effects: {grouping_name}\\n(95% Credible Intervals)')
    ax.set_xlabel('Treatment Effect')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'posterior_separation_{grouping_name}.png', dpi=300)
    plt.show()

    # Quantitative check
    mu_genre_samples = idata.posterior['mu_genre'].values
    genre_means = mu_genre_samples.mean(axis=(0, 1))
    genre_sds = mu_genre_samples.std(axis=(0, 1))

    mean_sep = genre_means.std()
    avg_uncertainty = genre_sds.mean()

    print(f"\nPosterior Separation for {grouping_name}:")
    print(f"  Between-group SD: {mean_sep:.4f}")
    print(f"  Avg within-group SD: {avg_uncertainty:.4f}")
    print(f"  Separation ratio: {mean_sep / avg_uncertainty:.2f}")

    if mean_sep > 2 * avg_uncertainty:
        print("  → ✅ Strong separation (groups clearly differ)")
    elif mean_sep > avg_uncertainty:
        print("  → ✓ Moderate separation")
    else:
        print("  → ⚠️  Weak separation (groups overlap heavily)")

for grouping_name, idata in models.items():
    check_posterior_separation(idata, grouping_name)
```

### Check 2: Shrinkage Patterns

**Is shrinkage helping?**

```python
def analyze_shrinkage(ite_estimates, hbm_estimates, creator_features, grouping_var):
    """
    Analyze how much and where shrinkage occurs.
    """
    merged = ite_estimates.merge(hbm_estimates, on='creator_id', suffixes=('_borusyak', '_hbm'))
    merged = merged.merge(creator_features[['creator_id', grouping_var]], on='creator_id')

    merged['shrinkage'] = abs(merged['effect_hat_hbm'] - merged['effect_hat_borusyak'])
    merged['shrinkage_pct'] = merged['shrinkage'] / abs(merged['effect_hat_borusyak']) * 100

    # Shrinkage by precision (SE)
    merged['precision'] = 1 / merged['se_borusyak']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Shrinkage vs precision
    axes[0].scatter(merged['precision'], merged['shrinkage'], alpha=0.5)
    axes[0].set_xlabel('Precision (1/SE)')
    axes[0].set_ylabel('Shrinkage (absolute)')
    axes[0].set_title('Shrinkage vs Precision\\n(Higher precision → Less shrinkage)')

    # Shrinkage by group
    sns.boxplot(data=merged, x=grouping_var, y='shrinkage_pct', ax=axes[1])
    axes[1].set_ylabel('Shrinkage (%)')
    axes[1].set_title(f'Shrinkage by {grouping_var}')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'shrinkage_analysis_{grouping_var}.png', dpi=300)
    plt.show()

    print(f"\nShrinkage Analysis for {grouping_var}:")
    print(f"  Median shrinkage: {merged['shrinkage'].median():.4f}")
    print(f"  Median shrinkage %: {merged['shrinkage_pct'].median():.1f}%")

    # Correlation with precision
    corr = merged[['precision', 'shrinkage']].corr().iloc[0, 1]
    print(f"  Correlation (precision, shrinkage): {corr:.3f}")

    if corr < -0.3:
        print("  → ✅ Good: Less precise estimates shrink more")
    else:
        print("  → ⚠️  Warning: Shrinkage pattern unexpected")

analyze_shrinkage(ite_estimates, hbm_estimates, creator_features, 'genre')
```

---

## Production Workflow

### Recommended Process

```python
# PHASE 1: Initial Exploration (do once)
# ========================================

# 1. Define candidates based on domain knowledge
candidates_primary = ['genre', 'size_tier', 'arpu_tier']
candidates_interactions = ['genre_x_size', 'genre_x_arpu']

# 2. Run EDA on historical data
for candidate in candidates_primary:
    quality_score = analyze_grouping_candidate(historical_df, candidate)
    check_group_sizes(historical_df, candidate)

# 3. Statistical tests
for candidate in candidates_primary:
    anova_test(historical_df, candidate)
    compute_eta_squared(historical_df, candidate)

# 4. Shortlist top 3-4 candidates (quality score > 20%)
shortlist = ['genre', 'arpu_tier', 'genre_x_arpu']  # Example


# PHASE 2: First Rollout (validate approach)
# ===========================================

# 1. Estimate ITEs with calibrated Borusyak
ites, _ = borusyak_imputation_estimator_calibrated(
    rollout_df,
    never_treated_ids,
    calibration_factor=2.5
)

# 2. Compare shortlisted groupings
comparison, models = compare_groupings_empirical(
    ites,
    creator_features,
    candidates=shortlist
)

# 3. Select best grouping
best_grouping = comparison.index[0]

# 4. Diagnostic checks
check_posterior_separation(models[best_grouping], best_grouping)
analyze_shrinkage(ites, hbm_estimates, creator_features, best_grouping)

# 5. Validate coverage on ground truth (if available)
if 'true_effects' in truth:
    coverage = compute_coverage(
        hbm_estimates['ci_lower'],
        hbm_estimates['ci_upper'],
        truth['true_effects']
    )
    print(f"HBM coverage: {coverage:.1%}")


# PHASE 3: Production (ongoing)
# ==============================

# 1. Use validated grouping for new rollouts
production_grouping = 'genre'  # From Phase 2

# 2. Monitor quarterly:
#    - Check if grouping quality persists
#    - Re-run LOO-IC comparison annually
#    - Update grouping if platform changes (new genres, etc.)

# 3. Alert if:
#    - LOO-IC degrades (grouping no longer optimal)
#    - Coverage drops below 90%
#    - Shrinkage patterns change dramatically
```

---

## Common Pitfalls

### ❌ Pitfall 1: Double-Dipping

**DON'T:**
```python
# Using same data to select features AND estimate effects
ites = borusyak_estimator(data)
selected_features = lasso(ites ~ all_features)  # Feature selection
hbm_estimates = hbm(ites, grouping=selected_features)  # Overfits!
```

**DO:**
```python
# Pre-specify candidates, compare using LOO-IC
candidates = ['genre', 'size', 'arpu']  # Domain-driven
comparison = az.compare({g: fit_hbm(ites, g) for g in candidates})
```

### ❌ Pitfall 2: Too Many Small Groups

```python
# Bad: 50 groups with 2-5 creators each
df['fine_grained_genre'] = df['sub_genre'].astype(str) + '_' + df['micro_segment']
# → Unstable estimates, overfitting

# Good: 5-10 groups with 15-30+ creators each
df['coarse_genre'] = df['genre'].map(genre_hierarchy)
```

### ❌ Pitfall 3: Ignoring Interpretability

```python
# Hard to explain to stakeholders
best_grouping = 'PCA_component_3_tertile'

# Easy to explain
best_grouping = 'content_type'  # Video, Livestream, Shorts
```

**Trade-off:** Sometimes slightly suboptimal but interpretable grouping is better for production.

### ❌ Pitfall 4: Not Validating on New Data

```python
# Fit once on 2023 data, use forever → Drift!

# Better: Quarterly re-validation
for quarter in ['2024-Q1', '2024-Q2', '2024-Q3']:
    validate_grouping_quality(quarter_data, current_grouping)
    if quality_score < threshold:
        trigger_grouping_review()
```

---

## Summary Checklist

Before finalizing a grouping for production:

- [ ] **Domain validation**: Makes intuitive sense to product/business teams
- [ ] **Statistical significance**: ANOVA p-value < 0.05
- [ ] **Effect size**: η² > 0.20 (explains >20% of variance)
- [ ] **Group sizes**: All groups have >15 creators
- [ ] **LOO-IC**: Best or within 2 points of best
- [ ] **Posterior separation**: Group effects clearly differ (separation ratio > 1.5)
- [ ] **Shrinkage patterns**: Sensible (more for low-precision estimates)
- [ ] **Coverage validation**: ≥93% on validation set
- [ ] **Interpretability**: Stakeholders understand and trust it
- [ ] **Monitoring plan**: Quarterly quality checks scheduled

---

## Quick Reference: Decision Tree

```
START
  ↓
Do you have domain knowledge about segments?
  ├─ YES → Start with 3-5 domain-driven candidates
  └─ NO → Use data-driven clustering + quartiles

  ↓
Run EDA: quality scores, ANOVA tests, η²
  ↓
Any candidates with quality score > 20%?
  ├─ NO → Consider not using HBM (too little structure)
  └─ YES → Continue with shortlist

  ↓
Fit HBM for each shortlisted candidate
  ↓
Compare using LOO-IC
  ↓
Best model has d_loo > 2 from second best?
  ├─ YES → Use best model
  └─ NO → Choose most interpretable among top models

  ↓
Run diagnostic checks (separation, shrinkage, coverage)
  ↓
All checks pass?
  ├─ NO → Revise candidates or grouping strategy
  └─ YES → Deploy to production with monitoring

  ↓
Quarterly: Re-validate quality scores and LOO-IC
  ↓
Quality degraded?
  ├─ YES → Re-run full analysis
  └─ NO → Continue with current grouping
```

---

**Remember:** The best grouping isn't always the most complex. Start simple (genre, size), validate rigorously, and only add complexity (interactions) if data clearly supports it.
