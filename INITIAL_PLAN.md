# Hierarchical Bayesian Modeling for Creator Experiment Readouts

## Project Plan & Implementation Guide

---

## 1. Problem Statement

We run A/B experiments on individual creators on a platform. The outcome metric is **average revenue per user** (continuous) in treatment vs. control. Many creators have small sample sizes (few hundred or fewer users), making individual experiment estimates noisy and unreliable. We want to give each creator a stable, trustworthy readout of their experiment's treatment effect.

Creators can be grouped by **genre** (e.g., comedy, music, gaming, education, lifestyle). We suspect treatment effects vary by genre (some content types respond differently to platform changes). We want to use the platform's aggregate data to improve individual estimates — especially for small creators — via hierarchical Bayesian modeling (partial pooling).

### What "better" means concretely

For a small creator with n=80 users, a raw frequentist estimate of treatment effect might be +$1.50 ± $3.00. That's useless as a readout. If the model knows that creators in their genre typically see effects around +$0.40 ± $0.30, we can produce a **shrinkage estimate** that's pulled toward the genre mean — say +$0.60 ± $0.50. This is more precise, better calibrated, and more actionable.

---

## 2. Architecture Overview

```
project/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_generation.py      # Synthetic data with known ground truth
│   ├── frequentist.py          # No-pooling and complete-pooling estimators
│   ├── hierarchical_model.py   # PyMC hierarchical model
│   ├── validation.py           # All validation/comparison logic
│   └── visualization.py        # All plots
├── notebooks/
│   └── walkthrough.ipynb       # Interactive narrative walkthrough
├── outputs/                    # Saved figures and results
└── tests/
    └── test_data_generation.py # Smoke tests
```

---

## 3. Dependencies

```
# requirements.txt
pymc>=5.10
arviz>=0.17
numpy>=1.24
scipy>=1.10
pandas>=1.5
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

**Installation note:** PyMC 5 uses PyTensor as its backend. On most machines, `pip install pymc` handles everything. If on Apple Silicon, `conda install -c conda-forge pymc` is often smoother. The NUTS sampler (No-U-Turn Sampler) is the default and works well for this problem size.

---

## 4. Module 1: Synthetic Data Generation (`data_generation.py`)

### Why synthetic data first

Synthetic data with known ground truth is essential. It's the only way to definitively answer "did the model recover the right answer?" With real data you never know the true treatment effect, so you can't measure estimation error.

### Data generating process (DGP)

This is the ground truth that we'll try to recover. The DGP should mimic the hierarchical structure we expect in real data.

```
Parameters to define:
- N_GENRES = 5 (e.g., comedy, music, gaming, education, lifestyle)
- N_CREATORS_PER_GENRE = 100 (500 total creators)
- Genre-level mean treatment effects: mu_genre[g] ~ Normal(0.5, 0.3)
    - These represent the "true" average treatment effect for each genre
    - Example draws: comedy=+0.80, music=+0.20, gaming=+0.55, education=+0.30, lifestyle=+0.60
- Within-genre creator variance: sigma_creator = 0.4
    - Controls how much individual creators deviate from their genre mean
- True individual creator effects: tau_creator[i] ~ Normal(mu_genre[genre_of_i], sigma_creator)
    - Each creator has their own true treatment effect, drawn around their genre mean
- Creator sample sizes: n_i ~ chosen to be HIGHLY VARIABLE
    - This is critical. Use a realistic heavy-tailed distribution.
    - Suggestion: draw from a log-normal or a mixture:
        - 40% of creators: n_i ~ Uniform(30, 100)    [small creators]
        - 35% of creators: n_i ~ Uniform(100, 500)   [medium creators]
        - 25% of creators: n_i ~ Uniform(500, 5000)  [large creators]
    - Split each creator's n_i roughly 50/50 into treatment and control
- Observation noise: sigma_obs = 2.0
    - Revenue per user has high individual variance (most users pay $0, some pay a lot)
    - This should be substantially larger than the treatment effects — that's what
      makes the problem hard and makes HBM valuable
```

### Data generation function

```python
def generate_experiment_data(seed=42) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
        df: DataFrame with columns [creator_id, genre, group ('treatment'/'control'), revenue]
            One row per user observation.
        truth: dict with keys:
            - 'genre_effects': array of shape (N_GENRES,) — true genre-level means
            - 'creator_effects': array of shape (N_CREATORS,) — true individual effects
            - 'sigma_creator': float — true within-genre SD
            - 'sigma_obs': float — true observation noise SD
            - 'creator_genre': array mapping creator_id -> genre index
            - 'creator_n': array of sample sizes per creator
    """
```

### What to generate per creator

For creator i with sample size n_i:
- Split into n_treat ≈ n_i/2 treatment users and n_control ≈ n_i/2 control users
- Control revenue per user: y_control ~ Normal(baseline, sigma_obs)
  - Use baseline = 5.0 (dollars) as a reasonable revenue baseline
- Treatment revenue per user: y_treatment ~ Normal(baseline + tau_creator[i], sigma_obs)
- The creator's **observed treatment effect** (what frequentist would estimate) is:
  mean(y_treatment) - mean(y_control)

### Key design choices

1. **Store raw user-level data**, not just creator summaries. This lets us compute standard errors properly and is more realistic.
2. **Use a fixed random seed** for reproducibility.
3. **Return the ground truth** separately so validation can compare estimates to it.
4. **Make the signal-to-noise ratio realistic.** If true effects are ~0.5 and observation noise is ~2.0, a creator with n=60 (30 per arm) will have SE ≈ 2.0 * sqrt(2/30) ≈ 0.52. So their 95% CI will be ±1.0 around the point estimate — very noisy relative to the true effect. This is the regime where HBM helps most.

---

## 5. Module 2: Frequentist Estimators (`frequentist.py`)

Implement two frequentist baselines. These are the "controls" for our comparison.

### Estimator A: No pooling (per-creator independent estimates)

For each creator independently:
1. Compute mean revenue in treatment group and control group
2. Treatment effect = mean_treatment - mean_control
3. Standard error = sqrt(var_treatment/n_treatment + var_control/n_control)
4. 95% confidence interval = effect ± 1.96 * SE
5. p-value from two-sample t-test (Welch's)

```python
def no_pooling_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        [creator_id, genre, effect_hat, se, ci_lower, ci_upper, p_value, n_total]
    One row per creator.
    """
```

This is what most platforms do today. It's unbiased but high-variance for small creators.

### Estimator B: Complete pooling (genre-level average applied to everyone)

1. Pool all creators within each genre
2. Compute a single treatment effect per genre (ignoring individual variation)
3. Assign each creator their genre's pooled estimate

```python
def complete_pooling_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with same columns as no_pooling.
    Every creator in a genre gets the same estimate.
    """
```

This is the opposite extreme: low variance but ignores real individual differences (high bias for creators who truly differ from their genre average).

### Why both matter

The whole point of HBM is that it's a principled middle ground — **partial pooling**. The comparison of all three (no pooling, complete pooling, partial pooling) is the key pedagogical and practical insight.

---

## 6. Module 3: Hierarchical Bayesian Model (`hierarchical_model.py`)

### Model specification

This is the core. The model should mirror the DGP (which in practice you'd need to hypothesize — here we have the advantage of knowing it).

```
Hierarchical model for treatment effect on revenue:

    # Hyperpriors (platform-level)
    mu_global ~ Normal(0, 1)              # Global mean treatment effect
    sigma_genre ~ HalfNormal(1)           # How much genres differ from each other

    # Genre-level priors
    mu_genre[g] ~ Normal(mu_global, sigma_genre)   for g in 1..G

    # Creator-level priors
    sigma_creator ~ HalfNormal(1)         # How much creators differ within a genre
    tau[i] ~ Normal(mu_genre[genre_of_i], sigma_creator)   for i in 1..N

    # Likelihood
    # For each creator i, the observed mean treatment effect d_i has known SE_i
    # (computed from the data), so:
    d_i ~ Normal(tau[i], SE_i)            for i in 1..N
```

### Critical implementation detail: work with summary statistics

You do NOT need to feed in raw user-level data. For a normal-normal model, the **sufficient statistics** are:
- `d_i` = observed mean difference (treatment - control) for creator i
- `SE_i` = standard error of that difference for creator i

This massively speeds up computation (500 observations instead of hundreds of thousands) and is mathematically equivalent.

### PyMC implementation sketch

```python
import pymc as pm

def fit_hierarchical_model(creator_summaries: pd.DataFrame) -> az.InferenceData:
    """
    Args:
        creator_summaries: DataFrame with columns
            [creator_id, genre_idx, effect_hat, se]

    Returns:
        ArviZ InferenceData object with posterior samples
    """
    genre_idx = creator_summaries['genre_idx'].values
    observed_effects = creator_summaries['effect_hat'].values
    observed_se = creator_summaries['se'].values
    n_genres = creator_summaries['genre_idx'].nunique()

    with pm.Model() as model:
        # Hyperpriors
        mu_global = pm.Normal('mu_global', mu=0, sigma=1)
        sigma_genre = pm.HalfNormal('sigma_genre', sigma=1)

        # Genre effects (use non-centered parameterization for better sampling)
        genre_offset = pm.Normal('genre_offset', mu=0, sigma=1, shape=n_genres)
        mu_genre = pm.Deterministic('mu_genre', mu_global + sigma_genre * genre_offset)

        # Creator effects
        sigma_creator = pm.HalfNormal('sigma_creator', sigma=1)
        creator_offset = pm.Normal('creator_offset', mu=0, sigma=1,
                                    shape=len(creator_summaries))
        tau = pm.Deterministic('tau',
                               mu_genre[genre_idx] + sigma_creator * creator_offset)

        # Likelihood: observed effect ~ Normal(true effect, known SE)
        obs = pm.Normal('obs', mu=tau, sigma=observed_se, observed=observed_effects)

        # Sample
        idata = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                          target_accept=0.9)

    return idata
```

### Key implementation notes

1. **Non-centered parameterization**: The `offset` trick (writing `mu_genre = mu_global + sigma_genre * offset` instead of `mu_genre ~ Normal(mu_global, sigma_genre)`) is critical for MCMC efficiency when group sizes are small or variance parameters are near zero. PyMC's NUTS sampler struggles with the centered version — you'll see divergent transitions. Always use non-centered for hierarchical models.

2. **Known observation SEs**: We treat SE_i as fixed/known (computed from data). This is standard practice and avoids modeling the full user-level likelihood, which would be much slower with no statistical benefit.

3. **Sampling parameters**:
   - `tune=1000`: warmup/adaptation steps (not included in posterior)
   - `draws=2000`: posterior samples per chain
   - `chains=4`: run 4 independent chains (for convergence diagnostics)
   - `target_accept=0.9`: slightly conservative step size (helps with hierarchical models)
   - Total: 8000 posterior draws. More than enough for this model.

4. **Expected runtime**: ~1-3 minutes on a laptop for 500 creators, 5 genres.

### Extracting estimates from the posterior

```python
def extract_hbm_estimates(idata, creator_summaries):
    """
    From the posterior, extract for each creator:
    - Posterior mean of tau[i] (point estimate)
    - Posterior SD of tau[i] (uncertainty)
    - 95% credible interval (HDI or quantile-based)

    Use arviz:
        az.summary(idata, var_names=['tau'])
    Or manually:
        tau_samples = idata.posterior['tau'].values  # shape (chains, draws, n_creators)
        tau_flat = tau_samples.reshape(-1, n_creators)
        point_estimates = tau_flat.mean(axis=0)
        ci_lower = np.percentile(tau_flat, 2.5, axis=0)
        ci_upper = np.percentile(tau_flat, 97.5, axis=0)
    """
```

---

## 7. Module 4: Validation (`validation.py`)

This is the most important module. It answers: "Does HBM actually work better, and how do we know?"

### Validation 1: Mean Squared Error to ground truth

Since we know the true creator effects from the DGP:

```python
def compute_mse(estimates: np.array, truth: np.array) -> float:
    return np.mean((estimates - truth) ** 2)

# Compute for each method:
# mse_no_pooling = MSE(frequentist per-creator estimates, true creator effects)
# mse_complete_pooling = MSE(genre mean applied to all, true creator effects)
# mse_hbm = MSE(HBM posterior means, true creator effects)
```

**Expected result**: HBM should have lowest MSE overall. The gap should be largest for small-sample creators.

Also compute MSE **stratified by creator sample size** (e.g., bins: n<100, 100-500, 500+). This shows where HBM helps most.

### Validation 2: Coverage of credible/confidence intervals

For each method, check: what fraction of creators have their TRUE effect inside the reported interval?

```python
def compute_coverage(ci_lower, ci_upper, truth, level=0.95):
    """
    Should be ~0.95 for a well-calibrated 95% interval.
    """
    covered = (truth >= ci_lower) & (truth <= ci_upper)
    return covered.mean()
```

**Expected results**:
- No-pooling frequentist: ~95% coverage (CIs are valid but very wide for small n)
- Complete pooling: terrible coverage (ignores individual variation, intervals too narrow)
- HBM: ~95% coverage with **much narrower intervals** than no-pooling

Also compute coverage stratified by sample size. For small creators, HBM should maintain ~95% coverage while having much narrower intervals than no-pooling.

### Validation 3: Interval width comparison

```python
def compute_avg_interval_width(ci_lower, ci_upper):
    return np.mean(ci_upper - ci_lower)
```

Narrower intervals with maintained coverage = strictly better. Compute overall and by sample-size bin.

### Validation 4: Shrinkage plot (THE key diagnostic)

This is the single most important visualization for understanding HBM behavior.

```
Scatter plot:
    x-axis: No-pooling (frequentist) estimate for each creator
    y-axis: HBM posterior mean for each creator
    Color: by genre
    Size: by sample size (larger points = more data)
    Reference line: y = x (no shrinkage)
    Horizontal lines: genre-level posterior means

What to look for:
    - Points should be pulled TOWARD genre means (shrinkage)
    - Small-sample creators (small points) should be pulled MORE
    - Large-sample creators should be near the y=x line (data dominates prior)
    - This is the visual proof that partial pooling is working
```

### Validation 5: MCMC diagnostics

Before trusting ANY results, verify the sampler worked:

```python
import arviz as az

# 1. R-hat (should be < 1.01 for all parameters)
summary = az.summary(idata, var_names=['mu_global', 'sigma_genre', 'sigma_creator',
                                        'mu_genre', 'tau'])
assert (summary['r_hat'] < 1.01).all()

# 2. Effective sample size (should be > 400 per chain, ideally > 1000)
assert (summary['ess_bulk'] > 400).all()

# 3. Divergent transitions (should be 0)
divergences = idata.sample_stats['diverging'].sum().values
assert divergences == 0

# 4. Trace plots (visual check for mixing)
az.plot_trace(idata, var_names=['mu_global', 'sigma_genre', 'sigma_creator'])

# 5. Posterior predictive check
# Generate predicted observed effects from posterior, compare distribution to actual
```

### Validation 6: Posterior predictive check

```python
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

az.plot_ppc(idata)  # Compares observed data distribution to model-predicted distribution
```

This answers: "If the model is correct, would it generate data that looks like what we observed?" If the distributions are wildly different, the model is misspecified.

### Validation 7: Simulation-based calibration (SBC) — advanced

This is the gold-standard validation. It answers: "Across many possible datasets from the prior, does the posterior have correct coverage?"

```
Procedure:
1. Draw parameters from the prior
2. Simulate data from those parameters
3. Fit the model
4. Check if the true parameters fall within the posterior as expected
5. Repeat 100-500 times
6. The rank statistics of the true values within the posterior samples
   should be uniformly distributed

This is computationally expensive (fitting the model hundreds of times)
but is the definitive test that the model + inference is correct.
Consider this a stretch goal. ArviZ has some SBC utilities.
```

---

## 8. Module 5: Visualization (`visualization.py`)

### Plot 1: Shrinkage plot (described above in Validation 4)

### Plot 2: MSE comparison bar chart

```
Grouped bar chart:
    Groups: sample size bins (small / medium / large / overall)
    Bars: No-pooling MSE, Complete-pooling MSE, HBM MSE
    Expected pattern: HBM wins everywhere, especially for small n
```

### Plot 3: Coverage vs. interval width scatter

```
For each creator, plot:
    x-axis: interval width
    y-axis: whether the true value was covered (0 or 1)
Overlay three series (no-pooling, complete-pooling, HBM) in different colors.
Add marginal annotations: overall coverage rate and mean width per method.
```

### Plot 4: Creator-level comparison (small multiples)

```
Pick 12-16 creators spanning small/medium/large sample sizes.
For each, show:
    - Frequentist point estimate ± 95% CI (blue)
    - HBM posterior mean ± 95% credible interval (orange)
    - True treatment effect (red dashed line)
    - Genre mean (gray dashed line)
Arrange in a grid, sorted by sample size.
This makes the shrinkage behavior tangible for individual cases.
```

### Plot 5: Posterior distributions for genre effects

```
For each genre, plot:
    - Posterior density of mu_genre[g] (from HBM)
    - Vertical line: true genre effect
    - Vertical line: genre-level frequentist estimate (complete pooling)
Shows how well the model recovers genre-level structure.
```

### Plot 6: Hyperparameter recovery

```
Posterior densities for sigma_genre and sigma_creator,
with vertical lines at true values.
If the model can recover these variance components, it's working correctly.
```

---

## 9. Notebook Walkthrough (`walkthrough.ipynb`)

Structure the notebook as a narrative:

1. **Setup**: imports, seed, generate data
2. **Explore the data**: show distribution of sample sizes, true effects by genre
3. **Frequentist estimates**: compute both, show how noisy the no-pooling estimates are for small creators
4. **Fit HBM**: run the PyMC model, show MCMC diagnostics
5. **Compare**: shrinkage plot, MSE comparison, coverage analysis
6. **Deep dive**: individual creator examples showing the value of partial pooling
7. **Conclusions**: when and why to use HBM, practical considerations

---

## 10. Implementation Order (for Claude Code)

Build and test in this order. Each step should produce runnable, testable output before moving on.

### Step 1: Environment setup
```bash
mkdir creator-hbm && cd creator-hbm
python -m venv venv && source venv/bin/activate
pip install pymc arviz numpy scipy pandas matplotlib seaborn jupyter
```
Verify: `python -c "import pymc; print(pymc.__version__)"` should print 5.x.

### Step 2: Data generation
Build `data_generation.py`. Test it independently:
```python
df, truth = generate_experiment_data()
print(f"Total observations: {len(df)}")
print(f"Creators: {df['creator_id'].nunique()}")
print(f"Sample size range: {df.groupby('creator_id').size().min()} - {df.groupby('creator_id').size().max()}")
print(f"Genre effects: {truth['genre_effects']}")
```

### Step 3: Frequentist baselines
Build `frequentist.py`. Test:
```python
no_pool = no_pooling_estimates(df)
print(f"Mean absolute error (no pooling): {np.mean(np.abs(no_pool['effect_hat'] - truth['creator_effects']))}")
```

### Step 4: Hierarchical model
Build `hierarchical_model.py`. This is the trickiest part. Test with a small subset first (50 creators) to iterate quickly, then scale up.

**Debugging priorities**:
- Zero divergent transitions
- R-hat < 1.01
- Reasonable posterior means (not all zeros, not exploding)

### Step 5: Validation functions
Build `validation.py`. Compute all metrics. Print a comparison table:
```
                    No-Pooling   Complete-Pooling   HBM
MSE (overall)         X.XX          X.XX           X.XX
MSE (small n)         X.XX          X.XX           X.XX
Coverage (95%)        X.XX          X.XX           X.XX
Mean CI width         X.XX          X.XX           X.XX
```

### Step 6: Visualizations
Build `visualization.py`. Generate all 6 plots.

### Step 7: Notebook
Assemble everything into the walkthrough notebook.

---

## 11. Expected Results & What to Look For

If everything is working correctly, you should see:

| Metric | No-Pooling | Complete-Pooling | HBM |
|--------|-----------|-----------------|-----|
| MSE (overall) | High | Medium | **Lowest** |
| MSE (small creators) | Very high | Medium | **Much lower** |
| MSE (large creators) | Low | Medium | **~Same as no-pooling** |
| 95% coverage | ~95% | <<95% | **~95%** |
| Mean interval width | Very wide | Narrow | **Medium (best tradeoff)** |

The headline finding: **HBM achieves near-nominal coverage with substantially narrower intervals**, especially for small creators. It does this by borrowing strength from the group, while still allowing individual variation.

---

## 12. Pitfalls & Troubleshooting

### Divergent transitions
If you see divergent transitions in PyMC output:
- Increase `target_accept` to 0.95 or 0.99
- Verify you're using non-centered parameterization
- Check for extreme data (e.g., a creator with n=2 might cause issues — consider filtering to n≥20)
- Try increasing `tune` steps

### Slow sampling
- Verify you're using summary statistics (d_i, SE_i), not raw user-level data
- 500 creators with 5 genres should take 1-5 minutes. If it's much longer, something is wrong.
- Check that shapes are correct in the model (common source of silent performance issues)

### HBM not beating frequentist
This could mean:
- The signal-to-noise ratio is too low (increase true effects or decrease sigma_obs)
- Creator sample sizes are too large (shrinkage only helps when data is limited)
- Bug in the model (check that genre indexing is correct)

### Coverage much lower than 95% for HBM
The model may be misspecified. Check:
- Are the priors too informative (too narrow)?
- Is the assumed normal likelihood appropriate?
- Are the SE_i being computed correctly?

---

## 13. Extension Roadmap (After Basic Version Works)

Once the simple version (single grouping by genre) is working and validated:

### Extension A: Crossed random effects (genre × audience segment)
Add a second grouping variable. The model becomes:
```
tau[i] ~ Normal(mu_genre[g_i] + mu_segment[s_i] + interaction[g_i, s_i], sigma_creator)
```
This is more realistic but harder to sample. May need stronger priors on the interaction term.

### Extension B: Non-normal outcomes
Real revenue data is heavily right-skewed (many $0, some large values). Consider:
- Log-transform revenue, model on log scale
- Use a hurdle/zero-inflated model (model P(revenue>0) and E[revenue|revenue>0] separately)
- Negative binomial or Gamma likelihood

### Extension C: Informative priors from historical data
This is what Google Ads does: use results from past experiments to set priors for new ones. You could:
- Fit the hierarchical model to a batch of completed experiments
- Use the fitted hyperparameters as priors for the next batch
- This creates a "learning" system that gets better over time

### Extension D: Time-varying effects
If experiments run at different times, treatment effects might drift. Could add a time component to the hierarchy.

### Extension E: Production deployment considerations
- Replace MCMC with variational inference (faster, less accurate) for real-time readouts
- Pre-compute genre priors nightly, apply them to new experiments as they complete
- Build confidence scores: "How much did the hierarchical model change this creator's estimate?"

---

## 14. Key References

- **Gelman & Hill (2006)**, *Data Analysis Using Regression and Multilevel/Hierarchical Models* — The canonical reference. Chapters 11-13 cover exactly this use case.
- **Gelman et al. (2013)**, *Bayesian Data Analysis (3rd ed.)* — Chapter 5 (hierarchical models) is the theoretical foundation. Free PDF available from Gelman's website.
- **McElreath (2020)**, *Statistical Rethinking (2nd ed.)* — Chapters 13-14. More accessible than Gelman. Has PyMC code translations available online.
- **PyMC documentation**: https://www.pymc.io/projects/docs/en/stable/ — The "Getting Started" and "Hierarchical Models" tutorials are directly relevant.
- **ArviZ documentation**: https://python.arviz.org/en/stable/ — For all posterior analysis and diagnostics.
- **"Eight Schools" example**: The classic pedagogical example of hierarchical modeling. Every Bayesian textbook covers it. It's a simpler version of exactly your problem (8 schools instead of 500 creators). PyMC's docs include it: search "Eight Schools PyMC".
- **Betancourt (2017)**, "A Conceptual Introduction to Hamiltonian Monte Carlo" — If you want to understand how the NUTS sampler works under the hood. Available on arXiv.

---

## 15. Glossary of Key Terms

| Term | Meaning |
|------|---------|
| **Partial pooling** | The HBM's core behavior: individual estimates are "shrunk" toward group means, with the amount of shrinkage determined by relative sample size and group-level variance |
| **Shrinkage** | The pulling of individual estimates toward the group mean. More shrinkage for noisier (small-n) estimates |
| **No pooling** | Estimating each creator independently (standard frequentist). Unbiased but high variance |
| **Complete pooling** | Ignoring individual differences, using only group-level estimates. Low variance but high bias |
| **Credible interval** | Bayesian analog of confidence interval. A 95% credible interval means "95% posterior probability the true value is in this range" |
| **Non-centered parameterization** | A reparameterization trick that helps MCMC samplers explore hierarchical models efficiently |
| **NUTS** | No-U-Turn Sampler. The default MCMC algorithm in PyMC. A variant of Hamiltonian Monte Carlo |
| **R-hat** | Convergence diagnostic. Compares within-chain and between-chain variance. Should be < 1.01 |
| **ESS** | Effective Sample Size. How many independent samples the MCMC chain is equivalent to. Should be > 400 |
| **Divergent transition** | A warning that the sampler failed to explore the posterior correctly. Should be 0 |
| **Posterior predictive check** | Generating fake data from the fitted model to see if it resembles the real data |
| **SBC** | Simulation-Based Calibration. Fitting the model to many simulated datasets to verify calibration |
| **Hyperprior** | Prior on parameters that govern other priors (e.g., the prior on sigma_genre) |