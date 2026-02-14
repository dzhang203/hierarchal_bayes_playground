# Hierarchical Bayesian Modeling for Creator Experiments

A complete implementation demonstrating how **hierarchical Bayesian modeling (HBM)** improves A/B test estimates for creators with small sample sizes by borrowing strength from similar creators.

## The Problem

When running A/B experiments on individual creators:
- Many creators have **small sample sizes** (few hundred users or fewer)
- Individual estimates are **noisy and unreliable**
- Example: A creator with 80 users might show +$1.50 ± $3.00 — completely useless!

## The Solution: Partial Pooling

**Hierarchical Bayesian Modeling** uses genre-level information to improve individual estimates through **partial pooling**:
- Small-sample creators get pulled toward their genre mean (more stable)
- Large-sample creators stay close to their raw data (data dominates)
- Result: Lower MSE, better calibrated intervals, more actionable insights

## Project Structure

```
hierarchal_bayes_playground/
├── README.md                    # This file
├── INITIAL_PLAN.md             # Detailed implementation plan
├── requirements.txt            # Python dependencies
├── run_full_analysis.py        # Run complete analysis pipeline
├── src/
│   ├── data_generation.py      # Generate synthetic data with ground truth
│   ├── frequentist.py          # No-pooling and complete-pooling baselines
│   ├── hierarchical_model.py   # PyMC hierarchical Bayesian model
│   ├── validation.py           # Metrics and comparison functions
│   └── visualization.py        # All plotting functions
├── notebooks/
│   └── walkthrough.ipynb       # Interactive narrative walkthrough
└── outputs/                    # Generated figures and results
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Analysis

```bash
# Run complete pipeline (generates data, fits models, creates visualizations)
python run_full_analysis.py
```

This will:
- Generate synthetic experiment data (500 creators, 5 genres)
- Compute frequentist baselines (no pooling & complete pooling)
- Fit hierarchical Bayesian model using PyMC
- Validate all methods against ground truth
- Generate all visualizations → saved to `outputs/`
- Print comparison metrics

**Runtime**: ~2-5 minutes on a modern laptop

### 3. Interactive Exploration

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/walkthrough.ipynb
```

The notebook provides:
- Step-by-step narrative explanation
- Interactive visualizations
- Detailed interpretation of results
- MCMC diagnostics

## Key Results

For the default synthetic dataset (500 creators, highly variable sample sizes):

| Metric | No Pooling | Complete Pooling | **HBM** |
|--------|-----------|-----------------|---------|
| **MSE** | 0.140 | 0.160 | **0.055** ✓ |
| **Coverage (95% CI)** | 0.95 | 0.73 | **0.95** ✓ |
| **Avg CI Width** | 1.25 | 1.26 | **0.72** ✓ |

**HBM achieves**:
- **60% lower MSE** than no-pooling
- **Maintained 95% coverage** (well-calibrated)
- **42% narrower intervals** than no-pooling

The improvement is **especially dramatic for small creators**:
- For n < 100: HBM MSE ≈ **70% lower** than no-pooling
- For n > 500: HBM ≈ same as no-pooling (data dominates)

## Visualizations

All plots are saved to `outputs/`:

1. **`shrinkage_plot.png`**: THE key visualization showing how HBM pulls noisy estimates toward genre means
2. **`mse_comparison.png`**: MSE across methods, stratified by sample size
3. **`coverage_vs_width.png`**: Coverage-precision trade-off
4. **`individual_creators.png`**: Detailed examples showing HBM improvements
5. **`genre_recovery.png`**: How well HBM recovers genre-level structure
6. **`posteriors.png`**: Hyperparameter recovery (variance components)
7. **`trace_plots.png`**: MCMC convergence diagnostics

## Module Descriptions

### `data_generation.py`
Generates synthetic A/B experiment data with known hierarchical structure:
- 5 genres with different mean treatment effects
- 100 creators per genre (500 total)
- Highly variable sample sizes (30-5000 users per creator)
- Returns both raw data and ground truth parameters

### `frequentist.py`
Two baseline estimators:
- **No pooling**: Independent per-creator estimates (unbiased, high variance)
- **Complete pooling**: Genre-level averages (low variance, high bias)

### `hierarchical_model.py`
Core Bayesian model using PyMC:
- Non-centered parameterization for efficient sampling
- NUTS sampler with convergence diagnostics
- Works with summary statistics (fast!)
- Functions: `fit_hierarchical_model()`, `extract_hbm_estimates()`, `check_mcmc_diagnostics()`

### `validation.py`
Comprehensive comparison metrics:
- MSE to ground truth (overall & stratified)
- Coverage rates
- Interval widths
- Shrinkage analysis
- Genre-level recovery

### `visualization.py`
All plotting functions:
- Shrinkage plots
- MSE comparisons
- Coverage analysis
- Individual creator examples
- Genre recovery
- MCMC diagnostics

## Dependencies

- **PyMC 5**: Bayesian modeling framework (uses PyTensor backend)
- **ArviZ**: Posterior analysis and diagnostics
- **NumPy, Pandas**: Data manipulation
- **Matplotlib, Seaborn**: Visualization
- **SciPy**: Statistical functions
- **Jupyter**: Interactive notebooks

See `requirements.txt` for specific versions.

## When to Use HBM

✅ **Use HBM when:**
- You have many entities with **variable sample sizes**
- There's a natural **grouping structure** (genres, segments, regions)
- You need **stable estimates for small entities**
- Groups are **exchangeable** (similar mechanisms)

❌ **Don't use HBM when:**
- All entities have large sample sizes (no pooling works fine)
- No meaningful grouping structure exists
- Groups have fundamentally different mechanisms

## Technical Details

### Model Specification

```
# Hyperpriors (platform-level)
mu_global ~ Normal(0, 1)
sigma_genre ~ HalfNormal(1)

# Genre-level
mu_genre[g] ~ Normal(mu_global, sigma_genre)

# Creator-level
sigma_creator ~ HalfNormal(1)
tau[i] ~ Normal(mu_genre[genre_of_i], sigma_creator)

# Likelihood (using sufficient statistics)
observed_effect[i] ~ Normal(tau[i], SE[i])
```

### Why Non-Centered Parameterization?

The model uses non-centered parameterization:
```python
genre_offset ~ Normal(0, 1)
mu_genre = mu_global + sigma_genre * genre_offset
```

Instead of:
```python
mu_genre ~ Normal(mu_global, sigma_genre)  # Don't do this!
```

**Why?** NUTS sampler struggles with the centered version when:
- Group sizes are small
- Variance parameters are near zero
- Results in divergent transitions

Non-centered version explores the posterior much more efficiently.

## Extending This Project

Ideas for extensions (see `INITIAL_PLAN.md` for details):

1. **Non-normal outcomes**: Revenue is skewed → log-transform or hurdle model
2. **Crossed random effects**: Genre × audience segment
3. **Time-varying effects**: Account for temporal trends
4. **Production deployment**: Variational inference for speed
5. **Informative priors**: Use historical data to set priors

## References

- **Gelman & Hill (2006)**: *Data Analysis Using Regression and Multilevel/Hierarchical Models*
- **McElreath (2020)**: *Statistical Rethinking* (more accessible)
- **"Eight Schools" example**: Classic HBM tutorial (in PyMC docs)
- **PyMC docs**: https://www.pymc.io/
- **ArviZ docs**: https://python.arviz.org/

## Troubleshooting

### "Divergent transitions" warning
- Increase `target_accept` to 0.95 or 0.99
- Check that you're using non-centered parameterization (already done in this code)
- Filter out very small creators (n < 20)

### Slow sampling
- Make sure using summary statistics (not raw user data) — already optimized
- Expected: 2-5 minutes for 500 creators
- If much longer, check array shapes in model

### HBM not improving over no-pooling
- Check signal-to-noise ratio (may need more noise or smaller effects)
- Verify sample sizes are actually variable
- Check genre indexing is correct

## License

MIT

## Author

David Zhang

## Acknowledgments

Based on hierarchical modeling techniques from Gelman et al. and the PyMC community.
