# Quick Start Guide

## 1. Setup (First Time Only)

```bash
# Activate virtual environment
source venv/bin/activate

# Verify installation
python -c "import pymc; import arviz; print('✓ All packages installed!')"
```

## 2. Run Analysis

### Option A: Full Automated Pipeline

```bash
python run_full_analysis.py
```

**What it does:**
- Generates synthetic data (500 creators, 5 genres)
- Fits all three models (no-pooling, complete-pooling, HBM)
- Validates against ground truth
- Creates all visualizations → `outputs/`
- Runtime: ~3-5 minutes

### Option B: Interactive Notebook

```bash
jupyter notebook notebooks/walkthrough.ipynb
```

**What you get:**
- Step-by-step narrative explanation
- Interactive code cells
- Live visualizations
- Detailed interpretation

## 3. View Results

### Visualizations
All plots saved to `outputs/`:

1. **`shrinkage_plot.png`** ⭐ - THE key plot showing partial pooling
2. **`mse_comparison.png`** - Performance across sample sizes
3. **`coverage_vs_width.png`** - Precision vs. calibration
4. **`individual_creators.png`** - Specific examples
5. **`genre_recovery.png`** - Genre-level validation
6. **`posteriors.png`** - Hyperparameter recovery
7. **`trace_plots.png`** - MCMC diagnostics

### Data Files
- **`comparison_results.csv`** - Full comparison table (all 500 creators)
- **`overall_metrics.csv`** - Summary metrics by method
- **`genre_recovery.csv`** - Genre-level validation

## 4. Understanding the Results

### Key Metrics

**HBM achieves ~60% reduction in MSE** vs. no-pooling:

```
No Pooling MSE:     0.14
Complete Pooling:   0.16
HBM:                0.06  ✓ BEST
```

**Especially dramatic for small creators:**

```
Small Creators (n < 100):
  No Pooling MSE:   0.28
  HBM MSE:          0.10  ← 65% improvement!

Large Creators (n > 500):
  No Pooling MSE:   0.008
  HBM MSE:          0.006  ← Similar (data dominates)
```

**HBM maintains proper coverage with narrower intervals:**

```
                Coverage    Avg CI Width
No Pooling:     95.0%       1.25
Complete Pool:  73.4%       1.26  ← Badly miscalibrated!
HBM:            96.8%       0.88  ← Best of both worlds
```

## 5. Test Individual Modules

```bash
# Test data generation
python src/data_generation.py

# Test frequentist baselines
python src/frequentist.py

# Test HBM (small example, ~30 seconds)
python src/hierarchical_model.py

# Test validation
python src/validation.py

# Test visualizations
python src/visualization.py
```

## 6. Customize Parameters

Edit `run_full_analysis.py` to change:

```python
# In generate_experiment_data():
n_genres=5,              # Number of content genres
n_creators_per_genre=100, # Creators per genre
sigma_creator=0.4,       # Within-genre variance
sigma_obs=2.0,           # Observation noise
seed=42                  # Random seed

# In fit_hierarchical_model():
draws=2000,              # Posterior samples per chain
tune=1000,               # Warmup steps
chains=4,                # Number of MCMC chains
target_accept=0.9        # Step size tuning (0.95 for fewer divergences)
```

## 7. Troubleshooting

### "Divergent transitions" warning

This means MCMC sampler had trouble exploring the posterior. Usually OK if:
- Number of divergences is small (< 5% of total draws)
- R-hat and ESS diagnostics pass
- Results look reasonable

**To fix:** Increase `target_accept` to 0.95 or 0.99 in `fit_hierarchical_model()`.

### Sampling is slow

Expected: 2-5 minutes for 500 creators.

If much longer:
- Check you're using summary statistics (already done)
- Reduce `draws` and `tune` for quick tests
- Use fewer chains for debugging (but 4+ for final results)

### Results don't match README

Random seed matters! Set `seed=42` everywhere for reproducibility.

## 8. Next Steps

1. **Read the notebook** (`notebooks/walkthrough.ipynb`) for full understanding
2. **Examine shrinkage plot** to see partial pooling in action
3. **Check INITIAL_PLAN.md** for extension ideas:
   - Non-normal outcomes (skewed revenue)
   - Crossed random effects (genre × segment)
   - Time-varying effects
   - Production deployment

## Need Help?

1. Check `README.md` for detailed documentation
2. Read `INITIAL_PLAN.md` for implementation details
3. Review PyMC docs: https://www.pymc.io/
4. Examine the code - it's well-commented!
