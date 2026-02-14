# How to Rerun the Analysis

## Quick Method: One Command

```bash
# Activate virtual environment
source venv/bin/activate

# Run complete analysis pipeline
python run_full_analysis.py
```

This will:
- ✓ Generate synthetic data with current parameters
- ✓ Fit all three models (no-pooling, complete-pooling, HBM)
- ✓ Validate and compare results
- ✓ Generate all visualizations → `outputs/`
- ✓ Save result tables → `outputs/*.csv`

**Runtime**: 2-5 minutes (faster with fewer creators)

---

## Interactive Method: Jupyter Notebook

```bash
# Activate virtual environment
source venv/bin/activate

# Launch Jupyter
jupyter notebook notebooks/walkthrough.ipynb
```

Then:
1. Click "Kernel" → "Restart & Run All"
2. Or run cells individually for step-by-step exploration

---

## Changing Parameters

### Method 1: Edit the Source File

Open `src/data_generation.py` and modify the default parameters:

```python
def generate_experiment_data(
    n_genres: int = 10,              # ← Change this
    n_creators_per_genre: int = 20,  # ← Change this
    genre_mean: float = 0.5,
    genre_std: float = 0.2,          # ← Change this
    sigma_creator: float = 0.4,
    sigma_obs: float = 2.0,
    baseline_revenue: float = 5.0,
    seed: int = 42                   # ← Change for different data
```

Then run: `python run_full_analysis.py`

### Method 2: In Python/Notebook

```python
from src.data_generation import generate_experiment_data

# Generate with custom parameters
df, truth = generate_experiment_data(
    n_genres=8,
    n_creators_per_genre=50,
    genre_std=0.4,
    seed=123  # Different random seed
)

# Then run the rest of the pipeline...
```

---

## Running Individual Components

### 1. Just Generate and Explore Data

```bash
source venv/bin/activate
python src/data_generation.py
```

### 2. Just Fit HBM

```bash
source venv/bin/activate
python src/hierarchical_model.py
```

### 3. Just Validate

```bash
source venv/bin/activate
python src/validation.py
```

### 4. Just Visualize

```bash
source venv/bin/activate
python src/visualization.py
```

### 5. Coverage-Precision Trade-off Plot

```bash
source venv/bin/activate
python visualize_coverage_tradeoff.py
```

---

## Customizing the HBM

To change MCMC sampling parameters, edit `run_full_analysis.py`:

```python
idata = fit_hierarchical_model(
    summaries,
    n_genres=truth['n_genres'],
    draws=2000,        # ← More draws = better estimates (slower)
    tune=1000,         # ← More tuning = better convergence
    chains=4,          # ← More chains = better diagnostics
    target_accept=0.9  # ← Higher = fewer divergences (slower)
)
```

**Common adjustments**:
- Getting divergences? → Increase `target_accept=0.95`
- Fast test? → Reduce `draws=1000, tune=500, chains=2`
- Production? → Increase `draws=3000, tune=1500, chains=4`

---

## Output Files

After running, check these locations:

### Visualizations
```
outputs/
├── shrinkage_plot.png              # Main HBM visualization
├── mse_comparison.png              # Performance comparison
├── coverage_vs_width.png           # Precision-calibration trade-off
├── coverage_precision_tradeoff.png # Small creator examples
├── individual_creators.png         # Specific creator comparisons
├── genre_recovery.png              # Genre-level validation
├── posteriors.png                  # Hyperparameter recovery
└── trace_plots.png                 # MCMC diagnostics
```

### Data Tables
```
outputs/
├── comparison_results.csv  # All 200 creators, all methods
├── overall_metrics.csv     # Summary comparison table
└── genre_recovery.csv      # Genre-level validation
```

---

## Troubleshooting

### "ModuleNotFoundError"
Make sure virtual environment is activated:
```bash
source venv/bin/activate
```

### "Divergent transitions"
Increase target_accept in `run_full_analysis.py`:
```python
target_accept=0.95  # or 0.99
```

### Slow sampling
- Reduce draws/tune for testing
- Check you have current parameters (fewer creators = faster)
- Expected: ~1 min per 100 creators with 4 chains

### Different results each run
Change the random seed:
```python
seed=42  # Use same seed for reproducibility
```

---

## Common Workflows

### Experiment with parameters
```bash
# 1. Edit src/data_generation.py
nano src/data_generation.py  # or use your editor

# 2. Rerun
python run_full_analysis.py

# 3. Compare with previous run
# Results saved in outputs/ (will be overwritten)
```

### Compare multiple configurations
```bash
# Save results between runs
cp -r outputs outputs_run1
python run_full_analysis.py

cp -r outputs outputs_run2
# Edit parameters
python run_full_analysis.py

# Now you have outputs_run1/ and outputs_run2/ to compare
```

### Production deployment
```bash
# Use more MCMC samples for final results
# Edit run_full_analysis.py:
#   draws=3000, tune=1500, target_accept=0.95

python run_full_analysis.py
```

---

## Quick Reference: Key Parameters

| Parameter | What it controls | Typical values |
|-----------|------------------|----------------|
| `n_genres` | Number of content genres | 5-15 |
| `n_creators_per_genre` | Creators per genre | 30-200 |
| `genre_std` | Between-genre variation | 0.2-0.5 |
| `sigma_creator` | Within-genre variation | 0.3-0.6 |
| `sigma_obs` | User-level noise | 1.5-3.0 |
| `seed` | Random seed | Any integer |
| `draws` | MCMC samples per chain | 1000-3000 |
| `target_accept` | MCMC step size tuning | 0.9-0.99 |

---

## Getting Help

- **Detailed docs**: See `README.md`
- **Coverage explanation**: See `COVERAGE_ANALYSIS.md`
- **Parameter changes**: See `PARAMETER_COMPARISON.md`
- **Original plan**: See `INITIAL_PLAN.md`
- **Quick start**: See `QUICKSTART.md`
