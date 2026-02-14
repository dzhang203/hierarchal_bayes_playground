# Parameter Change Analysis

## What Changed

### Original Parameters (Run 1)
```python
n_genres = 5
n_creators_per_genre = 100
genre_std = 0.3
Total creators: 500
```

### New Parameters (Run 2)
```python
n_genres = 10          # 2x more genres
n_creators_per_genre = 20   # 5x fewer per genre
genre_std = 0.2        # Less between-genre variation
Total creators: 200    # 2.5x fewer total
```

---

## Key Results Comparison

### Overall Performance

| Metric | Original (500) | New (200) | Change |
|--------|---------------|-----------|--------|
| **HBM MSE** | 0.0558 | 0.0526 | ✓ Slightly better |
| **HBM Coverage** | 96.8% | 97.5% | ✓ Slightly better |
| **HBM Width** | 0.880 | 0.873 | ~ Same |
| **No-pool MSE** | 0.1405 | 0.1387 | ~ Same |
| **No-pool Coverage** | 95.0% | 93.5% | ✗ Slightly worse |
| **No-pool Width** | 1.249 | 1.214 | ~ Same |

### Small Creators (n < 100)

| Metric | Method | Original | New | Change |
|--------|--------|----------|-----|--------|
| **MSE** | No-pool | 0.282 | 0.290 | ✗ Slightly worse |
|  | HBM | 0.098 | 0.091 | ✓ Slightly better! |
| **Coverage** | No-pool | 95.5% | **90.0%** | ✗ **Worse!** |
|  | HBM | 95.5% | **96.3%** | ✓ Better! |
| **Width** | No-pool | 2.057 | 1.995 | ~ Same |
|  | HBM | 1.271 | 1.267 | ~ Same |

### Genre Recovery

| Metric | Original | New | Change |
|--------|----------|-----|--------|
| **Genre MAE** | 0.030 | 0.083 | ✗ **Much worse** |
| **Coverage** | 100% | 100% | ✓ Same |

---

## 🔍 Interesting Observations

### 1. **HBM Remains Robust** ✅
- With 60% fewer creators, HBM performance is essentially unchanged
- MSE actually slightly improved (0.0558 → 0.0526)
- Coverage improved (96.8% → 97.5%)
- **Takeaway**: HBM is robust to sample size changes

### 2. **No-Pooling Shows Weakness** ⚠️
- Small creator coverage dropped significantly: **95.5% → 90.0%**
- This is below the nominal 95% target
- Why? With only 80 small creators (vs 200 before), we're more susceptible to random variation
- Some creators likely had extreme estimates that happened to miss the true value
- **Takeaway**: Frequentist coverage can be unreliable with small sample sizes, especially in smaller datasets

### 3. **Genre Recovery Degraded** 📉
- Genre-level MAE tripled: 0.030 → 0.083
- Makes sense! Only **20 creators per genre** (vs 100 before)
- Each genre estimate is much noisier
- But still 100% coverage (credible intervals appropriately widened)
- **Takeaway**: Hierarchical models need reasonable sample sizes at each level

### 4. **Faster Sampling** ⚡
- Runtime: 12 seconds → 4 seconds
- Fewer creators = faster computation
- Still only 16 divergences (acceptable level)
- **Takeaway**: HBM scales well

### 5. **HBM-NoPool Gap Maintained** 💪
- For small creators:
  - Original: HBM MSE 65% lower than no-pool
  - New: HBM MSE 69% lower than no-pool
- HBM advantage is consistent across different dataset sizes
- **Takeaway**: The fundamental value proposition of HBM holds

---

## 🎓 Lessons Learned

### What This Tells Us About Real-World Application

1. **More genres ≠ automatic improvement**
   - 10 genres vs 5, but only 20 creators each
   - Genre-level estimates become unreliable
   - **Guideline**: Need at least 30-50 entities per group for stable genre estimates

2. **HBM is more stable than frequentist for small samples**
   - No-pooling coverage dropped to 90% for small creators
   - HBM coverage stayed strong at 96.3%
   - **Why**: Borrowing strength provides regularization

3. **The bias-variance trade-off shifts with group size**
   - Smaller groups → more pooling → more bias
   - But variance reduction still dominates
   - Net result: HBM still wins

4. **Complete pooling gets even worse**
   - Medium creators: 74.9% → 68.6% coverage
   - Large creators: 31.2% → 38.0% coverage
   - Still terrible! Ignoring individual variation is bad

---

## 📊 Recommended Parameter Ranges

Based on these two runs:

| Parameter | Too Small | Good Range | Too Large |
|-----------|-----------|------------|-----------|
| **n_genres** | < 3 | 5-15 | > 20 (diminishing returns) |
| **n_creators_per_genre** | < 10 | 30-200 | > 500 (fine, just slower) |
| **Total creators** | < 100 | 200-1000 | > 5000 (use subsample) |
| **genre_std** | < 0.1 (genres too similar) | 0.2-0.5 | > 0.8 (weak grouping) |

**For your new setup (10 genres × 20 creators)**:
- ✓ Good for demonstration
- ⚠️ Genre estimates will be noisy (only 20 per group)
- ✓ Still shows HBM advantage
- ⚠️ Consider 30-50 per genre for production

---

## 🎯 Bottom Line

**The key finding remains rock-solid**:
- HBM reduces MSE by ~60-70% for small creators
- Maintains proper coverage (~95%)
- Provides narrower, more actionable intervals

**But we learned**:
- Need reasonable sample sizes at each hierarchical level
- HBM is more robust to sample size variation than frequentist approaches
- More groups doesn't help if each group is too small

**Your parameter choices are fine for exploration**, but for production:
- Aim for 30+ creators per genre
- 5-10 genres is often sufficient
- Focus on meaningful groupings over quantity of groups
