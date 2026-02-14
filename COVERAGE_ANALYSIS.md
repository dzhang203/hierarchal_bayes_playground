# Understanding Coverage and Why HBM Helps Small Creators

## The Key Question: Is HBM Actually Better for Small Creators?

**Yes, definitively!** But not for the reason you might initially think.

### The Results for Small Creators (n < 100)

| Method | MSE | Coverage | Avg CI Width |
|--------|-----|----------|--------------|
| No Pooling | 0.282 | **95.5%** ✓ | 2.057 |
| HBM | 0.098 | **95.5%** ✓ | 1.271 |

**Key insight**: HBM achieves the **same coverage** with:
- **65% lower MSE** (more accurate point estimates)
- **38% narrower intervals** (more precise/useful)

This is the "free lunch" from hierarchical modeling: **maintained calibration with increased precision**.

---

## What Coverage Actually Means

### Frequentist Coverage (Confidence Intervals)

**Definition**: If we repeated this experiment many times and computed a 95% CI each time, **95% of those intervals** would contain the true parameter value.

**Critically**: This is a statement about the *procedure*, not any single interval:
- For any specific interval, the true value is either inside or outside (we don't know which)
- But the procedure is calibrated: if you use it repeatedly, you'll be right 95% of the time

**Example**: For a small creator with true effect = $0.50:
- We might get CI: [-$0.80, +$2.80] (very wide!)
- We don't know if $0.50 is in there
- But if we ran 100 such experiments, ~95 of the CIs would contain their respective true values

### Bayesian Coverage (Credible Intervals)

**Definition**: Given the data we observed, there's a **95% probability** that the true parameter lies in this interval.

**Critically**: This is a statement about *this specific interval*:
- It's a probability statement about the parameter (treated as random)
- More intuitive: "I'm 95% confident the true effect is between X and Y"

**In practice**: When the model is well-specified and properly calibrated, Bayesian credible intervals have good frequentist coverage too. That's what we see here!

---

## What It Means to "Improve Coverage"

This is a trick question! **HBM doesn't improve coverage** — both methods achieve ~95.5% coverage.

### What HBM Actually Improves: The Precision-Calibration Trade-off

For small creators:

**No Pooling (Frequentist)**:
- Intervals are **very wide** (±$2.06 on average)
- This is *necessary* to achieve 95% coverage given the noise
- The estimate is unbiased but high-variance
- **Problem**: Wide intervals are not actionable ("The effect is somewhere between -$1 and +$3" is useless!)

**HBM**:
- Intervals are **much narrower** (±$1.27 on average)
- But *still* achieve 95% coverage
- How? By shrinking the point estimate toward the genre mean
- The shrunken estimate is more stable, so we can be more precise while maintaining calibration

### The Magic Trick

```
No Pooling Small Creator Example:
  Raw estimate: +$2.00 (very noisy)
  SE: ±$1.05 (high uncertainty)
  95% CI: [-$0.10, +$4.10] (very wide to be safe)

  True effect: +$0.60 (inside the interval ✓)

HBM Same Creator:
  Shrunken estimate: +$0.80 (pulled toward genre mean of $0.50)
  Posterior SD: ±$0.65 (lower uncertainty from pooling)
  95% CI: [-$0.47, +$2.07] (much narrower!)

  True effect: +$0.60 (still inside ✓)
```

HBM "knows" that extreme estimates from small samples are likely noise, so it shrinks them. This shrinkage:
1. **Reduces MSE** (point estimates closer to truth on average)
2. **Reduces variance** (estimates are more stable)
3. **Allows narrower intervals** while maintaining coverage

---

## Why Is Low Coverage a Problem?

### Actually: Low Coverage ISN'T the Problem for Frequentist Small Samples!

Looking at the results:
- No Pooling (small creators): Coverage = **95.5%** ✓ Excellent!
- No Pooling (medium creators): Coverage = **91.4%** ✓ Pretty good
- No Pooling (large creators): Coverage = **99.2%** ✓ Excellent!

**The frequentist approach has good coverage across all sample sizes!**

### The Real Problem: Width, Not Coverage

For small creators with no-pooling:
- Coverage is fine (95.5%)
- But intervals are **too wide to be useful** (±$2.06)

**Example decision problem**:
- Should creator X invest in treatment?
- Frequentist: "The effect is between -$1.00 and +$3.00 with 95% confidence"
- Decision: ??? (Can't decide - includes both big losses and big gains!)

With HBM:
- Coverage is the same (95.5%)
- But interval is narrower (±$1.27)
- HBM: "The effect is between -$0.50 and +$1.50 with 95% credibility"
- Decision: Probably yes (likely positive, limited downside)

### When Low Coverage IS a Problem: Complete Pooling

Look at complete pooling for medium/large creators:

| Size | Coverage |
|------|----------|
| Small (n<100) | 98.5% (overconservative) |
| Medium (100-500) | **74.9%** ❌ |
| Large (>500) | **31.2%** ❌ |

**This is bad!** For large creators:
- We claim 95% confidence
- But only 31% of intervals contain the true value
- We're wildly overconfident — intervals are too narrow
- Why? Complete pooling assumes all creators in a genre are identical (false!)

**Example**:
- Large creator has true effect = +$1.50
- But their genre mean is +$0.50
- Complete pooling gives them the genre mean
- CI: [+$0.30, +$0.70] (doesn't contain +$1.50!)
- We're falsely confident in the wrong answer

---

## Why HBM Is Better for Small Creators: The Full Picture

### 1. Lower MSE (More Accurate)
- Small creator raw estimate: very noisy (MSE = 0.282)
- HBM borrows strength from genre: more stable (MSE = 0.098)
- **65% reduction in estimation error**

### 2. Maintained Coverage (Properly Calibrated)
- Both achieve ~95% coverage
- Your intervals are trustworthy for decision-making
- Not overconfident (like complete pooling) or underconfident

### 3. Narrower Intervals (More Precise/Actionable)
- No pooling needs wide intervals to cover the noise
- HBM has less variance after shrinkage → can be more precise
- **38% narrower** while maintaining calibration

### 4. Adaptive Shrinkage
- Small creators: shrunk a lot (noisy data → lean on prior)
- Large creators: shrunk a little (data dominates → trust the data)
- Automatic! No manual tuning of when to pool vs. not pool

---

## The Bias-Variance Trade-off

**No Pooling**:
- Unbiased (on average, estimates equal true values)
- High variance (individual estimates are very noisy)
- Low MSE requires both low bias AND low variance
- High variance → high MSE for small samples

**HBM**:
- Slightly biased (shrinkage pulls toward genre mean)
- Much lower variance (pooling reduces noise)
- The bias is small and **more than compensated** by variance reduction
- Net result: much lower MSE

**Mathematical fact**: MSE = Bias² + Variance

For small creators:
- No Pooling: MSE = 0² + 0.282 = 0.282
- HBM: MSE ≈ 0.02² + 0.094 ≈ 0.098 (variance reduction >> bias increase)

---

## Intuitive Summary

**For a small creator with 60 users**:

**No Pooling (Standard Approach)**:
- Treats them as completely unique
- Estimates their effect from their data alone
- Result: Very noisy estimate, needs wide CI to be safe
- Coverage: Good ✓
- Usefulness: Limited (too uncertain)

**Complete Pooling**:
- Treats them as identical to their genre
- Ignores their individual data
- Result: Stable but wrong estimate
- Coverage: Good for small, bad for different creators ❌
- Usefulness: Misleading (falsely confident)

**HBM (Partial Pooling)**:
- Treats them as similar but not identical to their genre
- Blends their data with genre information
- More weight on genre when their data is noisy
- Result: Stable estimate, narrower CI that's still calibrated
- Coverage: Good ✓
- Usefulness: High (precise and trustworthy) ✓

---

## The Bottom Line

HBM is better for small creators because it achieves what seems impossible:

✅ **More accurate** (lower MSE)
✅ **More precise** (narrower intervals)
✅ **Still calibrated** (proper coverage)

This isn't magic — it's **borrowing strength** from similar creators. The genre structure contains information about what effect to expect, and HBM uses that information optimally.

For large creators, HBM is about the same as no-pooling (data dominates prior). For small creators, HBM is dramatically better. **This is exactly what we want from a hierarchical model.**
