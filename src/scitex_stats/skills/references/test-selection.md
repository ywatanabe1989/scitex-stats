---
name: test-selection
description: Decision tree for choosing the right statistical test based on data type, sample design, and assumptions.
---

# Statistical Test Selection Guide

## Decision Tree

### Step 1: What type of data?

- **Continuous** (measurements, scores) → Step 2
- **Categorical** (counts, frequencies) → chi2, fisher, mcnemar, cochran_q
- **Ordinal** (ranks) → Use non-parametric tests

### Step 2: How many groups?

- **1 group** vs known value → ttest_1samp / ks_1samp
- **2 groups** → Step 3
- **3+ groups** → Step 4

### Step 3: Two groups — paired or independent?

- **Independent**: ttest_ind (normal) / mannwhitneyu (non-normal)
- **Paired**: ttest_rel (normal) / wilcoxon (non-normal)

### Step 4: Three or more groups — paired or independent?

- **Independent**: anova (normal) / kruskal (non-normal)
- **Paired/Repeated**: anova_rm (normal) / friedman (non-normal)
- **Two factors**: anova_2way

### Step 5: After significant omnibus test

- Post-hoc: tukey_hsd (equal variance) / games_howell (unequal)
- Correction: bonferroni (conservative) / fdr_bh (liberal)

## Automatic Recommendation

```python
import scitex_stats as ss

# Pass your data — it checks normality, sample size, design
ss.recommend_tests(g1, g2, g3)
# Returns: ranked list of applicable tests with reasons
```

## When to Use Non-Parametric

- Shapiro-Wilk p < 0.05 (non-normal distribution)
- Small sample size (n < 20 per group)
- Ordinal data or heavy outliers
- When in doubt, non-parametric is safer (slightly less power)
