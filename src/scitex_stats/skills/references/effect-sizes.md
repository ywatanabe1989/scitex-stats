---
name: effect-sizes
description: Effect size measures, interpretation thresholds, and when to use each type.
---

# Effect Size Reference

## Available Effect Sizes

| Measure | Test Context | Small | Medium | Large |
|---------|-------------|-------|--------|-------|
| Cohen's d | t-tests | 0.2 | 0.5 | 0.8 |
| Cliff's delta | Mann-Whitney | 0.15 | 0.33 | 0.47 |
| Eta squared | ANOVA | 0.01 | 0.06 | 0.14 |
| Epsilon squared | Kruskal-Wallis | 0.01 | 0.06 | 0.14 |
| Probability of superiority | Any 2-group | 0.56 | 0.64 | 0.71 |
| Pearson r | Correlation | 0.1 | 0.3 | 0.5 |
| Cramér's V | Chi-square | 0.1 | 0.3 | 0.5 |

## Usage

```python
from scitex_stats import effect_sizes

# Standalone calculation
d = effect_sizes.cohens_d(g1, g2)
delta = effect_sizes.cliffs_delta(g1, g2)
eta2 = effect_sizes.eta_squared(ss_effect, ss_total)

# Automatically included in test results
result = ss.test_ttest_ind(g1, g2)
print(result["effect_size"])  # Cohen's d included
```

## Reporting

Always report effect sizes alongside p-values. A significant p-value with a tiny effect size may not be practically meaningful.

```
"t(38) = 2.45, p = .018, d = 0.77 [medium-to-large effect]"
```
