---
name: scitex-stats
description: Statistical testing with automatic test selection, effect sizes, power analysis, and publication formatting. Use when performing hypothesis tests, comparing groups, or formatting statistical results for papers.
allowed-tools: mcp__scitex__stats_*
---

# Statistical Testing with scitex-stats

## Quick Start

```python
import scitex_stats as ss

# Run a test directly
result = ss.run_test("brunner_munzel", g1, g2)

# Don't know which test? Build a context, get recommendations
ctx = ss.StatContext(n_groups=2, sample_sizes=[30, 32], outcome_type="continuous", design="between")
ss.recommend_tests(ctx)  # → ['brunner_munzel', 'ttest_ind', 'mannwhitneyu']

# Descriptive statistics
ss.describe(data)
```

## Common Workflows

### "I have two independent groups to compare"

```python
# Non-parametric (default choice — robust to non-normality and unequal variance)
result = ss.test_brunner_munzel(g1, g2)

# Parametric (normal data, equal variance)
result = ss.test_ttest_ind(g1, g2)

# Other non-parametric alternatives
result = ss.test_mannwhitneyu(g1, g2)  # assumes equal variance
```

### "I have two paired groups"

```python
# Parametric (normal differences)
result = ss.test_ttest_rel(before, after)

# Non-parametric
result = ss.test_wilcoxon(before, after)
```

### "I have 3+ groups"

```python
# One-way ANOVA (parametric)
result = ss.test_anova(g1, g2, g3)

# Kruskal-Wallis (non-parametric)
result = ss.test_kruskal(g1, g2, g3)

# Repeated measures
result = ss.test_anova_rm(subj1, subj2, subj3)
result = ss.test_friedman(subj1, subj2, subj3)

# Two-way ANOVA
result = ss.test_anova_2way(data, factor_a="treatment", factor_b="sex", value="score")
```

### "I need correlation"

```python
result = ss.test_pearson(x, y)      # Linear, normal
result = ss.test_spearman(x, y)     # Monotonic, any distribution
result = ss.test_kendall(x, y)      # Ordinal or small N
result = ss.test_theilsen(x, y)     # Robust regression slope
```

### "I have categorical data"

```python
result = ss.test_chi2(table)         # Independence test
result = ss.test_fisher(table)       # Small expected counts
result = ss.test_mcnemar(table)      # Paired binary outcomes
result = ss.test_cochran_q(table)    # Repeated binary measures
```

### "Check normality first"

```python
result = ss.test_normality(data)     # Shapiro + visual
result = ss.test_shapiro(data)       # Shapiro-Wilk
result = ss.test_ks_1samp(data)      # Kolmogorov-Smirnov
result = ss.test_ks_2samp(g1, g2)    # Two-sample KS
```

## Output Formats

Every test returns a dict with consistent keys:

```python
result = ss.test_ttest_ind(g1, g2)
# Keys: statistic, p_value, effect_size, ci_lower, ci_upper,
#        method, n1, n2, stars, interpretation
```

### Format for publication

```python
# Via MCP
stats_format_results(result)  # → "t(38) = 2.45, p = .018, d = 0.77"

# Via Python
ss.auto.p_to_stars(0.003)  # → "**"
```

## Unified Dispatcher

```python
# Run any test by name
ss.run_test("ttest_ind", g1, g2)
ss.run_test("anova", g1, g2, g3)
ss.run_test("pearson", x, y)

# List all available tests
ss.available_tests()
```

## Post-hoc & Corrections

```python
from scitex_stats import posthoc, correct

# After significant ANOVA
posthoc.tukey_hsd(g1, g2, g3)
posthoc.games_howell(g1, g2, g3)

# Multiple comparison correction
correct.bonferroni(p_values)
correct.fdr_bh(p_values)
correct.holm(p_values)
```

## Power Analysis

```python
from scitex_stats import power

# How many subjects do I need?
power.sample_size(effect_size=0.5, alpha=0.05, power=0.8, test="ttest_ind")

# What power do I have?
power.achieved_power(effect_size=0.5, n=30, alpha=0.05, test="ttest_ind")
```

## CLI Commands

```bash
# Start MCP server
scitex-stats mcp run
scitex-stats mcp install            # Install in Claude Desktop

# Browse API
scitex-stats list-python-apis       # List all public APIs
scitex-stats introspect api scitex_stats -v  # With docstrings

# Documentation
scitex-stats docs --list            # List doc pages
scitex-stats docs --tldr            # Quick-start summary
scitex-stats docs --json            # JSON for LLM consumption

# Skills (self-describing)
scitex-stats skills                     # Show help (list/get)
scitex-stats skills list                # List skill pages
scitex-stats skills get                 # Show main SKILL.md
scitex-stats skills get test-selection  # Show a reference page
```

## MCP Tools (for AI agents)

| Tool | Purpose |
|------|---------|
| `stats_recommend_tests` | Suggest appropriate test for your data |
| `stats_run_test` | Execute a statistical test |
| `stats_describe` | Descriptive statistics |
| `stats_effect_size` | Calculate effect size |
| `stats_normality_test` | Check normality assumption |
| `stats_correct_pvalues` | Multiple comparison correction |
| `stats_posthoc_test` | Post-hoc pairwise comparisons |
| `stats_power_analysis` | Power / sample size calculation |
| `stats_format_results` | APA-formatted result string |
| `stats_p_to_stars` | p-value → significance stars |

## Test Selection Guide

| Scenario | Parametric | Non-parametric |
|----------|-----------|----------------|
| 2 independent groups | `ttest_ind` | `brunner_munzel` (preferred), `mannwhitneyu` |
| 2 paired groups | `ttest_rel` | `wilcoxon` |
| 1 sample vs value | `ttest_1samp` | `ks_1samp` |
| 3+ independent groups | `anova` | `kruskal` |
| 3+ paired groups | `anova_rm` | `friedman` |
| 2-factor design | `anova_2way` | — |
| Linear association | `pearson` | `spearman` |
| 2x2 contingency | `chi2` | `fisher` |

## Specific Topics

* **Test selection decision tree** [references/test-selection.md](references/test-selection.md)
* **Effect size interpretation** [references/effect-sizes.md](references/effect-sizes.md)
