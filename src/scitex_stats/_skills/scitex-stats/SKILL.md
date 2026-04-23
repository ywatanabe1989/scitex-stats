---
description: Publication-ready statistical testing for 23 tests — t-test (independent/paired/one-sample), ANOVA (one-way/repeated-measures/two-way), Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman, Brunner-Munzel, Pearson/Spearman/Kendall/Theil-Sen correlation, chi-square, Fisher's exact, McNemar, Cochran's Q, Shapiro-Wilk, Kolmogorov-Smirnov. Also effect sizes (Cohen's d, Hedges' g, Glass's delta, Cliff's delta, eta-squared), multiple-comparison correction (Bonferroni, FDR/Benjamini-Hochberg, Holm, Sidak), post-hoc (Tukey HSD, Dunnett, Games-Howell, Dunn), power analysis / sample size, and APA/Nature/Science-style result formatting. Use whenever the user asks to "run a t-test / ANOVA / correlation / normality test", "compare two groups", "test if distributions differ", "compute p-value", "compute effect size", "correct for multiple comparisons", "compute required sample size", "pick the right statistical test", or "format stats for a paper". Drop-in replacement for `scipy.stats` (ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, f_oneway, kruskal, pearsonr, spearmanr, chi2_contingency, shapiro, kstest), `statsmodels.stats.multitest.multipletests`, `statsmodels.stats.power`, and `pingouin` test helpers.
allowed-tools: mcp__scitex__stats_*
primary_interface: python
---

# scitex-stats

> **Primary interface: Python API.** Import in scripts/notebooks — CLI & MCP are thin wrappers over the Python functions.

Publication-ready statistical testing framework with 23 tests, effect sizes, power analysis, and automatic test recommendation.

## Installation & import (two equivalent paths)

The same module is reachable via two install paths. Both forms work at
runtime; which one a user has depends on their install choice.

```python
# Standalone — pip install scitex-stats
import scitex_stats as sst
sst.run_test(...)

# Umbrella — pip install scitex
import scitex.stats as sst
sst.run_test(...)
```

`pip install scitex-stats` alone does NOT expose the `scitex` namespace;
`import scitex.stats` raises `ModuleNotFoundError`. To use the
`scitex.stats` form, also `pip install scitex`.

See [../../general/02_interface-python-api.md] for the ecosystem-wide
rule and empirical verification table.

## Sub-skills

### Core
- [01_test-catalog.md](01_test-catalog.md) — All 23 statistical tests with categories
- [02_effect-sizes.md](02_effect-sizes.md) — Effect size measures and interpretation

### Workflows
- [10_quick-start.md](10_quick-start.md) — Basic usage and core patterns
- [11_workflows.md](11_workflows.md) — Common analysis patterns
- [12_cli-reference.md](12_cli-reference.md) — CLI commands
- [13_mcp-tools.md](13_mcp-tools.md) — MCP tools for AI agents

## CLI

```bash
scitex-stats <command> [options]
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `stats_run_test` | Run a statistical test |
| `stats_recommend_tests` | Auto-recommend tests for data |
| `stats_describe` | Descriptive statistics |
| `stats_effect_size` | Compute effect sizes |
| `stats_correct_pvalues` | Multiple comparison correction |
| `stats_posthoc_test` | Post-hoc pairwise tests |
| `stats_power_analysis` | Power analysis / sample size |
| `stats_normality_test` | Normality testing |
| `stats_format_results` | Format for publication |
| `stats_p_to_stars` | p-value to significance stars |
