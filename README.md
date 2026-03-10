<!-- File: README.md -->

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-banner.png" alt="SciTeX Stats" width="400">
  </a>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/scitex-stats"><img src="https://badge.fury.io/py/scitex-stats.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/scitex-stats/"><img src="https://img.shields.io/pypi/pyversions/scitex-stats.svg" alt="Python Versions"></a>
  <a href="https://scitex-stats.readthedocs.io/"><img src="https://readthedocs.org/projects/scitex-stats/badge/?version=latest" alt="Documentation"></a>
  <a href="https://github.com/ywatanabe1989/scitex-stats/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ywatanabe1989/scitex-stats" alt="License"></a>
</p>

<p align="center">
  <a href="https://scitex.ai">scitex.ai</a> · <a href="https://scitex-stats.readthedocs.io/">docs</a> · <code>pip install scitex-stats</code>
</p>

---

**Publication-ready statistical testing framework with 23 tests, effect sizes, power analysis, and APA formatting.**

Part of the [SciTeX](https://scitex.ai) ecosystem — empowers both human researchers and AI agents.

## Installation

```bash
pip install scitex-stats

# With plotting support
pip install scitex-stats[plot]

# With MCP server for AI agents
pip install scitex-stats[mcp]

# Everything
pip install scitex-stats[all]
```

## Three Interfaces

| Interface | For | Description |
|-----------|-----|-------------|
| **Python API** | Human researchers | `import scitex_stats as ss` |
| **CLI Commands** | Terminal users | `scitex-stats mcp list-tools` |
| **MCP Tools** | AI agents | 10 tools for Claude/GPT integration |

<details>
<summary><strong>Python API</strong></summary>

<br>

**Run Tests** — 23 statistical tests

```python
import scitex_stats as ss

# Automatic test recommendation
recs = ss.recommend_tests(n_groups=2, paired=False, outcome_type="continuous")

# Run a test
result = ss.run_test("ttest_ind", data=[group1, group2])

# Publication-ready formatting
print(result["formatted"]["apa"])
# t(48) = 2.31, p = .025, d = 0.65
```

**Effect Sizes** — Cohen's d, Cliff's delta, eta squared, and more

```python
from scitex_stats import effect_sizes

d = effect_sizes.cohens_d(group1, group2)
delta = effect_sizes.cliffs_delta(group1, group2)
```

**Power Analysis** — Sample size calculation

```python
from scitex_stats import power

n = power.sample_size_ttest(effect_size=0.5, alpha=0.05, power=0.8)
```

**Multiple Comparisons** — Bonferroni, FDR, Holm, Sidak

```python
from scitex_stats import correct

adjusted = correct.fdr_bh(p_values, alpha=0.05)
```

**Post-hoc Tests** — Tukey HSD, Dunnett, Games-Howell

```python
from scitex_stats import posthoc

results = posthoc.tukey_hsd(groups, group_names=["A", "B", "C"])
```

</details>

<details>
<summary><strong>CLI Commands</strong></summary>

<br>

```bash
scitex-stats --help                          # Show all commands
scitex-stats --help-recursive                # Show all commands recursively
scitex-stats -V                              # Show version

# Python API introspection
scitex-stats list-python-apis                # List all public APIs
scitex-stats list-python-apis -v             # With docstrings
scitex-stats list-python-apis -vv            # Full documentation
scitex-stats list-python-apis --json         # JSON output

# MCP server management
scitex-stats mcp list-tools                  # List all MCP tools
scitex-stats mcp list-tools -v               # With signatures
scitex-stats mcp list-tools -vv              # With descriptions
scitex-stats mcp list-tools -vvv             # Full documentation
scitex-stats mcp doctor                      # Check server health
scitex-stats mcp installation                # Show Claude Desktop config
scitex-stats mcp start                       # Start MCP server
```

</details>

<details>
<summary><strong>MCP Tools — 10 tools for AI Agents</strong></summary>

<br>

| Tool | Description |
|------|-------------|
| `recommend_tests` | Recommend appropriate tests based on data characteristics |
| `run_test` | Execute a statistical test on provided data |
| `format_results` | Format results in journal style (APA, Nature, etc.) |
| `power_analysis` | Calculate statistical power or required sample size |
| `correct_pvalues` | Apply multiple comparison correction |
| `describe` | Calculate descriptive statistics |
| `effect_size` | Calculate effect size between groups |
| `normality_test` | Test whether data follows normal distribution |
| `posthoc_test` | Run post-hoc pairwise comparisons |
| `p_to_stars` | Convert p-value to significance stars |

**Claude Desktop** (`~/.config/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scitex-stats": {
      "command": "scitex-stats",
      "args": ["mcp", "start"]
    }
  }
}
```

</details>

## Available Tests

| Category | Tests |
|----------|-------|
| **Parametric** | t-test (ind, paired, 1-sample), ANOVA (1-way, RM, 2-way) |
| **Nonparametric** | Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman, Brunner-Munzel |
| **Correlation** | Pearson, Spearman, Kendall, Theil-Sen |
| **Categorical** | Chi-squared, Fisher exact, McNemar, Cochran's Q |
| **Normality** | Shapiro-Wilk, Kolmogorov-Smirnov (1-sample, 2-sample) |

## Documentation

Full documentation at [scitex-stats.readthedocs.io](https://scitex-stats.readthedocs.io/).

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
  <br>
  AGPL-3.0
</p>

<!-- EOF -->
