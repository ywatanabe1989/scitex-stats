# SciTeX Stats (<code>scitex-stats</code>)

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/scitex-logo-banner.png" alt="SciTeX Stats" width="400">
  </a>
</p>

<p align="center"><b>Publication-ready statistical testing with 23 tests, effect sizes, power analysis, and APA formatting</b></p>

<p align="center">
  <a href="https://scitex-stats.readthedocs.io/">Full Documentation</a> · <code>uv pip install scitex-stats[all]</code>
</p>

<!-- scitex-badges:start -->
<p align="center">
  <a href="https://pypi.org/project/scitex-stats/"><img src="https://img.shields.io/pypi/v/scitex-stats.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/scitex-stats/"><img src="https://img.shields.io/pypi/pyversions/scitex-stats.svg" alt="Python"></a>
  <a href="https://github.com/ywatanabe1989/scitex-stats/actions/workflows/test.yml"><img src="https://github.com/ywatanabe1989/scitex-stats/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/ywatanabe1989/scitex-stats/actions/workflows/install-test.yml"><img src="https://github.com/ywatanabe1989/scitex-stats/actions/workflows/install-test.yml/badge.svg" alt="Install Test"></a>
  <a href="https://codecov.io/gh/ywatanabe1989/scitex-stats"><img src="https://codecov.io/gh/ywatanabe1989/scitex-stats/graph/badge.svg" alt="Coverage"></a>
  <a href="https://scitex-stats.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/scitex-stats/badge/?version=latest" alt="Docs"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
</p>
<!-- scitex-badges:end -->

---

## Problem and Solution

| # | Problem | Solution |
|---|---------|----------|
| 1 | **Bare scipy returns `(statistic, p)`** -- effect size, CI, normality check, power each need manual follow-up calls | **Publication-ready** -- `stx.stats.run_test("ttest_ind", g1, g2, return_as="dataframe")` yields statistic + effect size (Cohen's d) + CI + normality + power in one DataFrame |
| 2 | **Test selection requires expertise** -- non-parametric vs parametric, paired vs independent, one-way vs repeated ANOVA | **Auto-recommend** -- `stx.stats.recommend_tests(data)` inspects distributions and suggests the right 2-3 tests |
| 3 | **APA formatting is manual** -- every paper spells out `t(58) = 2.34, p = .021, d = 0.60` by hand | **`format_results(style="apa")`** -- typed output strings in APA, MLA, or LaTeX directly from the result dataframe |

## Problem

Statistical testing in Python is fragmented across `scipy`, `statsmodels`, and `pingouin` — each with different interfaces and output conventions. Getting publication-ready results requires substantial manual work: computing effect sizes, running power analysis, formatting to APA or journal standards. AI agents face a further barrier: they cannot call Python libraries directly and need structured, tool-based access.

## Solution

scitex-stats provides a unified interface that covers the full statistical workflow:

- **23 statistical tests** with automatic recommendation based on data characteristics
- **Built-in effect sizes** (Cohen's d, Cliff's delta, eta squared), **power analysis**, and **APA-formatted output**
- **Four interfaces** — Python API, CLI, MCP server, and Skills — so human researchers and AI agents use the same engine

```mermaid
flowchart LR
    A[Raw Data] --> B{Recommend Test}
    B --> C[Run Test]
    C --> D[Effect Size]
    C --> E[Power Analysis]
    D --> F[APA Format]
    E --> F
    F --> G[Publication-Ready Result]

    style A fill:#4a90d9,stroke:#2c3e50,color:#fff
    style B fill:#f5a623,stroke:#2c3e50,color:#fff
    style C fill:#27ae60,stroke:#2c3e50,color:#fff
    style D fill:#8e44ad,stroke:#2c3e50,color:#fff
    style E fill:#8e44ad,stroke:#2c3e50,color:#fff
    style F fill:#e74c3c,stroke:#2c3e50,color:#fff
    style G fill:#2c3e50,stroke:#1a252f,color:#fff
```

*Figure 1. Statistical testing workflow. scitex-stats automates the full pipeline from raw data to publication-ready results: test recommendation based on data characteristics, test execution with effect size and power analysis, and APA-formatted output.*

Every test returns a **unified result dictionary** with consistent keys:

```json
{
  "test_method": "Student's t-test (independent)",
  "statistic": -3.210,
  "stat_symbol": "t",
  "alternative": "two-sided",
  "n_x": 30,
  "n_y": 30,
  "pvalue": 0.0022,
  "stars": "**",
  "alpha": 0.05,
  "significant": true,
  "effect_size": -0.829,
  "effect_size_metric": "Cohen's d",
  "effect_size_interpretation": "large",
  "power": 0.884,
  "H0": "μ(x) = μ(y)",
  "formatted": "t = -3.210, p = 0.0022, Cohen's d = -0.829, **"
}
```

*Table 3. Unified result format. All 23 tests return the same dictionary structure with test statistics, p-value, effect size with interpretation, statistical power, and APA-formatted string.*

## Architecture

```
scitex_stats/
├── _recommend.py        # StatContext + recommend_tests()
├── _run_test.py         # Unified test runner (23 tests)
├── effect_sizes/        # Cohen's d, Cliff's delta, eta², etc.
├── power/               # Power analysis + sample-size calculators
├── correct/             # FDR / Bonferroni / Holm corrections
├── posthoc/             # Tukey, Dunn, Nemenyi
├── format/              # APA / MLA / LaTeX / Nature formatters
├── _cli/                # Click group: scitex-stats ...
└── _mcp/                # MCP server: scitex-stats mcp start
```

```mermaid
flowchart TB
    Data[Raw arrays / DataFrame] --> Ctx[StatContext]
    Ctx --> Rec[recommend_tests]
    Rec --> Run[run_test]
    Run --> ES[effect_sizes]
    Run --> Pw[power]
    Run --> Res[Unified result dict]
    Res --> Corr[correct: FDR/Bonferroni]
    Res --> Post[posthoc: Tukey/Dunn]
    Res --> Fmt[format: APA/Nature/LaTeX]
    Fmt --> Pub[Publication-ready string]

    subgraph Surfaces ["Four surfaces — same engine"]
        Py[Python API]
        Cli[CLI]
        Mcp[MCP server]
        Sk[Skills]
    end
    Py -.-> Run
    Cli -.-> Run
    Mcp -.-> Run
    Sk -.-> Run

    style Pub fill:#27ae60,stroke:#2c3e50,color:#fff
    style Res fill:#4a90d9,stroke:#2c3e50,color:#fff
```

<p align="center"><sub><b>Figure 2.</b> Module + surface architecture. Every interface (Python, CLI, MCP, Skills) calls the same <code>run_test</code> engine; outputs are a unified dict that downstream formatters and corrections consume.</sub></p>

## Installation

Requires Python >= 3.10.

```bash
pip install scitex-stats

# With MCP server for AI agents
pip install scitex-stats[mcp]

# Everything
pip install scitex-stats[all]
```

## Quickstart

```python
import scitex_stats as ss

# Get test recommendation
ctx = ss.StatContext(n_groups=2, sample_sizes=[30, 30], outcome_type="continuous", design="between", paired=False)
recs = ss.recommend_tests(ctx)

# Run a test
result = ss.run_test("ttest_ind", data=group1, data2=group2)

# APA-formatted output
print(result["formatted"])
```

## Four Interfaces

<details open>
<summary><strong>Python API</strong></summary>

<br>

```python
import scitex_stats as ss

# Automatic test recommendation
ctx = ss.StatContext(n_groups=2, sample_sizes=[30, 30], outcome_type="continuous", design="between", paired=False)
recs = ss.recommend_tests(ctx)

# Run a test
result = ss.run_test("ttest_ind", data=group1, data2=group2)

# Effect sizes
from scitex_stats import effect_sizes
d = effect_sizes.cohens_d(group1, group2)

# Power analysis
from scitex_stats import power
n = power.sample_size_ttest(effect_size=0.5, alpha=0.05, power=0.8)

# Multiple comparison correction
from scitex_stats import correct
corrected = correct.correct_fdr(results)

# Post-hoc tests
from scitex_stats import posthoc
results = posthoc.posthoc_tukey(groups)
```

> **[Full API reference](https://scitex-stats.readthedocs.io/en/latest/api/scitex_stats.html)**

</details>

<details>
<summary><strong>CLI Commands</strong></summary>

<br>

```bash
scitex-stats --help-recursive                # Show all commands
scitex-stats list-python-apis                # List Python API tree
scitex-stats list-python-apis -v             # With docstrings
scitex-stats mcp list-tools                  # List MCP tools
scitex-stats mcp doctor                      # Check server health
scitex-stats mcp start                       # Start MCP server
```

> **[Full CLI reference](https://scitex-stats.readthedocs.io/en/latest/quickstart.html)**

</details>

<details>
<summary><strong>MCP Server — for AI Agents</strong></summary>

<br>

AI agents can run statistical tests and format publication-ready results autonomously.

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

*Table 1. MCP tools available for AI agent integration via `scitex-stats mcp start`.*

```bash
scitex-stats mcp start
```

> **[Full MCP specification](https://scitex-stats.readthedocs.io/en/latest/api/scitex_stats._mcp.html)**

</details>

<details>
<summary><strong>Skills — for AI Agent Discovery</strong></summary>

<br>

Skills provide workflow-oriented guides that AI agents query to discover capabilities and usage patterns.

```bash
scitex-stats skills list              # List available skill pages
scitex-stats skills get SKILL         # Show main skill page
scitex-dev skills export --package scitex-stats  # Export to Claude Code
```

| Skill | Content |
|-------|---------|
| `quick-start` | Basic usage and core patterns |
| `test-catalog` | All 23 statistical tests with categories |
| `effect-sizes` | Effect size measures and interpretation |
| `workflows` | Common analysis patterns |
| `cli-reference` | CLI commands |
| `mcp-tools` | MCP tools for AI agents |

</details>

## Demo

Three runnable examples ship under `examples/` — each one writes its outputs (CSV + JSON + figures) to a sibling `_out/` folder so GitHub viewers see real artefacts:

| Example | What it shows | Gallery |
|---------|---------------|---------|
| **`01_basic_ttest.py`** | Independent-samples t-test → APA-formatted result + box plot | <img src="docs/example_ttest_figure.png" alt="t-test demo" width="180"> |
| **`02_test_recommendation.py`** | `recommend_tests` selects the right test from a `StatContext` | see `examples/02_test_recommendation_out/results.txt` |
| **`03_multiple_comparison.py`** | Run-test → posthoc → FDR correction pipeline | see `examples/03_multiple_comparison_out/results.json` |

```mermaid
flowchart LR
    Data[Group 1 / Group 2 arrays] --> R[run_test 'ttest_ind']
    R --> Dict[Unified result dict]
    Dict --> APA[formatted: 't = -3.21, p = .002, **']
    Dict --> Box[scitex.plt box plot]
    Box --> Png[docs/example_ttest_figure.png]
    style APA fill:#27ae60,stroke:#2c3e50,color:#fff
    style Png fill:#27ae60,stroke:#2c3e50,color:#fff
```

<p align="center"><sub><b>Figure 3.</b> Demo flow. One <code>run_test</code> call yields APA strings <em>and</em> the data needed to draw the publication figure — both backed by the same unified result dict.</sub></p>

```bash
# Reproduce locally — outputs land in examples/01_basic_ttest_out/
python examples/01_basic_ttest.py
python examples/02_test_recommendation.py
python examples/03_multiple_comparison.py
```

## Choosing the Right Test

<p align="center">
  <img src="docs/decision_flowchart.png" alt="Statistical test decision flowchart" width="700">
</p>

*Figure 2. Decision flowchart for choosing a statistical test. Start with your data type, then follow the branches based on number of groups and study design. Brunner-Munzel is recommended as the default for two-group comparisons due to its robustness to unequal variances and non-normality.*

## Available Tests

| Category | Tests |
|----------|-------|
| **Parametric** | t-test (ind, paired, 1-sample), ANOVA (1-way, RM, 2-way) |
| **Nonparametric** | Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman, Brunner-Munzel |
| **Correlation** | Pearson, Spearman, Kendall, Theil-Sen |
| **Categorical** | Chi-squared, Fisher exact, McNemar, Cochran's Q |
| **Normality** | Shapiro-Wilk, Kolmogorov-Smirnov (1-sample, 2-sample) |

*Table 2. All 23 statistical tests organized by category.*

## Lint Rules

Detected by [scitex-linter](https://github.com/ywatanabe1989/scitex-linter) when this package is installed.

| Rule | Severity | Message |
|------|----------|---------|
| `STX-ST001` | warning | `scipy.stats.ttest_ind()` — use `stx.stats.ttest_ind()` for auto effect size + CI |
| `STX-ST002` | warning | `scipy.stats.mannwhitneyu()` — use `stx.stats.mannwhitneyu()` for auto effect size |
| `STX-ST003` | warning | `scipy.stats.pearsonr()` — use `stx.stats.pearsonr()` for auto CI + power |
| `STX-ST004` | warning | `scipy.stats.f_oneway()` — use `stx.stats.anova_oneway()` for post-hoc + effect sizes |
| `STX-ST005` | warning | `scipy.stats.wilcoxon()` — use `stx.stats.wilcoxon()` for auto effect size |
| `STX-ST006` | warning | `scipy.stats.kruskal()` — use `stx.stats.kruskal()` for post-hoc + effect sizes |

## Part of SciTeX

`scitex-stats` is part of [**SciTeX**](https://scitex.ai). Install via
the umbrella with `pip install scitex[stats]` to use as
`scitex.stats` (Python) or `scitex stats ...` (CLI).

```python
import scitex

@scitex.session
def main(CONFIG=scitex.INJECTED, plt=scitex.INJECTED):
    # Load data
    data = scitex.io.load("measurements.csv")

    # Run statistical test
    result = scitex.stats.run_test("ttest_ind", data=group1, data2=group2)
    scitex.io.save(result, "stats_result.csv")

    # Visualize with figrecipe (scitex.plt)
    fig, ax = scitex.plt.subplots()
    ax.plot_box([group1, group2], labels=["Control", "Treatment"])
    ax.set_xyt("Group", "Value", f"p = {result['pvalue']:.4f} {result['stars']}")
    scitex.io.save(fig, "comparison.png")  # Saves plot + CSV data

    return 0
```

<p align="center">
  <img src="docs/example_ttest_figure.png" alt="Example t-test visualization" width="450">
</p>

*Figure 3. Example output combining scitex.stats (statistical test) with scitex.plt (publication-ready figure). The box plot shows group comparison with individual data points, significance bracket, p-value, and effect size — all generated from the unified result dictionary.*

The ecosystem modules work together:

| Module | Package | Role |
|--------|---------|------|
| `scitex.stats` | [scitex-stats](https://github.com/ywatanabe1989/scitex-stats) | Statistical testing, effect sizes, power analysis |
| `scitex.plt` | [figrecipe](https://github.com/ywatanabe1989/figrecipe) | Publication-ready figures with auto CSV export |
| `scitex.io` | [scitex-io](https://github.com/ywatanabe1989/scitex-io) | Universal file I/O (30+ formats) |
| `scitex.clew` | [scitex-clew](https://github.com/ywatanabe1989/scitex-clew) | Reproducibility verification via hash DAGs |

The SciTeX system follows the Four Freedoms for Research below, inspired by [the Free Software Definition](https://www.gnu.org/philosophy/free-sw.en.html):

>Four Freedoms for Research
>
>0. The freedom to **run** your research anywhere — your machine, your terms.
>1. The freedom to **study** how every step works — from raw data to final manuscript.
>2. The freedom to **redistribute** your workflows, not just your papers.
>3. The freedom to **modify** any module and share improvements with the community.
>
>AGPL-3.0 — because we believe research infrastructure deserves the same freedoms as the software it runs on.

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
</p>

<!-- EOF -->
