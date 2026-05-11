# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # scitex_stats — Multiple Comparison Correction Quick Start
#
# Running many tests inflates the false-positive rate. `scitex_stats.correct` adjusts p-values across a family of comparisons — FDR (Benjamini-Hochberg), Bonferroni, Holm, and more.
#
# **What this notebook covers**
#
# 1. Build a family of pairwise comparison results.
# 2. Apply FDR (Benjamini-Hochberg) correction.
# 3. Read off the adjusted p-values and rejection decisions.
#
# Companion notebooks:
# - `01_basic_ttest.ipynb` — the unified result dict each entry comes from
# - `02_test_recommendation.ipynb` — picking the right test in the first place

# %%
import pandas as pd

from scitex_stats import correct

# %% [markdown]
# ## 1. A family of pairwise comparisons
#
# `correct_fdr` takes a list of result-dicts that each carry a `pvalue` key. The other keys (`var_x`, `var_y`, etc.) are preserved on the output so the corrected and original entries line up.

# %%
results = [
    {"pvalue": 0.010, "var_x": "A", "var_y": "B"},
    {"pvalue": 0.040, "var_x": "A", "var_y": "C"},
    {"pvalue": 0.030, "var_x": "A", "var_y": "D"},
    {"pvalue": 0.200, "var_x": "B", "var_y": "C"},
    {"pvalue": 0.005, "var_x": "B", "var_y": "D"},
    {"pvalue": 0.080, "var_x": "C", "var_y": "D"},
]
pd.DataFrame(results)

# %% [markdown]
# ## 2. Apply FDR (Benjamini-Hochberg)
#
# BH controls the **false discovery rate** — the expected proportion of false positives among the rejected hypotheses — at level α. It is the default for exploratory pairwise comparisons.

# %%
corrected = correct.correct_fdr(results, alpha=0.05, method="bh", verbose=False)
pd.DataFrame(corrected)

# %% [markdown]
# ## 3. Read the adjusted decisions
#
# Side-by-side comparison of original and adjusted p-values, with the rejection decision at α=0.05.

# %%
table = pd.DataFrame(
    {
        "comparison": [f"{o['var_x']} vs {o['var_y']}" for o in results],
        "p (original)": [o["pvalue"] for o in results],
        "p (adjusted)": [c["pvalue_adjusted"] for c in corrected],
        "rejected": [c["rejected"] for c in corrected],
    }
)
table

# %% [markdown]
# ## Where to next
#
# - **`01_basic_ttest.ipynb`** — the unified result dict that feeds these comparisons.
# - **`02_test_recommendation.ipynb`** — choosing the per-comparison test before correction.
# - `correct.correct_bonferroni`, `correct.correct_holm` — same interface, stricter family-wise control.
