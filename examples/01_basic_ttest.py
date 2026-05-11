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
# # scitex_stats — Basic t-test Quick Start
#
# `scitex_stats.run_test` is a single dispatcher for the 23 supported tests. It returns a **unified result dictionary** — test statistic, p-value, effect size, power, and an APA-formatted string — so the same downstream code reads any test.
#
# **What this notebook covers**
#
# 1. Generate two synthetic groups with a known effect.
# 2. Run an independent-samples t-test through `run_test`.
# 3. Inspect the unified result dict.
#
# Companion notebooks:
# - `02_test_recommendation.ipynb` — let `recommend_tests` pick the test for you
# - `03_multiple_comparison.ipynb` — FDR / Bonferroni corrections

# %%
import numpy as np
import pandas as pd

import scitex_stats as ss

ss.__version__

# %% [markdown]
# ## 1. Synthetic two-group data
#
# Two groups of `n=30`, separated by 0.5 σ. Cohen's d ≈ 0.5 is the textbook "medium" effect — visible but not overwhelming.

# %%
rng = np.random.default_rng(42)
group1 = rng.normal(loc=0.0, scale=1.0, size=30)
group2 = rng.normal(loc=0.5, scale=1.0, size=30)

print(
    f"group1: mean={group1.mean():+.3f}, sd={group1.std(ddof=1):.3f}, n={len(group1)}"
)
print(
    f"group2: mean={group2.mean():+.3f}, sd={group2.std(ddof=1):.3f}, n={len(group2)}"
)

# %% [markdown]
# ## 2. Run the test
#
# `run_test("ttest_ind", data=..., data2=...)` runs an independent-samples Student's t-test.

# %%
result = ss.run_test("ttest_ind", data=group1, data2=group2)
pd.Series(result).to_frame("value")

# %% [markdown]
# ## 3. Read the unified result
#
# Every test returns the same keys, so the same downstream formatting code applies to any of the 23 tests.

# %%
print(f"t-statistic            : {result['statistic']:+.4f}")
print(f"p-value                : {result['p_value']:.4f}")
print(
    f"Cohen's d (effect)     : {result['effect_size']:+.4f}  ({result['effect_size_interpretation']})"
)
print(f"power                  : {result['power']:.3f}")
print(f"significant @ α=0.05   : {result['significant']}")
print()
print(f"formatted (APA-ready)  : {result['formatted']}")

# %% [markdown]
# ## Where to next
#
# - **`02_test_recommendation.ipynb`** — `StatContext` + `recommend_tests` for picking the right test.
# - **`03_multiple_comparison.ipynb`** — FDR / Bonferroni / Holm corrections over a family of p-values.
