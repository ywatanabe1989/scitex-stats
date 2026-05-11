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
# # scitex_stats — Test Recommendation Quick Start
#
# Picking the right statistical test is half the work: parametric vs non-parametric, paired vs independent, one-way vs repeated measures. `StatContext` describes the design, `recommend_tests` returns the appropriate candidates ranked by suitability.
#
# **What this notebook covers**
#
# 1. Describe an experimental design with `StatContext`.
# 2. Ask `recommend_tests` for the top candidates.
# 3. Run the top recommendation through `run_test`.
#
# Companion notebooks:
# - `01_basic_ttest.ipynb` — the unified result dict
# - `03_multiple_comparison.ipynb` — corrections over a family of p-values

# %%
import numpy as np
import pandas as pd

import scitex_stats as ss

ss.__version__

# %% [markdown]
# ## 1. Describe the design
#
# Two independent groups, continuous outcome, unpaired. `StatContext` does not see the data — it captures the **design**, which is what determines test eligibility.

# %%
ctx = ss.StatContext(
    n_groups=2,
    sample_sizes=[30, 32],
    outcome_type="continuous",
    design="between",
    paired=False,
    has_control_group=False,
    n_factors=1,
)
ctx

# %% [markdown]
# ## 2. Get recommendations
#
# `recommend_tests` returns up to `top_k` test names, ordered by suitability for the described design.

# %%
recs = ss.recommend_tests(ctx, top_k=5)
pd.DataFrame({"rank": range(1, len(recs) + 1), "test": recs})

# %% [markdown]
# ## 3. Run the top recommendation
#
# The recommendation is a plain string, so it plugs straight into `run_test`.

# %%
rng = np.random.default_rng(0)
g1 = rng.normal(0.0, 1.0, 30)
g2 = rng.normal(0.5, 1.0, 32)

result = ss.run_test(recs[0], data=g1, data2=g2)
print(f"chosen test        : {recs[0]}")
print(f"formatted result   : {result['formatted']}")

# %% [markdown]
# ## Where to next
#
# - **`01_basic_ttest.ipynb`** — read the full unified result dict.
# - **`03_multiple_comparison.ipynb`** — what to do once you have many p-values.
