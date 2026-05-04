---
description: |
  [TOPIC] Quick Start
  [DETAILS] smallest end-to-end ttest_ind example with effect size + power.
tags: [scitex-stats-quick-start, scitex-stats]
---


# Quick Start

## Smallest end-to-end example — independent two-sample t-test

```python
import numpy as np
import scitex_stats as sst

rng = np.random.default_rng(0)
group_a = rng.normal(loc=0.0, scale=1.0, size=30)
group_b = rng.normal(loc=0.5, scale=1.0, size=30)

result = sst.test_ttest_ind(group_a, group_b)
print(result)
# {'statistic': -2.13, 'p_value': 0.037, 'effect_size': -0.55,
#  'ci': (-1.07, -0.03), 'power': 0.61, 'n1': 30, 'n2': 30, ...}
```

## Same call via the unified dispatcher

```python
result = sst.run_test("ttest_ind", group_a, group_b)
```

`run_test` accepts the same kwargs as the underlying `test_*` function and
also supports `return_as={"dict","dataframe","latex","json"}`.

## Auto-recommend a test for your design

```python
sst.recommend_tests(
    n_groups=2,
    sample_sizes=[30, 30],
    outcome="continuous",
    design="between",
    top_k=3,
)
# → ['test_ttest_ind', 'test_mannwhitneyu', 'test_brunner_munzel']
```

## Same workflow from the CLI

```bash
scitex-stats tests execute ttest_ind data.csv --x group_a --y group_b
scitex-stats tests recommend --n-groups 2 --sample-sizes 30,30
```

See [03_python-api.md](03_python-api.md) for the full Python surface and
[04_cli-reference.md](04_cli-reference.md) for the full CLI surface.
