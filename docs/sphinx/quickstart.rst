Quick Start
===========

Python API
----------

.. code-block:: python

   import scitex_stats as ss
   import numpy as np

   # Generate sample data
   rng = np.random.default_rng(42)
   group1 = rng.normal(0, 1, 30)
   group2 = rng.normal(0.5, 1, 30)

   # Step 1: Get test recommendation
   ctx = ss.StatContext(
       n_groups=2, sample_sizes=[30, 30],
       outcome_type="continuous", design="between", paired=False,
   )
   recs = ss.recommend_tests(ctx)
   print(recs)  # ['brunner_munzel', 'ttest_ind', 'mannwhitneyu']

   # Step 2: Run a test
   result = ss.run_test("ttest_ind", data=group1, data2=group2)

   # Step 3: APA-formatted output
   print(result["formatted"])
   # t = -2.923, p = 0.0049, Cohen's d = -0.755, **

   # Access individual fields
   print(f"H0: {result['H0']}")           # μ(x) = μ(y)
   print(f"Effect: {result['effect_size_interpretation']}")  # medium

Effect Sizes
-------------

.. code-block:: python

   from scitex_stats import effect_sizes
   d = effect_sizes.cohens_d(group1, group2)

Power Analysis
--------------

.. code-block:: python

   from scitex_stats import power
   n = power.sample_size_ttest(effect_size=0.5, alpha=0.05, power=0.8)

Multiple Comparison Correction
-------------------------------

.. code-block:: python

   from scitex_stats import correct
   corrected = correct.correct_fdr(results, alpha=0.05)

Post-hoc Tests
--------------

.. code-block:: python

   from scitex_stats import posthoc
   results = posthoc.posthoc_tukey(groups)

CLI
---

.. code-block:: bash

   # List available Python APIs
   scitex-stats list-python-apis -v

   # List MCP tools
   scitex-stats mcp list-tools -v

MCP Server
----------

.. code-block:: bash

   # Start the MCP server
   scitex-stats mcp start

   # Check health
   scitex-stats mcp doctor
