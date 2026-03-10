Examples
========

All examples are available in the ``examples/`` directory and produce both
human-readable (``.txt``) and structured (``.json``) output.

Example 1: Basic t-test
------------------------

.. code-block:: python

   import scitex_stats as ss
   import numpy as np

   rng = np.random.default_rng(42)
   group1 = rng.normal(loc=0.0, scale=1.0, size=30)
   group2 = rng.normal(loc=0.5, scale=1.0, size=30)

   result = ss.run_test("ttest_ind", data=group1, data2=group2)
   print(result["formatted"])
   # t = -2.923, p = 0.0049, Cohen's d = -0.755, **

Output:

.. code-block:: text

   Independent t-test
   ========================================
   t-statistic: -2.9233
   p-value: 0.0049
   Effect size (Cohen's d): -0.7548
   Formatted: t = -2.923, p = 0.0049, Cohen's d = -0.755, **


Example 2: Automatic Test Recommendation
-----------------------------------------

.. code-block:: python

   import scitex_stats as ss

   ctx = ss.StatContext(
       n_groups=2,
       sample_sizes=[30, 32],
       outcome_type="continuous",
       design="between",
       paired=False,
       has_control_group=False,
       n_factors=1,
   )
   recs = ss.recommend_tests(ctx, top_k=5)
   print(recs)
   # ['brunner_munzel', 'ttest_ind', 'mannwhitneyu']

Output:

.. code-block:: text

   Test Recommendations
   ========================================
     1. brunner_munzel
     2. ttest_ind
     3. mannwhitneyu


Example 3: Multiple Comparison Correction
-------------------------------------------

.. code-block:: python

   from scitex_stats import correct

   results = [
       {"pvalue": 0.01, "var_x": "A", "var_y": "B"},
       {"pvalue": 0.04, "var_x": "A", "var_y": "C"},
       {"pvalue": 0.03, "var_x": "A", "var_y": "D"},
       {"pvalue": 0.20, "var_x": "B", "var_y": "C"},
       {"pvalue": 0.005, "var_x": "B", "var_y": "D"},
       {"pvalue": 0.08, "var_x": "C", "var_y": "D"},
   ]

   corrected = correct.correct_fdr(results, alpha=0.05, method="bh")

Output:

.. code-block:: text

   Multiple Comparison Correction (FDR-BH)
   ==================================================
   Comparison   Original   Adjusted   Rejected
   ---------------------------------------------
          AvB     0.0100     0.0300        Yes
          AvC     0.0400     0.0600         No
          AvD     0.0300     0.0600         No
          BvC     0.2000     0.2000         No
          BvD     0.0050     0.0300        Yes
          CvD     0.0800     0.0960         No
