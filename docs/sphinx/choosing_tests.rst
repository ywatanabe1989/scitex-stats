Choosing the Right Test
=======================

One of the most common questions in statistics is *"Which test should I use?"*.
SciTeX Stats provides automatic test recommendation, but understanding the
decision logic helps you make informed choices.

Decision Flowchart
------------------

.. figure:: _static/decision_flowchart.png
   :alt: Statistical test decision flowchart
   :align: center
   :width: 90%

   **Figure 1.** Decision flowchart for choosing a statistical test. Start with
   your data type, then follow the branches based on your study design.

Step 1: What Type of Data?
--------------------------

.. list-table:: **Table 1.** Data type determines the family of tests available.
   :header-rows: 1
   :widths: 20 40 40

   * - Data Type
     - Description
     - Example
   * - **Categorical**
     - Counts or proportions in categories
     - Treatment success/failure, survey responses (Yes/No)
   * - **Continuous**
     - Measured values on a continuous scale
     - Blood pressure, reaction time, gene expression
   * - **Ordinal**
     - Ordered categories with no fixed distance
     - Likert scales, pain ratings, disease stages

Step 2: How Many Groups?
-------------------------

.. list-table:: **Table 2.** Number of groups determines whether to use two-sample or multi-sample tests.
   :header-rows: 1
   :widths: 15 25 30 30

   * - Groups
     - Design
     - Parametric
     - Nonparametric
   * - **1**
     - One-sample
     - One-sample t-test
     - Shapiro-Wilk (normality)
   * - **2**
     - Independent
     - Student's t-test
     - Brunner-Munzel*, Mann-Whitney U
   * - **2**
     - Paired
     - Paired t-test
     - Wilcoxon signed-rank
   * - **3+**
     - Independent
     - One-way ANOVA
     - Kruskal-Wallis
   * - **3+**
     - Paired/Repeated
     - Repeated-measures ANOVA
     - Friedman

\* *Brunner-Munzel is recommended as the default for two-group comparisons
because it does not assume equal variances or normality.*

Step 3: Check Assumptions
--------------------------

Before running a parametric test, verify:

1. **Normality** — Use ``ss.run_test("shapiro", data=x)``
2. **Equal variances** — Brunner-Munzel avoids this assumption entirely
3. **Independence** — Observations must be independent (unless paired design)

.. code-block:: python

   import scitex_stats as ss

   # Check normality
   norm_result = ss.run_test("shapiro", data=my_data)
   if norm_result["normal"]:
       print("Data appear normal → parametric tests OK")
   else:
       print("Data deviate from normality → use nonparametric tests")


Automatic Recommendation
-------------------------

Let SciTeX Stats decide for you:

.. code-block:: python

   import scitex_stats as ss

   ctx = ss.StatContext(
       n_groups=2,
       sample_sizes=[30, 32],
       outcome_type="continuous",
       design="between",
       paired=False,
   )
   recs = ss.recommend_tests(ctx, top_k=3)
   print(recs)
   # ['brunner_munzel', 'ttest_ind', 'mannwhitneyu']

The recommendation engine considers:

- Number of groups and sample sizes
- Outcome type (continuous, ordinal, categorical)
- Study design (between-subjects, within-subjects)
- Whether data are paired
- Sample size constraints (e.g., Fisher's exact for small contingency tables)

Correlation Tests
-----------------

For examining relationships between two variables:

.. list-table:: **Table 3.** Correlation tests by data type and assumptions.
   :header-rows: 1
   :widths: 20 30 50

   * - Test
     - Assumption
     - Use When
   * - **Pearson**
     - Linear relationship, normal data
     - Continuous data with suspected linear correlation
   * - **Spearman**
     - Monotonic relationship
     - Ordinal data, or continuous data with outliers
   * - **Kendall**
     - Monotonic relationship
     - Small samples, or many tied values

Categorical Tests
-----------------

For count data in contingency tables:

.. list-table:: **Table 4.** Categorical tests by design and sample size.
   :header-rows: 1
   :widths: 20 30 50

   * - Test
     - Design
     - Use When
   * - **Chi-squared**
     - Independent groups
     - Expected frequencies >= 5 in most cells
   * - **Fisher's exact**
     - Independent, 2x2
     - Small expected frequencies (< 5)
   * - **McNemar**
     - Paired, 2x2
     - Before/after binary outcome
   * - **Cochran's Q**
     - Paired, 2+ conditions
     - Repeated binary measurements
