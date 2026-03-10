SciTeX Stats
============

Publication-ready statistical testing framework with 23 tests, effect sizes,
power analysis, and MCP server.

Part of the `SciTeX <https://scitex.ai>`_ ecosystem.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index

Installation
------------

.. code-block:: bash

   pip install scitex-stats

Quick Start
-----------

.. code-block:: python

   import scitex_stats as ss

   # Recommend tests
   recs = ss.recommend_tests(n_groups=2, paired=False)

   # Run a test
   result = ss.run_test("ttest_ind", data=[group1, group2])

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
