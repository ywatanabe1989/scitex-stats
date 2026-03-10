Quick Start
===========

Running a Statistical Test
--------------------------

.. code-block:: python

   import scitex_stats as ss
   import numpy as np

   # Generate sample data
   group1 = np.random.normal(0, 1, 30)
   group2 = np.random.normal(0.5, 1, 30)

   # Get recommendations
   recs = ss.recommend_tests(n_groups=2, paired=False)

   # Run a t-test
   result = ss.run_test("ttest_ind", data=[group1, group2])

   # APA-formatted output
   print(result["formatted"]["apa"])

CLI Usage
---------

.. code-block:: bash

   # List available APIs
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
