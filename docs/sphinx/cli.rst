CLI Reference
=============

SciTeX Stats provides a command-line interface via the ``scitex-stats`` command.

Overview
--------

.. code-block:: bash

   scitex-stats --help-recursive    # Show all commands and subcommands

Core Commands
-------------

.. code-block:: bash

   scitex-stats list-python-apis         # List all Python API functions
   scitex-stats list-python-apis -v      # With signatures
   scitex-stats list-python-apis -vv     # With descriptions

MCP Subcommands
---------------

.. code-block:: bash

   scitex-stats mcp start           # Start MCP server
   scitex-stats mcp doctor          # Check MCP dependencies
   scitex-stats mcp install         # Show installation instructions
   scitex-stats mcp list-tools      # List available MCP tools
   scitex-stats mcp list-tools -v   # With signatures
