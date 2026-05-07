"""Smoke test for the ``scitex_stats._mcp`` subpackage."""


def test_mcp_subpackage_importable():
    from scitex_stats import _mcp  # noqa: F401

    assert _mcp is not None
