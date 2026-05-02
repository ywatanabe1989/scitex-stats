"""Smoke test for the ``scitex_stats._mcp._handlers`` subpackage."""


def test_handlers_subpackage_importable():
    from scitex_stats._mcp import _handlers  # noqa: F401

    assert _handlers is not None
