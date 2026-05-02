"""Smoke test for the ``scitex_stats._cli`` subpackage."""


def test_cli_subpackage_importable():
    from scitex_stats import _cli  # noqa: F401

    assert _cli is not None
