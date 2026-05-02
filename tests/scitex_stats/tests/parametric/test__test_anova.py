"""Smoke test for the ``scitex_stats.tests.parametric`` subpackage."""


def test_parametric_subpackage_importable():
    from scitex_stats.tests import parametric  # noqa: F401

    assert parametric is not None
