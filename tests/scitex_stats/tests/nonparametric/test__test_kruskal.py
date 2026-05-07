"""Smoke test for the ``scitex_stats.tests.nonparametric`` subpackage."""


def test_nonparametric_subpackage_importable():
    from scitex_stats.tests import nonparametric  # noqa: F401

    assert nonparametric is not None
