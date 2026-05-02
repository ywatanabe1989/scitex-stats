"""Smoke test for the ``scitex_stats.tests.normality`` subpackage."""


def test_normality_subpackage_importable():
    from scitex_stats.tests import normality  # noqa: F401

    assert normality is not None
