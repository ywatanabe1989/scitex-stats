"""Smoke test for the ``scitex_stats.tests.correlation`` subpackage."""


def test_correlation_subpackage_importable():
    from scitex_stats.tests import correlation  # noqa: F401

    assert correlation is not None
