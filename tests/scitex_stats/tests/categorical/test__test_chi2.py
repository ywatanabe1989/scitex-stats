"""Smoke test for the ``scitex_stats.tests.categorical`` subpackage."""


def test_categorical_subpackage_importable():
    from scitex_stats.tests import categorical  # noqa: F401

    assert categorical is not None
