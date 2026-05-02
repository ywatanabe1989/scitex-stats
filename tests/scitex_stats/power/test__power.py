"""Smoke test for the ``scitex_stats.power`` subpackage."""


def test_power_subpackage_importable():
    from scitex_stats import power  # noqa: F401

    assert power is not None
