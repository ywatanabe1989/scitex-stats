"""Smoke test for the ``scitex_stats.io`` subpackage."""


def test_io_subpackage_importable():
    from scitex_stats import io  # noqa: F401

    assert io is not None
