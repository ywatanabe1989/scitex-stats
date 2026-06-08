#!/usr/bin/env python3
# File: tests/scitex_stats/test__bundle_io.py

"""Tests for scitex_stats._bundle_io (stats ↔ SciTeX bundle I/O).

Mirrors ``src/scitex_stats/_bundle_io.py``. Two layers:

1. Integration (scitex-io present) — round-trips a run_test-shaped
   result dict through ``save_stats`` / ``load_stats`` and exercises
   ``test_result_to_stats`` directly. Skipped when scitex-io's bundle
   subpackage is unavailable.
2. Graceful degradation (scitex-io absent) — hides ``scitex_io`` from
   ``sys.modules`` (reversibly), reloads the module so its top-level
   bundle import fails, and asserts each public function raises the
   documented ImportError. No monkeypatch / mocker.

Each test covers exactly one observable contract.
"""

from __future__ import annotations

import importlib
import re
import sys

import pytest

# Integration layer requires scitex-io's bundle subpackage.
pytest.importorskip("scitex_io.bundle")

from scitex_stats import _bundle_io  # noqa: E402


# ---------------------------------------------------------------------------
# Shared result-dict builders
# ---------------------------------------------------------------------------
def _flat_result():
    """A legacy flat-format run_test-shaped result dict."""
    return {
        "name": "control vs treatment",
        "method": "t-test",
        "statistic": 2.5,
        "statistic_name": "t",
        "p_value": 0.003,
        "df": 18,
        "effect_size": 1.21,
        "ci95": [0.5, 1.8],
    }


def _nested_result():
    """A new nested-format result dict (as produced by test functions)."""
    return {
        "name": "nested comparison",
        "method": {"name": "t-test", "variant": "independent"},
        "results": {
            "statistic": 2.5,
            "statistic_name": "t",
            "p_value": 0.01,
            "df": 18,
            "significant": True,
            "alpha": 0.05,
            "effect_size": {
                "name": "d",
                "value": -0.83,
                "ci_lower": -1.4,
                "ci_upper": -0.2,
            },
        },
    }


# ---------------------------------------------------------------------------
# test_result_to_stats — flat format
# ---------------------------------------------------------------------------
def test_flat_result_produces_single_analysis():
    # Arrange
    result = _flat_result()

    # Act
    stats = _bundle_io.test_result_to_stats(result)

    # Assert
    assert len(stats.analyses) == 1


def test_flat_result_preserves_method_name():
    # Arrange
    result = _flat_result()

    # Act
    ad = _bundle_io.test_result_to_stats(result).analyses[0].to_dict()

    # Assert
    assert ad["method"]["name"] == "t-test"


def test_flat_result_preserves_p_value():
    # Arrange
    result = _flat_result()

    # Act
    ad = _bundle_io.test_result_to_stats(result).analyses[0].to_dict()

    # Assert
    assert ad["results"]["p_value"] == 0.003


def test_flat_result_preserves_effect_size():
    # Arrange
    result = _flat_result()

    # Act
    ad = _bundle_io.test_result_to_stats(result).analyses[0].to_dict()

    # Assert
    assert ad["results"]["effect_size"]["value"] == 1.21


def test_flat_result_records_comparison_name():
    # Arrange
    result = _flat_result()

    # Act
    ad = _bundle_io.test_result_to_stats(result).analyses[0].to_dict()

    # Assert
    assert ad["inputs"]["comparison_name"] == "control vs treatment"


# ---------------------------------------------------------------------------
# test_result_to_stats — nested format
# ---------------------------------------------------------------------------
def test_nested_result_preserves_method_variant():
    # Arrange
    result = _nested_result()

    # Act
    ad = _bundle_io.test_result_to_stats(result).analyses[0].to_dict()

    # Assert
    assert ad["method"]["variant"] == "independent"


def test_nested_result_preserves_p_value():
    # Arrange
    result = _nested_result()

    # Act
    ad = _bundle_io.test_result_to_stats(result).analyses[0].to_dict()

    # Assert
    assert ad["results"]["p_value"] == 0.01


def test_nested_result_preserves_effect_size():
    # Arrange
    result = _nested_result()

    # Act
    ad = _bundle_io.test_result_to_stats(result).analyses[0].to_dict()

    # Assert
    assert ad["results"]["effect_size"]["value"] == -0.83


# ---------------------------------------------------------------------------
# save_stats / load_stats round-trip
# ---------------------------------------------------------------------------
@pytest.fixture
def round_tripped(tmp_path):
    """Save a single flat result then load it back."""
    out = _bundle_io.save_stats([_flat_result()], tmp_path / "mystats")
    loaded = _bundle_io.load_stats(out)
    return out, loaded


def test_save_stats_creates_bundle_on_disk(round_tripped):
    # Arrange
    out, _ = round_tripped

    # Act
    exists = out.exists()

    # Assert
    assert exists


def test_round_trip_recovers_single_comparison(round_tripped):
    # Arrange
    _, loaded = round_tripped

    # Act
    n = len(loaded["comparisons"])

    # Assert
    assert n == 1


def test_round_trip_recovers_comparison_name(round_tripped):
    # Arrange
    _, loaded = round_tripped

    # Act
    name = loaded["comparisons"][0]["name"]

    # Assert
    assert name == "control vs treatment"


def test_round_trip_recovers_method(round_tripped):
    # Arrange
    _, loaded = round_tripped

    # Act
    method = loaded["comparisons"][0]["method"]

    # Assert
    assert method == "t-test"


def test_round_trip_recovers_p_value(round_tripped):
    # Arrange
    _, loaded = round_tripped

    # Act
    p_value = loaded["comparisons"][0]["p_value"]

    # Assert
    assert p_value == 0.003


def test_round_trip_recovers_effect_size(round_tripped):
    # Arrange
    _, loaded = round_tripped

    # Act
    es = loaded["comparisons"][0]["effect_size"]

    # Assert
    assert es == 1.21


def test_round_trip_recovers_ci95(round_tripped):
    # Arrange
    _, loaded = round_tripped

    # Act
    ci = loaded["comparisons"][0]["ci95"]

    # Assert
    assert ci == [0.5, 1.8]


def test_round_trip_computes_significance_band(round_tripped):
    # Arrange: p=0.003 falls in the 0.001 <= p < 0.01 band ("**").
    _, loaded = round_tripped

    # Act
    formatted = loaded["comparisons"][0]["formatted"]

    # Assert
    assert formatted == "**"


def test_round_trip_returns_metadata_dict(round_tripped):
    # Arrange
    _, loaded = round_tripped

    # Act
    metadata = loaded["metadata"]

    # Assert
    assert isinstance(metadata, dict)


def test_save_multiple_comparisons_round_trips_all(tmp_path):
    # Arrange
    out = _bundle_io.save_stats([_flat_result(), _nested_result()], tmp_path / "multi")

    # Act
    names = {c["name"] for c in _bundle_io.load_stats(out)["comparisons"]}

    # Assert
    assert names == {"control vs treatment", "nested comparison"}


def test_save_as_zip_uses_zip_suffix(tmp_path):
    # Arrange
    out = _bundle_io.save_stats([_flat_result()], tmp_path / "zipped", as_zip=True)

    # Act
    suffix = out.suffix

    # Assert
    assert suffix == ".zip"


def test_save_as_zip_round_trips(tmp_path):
    # Arrange
    out = _bundle_io.save_stats([_flat_result()], tmp_path / "zipped2", as_zip=True)

    # Act
    n = len(_bundle_io.load_stats(out)["comparisons"])

    # Assert
    assert n == 1


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------
def test_test_result_to_stats_reachable_from_package_root():
    # Arrange
    import scitex_stats

    # Act
    fn = getattr(scitex_stats, "test_result_to_stats", None)

    # Assert
    assert callable(fn)


def test_save_stats_reachable_from_package_root():
    # Arrange
    import scitex_stats

    # Act
    fn = getattr(scitex_stats, "save_stats", None)

    # Assert
    assert callable(fn)


def test_load_stats_reachable_from_package_root():
    # Arrange
    import scitex_stats

    # Act
    fn = getattr(scitex_stats, "load_stats", None)

    # Assert
    assert callable(fn)


def test_bundle_io_names_listed_in_dunder_all():
    # Arrange
    import scitex_stats

    names = {"test_result_to_stats", "save_stats", "load_stats"}

    # Act
    present = names.issubset(set(scitex_stats.__all__))

    # Assert
    assert present


# ---------------------------------------------------------------------------
# Graceful degradation (scitex-io absent)
# ---------------------------------------------------------------------------
_MSG = "scitex-io is required for stats bundle I/O; install scitex-stats[bundle]"


@pytest.fixture
def degraded_module():
    """Reload ``_bundle_io`` with ``scitex_io`` hidden, then restore.

    A ``None`` entry in ``sys.modules`` makes ``import scitex_io`` raise
    ImportError without touching the filesystem. Everything is restored in
    teardown so subsequent tests see the genuine bundle-backed module.
    """
    hidden = {
        name: sys.modules.get(name)
        for name in list(sys.modules)
        if name == "scitex_io" or name.startswith("scitex_io.")
    }
    hidden["scitex_stats._bundle_io"] = sys.modules.get("scitex_stats._bundle_io")
    try:
        for name in hidden:
            sys.modules.pop(name, None)
        sys.modules["scitex_io"] = None  # type: ignore[assignment]
        sys.modules["scitex_io.bundle"] = None  # type: ignore[assignment]
        yield importlib.import_module("scitex_stats._bundle_io")
    finally:
        for name in list(sys.modules):
            if name == "scitex_io" or name.startswith("scitex_io."):
                sys.modules.pop(name, None)
        sys.modules.pop("scitex_stats._bundle_io", None)
        for name, mod in hidden.items():
            if mod is not None:
                sys.modules[name] = mod
        importlib.import_module("scitex_stats._bundle_io")


def test_module_imports_cleanly_without_scitex_io(degraded_module):
    # Arrange
    module = degraded_module

    # Act
    available = module.BUNDLE_AVAILABLE

    # Assert
    assert available is False


def test_test_result_to_stats_raises_clear_importerror(degraded_module):
    # Arrange
    result = {"method": "t-test", "p_value": 0.5}

    # Act
    raises_ctx = pytest.raises(ImportError, match=re.escape(_MSG))

    # Assert
    with raises_ctx:
        degraded_module.test_result_to_stats(result)


def test_save_stats_raises_clear_importerror(degraded_module):
    # Arrange
    comparisons = [{"method": "t-test", "p_value": 0.5}]

    # Act
    raises_ctx = pytest.raises(ImportError, match=re.escape(_MSG))

    # Assert
    with raises_ctx:
        degraded_module.save_stats(comparisons, "x")


def test_load_stats_raises_clear_importerror(degraded_module):
    # Arrange
    path = "x"

    # Act
    raises_ctx = pytest.raises(ImportError, match=re.escape(_MSG))

    # Assert
    with raises_ctx:
        degraded_module.load_stats(path)


# EOF
