"""Tests for ``scitex_stats.io._bundle``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scitex_stats.io._bundle import (
    STATS_SCHEMA_SPEC,
    load_stats_bundle,
    save_stats_bundle,
    validate_stats_spec,
)

# ----- validate_stats_spec ------------------------------------------------- #


def test_empty_spec_is_valid():
    assert validate_stats_spec({}) == []


def test_minimal_valid_spec_with_only_schema():
    assert validate_stats_spec({"schema": STATS_SCHEMA_SPEC["name"]}) == []


def test_comparisons_must_be_list():
    errors = validate_stats_spec({"comparisons": {"not": "a list"}})
    assert any("must be a list" in e for e in errors)


def test_comparison_item_must_be_dict():
    errors = validate_stats_spec({"comparisons": ["not-a-dict", {"p_value": 0.04}]})
    assert any("must be a dictionary" in e for e in errors)


def test_p_value_out_of_range_flagged():
    errors = validate_stats_spec({"comparisons": [{"p_value": 1.5}]})
    assert any("between 0 and 1" in e for e in errors)


def test_p_value_must_be_numeric():
    errors = validate_stats_spec({"comparisons": [{"p_value": "low"}]})
    assert any("must be numeric" in e for e in errors)


def test_p_value_at_bounds_is_valid():
    assert validate_stats_spec({"comparisons": [{"p_value": 0.0}]}) == []
    assert validate_stats_spec({"comparisons": [{"p_value": 1.0}]}) == []


def test_effect_size_dict_value_must_be_numeric():
    errors = validate_stats_spec({"comparisons": [{"effect_size": {"value": "huge"}}]})
    assert any("effect_size.value" in e and "numeric" in e for e in errors)


def test_effect_size_scalar_must_be_numeric():
    errors = validate_stats_spec({"comparisons": [{"effect_size": "large"}]})
    assert any("must be numeric or dict" in e for e in errors)


def test_test_results_must_be_dict_or_list():
    errors = validate_stats_spec({"test_results": "summary"})
    assert any("dictionary or list" in e for e in errors)


def test_descriptive_must_be_dict():
    errors = validate_stats_spec({"descriptive": ["mean", "std"]})
    assert any("descriptive" in e and "dictionary" in e for e in errors)


def test_collects_multiple_errors():
    errors = validate_stats_spec(
        {
            "comparisons": [{"p_value": 1.5, "effect_size": "big"}],
            "descriptive": ["bad"],
        }
    )
    assert len(errors) >= 3


# ----- save_stats_bundle / load_stats_bundle ------------------------------- #


def test_save_then_load_round_trip(tmp_path: Path):
    spec = {
        "schema": STATS_SCHEMA_SPEC["name"],
        "comparisons": [{"p_value": 0.03, "effect_size": 0.42}],
    }
    save_stats_bundle({"spec": spec}, tmp_path)
    loaded = load_stats_bundle(tmp_path)
    assert loaded["spec"] == spec


def test_save_writes_stats_json(tmp_path: Path):
    save_stats_bundle({"spec": {"schema": "x"}}, tmp_path)
    assert (tmp_path / "stats.json").is_file()
    with open(tmp_path / "stats.json") as f:
        assert json.load(f) == {"schema": "x"}


def test_save_data_csv_round_trip(tmp_path: Path):
    df = pd.DataFrame({"group": ["a", "b", "c"], "value": [1.1, 2.2, 3.3]})
    save_stats_bundle({"spec": {}, "data": df}, tmp_path)
    loaded = load_stats_bundle(tmp_path)
    pd.testing.assert_frame_equal(loaded["data"].reset_index(drop=True), df)


def test_save_report_markdown(tmp_path: Path):
    save_stats_bundle({"spec": {}, "report": "# Results\n\nSignificant."}, tmp_path)
    assert (tmp_path / "report.md").is_file()
    assert "Significant" in (tmp_path / "report.md").read_text()


def test_load_missing_dir_returns_none_spec(tmp_path: Path):
    """Empty bundle dir → spec is None, no crash."""
    out = load_stats_bundle(tmp_path)
    assert out["spec"] is None


def test_save_creates_well_formed_json(tmp_path: Path):
    """Saved stats.json must be parseable by json.load."""
    save_stats_bundle({"spec": {"k": [1, 2, 3]}}, tmp_path)
    with open(tmp_path / "stats.json") as f:
        json.load(f)  # raises if malformed


def test_stats_schema_spec_has_required_metadata():
    assert "name" in STATS_SCHEMA_SPEC
    assert "version" in STATS_SCHEMA_SPEC
    assert "required_fields" in STATS_SCHEMA_SPEC
