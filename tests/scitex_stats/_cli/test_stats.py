"""Tests for ``scitex_stats._cli.stats`` worker functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scitex_stats._cli import stats as cli

# ----- run_tests_list ------------------------------------------------------ #


def test_run_tests_list_returns_zero_rc(capsys):
    # Arrange
    # Act
    rc = cli.run_tests_list(as_json=True)
    # Assert
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

def test_run_tests_list_returns_zero_payload(capsys):
    # Arrange
    rc = cli.run_tests_list(as_json=True)
    # Act
    payload = json.loads(capsys.readouterr().out)
    # Assert
    assert isinstance(payload, list) and len(payload) > 0


def test_run_tests_list_text_format_one_per_line(capsys):
    # Arrange
    cli.run_tests_list(as_json=False)
    # Act
    out = capsys.readouterr().out.strip().splitlines()
    # Assert
    assert len(out) > 0


# ----- run_format_pvalue --------------------------------------------------- #


def test_run_format_pvalue_significant_rc(capsys):
    # Arrange
    # Act
    rc = cli.run_format_pvalue(p=0.001)
    # Assert
    assert rc == 0
    out = capsys.readouterr().out.strip()

def test_run_format_pvalue_significant_case_2(capsys):
    # Arrange
    rc = cli.run_format_pvalue(p=0.001)
    # Act
    out = capsys.readouterr().out.strip()
    # Assert
    assert "*" in out


def test_run_format_pvalue_not_significant(capsys):
    # Arrange
    cli.run_format_pvalue(p=0.5)
    # Act
    out = capsys.readouterr().out.strip()
    # Assert
    assert "*" not in out


# ----- run_tests_describe with CSV ----------------------------------------- #


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "data.csv") -> str:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def test_run_tests_describe_csv_input_rc(tmp_path, capsys):
    # Arrange
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    # Act
    rc = cli.run_tests_describe(data=_write_csv(tmp_path, df), column="x")
    # Assert
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)

def test_run_tests_describe_csv_input_any_values_int_float(tmp_path, capsys):
    # Arrange
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    rc = cli.run_tests_describe(data=_write_csv(tmp_path, df), column="x")
    # Act
    payload = json.loads(capsys.readouterr().out)
    # Assert
    assert any(
        isinstance(v, (int, float)) and abs(v - 3.0) < 1e-9 for v in payload.values()
    )


def test_run_tests_describe_npy_input(tmp_path, capsys):
    # Arrange
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    p = tmp_path / "x.npy"
    np.save(p, arr)
    # Act
    rc = cli.run_tests_describe(data=str(p))
    # Assert
    assert rc == 0
    json.loads(capsys.readouterr().out)  # raises if malformed


def test_run_tests_describe_json_input(tmp_path, capsys):
    # Arrange
    p = tmp_path / "x.json"
    p.write_text(json.dumps([1.0, 2.0, 3.0]))
    # Act
    rc = cli.run_tests_describe(data=str(p))
    # Assert
    assert rc == 0


# ----- run_tests_execute --------------------------------------------------- #


def test_run_tests_execute_with_groups_rc(tmp_path, capsys):
    # Arrange
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "g1": rng.normal(0, 1, 30),
            "g2": rng.normal(2, 1, 30),
        }
    )
    # Act
    rc = cli.run_tests_execute(
        test_name="kruskal",
        data=_write_csv(tmp_path, df),
        groups="g1,g2",
    )
    # Assert
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    p = payload.get("pvalue", payload.get("p_value"))

def test_run_tests_execute_with_groups_case_2(tmp_path, capsys):
    # Arrange
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "g1": rng.normal(0, 1, 30),
            "g2": rng.normal(2, 1, 30),
        }
    )
    rc = cli.run_tests_execute(
        test_name="kruskal",
        data=_write_csv(tmp_path, df),
        groups="g1,g2",
    )
    payload = json.loads(capsys.readouterr().out)
    # Act
    p = payload.get("pvalue", payload.get("p_value"))
    # Assert
    assert p is not None

def test_run_tests_execute_with_groups_case_3(tmp_path, capsys):
    # Arrange
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "g1": rng.normal(0, 1, 30),
            "g2": rng.normal(2, 1, 30),
        }
    )
    rc = cli.run_tests_execute(
        test_name="kruskal",
        data=_write_csv(tmp_path, df),
        groups="g1,g2",
    )
    payload = json.loads(capsys.readouterr().out)
    # Act
    p = payload.get("pvalue", payload.get("p_value"))
    # Assert
    assert p < 0.05


def test_run_tests_execute_unknown_test_returns_nonzero_rc(tmp_path, capsys):
    # Arrange
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    # Act
    rc = cli.run_tests_execute(
        test_name="not_a_real_test", data=_write_csv(tmp_path, df), x="x"
    )
    # Assert
    assert rc == 1
    err = capsys.readouterr().err

def test_run_tests_execute_unknown_test_returns_nonzero_error_lower_err(tmp_path, capsys):
    # Arrange
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    rc = cli.run_tests_execute(
        test_name="not_a_real_test", data=_write_csv(tmp_path, df), x="x"
    )
    # Act
    err = capsys.readouterr().err
    # Assert
    assert "error" in err.lower()


# ----- _read_data internal helper ----------------------------------------- #


def test_read_data_unsupported_format_raises(tmp_path):
    # Arrange
    bogus = tmp_path / "data.bogus"
    # Act
    bogus.write_text("noop")
    # Assert
    with pytest.raises(ValueError):
        cli._read_data(str(bogus))


def test_read_data_csv_returns_dataframe(tmp_path):
    # Arrange
    df = pd.DataFrame({"x": [1, 2, 3]})
    # Act
    out = cli._read_data(_write_csv(tmp_path, df))
    # Assert
    assert out.equals(df)


def test_read_data_tsv(tmp_path):
    # Arrange
    p = tmp_path / "x.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n")
    # Act
    out = cli._read_data(str(p))
    # Assert
    assert list(out.columns) == ["a", "b"]


def test_select_column_one_column_dataframe_returns_array_ndarray(tmp_path):
    # Arrange
    df = pd.DataFrame({"x": [1, 2, 3]})
    # Act
    out = cli._select_column(df, name=None)
    # Assert
    assert isinstance(out, np.ndarray)

def test_select_column_one_column_dataframe_returns_array_list(tmp_path):
    # Arrange
    df = pd.DataFrame({"x": [1, 2, 3]})
    # Act
    out = cli._select_column(df, name=None)
    # Assert
    assert list(out) == [1, 2, 3]


def test_select_column_missing_raises(tmp_path):
    # Arrange
    # Act
    df = pd.DataFrame({"a": [1, 2]})
    # Assert
    with pytest.raises(SystemExit):
        cli._select_column(df, name="nonexistent")


# ----- run_tests_execute: --x/--y branches --------------------------------- #


def test_run_tests_execute_x_and_y_columns_rc(tmp_path, capsys):
    # Arrange
    p = tmp_path / "two.csv"
    rng_e = np.random.default_rng(0)
    pd.DataFrame(
        {"g1": rng_e.normal(0, 1, 30), "g2": rng_e.normal(0.5, 1, 30)}
    ).to_csv(p, index=False)
    rc = cli.run_tests_execute(
        test_name="ttest_ind", data=str(p), x="g1", y="g2", as_json=True
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0
    payload = json.loads(out)

def test_run_tests_execute_x_and_y_columns_value_payload_pvalue(tmp_path, capsys):
    # Arrange
    p = tmp_path / "two.csv"
    rng_e = np.random.default_rng(0)
    pd.DataFrame(
        {"g1": rng_e.normal(0, 1, 30), "g2": rng_e.normal(0.5, 1, 30)}
    ).to_csv(p, index=False)
    rc = cli.run_tests_execute(
        test_name="ttest_ind", data=str(p), x="g1", y="g2", as_json=True
    )
    out = capsys.readouterr().out
    # Act
    payload = json.loads(out)
    # Assert
    assert "p_value" in payload or "pvalue" in payload


def test_run_tests_execute_x_only_one_sample_rc(tmp_path, capsys):
    # Arrange
    p = tmp_path / "one.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(p, index=False)
    rc = cli.run_tests_execute(
        test_name="ttest_1samp",
        data=str(p),
        x="a",
        popmean=3.0,
        as_json=True,
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0
    payload = json.loads(out)

def test_run_tests_execute_x_only_one_sample_value_payload_pvalue(tmp_path, capsys):
    # Arrange
    p = tmp_path / "one.csv"
    pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(p, index=False)
    rc = cli.run_tests_execute(
        test_name="ttest_1samp",
        data=str(p),
        x="a",
        popmean=3.0,
        as_json=True,
    )
    out = capsys.readouterr().out
    # Act
    payload = json.loads(out)
    # Assert
    assert "p_value" in payload or "pvalue" in payload


def test_run_tests_execute_npy_data_fallback_rc(tmp_path, capsys):
    """No --x / --y / --groups → falls back to `df.to_numpy()` (or array)."""
    # Arrange
    p = tmp_path / "arr.npy"
    np.save(p, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    rc = cli.run_tests_execute(
        test_name="shapiro", data=str(p), as_json=True
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0
    payload = json.loads(out)

def test_run_tests_execute_npy_data_fallback_value_payload_pvalue(tmp_path, capsys):
    """No --x / --y / --groups → falls back to `df.to_numpy()` (or array)."""
    # Arrange
    p = tmp_path / "arr.npy"
    np.save(p, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    rc = cli.run_tests_execute(
        test_name="shapiro", data=str(p), as_json=True
    )
    out = capsys.readouterr().out
    # Act
    payload = json.loads(out)
    # Assert
    assert "p_value" in payload or "pvalue" in payload


# ----- run_tests_recommend ------------------------------------------------ #


def test_run_tests_recommend_json_rc(capsys):
    # Arrange
    rc = cli.run_tests_recommend(
        n_groups=2,
        sample_sizes="30,30",
        outcome="continuous",
        design="between",
        paired=False,
        top_k=3,
        as_json=True,
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0
    tests = json.loads(out)

def test_run_tests_recommend_json_list(capsys):
    # Arrange
    rc = cli.run_tests_recommend(
        n_groups=2,
        sample_sizes="30,30",
        outcome="continuous",
        design="between",
        paired=False,
        top_k=3,
        as_json=True,
    )
    out = capsys.readouterr().out
    # Act
    tests = json.loads(out)
    # Assert
    assert isinstance(tests, list) and 0 < len(tests) <= 3


def test_run_tests_recommend_plain_output_rc(capsys):
    # Arrange
    rc = cli.run_tests_recommend(
        n_groups=2,
        sample_sizes="25,25",
        outcome="continuous",
        design="between",
        paired=False,
        top_k=2,
        as_json=False,
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0

def test_run_tests_recommend_plain_output_strip(capsys):
    # Arrange
    rc = cli.run_tests_recommend(
        n_groups=2,
        sample_sizes="25,25",
        outcome="continuous",
        design="between",
        paired=False,
        top_k=2,
        as_json=False,
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert out.strip()


def test_run_tests_recommend_paired_forces_within_rc(capsys):
    """`paired=True` should override `design="between"` → "within"."""
    # Arrange
    rc = cli.run_tests_recommend(
        n_groups=2,
        sample_sizes="20,20",
        outcome="continuous",
        design="between",
        paired=True,
        top_k=3,
        as_json=True,
    )
    # Act
    out = capsys.readouterr().out
    # Assert
    assert rc == 0
    tests = json.loads(out)

def test_run_tests_recommend_paired_forces_within_ttest_ind(capsys):
    """`paired=True` should override `design="between"` → "within"."""
    # Arrange
    rc = cli.run_tests_recommend(
        n_groups=2,
        sample_sizes="20,20",
        outcome="continuous",
        design="between",
        paired=True,
        top_k=3,
        as_json=True,
    )
    out = capsys.readouterr().out
    # Act
    tests = json.loads(out)
    # Assert
    assert "ttest_ind" not in tests[:1]


# ----- _read_data: extra branches ----------------------------------------- #


def test_read_data_json_file(tmp_path):
    # Arrange
    p = tmp_path / "x.json"
    p.write_text(json.dumps([1, 2, 3]))
    # Act
    out = cli._read_data(str(p))
    # Assert
    assert np.array_equal(out, [1, 2, 3], equal_nan=True)


def test_read_data_stdin_path_reads_json(tmp_path):
    """Read stdin via a real temp file rather than monkeypatching the global.

    The no-mocks rule (PA-306) forbids ``monkeypatch.setattr("sys.stdin",
    ...)`` because it patches production process state. A temp-file-backed
    ``sys.stdin`` reassignment in a ``try/finally`` covers the same wire
    without leaving an after-effect on the test runner.
    """
    # Arrange
    import sys
    p = tmp_path / "stdin.json"
    p.write_text(json.dumps([7, 8, 9]))
    saved_stdin = sys.stdin
    f = open(p)
    sys.stdin = f
    # Act
    try:
        out = cli._read_data("-")
    finally:
        sys.stdin = saved_stdin
        f.close()
    # Assert
    assert np.array_equal(out, [7, 8, 9], equal_nan=True)


# ----- _emit -------------------------------------------------------------- #


def test_emit_plain_dict(capsys):
    # Arrange
    cli._emit({"k": 1, "v": [1, 2, 3]}, as_json=False)
    # Act
    out = capsys.readouterr().out
    # Assert
    assert "k" in out and "1" in out


def test_emit_plain_list(capsys):
    # Arrange
    cli._emit(["one", "two"], as_json=False)
    # Act
    out = capsys.readouterr().out
    # Assert
    assert "one" in out and "two" in out


# ----- _select_column ----------------------------------------------------- #


def test_select_column_pass_through_non_dataframe():
    # Arrange
    # Act
    arr = np.array([1, 2, 3])
    # Assert
    assert cli._select_column(arr, None) is arr


def test_select_column_named_lookup_2col_df():
    # Arrange
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    # Act
    out = cli._select_column(df, "b")
    # Assert
    assert np.array_equal(out, [3, 4], equal_nan=True)


def test_select_column_ambiguous_no_name_exits():
    # Arrange
    # Act
    df = pd.DataFrame({"a": [1], "b": [2]})
    # Assert
    with pytest.raises(SystemExit, match="specify --x"):
        cli._select_column(df, None)
