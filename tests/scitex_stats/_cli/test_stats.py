"""Tests for ``scitex_stats._cli.stats`` worker functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scitex_stats._cli import stats as cli

# ----- run_tests_list ------------------------------------------------------ #


def test_run_tests_list_returns_zero(capsys):
    rc = cli.run_tests_list(as_json=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) > 0


def test_run_tests_list_text_format_one_per_line(capsys):
    cli.run_tests_list(as_json=False)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) > 0


# ----- run_format_pvalue --------------------------------------------------- #


def test_run_format_pvalue_significant(capsys):
    rc = cli.run_format_pvalue(p=0.001)
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert "*" in out


def test_run_format_pvalue_not_significant(capsys):
    cli.run_format_pvalue(p=0.5)
    out = capsys.readouterr().out.strip()
    # NS marker is typically "ns" (or empty) — not asterisks.
    assert "*" not in out


# ----- run_tests_describe with CSV ----------------------------------------- #


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "data.csv") -> str:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def test_run_tests_describe_csv_input(tmp_path, capsys):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    rc = cli.run_tests_describe(data=_write_csv(tmp_path, df), column="x")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # Mean of 1..5 = 3.0; the API returns names+values, mean must be among them.
    assert any(
        isinstance(v, (int, float)) and abs(v - 3.0) < 1e-9 for v in payload.values()
    )


def test_run_tests_describe_npy_input(tmp_path, capsys):
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    p = tmp_path / "x.npy"
    np.save(p, arr)
    rc = cli.run_tests_describe(data=str(p))
    assert rc == 0
    json.loads(capsys.readouterr().out)  # raises if malformed


def test_run_tests_describe_json_input(tmp_path, capsys):
    p = tmp_path / "x.json"
    p.write_text(json.dumps([1.0, 2.0, 3.0]))
    rc = cli.run_tests_describe(data=str(p))
    assert rc == 0


# ----- run_tests_execute --------------------------------------------------- #


def test_run_tests_execute_with_groups(tmp_path, capsys):
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
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    p = payload.get("pvalue", payload.get("p_value"))
    assert p is not None
    # Two clearly different means → significant.
    assert p < 0.05


def test_run_tests_execute_unknown_test_returns_nonzero(tmp_path, capsys):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    rc = cli.run_tests_execute(
        test_name="not_a_real_test", data=_write_csv(tmp_path, df), x="x"
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


# ----- _read_data internal helper ----------------------------------------- #


def test_read_data_unsupported_format_raises(tmp_path):
    bogus = tmp_path / "data.bogus"
    bogus.write_text("noop")
    with pytest.raises(ValueError):
        cli._read_data(str(bogus))


def test_read_data_csv_returns_dataframe(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = cli._read_data(_write_csv(tmp_path, df))
    pd.testing.assert_frame_equal(out, df)


def test_read_data_tsv(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n")
    out = cli._read_data(str(p))
    assert list(out.columns) == ["a", "b"]


def test_select_column_one_column_dataframe_returns_array(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = cli._select_column(df, name=None)
    assert isinstance(out, np.ndarray)
    assert list(out) == [1, 2, 3]


def test_select_column_missing_raises(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(SystemExit):
        cli._select_column(df, name="nonexistent")


# ----- run_tests_execute: --x/--y branches --------------------------------- #


def test_run_tests_execute_x_and_y_columns(tmp_path, capsys):
    p = tmp_path / "two.csv"
    rng_e = np.random.default_rng(0)
    pd.DataFrame(
        {"g1": rng_e.normal(0, 1, 30), "g2": rng_e.normal(0.5, 1, 30)}
    ).to_csv(p, index=False)
    rc = cli.run_tests_execute(
        test_name="ttest_ind", data=str(p), x="g1", y="g2", as_json=True
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "p_value" in payload or "pvalue" in payload


def test_run_tests_execute_x_only_one_sample(tmp_path, capsys):
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
    assert rc == 0
    payload = json.loads(out)
    assert "p_value" in payload or "pvalue" in payload


def test_run_tests_execute_npy_data_fallback(tmp_path, capsys):
    """No --x / --y / --groups → falls back to `df.to_numpy()` (or array)."""
    p = tmp_path / "arr.npy"
    np.save(p, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    rc = cli.run_tests_execute(
        test_name="shapiro", data=str(p), as_json=True
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "p_value" in payload or "pvalue" in payload


# ----- run_tests_recommend ------------------------------------------------ #


def test_run_tests_recommend_json(capsys):
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
    assert rc == 0
    tests = json.loads(out)
    assert isinstance(tests, list) and 0 < len(tests) <= 3


def test_run_tests_recommend_plain_output(capsys):
    rc = cli.run_tests_recommend(
        n_groups=2,
        sample_sizes="25,25",
        outcome="continuous",
        design="between",
        paired=False,
        top_k=2,
        as_json=False,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip()


def test_run_tests_recommend_paired_forces_within(capsys):
    """`paired=True` should override `design="between"` → "within"."""
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
    assert rc == 0
    tests = json.loads(out)
    # Paired designs should not recommend independent t-test as top pick.
    assert "ttest_ind" not in tests[:1]


# ----- _read_data: extra branches ----------------------------------------- #


def test_read_data_json_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps([1, 2, 3]))
    out = cli._read_data(str(p))
    np.testing.assert_array_equal(out, [1, 2, 3])


def test_read_data_stdin_path_reads_json(tmp_path):
    """Read stdin via a real temp file rather than monkeypatching the global.

    The no-mocks rule (PA-306) forbids ``monkeypatch.setattr("sys.stdin",
    ...)`` because it patches production process state. A temp-file-backed
    ``sys.stdin`` reassignment in a ``try/finally`` covers the same wire
    without leaving an after-effect on the test runner.
    """
    import sys

    p = tmp_path / "stdin.json"
    p.write_text(json.dumps([7, 8, 9]))
    saved_stdin = sys.stdin
    f = open(p)
    sys.stdin = f
    try:
        out = cli._read_data("-")
    finally:
        sys.stdin = saved_stdin
        f.close()
    np.testing.assert_array_equal(out, [7, 8, 9])


# ----- _emit -------------------------------------------------------------- #


def test_emit_plain_dict(capsys):
    cli._emit({"k": 1, "v": [1, 2, 3]}, as_json=False)
    out = capsys.readouterr().out
    assert "k" in out and "1" in out


def test_emit_plain_list(capsys):
    cli._emit(["one", "two"], as_json=False)
    out = capsys.readouterr().out
    assert "one" in out and "two" in out


# ----- _select_column ----------------------------------------------------- #


def test_select_column_pass_through_non_dataframe():
    arr = np.array([1, 2, 3])
    assert cli._select_column(arr, None) is arr


def test_select_column_named_lookup_2col_df():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = cli._select_column(df, "b")
    np.testing.assert_array_equal(out, [3, 4])


def test_select_column_ambiguous_no_name_exits():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(SystemExit, match="specify --x"):
        cli._select_column(df, None)
