"""Tests for the less-common `_call_test` branches in `_dispatch`.

The integration tests cover two-sample / paired / one-sample / group
dispatch implicitly. These tests target the four still-uncovered
branches at lines 171, 208, 213-223:

- tuple unpacking when the test function returns `(result, fig)`
- `_ONE_SAMPLE_MEAN` (`ttest_1samp` with `popmean`)
- `_STACKED_GROUP` (Friedman expects column-stacked data)
- `_CONTINGENCY` (chi2 / fisher, with both `groups=` and
  `data + data2` paths)
- the final `raise ValueError` for an unrouted test_name.
"""

from __future__ import annotations

import numpy as np
import pytest

from scitex_stats._dispatch import _call_test, run_test

# ----- tuple unpacking (line 171) -------------------------------------- #


def test_run_test_unwraps_tuple_result_when_test_returns_pair():
    """When `plot=True`, some test functions return `(result, fig)`.
    `run_test` should unwrap to the dict before normalising."""
    rng = np.random.default_rng(0)
    g1 = rng.normal(0, 1, 25)
    g2 = rng.normal(0.5, 1, 25)
    g3 = rng.normal(1.0, 1, 25)
    out = run_test("anova", groups=[g1, g2, g3], plot=True, json_safe=True)
    assert isinstance(out, dict)
    assert "p_value" in out or "pvalue" in out


# ----- _ONE_SAMPLE_MEAN (line 208) ------------------------------------- #


def test_run_test_ttest_1samp_with_popmean():
    rng = np.random.default_rng(0)
    out = run_test("ttest_1samp", data=rng.normal(3.0, 1, 50), popmean=3.0)
    assert isinstance(out, dict)
    assert "p_value" in out or "pvalue" in out


# ----- _STACKED_GROUP (Friedman, line 213-214) ------------------------- #


def test_run_test_friedman_stacks_groups_into_2d():
    """Friedman expects subjects x conditions; `_call_test` builds
    that via `np.column_stack(groups)`."""
    rng = np.random.default_rng(0)
    n_subj = 12
    cond1 = rng.normal(0, 1, n_subj)
    cond2 = cond1 + rng.normal(0.4, 0.3, n_subj)
    cond3 = cond1 + rng.normal(0.8, 0.3, n_subj)
    out = run_test("friedman", groups=[cond1, cond2, cond3])
    assert isinstance(out, dict)


# ----- _CONTINGENCY (lines 216-221) ------------------------------------ #


def test_run_test_chi2_with_groups_2d_table():
    out = run_test("chi2", groups=[[10, 20], [30, 40]])
    assert isinstance(out, dict)
    assert "p_value" in out or "pvalue" in out


def test_run_test_chi2_via_data_plus_data2_rows():
    """When `groups` is None, `_call_test` builds the table via
    `np.vstack([data, data2])`."""
    out = run_test("chi2", data=[10, 20], data2=[30, 40])
    assert isinstance(out, dict)


def test_run_test_fisher_with_groups_2x2():
    out = run_test("fisher", groups=[[2, 5], [8, 3]])
    assert isinstance(out, dict)


# ----- unknown dispatch (line 223) ------------------------------------- #


def test_call_test_raises_on_unrouted_test_name():
    """`_call_test` should raise when a test_name slips through
    `run_test`'s alias check but isn't in any category set."""

    def _stub(*a, **kw):
        return {"unused": True}

    with pytest.raises(ValueError, match="No dispatch rule"):
        _call_test(
            _stub,
            "_definitely_unrouted_",
            data=None,
            data2=None,
            groups=None,
            alternative="two-sided",
            plot=False,
            popmean=0.0,
            return_as="dict",
        )
