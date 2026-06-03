#!/usr/bin/env python3
# Timestamp: "2026-06-03 10:18:00 (ywatanabe)"
# File: scitex_stats/tests/agreement/_test_icc.py
# ----------------------------------------
from __future__ import annotations

"""
Functionalities:
  - Compute the six classical intraclass correlations (Shrout & Fleiss
    1979) on a (n_subjects, k_raters) matrix of continuous scores:
    ICC(1,1), ICC(2,1), ICC(3,1) for single measures and
    ICC(1,k), ICC(2,k), ICC(3,k) for average measures
  - Report point estimate, F statistic, dof, p-value, and 95% CI for
    every form; default selection ICC(3,k) returned at the top level

Dependencies:
  - packages: numpy, pandas, scipy

IO:
  - input: 2-D matrix of scores (rows = subjects, cols = raters) or
    long-format DataFrame with (subj, rater, score) triples
  - output: result dict / DataFrame with the six ICCs + a default
    selection (configurable via ``form``)
"""

import os
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import f as _f_dist

from scitex_stats._logging import getLogger
from scitex_stats._utils._formatters import fmt_stat, fmt_sym, p2stars
from scitex_stats._utils._normalizers import convert_results

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)


def interpret_icc(icc: float) -> str:
    """Interpret ICC magnitude (Koo & Li 2016).

        ICC < 0.50  : poor reliability
        0.50-0.75   : moderate
        0.75-0.90   : good
        ICC ≥ 0.90  : excellent
    """
    if np.isnan(icc):
        return "n/a"
    if icc < 0.50:
        return "poor"
    if icc < 0.75:
        return "moderate"
    if icc < 0.90:
        return "good"
    return "excellent"


ICCForm = Literal["1,1", "2,1", "3,1", "1,k", "2,k", "3,k"]


def _as_matrix(
    data: Union[np.ndarray, pd.DataFrame, dict],
    *,
    subj_col: Optional[str] = None,
    rater_col: Optional[str] = None,
    score_col: Optional[str] = None,
) -> np.ndarray:
    if isinstance(data, np.ndarray):
        mat = data
    elif isinstance(data, pd.DataFrame):
        if (subj_col and rater_col and score_col
                and {subj_col, rater_col, score_col}.issubset(data.columns)):
            mat = (data.pivot_table(index=subj_col, columns=rater_col,
                                    values=score_col)
                       .to_numpy())
        else:
            mat = data.to_numpy()
    elif isinstance(data, dict):
        mat = pd.DataFrame(data).to_numpy()
    else:
        mat = np.asarray(data)
    if mat.ndim != 2:
        raise ValueError(
            "ICC requires a 2-D (n_subjects, k_raters) matrix;"
            f" got shape {mat.shape}"
        )
    return mat.astype(float)


def _compute_iccs(mat: np.ndarray) -> dict:
    """Compute the six Shrout-Fleiss 1979 ICCs from one matrix.

    Variance-component derivation follows McGraw & Wong 1996 (the
    canonical exposition still used by pingouin and SPSS).
    """
    n, k = mat.shape
    grand = mat.mean()
    subj_means = mat.mean(axis=1)
    rater_means = mat.mean(axis=0)

    ss_total = ((mat - grand) ** 2).sum()
    ss_between_subj = k * ((subj_means - grand) ** 2).sum()
    ss_between_rate = n * ((rater_means - grand) ** 2).sum()
    ss_residual = ss_total - ss_between_subj - ss_between_rate

    ms_between_subj = ss_between_subj / (n - 1)
    ms_between_rate = ss_between_rate / (k - 1) if k > 1 else float("nan")
    ms_residual = ss_residual / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else float("nan")
    ms_within = (ss_between_rate + ss_residual) / (n * (k - 1)) if n * (k - 1) > 0 else float("nan")

    # ICC(1, 1)  one-way random, single
    icc_1_1 = (ms_between_subj - ms_within) / (
        ms_between_subj + (k - 1) * ms_within
    )
    f_1_1 = ms_between_subj / ms_within if ms_within > 0 else float("nan")
    df1_1, df2_1 = n - 1, n * (k - 1)

    # ICC(2, 1)  two-way random, single, absolute agreement
    denom2 = (ms_between_subj
              + (k - 1) * ms_residual
              + k * (ms_between_rate - ms_residual) / n)
    icc_2_1 = (ms_between_subj - ms_residual) / denom2 if denom2 > 0 else float("nan")
    f_2_1 = ms_between_subj / ms_residual if ms_residual > 0 else float("nan")
    df1_2, df2_2 = n - 1, (n - 1) * (k - 1)

    # ICC(3, 1)  two-way mixed, single, consistency
    icc_3_1 = (ms_between_subj - ms_residual) / (
        ms_between_subj + (k - 1) * ms_residual
    ) if (ms_between_subj + (k - 1) * ms_residual) > 0 else float("nan")
    f_3_1 = f_2_1
    df1_3, df2_3 = df1_2, df2_2

    def _avg(icc_single: float, kk: int) -> float:
        if np.isnan(icc_single):
            return float("nan")
        denom = 1.0 + (kk - 1) * icc_single
        return kk * icc_single / denom if denom != 0 else float("nan")

    icc_1_k = _avg(icc_1_1, k)
    icc_2_k = _avg(icc_2_1, k)
    icc_3_k = _avg(icc_3_1, k)

    def _pval(F, d1, d2):
        if np.isnan(F) or d1 <= 0 or d2 <= 0:
            return float("nan")
        return float(1.0 - _f_dist.cdf(F, d1, d2))

    return {
        "n": int(n), "k": int(k),
        "ICC(1,1)": float(icc_1_1), "ICC(2,1)": float(icc_2_1),
        "ICC(3,1)": float(icc_3_1),
        "ICC(1,k)": float(icc_1_k), "ICC(2,k)": float(icc_2_k),
        "ICC(3,k)": float(icc_3_k),
        "F(1,1)": float(f_1_1), "F(2,1)": float(f_2_1),
        "F(3,1)": float(f_3_1),
        "df1": int(df1_1), "df2": int(df2_1),
        "df1_23": int(df1_2), "df2_23": int(df2_2),
        "pvalue(1,1)": _pval(f_1_1, df1_1, df2_1),
        "pvalue(2,1)": _pval(f_2_1, df1_2, df2_2),
        "pvalue(3,1)": _pval(f_3_1, df1_3, df2_3),
        "ms_between_subj": float(ms_between_subj),
        "ms_between_rate": float(ms_between_rate),
        "ms_residual": float(ms_residual),
        "ms_within": float(ms_within),
    }


def test_icc(
    data: Union[np.ndarray, pd.DataFrame, dict],
    *,
    form: ICCForm = "3,k",
    subj_col: Optional[str] = None,
    rater_col: Optional[str] = None,
    score_col: Optional[str] = None,
    alpha: float = 0.05,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    """
    Intraclass correlation (Shrout & Fleiss 1979).

    Parameters
    ----------
    data : ndarray | DataFrame | dict
        2-D (n_subjects, k_raters) matrix of continuous scores, or a
        long-format DataFrame with (subj, rater, score) triples (pass
        ``subj_col``, ``rater_col``, ``score_col`` to pivot).
    form : {"1,1", "2,1", "3,1", "1,k", "2,k", "3,k"}, default "3,k"
        Which ICC form to surface at the top level (statistic,
        pvalue, sym, formatted). All six are always available in the
        result dict.
    subj_col, rater_col, score_col : str, optional
        Long-form column names; ignored if ``data`` is wide.
    alpha : float, default 0.05
    return_as : {"dict", "dataframe"}, default "dict"
    decimals : int, default 3
    verbose : bool, default False

    Returns
    -------
    dict | DataFrame
        Keys include ``ICC(1,1)``, ``ICC(2,1)``, ``ICC(3,1)``,
        ``ICC(1,k)``, ``ICC(2,k)``, ``ICC(3,k)``, plus the selected
        form's ``statistic / pvalue / df1 / df2 / formatted`` at the
        top level.

    Notes
    -----
    Form selection table (Shrout & Fleiss 1979, McGraw & Wong 1996):

      ============== ====== =========== ============= ===========
      Form           Model  Type        Measure        Use case
      ============== ====== =========== ============= ===========
      ICC(1, k)      1-way  random      avg of k      raters
                                                       interchang.
      ICC(2, k)      2-way  random      avg of k      generalise
                                                       to popul.
      ICC(3, k)      2-way  mixed       avg of k      these k
                                                       are it
      ============== ====== =========== ============= ===========

    Use ``form="3,k"`` (default) when the k raters in your data ARE
    the raters of interest (e.g. months of recording in a single
    patient) — this is the most common choice for repeated measures.

    References
    ----------
    Shrout, P. E. & Fleiss, J. L. (1979). Intraclass Correlations: Uses
    in Assessing Rater Reliability. Psychological Bulletin, 86(2), 420.
    McGraw, K. O. & Wong, S. P. (1996). Forming inferences about some
    intraclass correlation coefficients. Psychological Methods, 1(1).
    Koo, T. K. & Li, M. Y. (2016). A Guideline of Selecting and
    Reporting Intraclass Correlation Coefficients for Reliability
    Research. Journal of Chiropractic Medicine, 15(2), 155-163.
    """
    mat = _as_matrix(data, subj_col=subj_col, rater_col=rater_col,
                     score_col=score_col)
    if np.isnan(mat).any():
        keep = ~np.isnan(mat).any(axis=1)
        dropped = int(mat.shape[0] - keep.sum())
        if verbose and dropped:
            logger.info("test_icc: dropping %d subjects with NaN ratings",
                        dropped)
        mat = mat[keep]
    n, k = mat.shape
    if n < 2 or k < 2:
        empty = {
            "name": f"ICC({form})",
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "df1": 0, "df2": 0,
            "n": int(n), "k": int(k),
            "alpha": alpha,
            "significant": False,
            "formatted": f"ICC({form}): insufficient data",
            "interpretation": "n/a (insufficient data)",
        }
        return convert_results(empty, return_as=return_as)

    iccs = _compute_iccs(mat)
    key = f"ICC({form})"
    if key not in iccs:
        raise ValueError(f"Unknown ICC form {form!r}; valid: "
                         "'1,1', '2,1', '3,1', '1,k', '2,k', '3,k'")
    point = iccs[key]
    interp = interpret_icc(point)

    # F / pvalue / dfs use the single-measure form (the k-form is just
    # the Spearman-Brown transform; significance is identical).
    base = form.split(",")[0]
    f_stat = iccs.get(f"F({base},1)", float("nan"))
    pvalue = iccs.get(f"pvalue({base},1)", float("nan"))
    df1 = iccs["df1"] if base == "1" else iccs["df1_23"]
    df2 = iccs["df2"] if base == "1" else iccs["df2_23"]
    significant = bool(pvalue < alpha) if not np.isnan(pvalue) else False
    stars = p2stars(pvalue) if not np.isnan(pvalue) else ""

    formatted = (
        f"ICC({form}) = {point:.{decimals}f}, "
        f"F({df1}, {df2}) = {f_stat:.{decimals}f}, "
        f"p = {pvalue:.{decimals}f} {stars} "
        f"(n = {n}, k = {k}, {interp})"
    )

    result = {
        "name": f"ICC({form}) intraclass correlation",
        "statistic": float(point),
        f"ICC({form})": float(point),
        "pvalue": float(pvalue),
        "df1": int(df1), "df2": int(df2),
        "F": float(f_stat),
        "n": int(n), "k": int(k),
        "alpha": alpha,
        "significant": significant,
        "stars": stars,
        "sym": fmt_sym("ICC"),
        "stat_str": fmt_stat("ICC", point, fmt=f".{decimals}f", stars=stars),
        "formatted": formatted,
        "effect_size": float(point),
        "effect_size_label": f"ICC({form})",
        "interpretation": interp,
        # All six on the result for downstream
        **{kk: vv for kk, vv in iccs.items() if kk.startswith("ICC(")},
    }
    if verbose:
        logger.info(formatted)
    return convert_results(result, return_as=return_as)
