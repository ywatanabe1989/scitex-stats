#!/usr/bin/env python3
# File: src/scitex_stats/_cli/stats.py
"""CLI verbs for statistical operations — wraps the public Python API.

Subcommands registered:
    describe        - Compute descriptive statistics from a CSV / table
    list-tests      - Print all available statistical test names
    run-test        - Run a named statistical test on CSV data
    recommend-tests - Recommend tests for a (n_groups, sample_sizes, outcome_type) context
    format-pvalue   - Format a p-value into significance stars
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _read_data(path: str) -> "Any":
    """Read CSV / NumPy file / stdin JSON into a numpy / pandas object.

    Supported input formats (auto-detected from extension):
      - .csv  → pandas.DataFrame (uses pd.read_csv)
      - .npy  → numpy.ndarray
      - .json → numpy.ndarray (loads list-of-list as 2D, list as 1D)
      - "-"   → read JSON from stdin
    """
    import numpy as np

    if path == "-":
        return np.asarray(json.load(sys.stdin))
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".json"):
        with open(path) as f:
            return np.asarray(json.load(f))
    if path.endswith(".csv") or path.endswith(".tsv"):
        import pandas as pd

        sep = "\t" if path.endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(
        f"Unsupported data format: {path} (use .csv / .tsv / .npy / .json or '-' for stdin JSON)"
    )


def _select_column(df, name: "str | None"):
    """Pick a single column from a DataFrame, returning np.ndarray. Pass-through if not DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        return df
    if not isinstance(df, pd.DataFrame):
        return df
    if name is None:
        if df.shape[1] == 1:
            return df.iloc[:, 0].to_numpy()
        raise SystemExit(f"Error: data has {df.shape[1]} columns; specify --x / --y")
    if name not in df.columns:
        raise SystemExit(
            f"Error: column {name!r} not found. Available: {list(df.columns)}"
        )
    return df[name].to_numpy()


def _emit(payload: "dict | list", as_json: bool = True, indent: int = 2):
    """Print payload to stdout. JSON if as_json else readable table."""
    if as_json:
        print(json.dumps(payload, indent=indent, default=str))
    else:
        if isinstance(payload, list):
            for item in payload:
                print(item)
        else:
            for k, v in payload.items():
                print(f"  {k:25s} {v}")


# 1. Main entry — register on parent subparsers
# ----------------------------------------


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register all stats CLI verbs on the parent ArgumentParser."""
    _register_describe(subparsers)
    _register_list_tests(subparsers)
    _register_run_test(subparsers)
    _register_recommend_tests(subparsers)
    _register_format_pvalue(subparsers)


# 2. Per-verb registrations
# ----------------------------------------


def _register_describe(sp):
    p = sp.add_parser(
        "describe-table",
        help="Compute descriptive statistics from a CSV/NPY/JSON file.",
        description=(
            "Compute descriptive statistics (mean, std, min, max, quartiles, n).\n"
            "Reads CSV, NPY, JSON, or stdin JSON. Output is JSON by default."
        ),
    )
    p.add_argument(
        "data", help="Path to data (.csv/.tsv/.npy/.json) or '-' for stdin JSON."
    )
    p.add_argument(
        "--column",
        "-c",
        default=None,
        help="Column to describe (CSV only). Defaults to all numeric.",
    )
    p.add_argument(
        "--funcs",
        default=None,
        help="Comma-separated funcs to compute (e.g. 'mean,std,median').",
    )
    p.add_argument(
        "--no-json", action="store_true", help="Pretty-print instead of JSON."
    )
    p.set_defaults(func=_run_describe)


def _run_describe(args: argparse.Namespace) -> int:
    import scitex_stats as ss

    data = _read_data(args.data)
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame) and args.column:
            data = data[args.column].to_numpy()
        elif hasattr(data, "to_numpy"):
            data = data.to_numpy()
    except ImportError:
        pass

    if args.funcs:
        values, names = ss.describe(data, funcs=args.funcs.split(","))
    else:
        values, names = ss.describe(data)
    payload = dict(
        zip(names, [v.tolist() if hasattr(v, "tolist") else v for v in values])
    )
    _emit(payload, as_json=not args.no_json)
    return 0


def _register_list_tests(sp):
    p = sp.add_parser(
        "list-tests",
        help="List all available statistical test names.",
        description="Print the canonical list of test names accepted by `run-test`.",
    )
    p.add_argument(
        "--no-json", action="store_true", help="One name per line instead of JSON list."
    )
    p.set_defaults(func=_run_list_tests)


def _run_list_tests(args: argparse.Namespace) -> int:
    import scitex_stats as ss

    tests = ss.available_tests()
    if args.no_json:
        for t in tests:
            print(t)
    else:
        print(json.dumps(tests, indent=2))
    return 0


def _register_run_test(sp):
    p = sp.add_parser(
        "run-test",
        help="Run a named statistical test on CSV/NPY/JSON data.",
        description=(
            "Run any test from `list-tests`. Examples:\n"
            "  scitex-stats run-test ttest_ind data.csv --x group_a --y group_b\n"
            "  scitex-stats run-test anova data.csv --groups col1,col2,col3\n"
            "  scitex-stats run-test pearson data.csv --x x --y y\n"
            "  scitex-stats run-test chi2 contingency.csv  (whole table)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("test_name", help="Test name (see `list-tests`).")
    p.add_argument(
        "data", help="Path to data (.csv/.tsv/.npy/.json) or '-' for stdin JSON."
    )
    p.add_argument("--x", default=None, help="Column for the first sample.")
    p.add_argument("--y", default=None, help="Column for the second sample.")
    p.add_argument(
        "--groups",
        default=None,
        help="Comma-separated columns for K groups (anova/kruskal).",
    )
    p.add_argument(
        "--popmean",
        type=float,
        default=0.0,
        help="Population mean (for 1-sample tests).",
    )
    p.add_argument(
        "--alternative",
        default="two-sided",
        choices=["two-sided", "greater", "less"],
        help="Alternative hypothesis.",
    )
    p.add_argument(
        "--no-json", action="store_true", help="Pretty-print instead of JSON."
    )
    p.set_defaults(func=_run_run_test)


def _run_run_test(args: argparse.Namespace) -> int:
    import scitex_stats as ss

    df = _read_data(args.data)
    kwargs: dict[str, Any] = {
        "alternative": args.alternative,
        "popmean": args.popmean,
        "json_safe": True,
    }

    # Build the data argument(s) based on which flags were given
    if args.groups:
        cols = [c.strip() for c in args.groups.split(",")]
        try:
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                kwargs["groups"] = [df[c].to_numpy() for c in cols]
            else:
                raise SystemExit("--groups requires CSV / DataFrame input.")
        except ImportError:
            raise SystemExit("pandas required for --groups")
    elif args.x and args.y:
        kwargs["data"] = _select_column(df, args.x)
        kwargs["data2"] = _select_column(df, args.y)
    elif args.x:
        kwargs["data"] = _select_column(df, args.x)
    else:
        # whole table — caller used `chi2 contingency.csv` style
        try:
            import pandas as pd

            kwargs["data"] = df.to_numpy() if isinstance(df, pd.DataFrame) else df
        except ImportError:
            kwargs["data"] = df

    try:
        result = ss.run_test(args.test_name, **kwargs)
    except Exception as e:
        print(
            json.dumps({"error": str(e), "test": args.test_name}, indent=2),
            file=sys.stderr,
        )
        return 1

    _emit(result, as_json=not args.no_json)
    return 0


def _register_recommend_tests(sp):
    p = sp.add_parser(
        "recommend-tests",
        help="Recommend statistical tests for a study design.",
        description=(
            "Print top-K recommended tests given a study design.\n"
            "Examples:\n"
            "  scitex-stats recommend-tests --n-groups 2 --sample-sizes 30,28 --outcome continuous\n"
            "  scitex-stats recommend-tests --n-groups 3 --sample-sizes 20,20,20 --outcome continuous --paired"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n-groups", type=int, required=True, help="Number of groups (1, 2, K)."
    )
    p.add_argument(
        "--sample-sizes",
        required=True,
        help="Comma-separated per-group sample sizes (e.g. 30,28).",
    )
    p.add_argument(
        "--outcome",
        default="continuous",
        choices=["continuous", "ordinal", "binary", "categorical"],
        help="Outcome variable type.",
    )
    p.add_argument(
        "--design",
        default="between",
        choices=["between", "within", "mixed"],
        help="Experimental design.",
    )
    p.add_argument(
        "--paired",
        action="store_true",
        help="Paired/related samples (also sets --design=within).",
    )
    p.add_argument("--top-k", type=int, default=3, help="How many tests to return.")
    p.add_argument(
        "--no-json", action="store_true", help="One name per line instead of JSON."
    )
    p.set_defaults(func=_run_recommend_tests)


def _run_recommend_tests(args: argparse.Namespace) -> int:
    import scitex_stats as ss

    sizes = [int(s) for s in args.sample_sizes.split(",")]
    design = "within" if args.paired else args.design
    ctx = ss.StatContext(
        n_groups=args.n_groups,
        sample_sizes=sizes,
        outcome_type=args.outcome,
        design=design,
        paired=args.paired,
    )
    tests = ss.recommend_tests(ctx, top_k=args.top_k)
    if args.no_json:
        for t in tests:
            print(t)
    else:
        print(json.dumps(tests, indent=2))
    return 0


def _register_format_pvalue(sp):
    p = sp.add_parser(
        "format-pvalue",
        help="Convert a p-value to significance stars.",
        description="Print significance stars for a p-value (e.g. 0.001 → '***').",
    )
    p.add_argument("p", type=float, help="The p-value (0-1).")
    p.add_argument("--style", default=None, help="Style ID (default: built-in).")
    p.set_defaults(func=_run_format_pvalue)


def _run_format_pvalue(args: argparse.Namespace) -> int:
    import scitex_stats as ss

    stars = ss.p_to_stars(args.p, style=args.style)
    print(stars)
    return 0


# EOF
