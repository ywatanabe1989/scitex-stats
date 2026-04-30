#!/usr/bin/env python3
# File: src/scitex_stats/_cli/stats.py
"""CLI verbs for statistical operations — wraps the public Python API.

Layout (per scitex-dev `_skills/general/03_interface_02_cli/`):
    tests <verb>      Noun group with 4+ sibling verbs:
        list          List available test names
        run <name>    Run a named test on data
        describe      Compute descriptive statistics from data
        recommend     Recommend tests for a study design
    format-pvalue P   Compound leaf (only 1 pvalue verb — no group)
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
      - .tsv  → pandas.DataFrame (sep='\\t')
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
    """Print payload to stdout. JSON if as_json else readable."""
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
    """Register the `tests` noun group + `format-pvalue` leaf on the parent."""
    _register_tests_group(subparsers)
    _register_format_pvalue(subparsers)


# 2. `tests` noun group
# ----------------------------------------


def _register_tests_group(subparsers: argparse._SubParsersAction) -> None:
    """Register `scitex-stats tests <verb>` group."""
    tests_parser = subparsers.add_parser(
        "tests",
        help="Statistical tests — list / run / describe / recommend.",
        description=(
            "Noun group for the 23-test framework. Run `scitex-stats tests --help`\n"
            "to see all verbs."
        ),
    )
    tests_sub = tests_parser.add_subparsers(dest="tests_command", title="Verbs")
    _register_tests_list(tests_sub)
    _register_tests_execute(tests_sub)
    _register_tests_describe(tests_sub)
    _register_tests_recommend(tests_sub)
    tests_parser.set_defaults(func=lambda a: tests_parser.print_help() or 0)


def _register_tests_list(sp):
    p = sp.add_parser(
        "list",
        help="List all available statistical test names.",
        description=(
            "Print the canonical list of test names accepted by `tests execute`.\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats tests list\n"
            "  $ scitex-stats tests list --no-json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        default=True,
        help="Emit JSON list (default).",
    )
    p.add_argument(
        "--no-json",
        dest="as_json",
        action="store_false",
        help="Plain text — one name per line.",
    )
    p.set_defaults(func=_run_tests_list)


def _run_tests_list(args: argparse.Namespace) -> int:
    import scitex_stats as ss

    tests = ss.available_tests()
    if args.as_json:
        print(json.dumps(tests, indent=2))
    else:
        for t in tests:
            print(t)
    return 0


def _register_tests_execute(sp):
    p = sp.add_parser(
        "execute",
        help="Execute a named statistical test on CSV/NPY/JSON data.",
        description=(
            "Run any test listed by `tests list`.\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats tests execute ttest_ind data.csv --x group_a --y group_b\n"
            "  $ scitex-stats tests execute anova data.csv --groups col1,col2,col3\n"
            "  $ scitex-stats tests execute pearson data.csv --x x --y y\n"
            "  $ scitex-stats tests execute chi2 contingency.csv  (whole table)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("test_name", help="Test name (see `tests list`).")
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
        "--json",
        dest="as_json",
        action="store_true",
        default=True,
        help="JSON output (default).",
    )
    p.add_argument(
        "--no-json",
        dest="as_json",
        action="store_false",
        help="Plain key/value pairs.",
    )
    p.set_defaults(func=_run_tests_execute)


def _run_tests_execute(args: argparse.Namespace) -> int:
    import scitex_stats as ss

    df = _read_data(args.data)
    kwargs: dict[str, Any] = {
        "alternative": args.alternative,
        "popmean": args.popmean,
        "json_safe": True,
    }

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

    _emit(result, as_json=args.as_json)
    return 0


def _register_tests_describe(sp):
    p = sp.add_parser(
        "describe",
        help="Compute descriptive statistics from a CSV/NPY/JSON file.",
        description=(
            "Compute descriptive statistics (mean, std, min, max, quartiles, n).\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats tests describe data.csv -c group_a\n"
            "  $ scitex-stats tests describe data.npy --funcs mean,std,median\n"
            "  $ cat numbers.json | scitex-stats tests describe -"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--json",
        dest="as_json",
        action="store_true",
        default=True,
        help="JSON output (default).",
    )
    p.add_argument(
        "--no-json",
        dest="as_json",
        action="store_false",
        help="Plain key/value pairs.",
    )
    p.set_defaults(func=_run_tests_describe)


def _run_tests_describe(args: argparse.Namespace) -> int:
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
    _emit(payload, as_json=args.as_json)
    return 0


def _register_tests_recommend(sp):
    p = sp.add_parser(
        "recommend",
        help="Recommend statistical tests for a study design.",
        description=(
            "Print top-K recommended tests for a given (n_groups, sample_sizes,\n"
            "outcome_type, design) context.\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats tests recommend --n-groups 2 --sample-sizes 30,28 --outcome continuous\n"
            "  $ scitex-stats tests recommend --n-groups 3 --sample-sizes 20,20,20 --outcome continuous --paired"
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
        "--json",
        dest="as_json",
        action="store_true",
        default=True,
        help="JSON output (default).",
    )
    p.add_argument(
        "--no-json",
        dest="as_json",
        action="store_false",
        help="One name per line.",
    )
    p.set_defaults(func=_run_tests_recommend)


def _run_tests_recommend(args: argparse.Namespace) -> int:
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
    if args.as_json:
        print(json.dumps(tests, indent=2))
    else:
        for t in tests:
            print(t)
    return 0


# 3. `format-pvalue` compound leaf
# ----------------------------------------


def _register_format_pvalue(sp):
    p = sp.add_parser(
        "format-pvalue",
        help="Convert a p-value to significance stars.",
        description=(
            "Print significance stars for a p-value (e.g. 0.001 → '***').\n"
            "\n"
            "Example:\n"
            "  $ scitex-stats format-pvalue 0.001\n"
            "  $ scitex-stats format-pvalue 0.5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
