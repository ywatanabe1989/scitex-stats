#!/usr/bin/env python3
# File: examples/02_test_recommendation.py
"""Automatic test recommendation example."""

from pathlib import Path

import scitex_stats as ss

OUT_DIR = Path(__file__).parent / "02_test_recommendation_out"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Create a statistical context for a two-group comparison
    ctx = ss.StatContext(
        n_groups=2,
        sample_sizes=[30, 32],
        outcome_type="continuous",
        design="between",
        paired=False,
        has_control_group=False,
        n_factors=1,
    )

    # Get recommended tests (returns list of test name strings)
    recs = ss.recommend_tests(ctx, top_k=5)

    lines = ["Test Recommendations", "=" * 40]
    for rank, test_name in enumerate(recs, 1):
        lines.append(f"  {rank}. {test_name}")
    output = "\n".join(lines)
    print(output)

    (OUT_DIR / "results.txt").write_text(output + "\n")

    # Save as JSON
    import json

    (OUT_DIR / "results.json").write_text(
        json.dumps({"context": str(ctx), "recommendations": recs}, indent=2) + "\n"
    )

    print(f"\nSaved to {OUT_DIR}/")
    print("  results.txt  — human-readable summary")
    print("  results.json — structured data")


if __name__ == "__main__":
    main()
