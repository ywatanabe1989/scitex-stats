#!/usr/bin/env python3
# File: examples/02_test_recommendation.py
"""Automatic test recommendation example."""

import scitex_stats as ss


def main():
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

    print("Test Recommendations")
    print("=" * 40)
    for rank, test_name in enumerate(recs, 1):
        print(f"  {rank}. {test_name}")


if __name__ == "__main__":
    main()
