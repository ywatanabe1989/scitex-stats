#!/usr/bin/env python3
# File: examples/02_test_recommendation.py
"""Automatic test recommendation example."""

import scitex_stats as ss


def main():
    # Get recommendations for a two-group comparison
    recs = ss.recommend_tests(
        n_groups=2,
        paired=False,
        outcome_type="continuous",
    )

    print("Test Recommendations")
    print("=" * 40)
    for rec in recs:
        print(f"  {rec['test_name']}: {rec['reason']}")


if __name__ == "__main__":
    main()
