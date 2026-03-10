#!/usr/bin/env python3
# File: examples/01_basic_ttest.py
"""Basic t-test example with publication-ready output."""

import numpy as np

import scitex_stats as ss


def main():
    # Generate sample data
    rng = np.random.default_rng(42)
    group1 = rng.normal(loc=0.0, scale=1.0, size=30)
    group2 = rng.normal(loc=0.5, scale=1.0, size=30)

    # Run independent t-test via dispatcher
    result = ss.run_test("ttest_ind", data=group1, data2=group2)

    print("Independent t-test")
    print("=" * 40)
    print(f"t-statistic: {result['statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {result['effect_size']:.4f}")
    print(f"Formatted: {result['formatted']}")


if __name__ == "__main__":
    main()
