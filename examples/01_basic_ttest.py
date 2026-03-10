#!/usr/bin/env python3
# File: examples/01_basic_ttest.py
"""Basic t-test example with publication-ready output."""

from pathlib import Path

import numpy as np

import scitex_stats as ss

OUT_DIR = Path(__file__).parent / "01_basic_ttest_out"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Generate sample data
    rng = np.random.default_rng(42)
    group1 = rng.normal(loc=0.0, scale=1.0, size=30)
    group2 = rng.normal(loc=0.5, scale=1.0, size=30)

    # Run independent t-test via dispatcher
    result = ss.run_test("ttest_ind", data=group1, data2=group2)

    lines = [
        "Independent t-test",
        "=" * 40,
        f"t-statistic: {result['statistic']:.4f}",
        f"p-value: {result['p_value']:.4f}",
        f"Effect size (Cohen's d): {result['effect_size']:.4f}",
        f"Formatted: {result['formatted']}",
    ]
    output = "\n".join(lines)
    print(output)

    (OUT_DIR / "results.txt").write_text(output + "\n")

    # Save full result as JSON
    import json

    json_safe = {k: (float(v) if hasattr(v, "item") else v) for k, v in result.items()}
    (OUT_DIR / "results.json").write_text(
        json.dumps(json_safe, indent=2, default=str) + "\n"
    )

    print(f"\nSaved to {OUT_DIR}/")
    print("  results.txt  — human-readable summary")
    print("  results.json — full unified result dict")


if __name__ == "__main__":
    main()
