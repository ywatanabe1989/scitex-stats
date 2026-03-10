#!/usr/bin/env python3
# File: examples/03_multiple_comparison.py
"""Multiple comparison correction example."""

from scitex_stats import correct


def main():
    # Simulate p-values from multiple tests
    p_values = [0.01, 0.04, 0.03, 0.20, 0.005, 0.08]

    # Apply FDR correction (Benjamini-Hochberg)
    adjusted = correct.fdr_bh(p_values, alpha=0.05)

    print("Multiple Comparison Correction (FDR-BH)")
    print("=" * 50)
    print(f"{'Original':>10} {'Adjusted':>10} {'Significant':>12}")
    print("-" * 35)
    for orig, adj in zip(p_values, adjusted["adjusted_pvalues"]):
        sig = "Yes" if adj < 0.05 else "No"
        print(f"{orig:>10.4f} {adj:>10.4f} {sig:>12}")


if __name__ == "__main__":
    main()
